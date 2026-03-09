"""
FastAPI dependency injection for authentication and authorization.

Two dependencies:
  - get_current_user: decodes JWT → UserContext (who is making the request)
  - require_role:     dependency factory → guards routes by role
"""
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.jwt import decode_access_token

# Tells FastAPI where clients get tokens from.
# This powers the "Authorize" button in /docs (Swagger UI).
# tokenUrl must match your login endpoint exactly.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@dataclass
class UserContext:
    """
    Lightweight identity object extracted from the JWT.
    Passed into every route that requires authentication.
    No database call — everything comes from the token claims.
    """
    user_id: UUID
    organization_id: UUID   # the tenant — replaces hardcoded TENANT_ID
    role: str               # 'owner' | 'admin' | 'finance' | 'employee'


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> UserContext:
    """
    Core auth dependency. Injected into any route that requires login.

    FastAPI calls this automatically when a route declares:
        current_user: UserContext = Depends(get_current_user)

    Flow:
      1. OAuth2PasswordBearer extracts the Bearer token from Authorization header
      2. decode_access_token verifies signature + expiry (raises 401 if invalid)
      3. We build UserContext from the claims — zero DB lookup
    """
    payload = decode_access_token(token)

    return UserContext(
        user_id=UUID(payload["sub"]),
        organization_id=UUID(payload["org_id"]),
        role=payload["role"],
    )


def require_role(*roles: str):
    """
    Dependency factory for role-based access control (RBAC).

    Usage in a route:
        current_user = Depends(require_role("admin", "owner"))

    Why a factory (function returning a function)?
      - Lets you parameterize the dependency at definition time
      - Each call to require_role() creates a fresh dependency with its own allowed roles
      - FastAPI caches dependencies per request — this is safe and efficient
    """
    def _guard(current_user: UserContext = Depends(get_current_user)) -> UserContext:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not allowed. Required: {list(roles)}",
            )
        return current_user

    return _guard

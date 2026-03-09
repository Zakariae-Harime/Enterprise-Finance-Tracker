"""
Organization API Routes

Endpoints for managing the current user's organization:
  GET  /organizations/me         - Get org details + member count
  GET  /organizations/members    - List all members in the org
  POST /organizations/members/invite - Add a new member (owner/admin only)

All routes are tenant-scoped: every query filters by current_user.organization_id.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime
from typing import Optional

from src.auth.dependencies import get_current_user, require_role, UserContext
from src.api.dependencies import get_db_pool

router = APIRouter(
    prefix="/organizations",
    tags=["organizations"],
)


# ── Schemas (small enough to live here, not worth a separate file) ─────────

class OrganizationResponse(BaseModel):
    org_id: UUID
    name: str
    slug: str
    plan: str
    member_count: int
    created_at: datetime


class MemberResponse(BaseModel):
    user_id: UUID
    email: str
    full_name: Optional[str]
    role: str
    joined_at: datetime


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str  # 'admin' | 'finance' | 'employee'


class InviteMemberResponse(BaseModel):
    message: str
    user_id: UUID
    role: str


# GET /organizations/me 

@router.get(
    "/me",
    response_model=OrganizationResponse,
    summary="Get current organization details",
)
async def get_my_organization(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
) -> OrganizationResponse:
    """
    Returns the organization the authenticated user belongs to,
    plus a live member count (COUNT from organization_members).
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT o.id, o.name, o.slug, o.plan, o.created_at,
                   COUNT(om.id) AS member_count
            FROM organizations o
            JOIN organization_members om ON om.organization_id = o.id
            WHERE o.id = $1
            GROUP BY o.id
            """,
            current_user.organization_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Organization not found")

    return OrganizationResponse(
        org_id=row["id"],
        name=row["name"],
        slug=row["slug"],
        plan=row["plan"],
        member_count=row["member_count"],
        created_at=row["created_at"],
    )


#  GET /organizations/members 

@router.get(
    "/members",
    response_model=list[MemberResponse],
    summary="List all members in the organization",
)
async def list_members(
    current_user: UserContext = Depends(get_current_user),
    db_pool=Depends(get_db_pool),
) -> list[MemberResponse]:
    """
    Returns all users in the current tenant's organization.
    Joins organization_members with users to get email + full_name.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT u.id AS user_id, u.email, u.full_name,
                   om.role, om.created_at AS joined_at
            FROM organization_members om
            JOIN users u ON u.id = om.user_id
            WHERE om.organization_id = $1
            ORDER BY om.created_at ASC
            """,
            current_user.organization_id,
        )

    return [
        MemberResponse(
            user_id=row["user_id"],
            email=row["email"],
            full_name=row["full_name"],
            role=row["role"],
            joined_at=row["joined_at"],
        )
        for row in rows
    ]


# POST /organizations/members/invite 

@router.post(
    "/members/invite",
    response_model=InviteMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Invite an existing user to the organization",
)
async def invite_member(
    request: InviteMemberRequest,
    current_user: UserContext = Depends(require_role("owner", "admin")),
    db_pool=Depends(get_db_pool),
) -> InviteMemberResponse:
    """
    Adds an existing user (by email) to the current organization with a given role.

    Guards: only 'owner' or 'admin' can invite.
    The invited user must already have an account (registered via /auth/register).
    """
    VALID_ROLES = {"admin", "finance", "employee"}
    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role '{request.role}'. Must be one of: {sorted(VALID_ROLES)}",
        )

    async with db_pool.acquire() as conn:
        # Look up user by email
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            request.email,
        )
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"No user found with email '{request.email}'. They must register first.",
            )

        # Check not already a member
        existing = await conn.fetchrow(
            "SELECT id FROM organization_members WHERE organization_id = $1 AND user_id = $2",
            current_user.organization_id,
            user["id"],
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"User '{request.email}' is already a member of this organization.",
            )

        # Add to org
        await conn.execute(
            """
            INSERT INTO organization_members (organization_id, user_id, role)
            VALUES ($1, $2, $3)
            """,
            current_user.organization_id,
            user["id"],
            request.role,
        )

    return InviteMemberResponse(
        message=f"User '{request.email}' added as '{request.role}'",
        user_id=user["id"],
        role=request.role,
    )

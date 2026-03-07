"""
Pydantic V2 schemas for authentication endpoints.

Schemas define the shape of:
  - What the client sends (Request models)
  - What the server returns (Response models)

FastAPI uses these for:
  - Automatic input validation (wrong type → 422 before your code runs)
  - Auto-generated Swagger UI documentation
  - Response serialization (converts Python objects to JSON)
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
import re

# Request Models 

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, description="Min 8 chars, 1 uppercase, 1 digit, 1 special")
    full_name: str = Field(min_length=1, max_length=255)
    org_name: str = Field(min_length=1, max_length=255, description="Your company name")

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character")
        return v
    
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


# Response Models

class TokenResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900          # 15 minutes in seconds


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    email: str
    full_name: str | None
    organization_id: str
    role: str


class MessageResponse(BaseModel):
    message: str

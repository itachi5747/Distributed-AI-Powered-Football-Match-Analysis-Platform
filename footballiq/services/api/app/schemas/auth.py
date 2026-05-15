import re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)
    organisation: Optional[str] = Field(None, max_length=100)

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one digit")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 1800


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    email: str
    full_name: Optional[str]
    organisation: Optional[str]
    plan: str
    is_active: bool
    created_at: str


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class ApiKeyResponse(BaseModel):
    id: str
    label: str
    key: Optional[str] = None
    key_prefix: str
    created_at: str
    last_used_at: Optional[str]
    expires_at: Optional[str]
    is_active: bool

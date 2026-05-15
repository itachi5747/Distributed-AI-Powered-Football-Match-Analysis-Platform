from datetime import datetime, timezone
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.auth import (
    ApiKeyResponse,
    CreateApiKeyRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services import auth_service

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
bearer_scheme = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.register_user(db, data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        organisation=user.organisation,
        plan=user.plan.value,
        is_active=user.is_active,
        created_at=str(user.created_at),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    user = await auth_service.authenticate_user(db, data)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access_token = auth_service.create_access_token(str(user.id), user.email, user.plan.value)
    refresh_token = auth_service.create_refresh_token()
    await auth_service.store_refresh_token(redis, refresh_token, str(user.id))

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    data: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    user_id = await auth_service.resolve_refresh_token(redis, data.refresh_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user = await db.scalar(select(User).where(User.id == user_id, User.is_active.is_(True)))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    access_token = auth_service.create_access_token(str(user.id), user.email, user.plan.value)
    new_refresh = auth_service.create_refresh_token()
    await auth_service.store_refresh_token(redis, new_refresh, str(user.id))
    await redis.delete(f"refresh:{data.refresh_token}")

    return TokenResponse(access_token=access_token, refresh_token=new_refresh)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    try:
        payload = auth_service.decode_access_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    jti = payload.get("jti")
    exp = payload.get("exp")
    if jti and exp:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        expire_seconds = max(1, int(exp) - now_ts)
        await auth_service.blacklist_token(redis, jti, expire_seconds)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        organisation=current_user.organisation,
        plan=current_user.plan.value,
        is_active=current_user.is_active,
        created_at=str(current_user.created_at),
    )


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: CreateApiKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    api_key, raw_key = await auth_service.create_user_api_key(
        db,
        str(current_user.id),
        data.label,
        data.expires_in_days,
    )
    return ApiKeyResponse(
        id=str(api_key.id),
        label=api_key.label,
        key=raw_key,
        key_prefix=api_key.key_prefix,
        created_at=str(api_key.created_at),
        last_used_at=str(api_key.last_used_at) if api_key.last_used_at else None,
        expires_at=str(api_key.expires_at) if api_key.expires_at else None,
        is_active=api_key.is_active,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await db.scalars(
        select(ApiKey).where(ApiKey.user_id == current_user.id).order_by(ApiKey.created_at.desc())
    )
    return [
        ApiKeyResponse(
            id=str(row.id),
            label=row.label,
            key_prefix=row.key_prefix,
            created_at=str(row.created_at),
            last_used_at=str(row.last_used_at) if row.last_used_at else None,
            expires_at=str(row.expires_at) if row.expires_at else None,
            is_active=row.is_active,
        )
        for row in rows.all()
    ]


@router.delete("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    api_key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    key = await db.scalar(select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.user_id == current_user.id))
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    key.is_active = False
    await db.flush()

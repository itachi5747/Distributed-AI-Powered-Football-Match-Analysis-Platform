import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

import redis.asyncio as aioredis
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str, plan: str) -> str:
    jti = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "email": email,
        "plan": plan,
        "jti": jti,
        "type": "access",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token() -> str:
    return secrets.token_urlsafe(64)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    raw = secrets.token_urlsafe(32)
    full_key = f"fiq_live_{raw}"
    prefix = full_key[:16]
    key_hash = hash_api_key(full_key)
    return full_key, prefix, key_hash


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    existing = await db.scalar(select(User).where(User.email == data.email))
    if existing:
        raise ValueError("Email already registered")

    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
        organisation=data.organisation,
    )
    db.add(user)
    await db.flush()
    return user


async def authenticate_user(db: AsyncSession, data: LoginRequest) -> Optional[User]:
    user = await db.scalar(select(User).where(User.email == data.email, User.is_active.is_(True)))
    if not user or not verify_password(data.password, user.password_hash):
        return None
    user.last_login_at = datetime.utcnow()
    await db.flush()
    return user


async def store_refresh_token(redis: aioredis.Redis, refresh_token: str, user_id: str) -> None:
    ttl = settings.JWT_REFRESH_EXPIRE_DAYS * 24 * 3600
    await redis.setex(f"refresh:{refresh_token}", ttl, user_id)


async def resolve_refresh_token(redis: aioredis.Redis, refresh_token: str) -> Optional[str]:
    user_id = await redis.get(f"refresh:{refresh_token}")
    return user_id


async def blacklist_token(redis: aioredis.Redis, jti: str, expire_seconds: int) -> None:
    await redis.setex(f"jwt:blacklist:{jti}", expire_seconds, "1")


async def is_token_blacklisted(redis: aioredis.Redis, jti: str) -> bool:
    return bool(await redis.exists(f"jwt:blacklist:{jti}"))


async def create_user_api_key(
    db: AsyncSession,
    user_id: str,
    label: str,
    expires_in_days: int | None,
) -> tuple[ApiKey, str]:
    full_key, prefix, key_hash = generate_api_key()
    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    api_key = ApiKey(
        user_id=uuid.UUID(user_id),
        label=label,
        key_hash=key_hash,
        key_prefix=prefix,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.flush()
    return api_key, full_key

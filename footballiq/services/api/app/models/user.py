import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Enum as SAEnum, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class UserPlan(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(60), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    organisation: Mapped[str | None] = mapped_column(String(100))
    plan: Mapped[UserPlan] = mapped_column(
        SAEnum(UserPlan, name="userplan", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=UserPlan.FREE,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    storage_quota_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=10_737_418_240)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(nullable=True)

    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    matches: Mapped[list["Match"]] = relationship(
        "Match",
        back_populates="user",
        cascade="all, delete-orphan",
    )

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, Enum as SAEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class MatchStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    team_a_name: Mapped[str] = mapped_column(String(100), nullable=False, default="Team A")
    team_b_name: Mapped[str] = mapped_column(String(100), nullable=False, default="Team B")
    team_a_jersey_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    team_b_jersey_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    match_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(200), nullable=True)
    competition: Mapped[str | None] = mapped_column(String(200), nullable=True)

    video_url: Mapped[str] = mapped_column(Text, nullable=False)
    annotated_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=25.0)
    resolution_w: Mapped[int] = mapped_column(Integer, nullable=False, default=1920)
    resolution_h: Mapped[int] = mapped_column(Integer, nullable=False, default=1080)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[MatchStatus] = mapped_column(
        SAEnum(MatchStatus, name="matchstatus", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=MatchStatus.QUEUED,
        index=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_live: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    analysis_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    homography_matrix: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    processing_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="matches")

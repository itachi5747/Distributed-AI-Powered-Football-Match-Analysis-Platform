"""create matches table

Revision ID: 20260515_0002
Revises: 20260515_0001
Create Date: 2026-05-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260515_0002"
down_revision = "20260515_0001"
branch_labels = None
depends_on = None


match_status_enum = postgresql.ENUM(
    "queued", "processing", "completed", "failed", "cancelled", name="matchstatus", create_type=False
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'matchstatus') THEN
                CREATE TYPE matchstatus AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled');
            END IF;
        END
        $$;
        """
    )

    op.create_table(
        "matches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("team_a_name", sa.String(length=100), nullable=False, server_default="Team A"),
        sa.Column("team_b_name", sa.String(length=100), nullable=False, server_default="Team B"),
        sa.Column("team_a_jersey_color", sa.String(length=7), nullable=True),
        sa.Column("team_b_jersey_color", sa.String(length=7), nullable=True),
        sa.Column("match_date", sa.Date(), nullable=True),
        sa.Column("venue", sa.String(length=200), nullable=True),
        sa.Column("competition", sa.String(length=200), nullable=True),
        sa.Column("video_url", sa.Text(), nullable=False),
        sa.Column("annotated_url", sa.Text(), nullable=True),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=False, server_default="25"),
        sa.Column("resolution_w", sa.Integer(), nullable=False, server_default="1920"),
        sa.Column("resolution_h", sa.Integer(), nullable=False, server_default="1080"),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column("processed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", match_status_enum, nullable=False, server_default="queued"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("is_live", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("analysis_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("homography_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("processing_started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    op.create_index("ix_matches_user_id", "matches", ["user_id"], unique=False)
    op.create_index("ix_matches_status", "matches", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_matches_status", table_name="matches")
    op.drop_index("ix_matches_user_id", table_name="matches")
    op.drop_table("matches")
    match_status_enum.drop(op.get_bind(), checkfirst=True)

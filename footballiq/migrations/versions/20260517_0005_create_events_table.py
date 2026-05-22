"""create events table

Revision ID: 20260517_0005
Revises: 20260517_0004
Create Date: 2026-05-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260517_0005"
down_revision = "20260517_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create event type enum
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'eventtype') THEN
                CREATE TYPE eventtype AS ENUM ('goal', 'shot', 'pass', 'tackle', 'foul', 'corner', 'offside', 'save', 'dribble', 'interception', 'clearance', 'free_kick', 'cross', 'header');
            END IF;
        END
        $$;
        """
    )

    # Create event outcome enum
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'eventoutcome') THEN
                CREATE TYPE eventoutcome AS ENUM ('success', 'failure', 'on_target', 'off_target', 'blocked', 'saved', 'goal');
            END IF;
        END
        $$;
        """
    )

    # Create events table
    op.create_table(
        "events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("match_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("type", postgresql.ENUM('goal', 'shot', 'pass', 'tackle', 'foul', 'corner', 'offside', 'save', 'dribble', 'interception', 'clearance', 'free_kick', 'cross', 'header', name='eventtype', create_type=False), nullable=False),
        sa.Column("timestamp_ms", sa.Float(), nullable=False),
        sa.Column("match_minute", sa.Integer(), nullable=True),
        sa.Column("team", sa.String(length=10), nullable=False),
        sa.Column("player_track_id", sa.Integer(), nullable=True),
        sa.Column("secondary_player_track_id", sa.Integer(), nullable=True),
        sa.Column("location_x_m", sa.Float(), nullable=True),
        sa.Column("location_y_m", sa.Float(), nullable=True),
        sa.Column("outcome", postgresql.ENUM('success', 'failure', 'on_target', 'off_target', 'blocked', 'saved', 'goal', name='eventoutcome', create_type=False), nullable=True),
        sa.Column("xg_value", sa.Float(), nullable=True),
        sa.Column("shot_distance_m", sa.Float(), nullable=True),
        sa.Column("shot_angle_deg", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_events_match_id", "events", ["match_id"])
    op.create_index("ix_events_timestamp_ms", "events", ["timestamp_ms"])


def downgrade() -> None:
    op.drop_index("ix_events_timestamp_ms", table_name="events")
    op.drop_index("ix_events_match_id", table_name="events")
    op.drop_table("events")
    op.execute("DROP TYPE IF EXISTS eventoutcome")
    op.execute("DROP TYPE IF EXISTS eventtype")
"""create frame detection tables

Revision ID: 20260517_0003
Revises: 20260515_0002
Create Date: 2026-05-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260517_0003"
down_revision = "20260515_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create frames table
    op.create_table(
        "frames",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("match_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_frames_match_id", "frames", ["match_id"])
    op.create_index("ix_frames_frame_number", "frames", ["frame_number"])

    # Create frame_detections table
    op.create_table(
        "frame_detections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("class_label", sa.String(length=20), nullable=False),
        sa.Column("team", sa.String(length=20), nullable=True),
        sa.Column("bbox_norm", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("bbox_px", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("position_px_x", sa.Float(), nullable=True),
        sa.Column("position_px_y", sa.Float(), nullable=True),
        sa.Column("position_m_x", sa.Float(), nullable=True),
        sa.Column("position_m_y", sa.Float(), nullable=True),
        sa.Column("speed_kmh", sa.Float(), nullable=True),
        sa.Column("acceleration_ms2", sa.Float(), nullable=True),
        sa.Column("action", sa.String(length=20), nullable=True),
        sa.Column("jersey_color_hsv", postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.ForeignKeyConstraint(["frame_id"], ["frames.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_frame_detections_frame_id", "frame_detections", ["frame_id"])
    op.create_index("ix_frame_detections_track_id", "frame_detections", ["track_id"])


def downgrade() -> None:
    op.drop_index("ix_frame_detections_track_id", table_name="frame_detections")
    op.drop_index("ix_frame_detections_frame_id", table_name="frame_detections")
    op.drop_table("frame_detections")
    op.drop_index("ix_frames_frame_number", table_name="frames")
    op.drop_index("ix_frames_match_id", table_name="frames")
    op.drop_table("frames")
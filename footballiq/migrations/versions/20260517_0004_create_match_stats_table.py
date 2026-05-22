"""create match stats table

Revision ID: 20260517_0004
Revises: 20260517_0003
Create Date: 2026-05-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260517_0004"
down_revision = "20260517_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create match_stats table
    op.create_table(
        "match_stats",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("match_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("possession_a_pct", sa.Float(), nullable=True),
        sa.Column("possession_b_pct", sa.Float(), nullable=True),
        sa.Column("goals_a", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("goals_b", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("shots_a", sa.Integer(), nullable=True),
        sa.Column("shots_on_target_a", sa.Integer(), nullable=True),
        sa.Column("shots_b", sa.Integer(), nullable=True),
        sa.Column("shots_on_target_b", sa.Integer(), nullable=True),
        sa.Column("xg_a", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("xg_b", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("passes_attempted_a", sa.Integer(), nullable=True),
        sa.Column("passes_completed_a", sa.Integer(), nullable=True),
        sa.Column("pass_accuracy_a", sa.Float(), nullable=True),
        sa.Column("passes_attempted_b", sa.Integer(), nullable=True),
        sa.Column("passes_completed_b", sa.Integer(), nullable=True),
        sa.Column("pass_accuracy_b", sa.Float(), nullable=True),
        sa.Column("tackles_a", sa.Integer(), nullable=True),
        sa.Column("tackles_won_a", sa.Integer(), nullable=True),
        sa.Column("tackles_b", sa.Integer(), nullable=True),
        sa.Column("tackles_won_b", sa.Integer(), nullable=True),
        sa.Column("fouls_a", sa.Integer(), nullable=True),
        sa.Column("fouls_b", sa.Integer(), nullable=True),
        sa.Column("corners_a", sa.Integer(), nullable=True),
        sa.Column("corners_b", sa.Integer(), nullable=True),
        sa.Column("offsides_a", sa.Integer(), nullable=True),
        sa.Column("offsides_b", sa.Integer(), nullable=True),
        sa.Column("total_distance_a_km", sa.Float(), nullable=True),
        sa.Column("total_distance_b_km", sa.Float(), nullable=True),
        sa.Column("avg_speed_a_kmh", sa.Float(), nullable=True),
        sa.Column("avg_speed_b_kmh", sa.Float(), nullable=True),
        sa.Column("max_speed_a_kmh", sa.Float(), nullable=True),
        sa.Column("max_speed_b_kmh", sa.Float(), nullable=True),
        sa.Column("sprints_a", sa.Integer(), nullable=True),
        sa.Column("sprints_b", sa.Integer(), nullable=True),
        sa.Column("formation_a", sa.String(length=10), nullable=True),
        sa.Column("formation_b", sa.String(length=10), nullable=True),
        sa.Column("pressing_intensity_a", sa.Float(), nullable=True),
        sa.Column("pressing_intensity_b", sa.Float(), nullable=True),
        sa.Column("momentum_a", sa.Float(), nullable=True),
        sa.Column("momentum_b", sa.Float(), nullable=True),
        sa.Column("last_frame_number", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_match_stats_match_id", "match_stats", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_match_stats_match_id", table_name="match_stats")
    op.drop_table("match_stats")
import asyncio
import os
import json

import redis.asyncio as aioredis
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from shared.messaging import consume_queue
from shared.schemas import FrameMessage
from shared.constants import QUEUE_STATS, REDIS_PUBSUB_MATCH
from worker.calculators.possession import PossessionAccumulator
from worker.calculators.speed import SpeedAccumulator
from worker.calculators.events import EventDetector
from worker.calculators.heatmap import HeatmapBuilder

DB_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ["REDIS_URL"]

engine = create_async_engine(DB_URL, pool_pre_ping=True)
AsyncSession = async_sessionmaker(engine, expire_on_commit=False)

# In-memory state per match (keyed by match_id)
accumulators: dict[str, dict] = {}
SNAPSHOT_EVERY_N_FRAMES = 25  # write to DB every 25 processed frames


def get_or_create(match_id: str, fps: float = 25.0) -> dict:
    if match_id not in accumulators:
        accumulators[match_id] = {
            "possession": PossessionAccumulator(),
            "speed": SpeedAccumulator(fps=fps),
            "events": EventDetector(fps=fps),
            "heatmap": HeatmapBuilder(),
            "goals_a": 0,
            "goals_b": 0,
            "shots_a": 0,
            "shots_b": 0,
            "passes_a": 0,
            "passes_b": 0,
            "tackles_a": 0,
            "tackles_b": 0,
            "frame_count": 0,
            "fps": fps,
        }
    return accumulators[match_id]


async def on_frame(payload: dict) -> None:
    msg = FrameMessage(**payload)
    match_id = msg.match_id
    acc = get_or_create(match_id)
    acc["frame_count"] += 1
    dets = [d.model_dump() for d in msg.detections]
    ball = msg.ball_position_m.model_dump() if msg.ball_position_m else None

    # Update all accumulators
    acc["possession"].update(ball, dets)
    speeds = acc["speed"].update(dets, msg.frame_number)
    new_events = acc["events"].update({
        "ball_position_m": ball,
        "detections": dets,
        "frame_number": msg.frame_number,
        "timestamp_ms": msg.timestamp_ms,
    })
    acc["heatmap"].update(dets)

    # Tally events
    for ev in new_events:
        t = ev["type"]
        team = ev.get("team", "")
        if t == "goal":
            if team == "team_a":
                acc["goals_a"] += 1
            else:
                acc["goals_b"] += 1
        elif t == "shot":
            if team == "team_a":
                acc["shots_a"] += 1
            else:
                acc["shots_b"] += 1
        elif t == "pass":
            if team == "team_a":
                acc["passes_a"] += 1
            else:
                acc["passes_b"] += 1
        elif t == "tackle":
            if team == "team_a":
                acc["tackles_a"] += 1
            else:
                acc["tackles_b"] += 1

    # Snapshot to DB + Redis every N frames
    if acc["frame_count"] % SNAPSHOT_EVERY_N_FRAMES == 0:
        await write_snapshot(match_id, acc, msg.frame_number, new_events)


async def write_snapshot(match_id: str, acc: dict, frame_no: int, new_events: list) -> None:
    poss = acc["possession"].stats
    speed = acc["speed"].team_summary()

    snapshot = {
        **poss,
        "goals_a": acc["goals_a"],
        "goals_b": acc["goals_b"],
        "shots_a": acc["shots_a"],
        "shots_b": acc["shots_b"],
        "passes_completed_a": acc["passes_a"],
        "passes_completed_b": acc["passes_b"],
        "tackles_a": acc["tackles_a"],
        "tackles_b": acc["tackles_b"],
        "total_distance_a_km": speed.get("team_a", {}).get("total_distance_km", 0),
        "total_distance_b_km": speed.get("team_b", {}).get("total_distance_km", 0),
        "max_speed_a_kmh": speed.get("team_a", {}).get("max_speed_kmh", 0),
        "max_speed_b_kmh": speed.get("team_b", {}).get("max_speed_kmh", 0),
        "sprints_a": speed.get("team_a", {}).get("sprints", 0),
        "sprints_b": speed.get("team_b", {}).get("sprints", 0),
        "last_frame_number": frame_no,
    }

    # Write to PostgreSQL
    async with AsyncSession() as session:
        await session.execute(
            sqlalchemy.text("""
                INSERT INTO match_stats (id, match_id, updated_at, possession_a_pct, possession_b_pct,
                    goals_a, goals_b, shots_a, shots_b, shots_on_target_a, shots_on_target_b,
                    xg_a, xg_b, passes_attempted_a, passes_attempted_b, passes_completed_a,
                    passes_completed_b, pass_accuracy_a, pass_accuracy_b, tackles_a, tackles_b,
                    tackles_won_a, tackles_won_b, fouls_a, fouls_b, corners_a, corners_b,
                    offsides_a, offsides_b, total_distance_a_km, total_distance_b_km,
                    avg_speed_a_kmh, avg_speed_b_kmh, max_speed_a_kmh, max_speed_b_kmh,
                    sprints_a, sprints_b, formation_a, formation_b,
                    pressing_intensity_a, pressing_intensity_b, momentum_a, momentum_b,
                    last_frame_number)
                VALUES (gen_random_uuid(), :match_id, now(), :pa, :pb, :ga, :gb, :sa, :sb,
                        :sona, :sonb, :xga, :xgb, :paa, :pab, :pca, :pcb, :paca, :pacb,
                        :ta, :tb, :twa, :twb, :fa, :fb, :ca, :cb, :osa, :osb,
                        :da, :db, :aspeeda, :aspeedb, :msa, :msb, :spa, :spb,
                        :forma, :formb, :pia, :pib, :moma, :momb, :lfn)
                ON CONFLICT (match_id) DO UPDATE SET
                    updated_at=now(), possession_a_pct=:pa, possession_b_pct=:pb,
                    goals_a=:ga, goals_b=:gb, shots_a=:sa, shots_b=:sb,
                    shots_on_target_a=:sona, shots_on_target_b=:sonb,
                    xg_a=:xga, xg_b=:xgb,
                    passes_attempted_a=:paa, passes_attempted_b=:pab,
                    passes_completed_a=:pca, passes_completed_b=:pcb,
                    pass_accuracy_a=:paca, pass_accuracy_b=:pacb,
                    tackles_a=:ta, tackles_b=:tb,
                    tackles_won_a=:twa, tackles_won_b=:twb,
                    fouls_a=:fa, fouls_b=:fb,
                    corners_a=:ca, corners_b=:cb,
                    offsides_a=:osa, offsides_b=:osb,
                    total_distance_a_km=:da, total_distance_b_km=:db,
                    avg_speed_a_kmh=:aspeeda, avg_speed_b_kmh=:aspeedb,
                    max_speed_a_kmh=:msa, max_speed_b_kmh=:msb,
                    sprints_a=:spa, sprints_b=:spb,
                    formation_a=:forma, formation_b=:formb,
                    pressing_intensity_a=:pia, pressing_intensity_b=:pib,
                    momentum_a=:moma, momentum_b=:momb,
                    last_frame_number=:lfn
            """),
            {
                "match_id": match_id,
                "pa": snapshot["possession_a_pct"],
                "pb": snapshot["possession_b_pct"],
                "ga": snapshot["goals_a"],
                "gb": snapshot["goals_b"],
                "sa": snapshot["shots_a"],
                "sb": snapshot["shots_b"],
                "sona": snapshot.get("shots_on_target_a", 0),
                "sonb": snapshot.get("shots_on_target_b", 0),
                "xga": snapshot.get("xg_a", 0.0),
                "xgb": snapshot.get("xg_b", 0.0),
                "paa": snapshot.get("passes_attempted_a", 0),
                "pab": snapshot.get("passes_attempted_b", 0),
                "pca": snapshot["passes_completed_a"],
                "pcb": snapshot["passes_completed_b"],
                "paca": snapshot.get("pass_accuracy_a", 0.0),
                "pacb": snapshot.get("pass_accuracy_b", 0.0),
                "ta": snapshot["tackles_a"],
                "tb": snapshot["tackles_b"],
                "twa": snapshot.get("tackles_won_a", 0),
                "twb": snapshot.get("tackles_won_b", 0),
                "fa": snapshot.get("fouls_a", 0),
                "fb": snapshot.get("fouls_b", 0),
                "ca": snapshot.get("corners_a", 0),
                "cb": snapshot.get("corners_b", 0),
                "osa": snapshot.get("offsides_a", 0),
                "osb": snapshot.get("offsides_b", 0),
                "da": snapshot["total_distance_a_km"],
                "db": snapshot["total_distance_b_km"],
                "aspeeda": snapshot.get("avg_speed_a_kmh", 0.0),
                "aspeedb": snapshot.get("avg_speed_b_kmh", 0.0),
                "msa": snapshot["max_speed_a_kmh"],
                "msb": snapshot["max_speed_b_kmh"],
                "spa": snapshot["sprints_a"],
                "spb": snapshot["sprints_b"],
                "forma": snapshot.get("formation_a", ""),
                "formb": snapshot.get("formation_b", ""),
                "pia": snapshot.get("pressing_intensity_a", 0.0),
                "pib": snapshot.get("pressing_intensity_b", 0.0),
                "moma": snapshot.get("momentum_a", 0.0),
                "momb": snapshot.get("momentum_b", 0.0),
                "lfn": frame_no,
            },
        )
        await session.commit()

    # Write to Redis for instant WebSocket delivery
    redis = aioredis.from_url(REDIS_URL)
    await redis.set(f"match:{match_id}:stats", json.dumps(snapshot), ex=86400)
    await redis.publish(f"pubsub:match:{match_id}", json.dumps({
        "event": "stats_update",
        "match_id": match_id,
        "frame_number": frame_no,
        **snapshot,
    }))
    await redis.aclose()


async def main() -> None:
    print(f"[Stats Engine] Starting — consuming {QUEUE_STATS}")
    await consume_queue(QUEUE_STATS, on_frame, prefetch_count=10)


if __name__ == "__main__":
    asyncio.run(main())
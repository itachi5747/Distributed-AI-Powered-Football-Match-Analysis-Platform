import asyncio
import os
import tempfile
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


import cv2
import numpy as np
import sqlalchemy
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from shared.messaging import consume_queue, publish_message
from shared.schemas import DetectionObject, FrameMessage, InferenceJobMessage, Point2D, ReportJobMessage
from shared.storage import download_file

from worker.pipeline.ball_tracker import BallKalmanTracker
from worker.pipeline.detector import (
    CLASS_BALL,
    CLASS_GOALKEEPER,
    CLASS_GOALPOST,
    CLASS_PLAYER,
    CLASS_REFEREE,
    FootballDetector,
)
from worker.pipeline.homography import FieldHomography
from worker.pipeline.team_classifier import TeamClassifier
from worker.pipeline.tracker import FootballTracker

DB_URL = os.environ["DATABASE_URL"]
QUEUE_INFERENCE = os.environ.get("QUEUE_INFERENCE", "footballiq.inference")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))

engine = create_async_engine(DB_URL, pool_pre_ping=True)
AsyncSession = async_sessionmaker(engine, expire_on_commit=False)

CLASS_LABEL_MAP = {
    CLASS_PLAYER: "player",
    CLASS_BALL: "ball",
    CLASS_REFEREE: "referee",
    CLASS_GOALKEEPER: "goalkeeper",
    CLASS_GOALPOST: "goalpost",
}


def _extract_bucket_key(video_url: str) -> tuple[str, str]:
    parsed = urlparse(video_url)
    path = parsed.path.lstrip("/")
    if not path or "/" not in path:
        raise ValueError(f"Invalid video URL path: {video_url}")
    bucket, key = path.split("/", 1)
    return bucket, key


async def _fetch_match_context(match_id: str) -> tuple[str, str, str]:
    async with AsyncSession() as session:
        result = await session.execute(
            sqlalchemy.text("SELECT user_id::text, status::text, title FROM matches WHERE id = :id"),
            {"id": match_id},
        )
        row = result.first()
        if not row:
            raise ValueError(f"Match not found: {match_id}")
        return row[0], row[1], row[2]


async def update_match_status(
    match_id: str,
    status: str,
    processed: int = 0,
    total_frames: int | None = None,
    reason: str | None = None,
) -> None:
    started = datetime.utcnow() if status == "processing" else None
    completed = datetime.utcnow() if status in ("completed", "failed", "cancelled") else None

    async with AsyncSession() as session:
        await session.execute(
            sqlalchemy.text(
                """
                UPDATE matches
                SET
                    status = :status,
                    processed_frames = :processed,
                    total_frames = COALESCE(:total_frames, total_frames),
                    failure_reason = :reason,
                    processing_started_at = COALESCE(:started, processing_started_at),
                    completed_at = COALESCE(:completed, completed_at)
                WHERE id = :match_id
                """
            ),
            {
                "status": status,
                "processed": processed,
                "total_frames": total_frames,
                "reason": reason,
                "started": started,
                "completed": completed,
                "match_id": match_id,
            },
        )
        await session.commit()


def _ball_from_tracked(tracked) -> tuple[float, float] | None:
    if tracked.class_id is None:
        return None

    indices = np.where(tracked.class_id == CLASS_BALL)[0]
    if len(indices) == 0:
        return None

    conf = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked.xyxy))
    best_i = indices[np.argmax(conf[indices])]
    x1, y1, x2, y2 = tracked.xyxy[best_i]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def _team_label(raw: str):
    if raw == "team_a":
        return "team_a"
    if raw == "team_b":
        return "team_b"
    if raw == "referee":
        return "referee"
    return "unknown"


async def process_match(job: InferenceJobMessage) -> None:
    match_id = job.match_id
    print(f"[Inference] Starting match {match_id}")

    user_id, _, _ = await _fetch_match_context(match_id)
    tmp_path = None

    try:
        await update_match_status(match_id, "processing", processed=0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        bucket, key = _extract_bucket_key(job.video_url)
        download_file(bucket, key, tmp_path)

        detector = FootballDetector()
        tracker = FootballTracker()
        classifier = TeamClassifier()
        ball_tracker = BallKalmanTracker()
        homography = FieldHomography()

        if job.analysis_config.get("homography_corners"):
            homography.calibrate_manual(job.analysis_config["homography_corners"])

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV failed to open downloaded video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        await update_match_status(match_id, "processing", processed=0, total_frames=total_frames)

        frame_number = 0
        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % max(FRAME_SKIP, 1) != 0:
                continue

            timestamp_ms = (frame_number / max(fps, 1.0)) * 1000.0

            raw = detector.detect(frame)
            tracked = tracker.update(raw)

            if frame_number % 50 == 1:
                classifier.fit(frame, tracked, [CLASS_PLAYER, CLASS_GOALKEEPER])

            ball_raw = _ball_from_tracked(tracked)
            ball_smoothed_px = ball_tracker.update(ball_raw)
            ball_smoothed_m = (
                homography.pixel_to_metres(ball_smoothed_px[0], ball_smoothed_px[1])
                if ball_smoothed_px is not None
                else None
            )

            detections = []
            if tracked.tracker_id is not None and tracked.class_id is not None:
                h, w = frame.shape[:2]
                for i, bbox in enumerate(tracked.xyxy):
                    class_id = int(tracked.class_id[i])
                    if class_id not in CLASS_LABEL_MAP:
                        continue

                    track_id = int(tracked.tracker_id[i])
                    conf = float(tracked.confidence[i])
                    cx = float((bbox[0] + bbox[2]) / 2)
                    cy = float((bbox[1] + bbox[3]) / 2)

                    team = "unknown"
                    if class_id in (CLASS_PLAYER, CLASS_GOALKEEPER):
                        team = _team_label(classifier.classify(frame, track_id, bbox))

                    pos_m = homography.pixel_to_metres(cx, cy)

                    detections.append(
                        DetectionObject(
                            track_id=track_id,
                            class_label=CLASS_LABEL_MAP[class_id],
                            team=team,
                            bbox_norm=[float(bbox[0] / w), float(bbox[1] / h), float(bbox[2] / w), float(bbox[3] / h)],
                            bbox_px=[float(v) for v in bbox],
                            confidence=conf,
                            position_px=Point2D(x=cx, y=cy),
                            position_m=Point2D(x=pos_m[0], y=pos_m[1]) if pos_m else None,
                        )
                    )

            frame_msg = FrameMessage(
                match_id=match_id,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                detections=detections,
                ball_position_px=(
                    Point2D(x=ball_smoothed_px[0], y=ball_smoothed_px[1]) if ball_smoothed_px else None
                ),
                ball_position_m=(Point2D(x=ball_smoothed_m[0], y=ball_smoothed_m[1]) if ball_smoothed_m else None),
                homography_matrix=homography.to_flat_list(),
            )
            await publish_message("stats", frame_msg)

            processed += 1
            if processed % 100 == 0:
                await update_match_status(match_id, "processing", processed=processed, total_frames=total_frames)

        cap.release()

        await update_match_status(match_id, "completed", processed=processed, total_frames=total_frames)
        await publish_message(
            "reports",
            ReportJobMessage(match_id=match_id, user_id=user_id),
        )
        print(f"[Inference] Match {match_id} completed. Processed={processed}")

    except Exception as exc:
        await update_match_status(match_id, "failed", reason=str(exc))
        print(f"[Inference] Match {match_id} failed: {exc}")
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def on_message(payload: dict) -> None:
    job = InferenceJobMessage(**payload)
    await process_match(job)


async def main() -> None:
    print(f"[Inference Worker] Listening on queue: {QUEUE_INFERENCE}")
    await consume_queue(QUEUE_INFERENCE, on_message, prefetch_count=1)


if __name__ == "__main__":
    asyncio.run(main())

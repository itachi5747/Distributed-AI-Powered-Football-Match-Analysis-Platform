import json
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.match import Match
from shared.schemas import InferenceJobMessage
from shared.storage import upload_file
from shared.messaging import publish_message

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
MAX_FILE_SIZE_FREE = 2 * 1024**3
MAX_FILE_SIZE_PRO = 10 * 1024**3


def validate_video_file(filename: str, file_size: int, plan: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    max_size = MAX_FILE_SIZE_PRO if plan in ("pro", "enterprise") else MAX_FILE_SIZE_FREE
    if file_size > max_size:
        raise ValueError(
            f"File too large ({file_size / 1024**3:.2f} GB). Limit: {max_size / 1024**3:.2f} GB"
        )


def extract_video_metadata(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError("No video stream found")

    fps_raw = video_stream.get("r_frame_rate", "25/1")
    num, den = fps_raw.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 25.0

    duration = float(data.get("format", {}).get("duration", 0) or 0)
    total_frames = int(duration * fps) if duration > 0 else None

    return {
        "fps": round(fps, 3),
        "duration_seconds": duration,
        "resolution_w": int(video_stream.get("width", 1920)),
        "resolution_h": int(video_stream.get("height", 1080)),
        "total_frames": total_frames,
    }


async def create_match_from_upload(
    db: AsyncSession,
    user_id: str,
    title: str,
    team_a_name: str,
    team_b_name: str,
    filename: str,
    source_file,
    file_size: int,
    analysis_config: dict,
) -> Match:
    ext = os.path.splitext(filename)[1].lower() or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        source_file.seek(0)
        shutil.copyfileobj(source_file, tmp)
        tmp.flush()

        try:
            metadata = extract_video_metadata(tmp.name)
        except Exception:
            metadata = {
                "fps": 25.0,
                "duration_seconds": 0.0,
                "resolution_w": 1920,
                "resolution_h": 1080,
                "total_frames": None,
            }

        key = f"uploads/{user_id}/{uuid.uuid4()}/{filename}"
        tmp.seek(0)
        content_type = "video/mp4" if ext == ".mp4" else "application/octet-stream"
        video_url = upload_file(settings.MINIO_BUCKET_VIDEOS, key, tmp, content_type=content_type)

    match = Match(
        user_id=uuid.UUID(user_id),
        title=title,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        video_url=video_url,
        file_size_bytes=file_size,
        analysis_config=analysis_config,
        **metadata,
    )
    db.add(match)
    await db.flush()

    job = InferenceJobMessage(
        match_id=str(match.id),
        video_url=video_url,
        analysis_config=analysis_config,
    )
    await publish_message("inference", job)

    return match

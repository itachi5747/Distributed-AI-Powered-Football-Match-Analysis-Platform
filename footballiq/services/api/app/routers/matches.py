import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.match import Match, MatchStatus
from app.models.user import User
from app.services.ingestion_service import create_match_from_upload, validate_video_file

router = APIRouter(prefix="/api/v1/matches", tags=["Matches"])


def _parse_analysis_config(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError
        return parsed
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="analysis_config must be valid JSON object") from exc


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_match(
    file: UploadFile = File(...),
    title: str = Form(..., max_length=200),
    team_a_name: str = Form("Team A", max_length=100),
    team_b_name: str = Form("Team B", max_length=100),
    analysis_config: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    try:
        validate_video_file(file.filename, file_size, current_user.plan.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    match = await create_match_from_upload(
        db=db,
        user_id=str(current_user.id),
        title=title,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        filename=file.filename,
        source_file=file.file,
        file_size=file_size,
        analysis_config=_parse_analysis_config(analysis_config),
    )

    return {
        "success": True,
        "data": {
            "match_id": str(match.id),
            "title": match.title,
            "status": match.status.value,
            "message": "Video uploaded successfully. Processing queued.",
        },
    }


@router.get("")
async def list_matches(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status_filter: MatchStatus | None = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(Match).where(Match.user_id == current_user.id)
    if status_filter:
        query = query.where(Match.status == status_filter)

    query = query.order_by(Match.created_at.desc()).offset((page - 1) * limit).limit(limit)
    rows = (await db.scalars(query)).all()

    total_query = select(func.count()).select_from(Match).where(Match.user_id == current_user.id)
    if status_filter:
        total_query = total_query.where(Match.status == status_filter)
    total = await db.scalar(total_query)

    return {
        "success": True,
        "data": [
            {
                "id": str(m.id),
                "title": m.title,
                "status": m.status.value,
                "team_a": m.team_a_name,
                "team_b": m.team_b_name,
                "duration_seconds": m.duration_seconds,
                "created_at": str(m.created_at),
            }
            for m in rows
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total or 0,
            "has_next": (page * limit) < (total or 0),
        },
    }


@router.get("/{match_id}")
async def get_match(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = await db.scalar(select(Match).where(Match.id == match_id, Match.user_id == current_user.id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    total_frames = max(match.total_frames or 1, 1)
    progress = round(match.processed_frames / total_frames, 4)

    return {
        "success": True,
        "data": {
            "id": str(match.id),
            "title": match.title,
            "status": match.status.value,
            "progress": progress,
            "team_a_name": match.team_a_name,
            "team_b_name": match.team_b_name,
            "fps": match.fps,
            "duration_seconds": match.duration_seconds,
            "resolution": f"{match.resolution_w}x{match.resolution_h}",
            "video_url": match.video_url,
            "annotated_url": match.annotated_url,
            "created_at": str(match.created_at),
            "completed_at": str(match.completed_at) if match.completed_at else None,
        },
    }


@router.patch("/{match_id}")
async def update_match(
    match_id: uuid.UUID,
    title: str | None = Form(None, max_length=200),
    team_a_name: str | None = Form(None, max_length=100),
    team_b_name: str | None = Form(None, max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = await db.scalar(select(Match).where(Match.id == match_id, Match.user_id == current_user.id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if title is not None:
        match.title = title
    if team_a_name is not None:
        match.team_a_name = team_a_name
    if team_b_name is not None:
        match.team_b_name = team_b_name

    await db.flush()

    return {
        "success": True,
        "data": {
            "id": str(match.id),
            "title": match.title,
            "team_a_name": match.team_a_name,
            "team_b_name": match.team_b_name,
        },
    }


@router.get("/{match_id}/status")
async def get_match_status(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = await db.scalar(select(Match).where(Match.id == match_id, Match.user_id == current_user.id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    total_frames = max(match.total_frames or 1, 1)
    progress = round(match.processed_frames / total_frames, 4)

    return {
        "success": True,
        "data": {
            "match_id": str(match.id),
            "status": match.status.value,
            "progress": progress,
            "processed_frames": match.processed_frames,
            "total_frames": match.total_frames,
            "failure_reason": match.failure_reason,
        },
    }


@router.get("/{match_id}/frames/{frame_number}")
async def get_annotated_frame(
    match_id: uuid.UUID,
    frame_number: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ = frame_number
    match = await db.scalar(select(Match).where(Match.id == match_id, Match.user_id == current_user.id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    raise HTTPException(status_code=404, detail="Annotated frame not available yet")


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_match(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    match = await db.scalar(select(Match).where(Match.id == match_id, Match.user_id == current_user.id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    await db.delete(match)

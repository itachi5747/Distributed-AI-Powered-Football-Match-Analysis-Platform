import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TeamLabel(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"


class ClassLabel(str, Enum):
    PLAYER = "player"
    BALL = "ball"
    REFEREE = "referee"
    GOALKEEPER = "goalkeeper"
    GOALPOST = "goalpost"


class ActionLabel(str, Enum):
    STANDING = "standing"
    JOGGING = "jogging"
    SPRINTING = "sprinting"
    PASSING = "passing"
    RECEIVING = "receiving"
    SHOOTING = "shooting"
    HEADING = "heading"
    TACKLING = "tackling"
    DRIBBLING = "dribbling"
    SAVING = "saving"
    CELEBRATING = "celebrating"


class Point2D(BaseModel):
    x: float
    y: float


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float


class DetectionObject(BaseModel):
    track_id: int
    class_label: ClassLabel
    team: TeamLabel = TeamLabel.UNKNOWN
    bbox_norm: List[float] = Field(..., min_length=4, max_length=4)
    bbox_px: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_px: Optional[Point2D] = None
    position_m: Optional[Point2D] = None
    speed_kmh: Optional[float] = None
    acceleration_ms2: Optional[float] = None
    keypoints: Optional[List[Keypoint]] = None
    action: Optional[ActionLabel] = None
    action_confidence: Optional[float] = None
    jersey_color_hsv: Optional[List[int]] = None


class FrameMessage(BaseModel):
    match_id: str
    frame_number: int
    timestamp_ms: float
    detections: List[DetectionObject]
    ball_position_px: Optional[Point2D] = None
    ball_position_m: Optional[Point2D] = None
    ball_velocity_mps: Optional[float] = None
    ball_confidence: Optional[float] = None
    homography_matrix: Optional[List[float]] = None


class InferenceJobMessage(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match_id: str
    video_url: str
    analysis_config: dict
    priority: int = 5


class ReportJobMessage(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match_id: str
    user_id: str

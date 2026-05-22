import math
from shared.constants import PITCH_LENGTH_M, PITCH_WIDTH_M, GOAL_WIDTH_M, POSSESSION_THRESHOLD_M
GOAL_X_MIN = 100.0  # metres — goal zone x range
GOAL_X_MAX = PITCH_LENGTH_M
GOAL_Y_MIN = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2  # 30.34m
GOAL_Y_MAX = (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2  # 37.66m
SHOT_VELOCITY_THRESHOLD_MPS = 15.0  # ball must move > this to be a shot
PASS_MAX_TIME_FRAMES = 50  # max frames between pass start and receive


class EventDetector:
    """
    Stateful detector for: goals, shots, passes, tackles.
    Called with each FrameMessage payload.
    """

    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.prev_ball_pos = None
        self.prev_owner_id = None
        self.prev_owner_team = None
        self.prev_frame = 0
        self.detected_events = []

    def update(self, frame_msg: dict) -> list:
        """Returns list of new event dicts detected in this frame."""
        new_events = []
        ball = frame_msg.get("ball_position_m")
        dets = frame_msg.get("detections", [])
        frame_no = frame_msg.get("frame_number", 0)
        ts = frame_msg.get("timestamp_ms", 0)

        # ── Ball velocity ──────────────────────────────────────────
        ball_velocity_mps = 0.0
        if ball and self.prev_ball_pos:
            dx = ball["x"] - self.prev_ball_pos["x"]
            dy = ball["y"] - self.prev_ball_pos["y"]
            dist_m = math.sqrt(dx**2 + dy**2)
            elapsed_s = (frame_no - self.prev_frame) / self.fps
            if elapsed_s > 0:
                ball_velocity_mps = dist_m / elapsed_s

        # ── Goal detection ─────────────────────────────────────────
        if ball and self.prev_ball_pos and self.prev_ball_pos["x"] < GOAL_X_MIN and ball["x"] >= GOAL_X_MIN and GOAL_Y_MIN <= ball["y"] <= GOAL_Y_MAX:
            scoring_team = self.prev_owner_team or "team_a"
            new_events.append({
                "type": "goal",
                "team": scoring_team,
                "timestamp_ms": ts,
                "frame_number": frame_no,
                "player_track_id": self.prev_owner_id,
                "location_x_m": ball["x"] if ball else None,
                "location_y_m": ball["y"] if ball else None,
                "confidence": 0.95,
            })

        # ── Shot detection ─────────────────────────────────────────
        if ball_velocity_mps > SHOT_VELOCITY_THRESHOLD_MPS and self.prev_owner_id:
            new_events.append({
                "type": "shot",
                "team": self.prev_owner_team or "unknown",
                "timestamp_ms": ts,
                "frame_number": frame_no,
                "player_track_id": self.prev_owner_id,
                "location_x_m": ball["x"] if ball else None,
                "location_y_m": ball["y"] if ball else None,
                "confidence": min(ball_velocity_mps / 30.0, 1.0),
            })

        # ── Possession change → pass or tackle ────────────────────
        if ball:
            curr_owner_id, curr_owner_team = self._closest_player(ball, dets)
            if (self.prev_owner_id is not None and curr_owner_id is not None and curr_owner_id != self.prev_owner_id):
                if curr_owner_team == self.prev_owner_team:
                    # Same team → pass
                    new_events.append({
                        "type": "pass",
                        "team": self.prev_owner_team,
                        "timestamp_ms": ts,
                        "frame_number": frame_no,
                        "player_track_id": self.prev_owner_id,
                        "secondary_player_track_id": curr_owner_id,
                        "outcome": "success",
                        "confidence": 0.80,
                    })
                else:
                    # Different team → tackle / interception
                    new_events.append({
                        "type": "tackle",
                        "team": curr_owner_team,
                        "timestamp_ms": ts,
                        "frame_number": frame_no,
                        "player_track_id": curr_owner_id,
                        "secondary_player_track_id": self.prev_owner_id,
                        "outcome": "success",
                        "confidence": 0.75,
                    })
            self.prev_owner_id = curr_owner_id
            self.prev_owner_team = curr_owner_team
            self.prev_ball_pos = ball
            self.prev_frame = frame_no
        return new_events

    def _closest_player(self, ball, dets):
        min_dist = float("inf")
        closest_id, closest_team = None, None
        for det in dets:
            if det.get("class_label") not in ("player", "goalkeeper"):
                continue
            pos = det.get("position_m")
            if not pos:
                continue
            dist = math.sqrt((pos["x"] - ball["x"]) ** 2 + (pos["y"] - ball["y"]) ** 2)
            if dist < min_dist and dist < POSSESSION_THRESHOLD_M:
                min_dist = dist
                closest_id = det["track_id"]
                closest_team = det.get("team", "unknown")
        return closest_id, closest_team
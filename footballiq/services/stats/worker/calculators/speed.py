import math
from collections import defaultdict
from shared.constants import SPRINT_THRESHOLD_KMH, HIGH_INTENSITY_KMH


class SpeedAccumulator:
    """Track distance and speed per player across frames."""

    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.prev_positions: dict[int, dict] = {}  # track_id → {x, y}
        self.speed_history: dict[int, list] = defaultdict(list)  # last 5 speeds
        self.total_distance: dict[int, float] = defaultdict(float)
        self.team_lookup: dict[int, str] = {}
        self.max_speed: dict[int, float] = defaultdict(float)
        self.sprint_active: dict[int, bool] = defaultdict(False)
        self.sprint_count: dict[int, int] = defaultdict(int)

    def update(self, detections: list, frame_number: int) -> dict:
        """
        Process one frame of detections.
        Returns per-player speed dict: {track_id: speed_kmh}
        """
        speeds = {}
        for det in detections:
            if det.get("class_label") not in ("player", "goalkeeper"):
                continue
            track_id = det["track_id"]
            pos = det.get("position_m")
            team = det.get("team", "unknown")
            if pos is None:
                continue
            self.team_lookup[track_id] = team
            if track_id in self.prev_positions:
                prev = self.prev_positions[track_id]
                dist_m = math.sqrt((pos["x"] - prev["x"]) ** 2 + (pos["y"] - prev["y"]) ** 2)
                # Sanity check: player can't move > 12m per frame at 25fps
                if dist_m < 12.0:
                    speed_mps = dist_m * self.fps
                    speed_kmh = speed_mps * 3.6
                    # Rolling average over 5 frames
                    hist = self.speed_history[track_id]
                    hist.append(speed_kmh)
                    if len(hist) > 5:
                        hist.pop(0)
                    smooth_speed = sum(hist) / len(hist)
                    self.total_distance[track_id] += dist_m / 1000  # to km
                    self.max_speed[track_id] = max(self.max_speed[track_id], smooth_speed)
                    speeds[track_id] = smooth_speed
                    # Sprint detection
                    was_sprinting = self.sprint_active[track_id]
                    is_sprinting = smooth_speed >= SPRINT_THRESHOLD_KMH
                    self.sprint_active[track_id] = is_sprinting
                    if is_sprinting and not was_sprinting:
                        self.sprint_count[track_id] += 1
                self.prev_positions[track_id] = pos
            else:
                # First time seeing this player
                self.prev_positions[track_id] = pos
        return speeds

    def team_summary(self) -> dict:
        result = {}
        for team in ("team_a", "team_b"):
            ids = [tid for tid, t in self.team_lookup.items() if t == team]
            if not ids:
                result[team] = {}
                continue
            team_speeds = [s for i in ids for s in self.speed_history.get(i, [0])]
            result[team] = {
                "total_distance_km": round(sum(self.total_distance[i] for i in ids), 1),
                "avg_speed_kmh": round(sum(team_speeds) / max(len(team_speeds), 1), 1),
                "max_speed_kmh": round(max((self.max_speed[i] for i in ids), default=0.0), 1),
                "sprints": sum(self.sprint_count[i] for i in ids),
            }
        return result
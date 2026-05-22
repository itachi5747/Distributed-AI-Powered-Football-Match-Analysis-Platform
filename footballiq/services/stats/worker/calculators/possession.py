import math
from shared.constants import POSSESSION_THRESHOLD_M


class PossessionAccumulator:
    """
    Track which team has possession each frame.
    Ball ownership = team whose player is closest to ball AND within THRESHOLD.
    """

    def __init__(self):
        self.frames: dict[str, int] = {"team_a": 0, "team_b": 0, "contested": 0}

    def update(self, ball_position_m, detections) -> str:
        """Returns current possession owner: 'team_a', 'team_b', or 'contested'."""
        if ball_position_m is None:
            self.frames["contested"] += 1
            return "contested"

        min_dist = float("inf")
        owner = "contested"
        for det in detections:
            if det.get("class_label") not in ("player", "goalkeeper"):
                continue
            pos = det.get("position_m")
            if pos is None:
                continue
            dist = math.sqrt((pos["x"] - ball_position_m["x"]) ** 2 + (pos["y"] - ball_position_m["y"]) ** 2)
            if dist < min_dist:
                min_dist = dist
                owner = det.get("team", "contested")
            if min_dist > POSSESSION_THRESHOLD_M:
                owner = "contested"
        self.frames[owner] += 1
        return owner

    @property
    def stats(self) -> dict:
        total = max(sum(self.frames.values()), 1)
        a = self.frames["team_a"] / total * 100
        b = self.frames["team_b"] / total * 100
        return {
            "possession_a_pct": round(a, 1),
            "possession_b_pct": round(b, 1),
            "possession_a_seconds": self.frames["team_a"] / 25,  # approx at 25fps
            "possession_b_seconds": self.frames["team_b"] / 25,
        }
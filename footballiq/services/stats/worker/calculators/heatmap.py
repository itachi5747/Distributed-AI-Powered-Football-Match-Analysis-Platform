import numpy as np
from scipy.ndimage import gaussian_filter
from shared.constants import PITCH_LENGTH_M, PITCH_WIDTH_M, HEATMAP_BINS_X, HEATMAP_BINS_Y


class HeatmapBuilder:
    """Maintain 2D position histograms per team and per player."""

    def __init__(self):
        self.team_positions: dict[str, list] = {"team_a": [], "team_b": []}
        self.player_positions: dict[int, list] = {}

    def update(self, detections: list):
        for det in detections:
            if det.get("class_label") not in ("player", "goalkeeper"):
                continue
            pos = det.get("position_m")
            if not pos:
                continue
            team = det.get("team")
            track_id = det.get("track_id")
            if team in self.team_positions:
                self.team_positions[team].append((pos["x"], pos["y"]))
            if track_id is not None:
                if track_id not in self.player_positions:
                    self.player_positions[track_id] = []
                self.player_positions[track_id].append((pos["x"], pos["y"]))

    def build_team_heatmap(self, team: str) -> list:
        return self._build(self.team_positions.get(team, []))

    def build_player_heatmap(self, track_id: int) -> list:
        return self._build(self.player_positions.get(track_id, []))

    def _build(self, positions: list) -> list:
        if not positions:
            return [0.0] * (HEATMAP_BINS_X * HEATMAP_BINS_Y)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        heatmap, _, _ = np.histogram2d(
            xs, ys,
            bins=(HEATMAP_BINS_X, HEATMAP_BINS_Y),
            range=[[0, PITCH_LENGTH_M], [0, PITCH_WIDTH_M]]
        )
        heatmap = gaussian_filter(heatmap.astype(float), sigma=1.5)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.flatten().tolist()
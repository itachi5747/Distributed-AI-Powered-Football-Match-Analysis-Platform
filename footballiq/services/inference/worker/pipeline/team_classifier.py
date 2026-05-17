from collections import Counter, defaultdict
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

GREEN_LOWER = np.array([36, 40, 40])
GREEN_UPPER = np.array([86, 255, 255])
HISTORY_FRAMES = 10


class TeamClassifier:
    def __init__(self) -> None:
        self.kmeans: Optional[KMeans] = None
        self.vote_history: dict[int, list[str]] = defaultdict(list)
        self.cluster_to_team: dict[int, str] = {}
        self.fitted = False

    def _crop_jersey(self, frame: np.ndarray, bbox_px: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox_px]
        jersey_h = int((y2 - y1) * 0.4)
        crop = frame[max(0, y1) : max(0, y1 + jersey_h), max(0, x1) : max(0, x2)]
        return crop if crop.size > 0 else None

    def _extract_colors(self, crop: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        non_grass = ~grass_mask.astype(bool)
        if not non_grass.any():
            return None
        return hsv[non_grass].reshape(-1, 3).astype(np.float32)

    def fit(self, frame: np.ndarray, detections: sv.Detections, player_class_ids: list[int]) -> None:
        all_colors = []
        class_ids = detections.class_id
        if class_ids is None:
            return

        for i, bbox in enumerate(detections.xyxy):
            if int(class_ids[i]) not in player_class_ids:
                continue
            crop = self._crop_jersey(frame, bbox)
            if crop is None:
                continue
            colors = self._extract_colors(crop)
            if colors is not None and len(colors) >= 5:
                sample = colors[np.random.choice(len(colors), min(50, len(colors)), replace=False)]
                all_colors.append(sample)

        if len(all_colors) < 3:
            return

        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.kmeans.fit(np.vstack(all_colors))
        self.fitted = True

        centres = self.kmeans.cluster_centers_
        ref_cluster = int(np.argmin(centres[:, 1]))
        teams = [i for i in range(3) if i != ref_cluster]
        self.cluster_to_team = {
            teams[0]: "team_a",
            teams[1]: "team_b",
            ref_cluster: "referee",
        }

    def classify(self, frame: np.ndarray, track_id: int, bbox_px: np.ndarray) -> str:
        if not self.fitted or self.kmeans is None:
            return "unknown"

        crop = self._crop_jersey(frame, bbox_px)
        if crop is None:
            return "unknown"

        colors = self._extract_colors(crop)
        if colors is None or len(colors) == 0:
            return "unknown"

        sample = colors[np.random.choice(len(colors), min(30, len(colors)), replace=False)]
        preds = self.kmeans.predict(sample)
        cluster_id = int(np.bincount(preds).argmax())
        label = self.cluster_to_team.get(cluster_id, "unknown")

        history = self.vote_history[track_id]
        history.append(label)
        if len(history) > HISTORY_FRAMES:
            history.pop(0)

        return Counter(history).most_common(1)[0][0]

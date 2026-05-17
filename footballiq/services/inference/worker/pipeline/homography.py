from typing import Optional

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

FIELD_CORNERS_M = np.array(
    [[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]],
    dtype=np.float32,
)


class FieldHomography:
    def __init__(self) -> None:
        self.H: Optional[np.ndarray] = None

    def calibrate_manual(self, corner_pixels: list[dict]) -> bool:
        if len(corner_pixels) != 4:
            return False

        src = np.array([[c["x"], c["y"]] for c in corner_pixels], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, FIELD_CORNERS_M, cv2.RANSAC, 5.0)
        return self.H is not None

    def pixel_to_metres(self, px: float, py: float) -> Optional[tuple[float, float]]:
        if self.H is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        x_m = float(result[0][0][0])
        y_m = float(result[0][0][1])
        x_m = max(0.0, min(PITCH_LENGTH_M, x_m))
        y_m = max(0.0, min(PITCH_WIDTH_M, y_m))
        return x_m, y_m

    def to_flat_list(self) -> Optional[list[float]]:
        if self.H is None:
            return None
        return self.H.flatten().tolist()

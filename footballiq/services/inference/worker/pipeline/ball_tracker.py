from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter


class BallKalmanTracker:
    def __init__(self, fps: float = 25.0) -> None:
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0 / max(fps, 1.0)

        self.kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.R = np.diag([10.0, 10.0])
        self.kf.Q = np.eye(4) * 0.1
        self.kf.P = np.eye(4) * 1000

        self.initialized = False
        self.frames_missing = 0
        self.max_missing = 30

    def update(self, ball_px: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if ball_px is not None:
            self.frames_missing = 0
            if not self.initialized:
                self.kf.x = np.array([[ball_px[0]], [ball_px[1]], [0.0], [0.0]], dtype=np.float32)
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.update(np.array([[ball_px[0]], [ball_px[1]]], dtype=np.float32))
        else:
            if not self.initialized:
                return None
            self.frames_missing += 1
            if self.frames_missing > self.max_missing:
                return None
            self.kf.predict()

        return float(self.kf.x[0]), float(self.kf.x[1])

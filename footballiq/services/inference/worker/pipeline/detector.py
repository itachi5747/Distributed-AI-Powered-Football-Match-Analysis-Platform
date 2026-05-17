import os

import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = os.path.join(
    os.environ.get("MODEL_DIR", "/models"),
    os.environ.get("DETECTION_MODEL", "yolov8x.pt"),
)
CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.45"))

CLASS_PLAYER = 0
CLASS_BALL = 1
CLASS_REFEREE = 2
CLASS_GOALKEEPER = 3
CLASS_GOALPOST = 4


class FootballDetector:
    def __init__(self) -> None:
        self.model = YOLO(MODEL_PATH)
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=0.45,
            max_det=100,
            verbose=False,
            half=self.device == "cuda",
        )
        return sv.Detections.from_ultralytics(results[0])

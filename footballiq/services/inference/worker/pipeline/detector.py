import os

import numpy as np
import supervision as sv
from ultralytics import YOLO

model_dir = os.environ.get("MODEL_DIR", "/models")
if model_dir == "/models":
    try:
        if not os.path.exists("/models"):
            os.makedirs("/models", exist_ok=True)
        elif not os.access("/models", os.W_OK):
            raise PermissionError()
    except (PermissionError, OSError):
        # Fallback to local 'models' folder in the project root if /models is not writable
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
        os.makedirs(model_dir, exist_ok=True)

MODEL_PATH = os.path.join(
    model_dir,
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
        import torch
        self.model = YOLO(MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

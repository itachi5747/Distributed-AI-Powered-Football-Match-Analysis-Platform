import supervision as sv


class FootballTracker:
    def __init__(self, fps: float = 25.0) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=int(max(fps, 1.0) * 1.5),
            minimum_matching_threshold=0.8,
            frame_rate=int(max(fps, 1.0)),
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        self.tracker.reset()

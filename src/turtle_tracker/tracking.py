from dataclasses import dataclass
from datetime import datetime

from .calibration import HomographyCalibration
from .vision import Detection


@dataclass
class Track:
    x: float
    y: float
    timestamp: datetime
    speed: float = 0.0


class PositionTracker:
    def __init__(self, calibration: HomographyCalibration):
        self.calibration = calibration
        self.previous: Track | None = None

    def update(self, detection: Detection, timestamp: datetime) -> Track:
        x, y = self.calibration.pixel_to_meters(detection.x_pixel, detection.y_pixel)
        speed = 0.0
        if self.previous:
            elapsed = max((timestamp - self.previous.timestamp).total_seconds(), 0.001)
            speed = ((x - self.previous.x) ** 2 + (y - self.previous.y) ** 2) ** 0.5 / elapsed
        track = Track(x, y, timestamp, speed)
        self.previous = track
        return track

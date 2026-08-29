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


@dataclass
class DoorCrossing:
    event: str  # "entered_house" or "left_house"
    timestamp: datetime


class DoorCrossingTracker:
    """Tracks which side of the door threshold the tortoise is on and reports crossings."""

    def __init__(self) -> None:
        self.last_side: str | None = None

    def update(self, side: str | None, timestamp: datetime) -> DoorCrossing | None:
        if side is None:
            return None
        crossing = None
        if self.last_side is not None and side != self.last_side:
            event = "entered_house" if side == "inside" else "left_house"
            crossing = DoorCrossing(event, timestamp)
        self.last_side = side
        return crossing

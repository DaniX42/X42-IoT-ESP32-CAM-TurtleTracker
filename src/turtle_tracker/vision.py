from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x_pixel: float
    y_pixel: float
    confidence: float


ENCLOSURE_POLYGON_SOURCE = np.array(
    [[326, 200], [336, 244], [272, 309], [148, 263], [102, 256], [88, 219], [65, 176], [203, 121]],
    dtype=np.float32,
)


def scaled_enclosure_polygon(width: int, height: int) -> np.ndarray:
    scale = np.array([width / 640, height / 480], dtype=np.float32)
    polygon = ENCLOSURE_POLYGON_SOURCE * scale
    center = polygon.mean(axis=0)
    return np.rint(center + (polygon - center) * 1.01).astype(np.int32)


def draw_enclosure_overlay(image: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    polygon = scaled_enclosure_polygon(image.shape[1], image.shape[0])
    turquoise = (208, 224, 64)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end.astype(np.float32) - start.astype(np.float32)
        edge_length = float(np.linalg.norm(edge))
        if edge_length == 0:
            continue
        direction = edge / edge_length
        distance = 0.0
        while distance < edge_length:
            dash_end = min(distance + 14.0, edge_length)
            dash_start_point = np.rint(start + direction * distance).astype(np.int32)
            dash_end_point = np.rint(start + direction * dash_end).astype(np.int32)
            cv2.line(overlay, tuple(dash_start_point), tuple(dash_end_point), turquoise, 2)
            distance += 24.0
    return overlay


# Door camera door-zone geometry, defined in the raw (unrotated) VGA sensor
# orientation used by the detector, matching the 3 reference lines marked on
# the mounting photo (left frame edge, right frame edge, threshold).
# Adjust these to match the actual mounted camera; see docs/calibration.md.
DOOR_ZONE_LEFT_SOURCE = np.array([[220, 60], [190, 300]], dtype=np.float32)
DOOR_ZONE_RIGHT_SOURCE = np.array([[420, 60], [450, 300]], dtype=np.float32)
DOOR_THRESHOLD_SOURCE = np.array([[150, 190], [470, 190]], dtype=np.float32)


def door_corridor_polygon() -> np.ndarray:
    return np.array(
        [DOOR_ZONE_LEFT_SOURCE[0], DOOR_ZONE_RIGHT_SOURCE[0], DOOR_ZONE_RIGHT_SOURCE[1], DOOR_ZONE_LEFT_SOURCE[1]],
        dtype=np.float32,
    )


def _line_side(x_pixel: float, y_pixel: float, line: np.ndarray) -> float:
    direction = line[1] - line[0]
    to_point = np.array([x_pixel, y_pixel], dtype=np.float32) - line[0]
    return float(direction[0] * to_point[1] - direction[1] * to_point[0])


def classify_door_detection(x_pixel: float, y_pixel: float) -> str | None:
    """Return "inside", "outside", or None if the point is outside the door corridor."""
    polygon = door_corridor_polygon()
    if cv2.pointPolygonTest(polygon, (x_pixel, y_pixel), False) < 0:
        return None
    return "inside" if _line_side(x_pixel, y_pixel, DOOR_THRESHOLD_SOURCE) > 0 else "outside"


def draw_door_calibration_overlay(image: np.ndarray) -> np.ndarray:
    """Draw the door-zone lines for calibration only; never used on the live door feed."""
    overlay = image.copy()
    yellow = (0, 215, 255)
    magenta = (255, 0, 255)
    for line, color in ((DOOR_ZONE_LEFT_SOURCE, yellow), (DOOR_ZONE_RIGHT_SOURCE, yellow), (DOOR_THRESHOLD_SOURCE, magenta)):
        start, end = line.astype(np.int32)
        cv2.line(overlay, tuple(start), tuple(end), color, 2)
    return overlay


class MotionDetector:
    """Small OpenCV baseline detector; replace with a tortoise model later."""

    def __init__(self) -> None:
        self._background = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=True)

    def detect(self, image: np.ndarray) -> Detection | None:
        mask = self._background.apply(image)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 25:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        return Detection(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], min(1.0, area / 5000))


def decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Payload is not a valid JPEG image")
    return image

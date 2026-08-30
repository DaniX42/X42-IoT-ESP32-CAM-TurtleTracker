from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x_pixel: float
    y_pixel: float
    confidence: float
    x_min: int = 0  # Bounding box for motion crop
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0


ENCLOSURE_POLYGON_SOURCE = np.array(
    [[326, 200], [336, 244], [272, 309], [148, 263], [102, 256], [88, 219], [65, 176], [203, 121]],
    dtype=np.float32,
)


def scaled_enclosure_polygon(width: int, height: int) -> np.ndarray:
    scale = np.array([width / 640, height / 480], dtype=np.float32)
    polygon = ENCLOSURE_POLYGON_SOURCE * scale
    center = polygon.mean(axis=0)
    return np.rint(center + (polygon - center) * 1.01).astype(np.int32)


def in_enclosure_polygon(x_pixel: float, y_pixel: float, width: int, height: int) -> bool:
    """Reject motion detected outside the calibrated enclosure area (e.g. plants, shadows outside the fence)."""
    polygon = scaled_enclosure_polygon(width, height)
    return cv2.pointPolygonTest(polygon, (x_pixel, y_pixel), False) >= 0


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


# Door camera entrance zone, defined as fractions (0..1) of the raw (unrotated)
# frame so it scales to the camera's actual resolution. The actual door opening
# is vertical in the image. Using the current live-view orientation: shift in
# X-coordinates moves the lines up/down visually.
DOOR_NEAR_LINE_FRACTION = np.array([[0.24625, 0.00], [0.24625, 0.70]], dtype=np.float32)
DOOR_FAR_LINE_FRACTION = np.array([[0.48625, 0.00], [0.48625, 0.70]], dtype=np.float32)


def _scale_fraction_line(line: np.ndarray, width: int, height: int) -> np.ndarray:
    return line * np.array([width, height], dtype=np.float32)


def door_entrance_lines(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        _scale_fraction_line(DOOR_NEAR_LINE_FRACTION, width, height),
        _scale_fraction_line(DOOR_FAR_LINE_FRACTION, width, height),
    )


def classify_door_detection(x_pixel: float, y_pixel: float, width: int, height: int) -> str:
    """Return the door-zone state for the current turtle position."""
    top_third = height / 3
    middle_third = 2 * height / 3
    if y_pixel < top_third:
        return "outside"
    if y_pixel < middle_third:
        return "buffer"
    return "inside"


def draw_door_calibration_overlay(image: np.ndarray) -> np.ndarray:
    """Draw the door boundary lines for calibration or live preview."""
    overlay = image.copy()
    height, width = image.shape[:2]
    near_line, far_line = door_entrance_lines(width, height)
    yellow = (0, 215, 255)
    magenta = (255, 0, 255)
    for line, color in ((near_line, yellow), (far_line, magenta)):
        start, end = line.astype(np.int32)
        cv2.line(overlay, tuple(start), tuple(end), color, 2)
    return overlay


def draw_position_map(
    length_meters: float,
    width_meters: float,
    x: float | None,
    y: float | None,
    inside_house: bool,
) -> np.ndarray:
    """Schematic top-down enclosure map with the tortoise position, or "@Home" when inside the house."""
    scale = 80.0
    margin = 40
    enclosure_width_px = round(length_meters * scale)
    enclosure_height_px = round(width_meters * scale)
    image = np.full((enclosure_height_px + margin * 2, enclosure_width_px + margin * 2, 3), (60, 130, 60), dtype=np.uint8)
    top_left = (margin, margin)
    bottom_right = (margin + enclosure_width_px, margin + enclosure_height_px)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)

    if inside_house:
        house_point = (margin, margin + enclosure_height_px // 2)
        cv2.circle(image, house_point, 10, (0, 165, 255), -1)
        cv2.putText(image, "@Home", (house_point[0] + 16, house_point[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    elif x is not None and y is not None:
        clamped_x = min(max(x, 0.0), length_meters)
        clamped_y = min(max(y, 0.0), width_meters)
        out_of_bounds = clamped_x != x or clamped_y != y
        point = (margin + round(clamped_x * scale), margin + round(clamped_y * scale))
        color = (0, 165, 255) if out_of_bounds else (0, 0, 255)
        cv2.circle(image, point, 8, color, -1)
        label = f"{x:.1f}m, {y:.1f}m" + (" (out of range)" if out_of_bounds else "")
        cv2.putText(image, label, (point[0] + 12, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "no data", (margin, margin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


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
        x_center = moments["m10"] / moments["m00"]
        y_center = moments["m01"] / moments["m00"]
        x_min, y_min, w, h = cv2.boundingRect(contour)
        x_max = min(x_min + w, image.shape[1])
        y_max = min(y_min + h, image.shape[0])
        return Detection(x_center, y_center, min(1.0, area / 5000), x_min, y_min, x_max, y_max)


def decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Payload is not a valid JPEG image")
    return image


def draw_detection_overlay(image: np.ndarray, detection: Detection, timestamp: datetime | None = None) -> np.ndarray:
    """Draw a marker at the detected turtle position with elapsed time info."""
    overlay = image.copy()
    x_pixel = int(detection.x_pixel)
    y_pixel = int(detection.y_pixel)
    # Draw a cyan circle around the detection center
    cv2.circle(overlay, (x_pixel, y_pixel), 20, (255, 255, 0), 2)  # Cyan circle
    # Draw a small crosshair
    cv2.line(overlay, (x_pixel - 10, y_pixel), (x_pixel + 10, y_pixel), (255, 255, 0), 2)
    cv2.line(overlay, (x_pixel, y_pixel - 10), (x_pixel, y_pixel + 10), (255, 255, 0), 2)
    # Draw confidence text (rotated 90° right)
    confidence_text = f"{detection.confidence * 100:.0f}%"
    cv2.putText(overlay, confidence_text, (x_pixel + 25, y_pixel - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    # Draw elapsed time if timestamp provided
    if timestamp is not None:
        now = datetime.now(timezone.utc)
        elapsed = (now - timestamp).total_seconds()
        time_text = f"{int(elapsed)}s ago"
        cv2.putText(overlay, time_text, (x_pixel + 25, y_pixel + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return overlay

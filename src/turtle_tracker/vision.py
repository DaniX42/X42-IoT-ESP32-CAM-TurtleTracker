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
PREVIEW_CROP_DISPLAY = (100, 280, 330, 600)


def scaled_enclosure_polygon(width: int, height: int) -> np.ndarray:
    scale = np.array([width / 640, height / 480], dtype=np.float32)
    polygon = ENCLOSURE_POLYGON_SOURCE * scale
    center = polygon.mean(axis=0)
    return np.rint(center + (polygon - center) * 1.01).astype(np.int32)


def enclosure_mask(image: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [scaled_enclosure_polygon(image.shape[1], image.shape[0])], 255)
    return mask


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
        dash_length = 14.0
        gap_length = 10.0
        distance = 0.0
        while distance < edge_length:
            dash_end = min(distance + dash_length, edge_length)
            dash_start_point = np.rint(start + direction * distance).astype(np.int32)
            dash_end_point = np.rint(start + direction * dash_end).astype(np.int32)
            cv2.line(overlay, tuple(dash_start_point), tuple(dash_end_point), turquoise, 2)
            distance += dash_length + gap_length
    return overlay


def crop_preview(image: np.ndarray) -> np.ndarray:
    x_start, y_start, x_end, y_end = PREVIEW_CROP_DISPLAY
    x_scale = image.shape[1] / 480
    y_scale = image.shape[0] / 640
    x_start, x_end = np.rint(np.array([x_start, x_end]) * x_scale).astype(int)
    y_start, y_end = np.rint(np.array([y_start, y_end]) * y_scale).astype(int)
    return image[max(0, y_start):min(image.shape[0], y_end), max(0, x_start):min(image.shape[1], x_end)]


class MotionDetector:
    """Small OpenCV baseline detector; replace with a tortoise model later."""

    def __init__(self) -> None:
        self._background = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=True)

    def detect(self, image: np.ndarray) -> Detection | None:
        mask = self._background.apply(image)
        mask = cv2.bitwise_and(mask, enclosure_mask(image))
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

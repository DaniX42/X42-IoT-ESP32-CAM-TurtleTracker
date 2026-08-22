from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x_pixel: float
    y_pixel: float
    confidence: float


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

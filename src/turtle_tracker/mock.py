import cv2
import numpy as np


def mock_jpeg(width: int = 640, height: int = 360, x: int | None = None, y: int | None = None) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (55, 105, 55)
    cv2.rectangle(image, (8, 8), (width - 8, height - 8), (180, 180, 180), 2)
    cv2.circle(image, (x or width // 2, y or height // 2), 28, (35, 70, 130), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode mock image")
    return encoded.tobytes()

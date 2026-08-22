from pathlib import Path

import cv2
import numpy as np


class HomographyCalibration:
    def __init__(self, image_points: list[list[float]], enclosure_length: float = 7.0, enclosure_width: float = 2.5):
        if len(image_points) != 4:
            raise ValueError("Exactly four image reference points are required")
        source = np.asarray(image_points, dtype=np.float32)
        target = np.asarray([[0, 0], [enclosure_length, 0], [enclosure_length, enclosure_width], [0, enclosure_width]], dtype=np.float32)
        matrix, _ = cv2.findHomography(source, target)
        if matrix is None:
            raise ValueError("Reference points cannot define a homography")
        self.matrix = matrix

    def pixel_to_meters(self, x_pixel: float, y_pixel: float) -> tuple[float, float]:
        point = np.asarray([[[x_pixel, y_pixel]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)[0, 0]
        return float(transformed[0]), float(transformed[1])

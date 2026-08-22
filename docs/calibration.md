# Calibration

## Ziel

Umrechnung Bildkoordinaten in reale Gehegekoordinaten.

## Bekannte Maße

Gehege:

# Calibration

Calibration maps image coordinates to enclosure coordinates in metres.

The enclosure defaults to 7.0 m x 2.5 m. Mark four image reference points in clockwise order, starting at the enclosure origin. `HomographyCalibration` uses `cv2.findHomography()` and maps those points to `(0, 0)`, `(7, 0)`, `(7, 2.5)`, and `(0, 2.5)`.

The initial development app uses the four corners of a 640 x 360 image. Real camera calibration should provide measured pixel coordinates before deployment.
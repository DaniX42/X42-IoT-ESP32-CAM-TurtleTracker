# Calibration

## Ziel

Umrechnung Bildkoordinaten in reale Gehegekoordinaten.

## Bekannte Maße

Gehege:

# Calibration

Calibration maps image coordinates to enclosure coordinates in metres.

The enclosure defaults to 7.0 m x 2.5 m. Mark four image reference points in clockwise order, starting at the enclosure origin. `HomographyCalibration` uses `cv2.findHomography()` and maps those points to `(0, 0)`, `(7, 0)`, `(7, 2.5)`, and `(0, 2.5)`.

The initial development app uses the four corners of a 640 x 360 image. Real camera calibration should provide measured pixel coordinates before deployment.

## Door zone calibration

`turtle-cam-door` is mounted above the entrance. Two reference lines mark the entrance gap in `vision.py`, defined as fractions (0-1) of the frame so they scale to the camera's actual resolution (e.g. 1600x1200 UXGA), not a fixed pixel size:

- `DOOR_NEAR_LINE_FRACTION`: the garden-side edge of the entrance gap.
- `DOOR_FAR_LINE_FRACTION`: the house-side edge (further from the camera, e.g. under a ledge).

The midpoint between the two lines is the crossing threshold; the side of this midpoint (`inside` vs `outside`) determines direction. There is no separate left/right corridor filter - the full frame width is considered.

To calibrate: mark the entrance boundary on a photo from `GET /api/frames/turtle-cam-door/latest` (rotated for display), convert the marked points back to the raw sensor orientation (rotate 90° counter-clockwise) and to fractions of the frame size, update the two constants in `vision.py`, then verify with `GET /api/frames/turtle-cam-door/calibration`.
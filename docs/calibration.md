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

`turtle-cam-door` is mounted above the entrance. Three reference lines define the detection zone in `vision.py`, in the raw (unrotated) VGA sensor orientation:

- `DOOR_ZONE_LEFT_SOURCE` / `DOOR_ZONE_RIGHT_SOURCE`: the corridor edges (left/right door frame), used to ignore motion outside the doorway.
- `DOOR_THRESHOLD_SOURCE`: the crossing line; the side of this line (`inside` vs `outside`) determines direction.

To calibrate: mark the entrance boundary on a photo from `GET /api/frames/turtle-cam-door/latest` (rotated for display), convert the marked points back to the raw sensor orientation (rotate 90° counter-clockwise), update the three constants in `vision.py`, then verify with `GET /api/frames/turtle-cam-door/calibration`.
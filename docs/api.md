# REST API

The service exposes OpenAPI documentation at `/docs`.

## Current position

`GET /api/position` returns the latest position or `404` if no frame has produced a detection.

```json
{
  "timestamp": "2026-08-22T10:00:00Z",
  "x": 2.3,
  "y": 1.7,
  "inside_house": false,
  "speed": 0.04,
  "confidence": 0.72
}
```

## History

`GET /api/history?limit=1000` returns recent positions, newest first.

## Heatmap data

`GET /api/heatmap?grid=0.5` returns counts grouped into square metre bins.

## Frame ingestion

`POST /api/frames/{camera_id}` accepts a JPEG in multipart field `file`.

`POST /api/mock/frame` runs the same path using a generated JPEG while camera hardware is unavailable.

## Latest frame

`GET /api/frames/{camera_id}/latest` returns the latest valid JPEG received for the camera with `Content-Type: image/jpeg`. It returns `404` until the first frame has been received. The endpoint is intended for Home Assistant's Generic Camera integration.

`turtle-cam-door` frames never carry an overlay on this endpoint.

## Door calibration frame

`GET /api/frames/turtle-cam-door/calibration` returns the latest door-camera JPEG with the door-zone lines (left/right corridor edges and threshold) drawn on top, for calibrating `vision.DOOR_ZONE_LEFT_SOURCE` / `DOOR_ZONE_RIGHT_SOURCE` / `DOOR_THRESHOLD_SOURCE`. It is calibration-only and never served on `/latest`.

## Door crossing detection

Motion detected on the `turtle-cam-door` camera is classified as `inside` or `outside` the door threshold. A side change writes an `entered_house` or `left_house` row to the `events` table and updates `inside_house` on the next enclosure position.

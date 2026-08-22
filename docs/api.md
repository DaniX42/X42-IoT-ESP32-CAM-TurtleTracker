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

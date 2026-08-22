# Turtle Tracker

A camera-based position tracking service for one untagged tortoise in a 7 m x 2.5 m outdoor enclosure.

## Current skeleton

- FastAPI REST backend with OpenAPI documentation
- SQLite persistence for positions and events
- Multipart JPEG ingestion for ESP32-CAM devices
- OpenCV MOG2 motion detection baseline
- Four-point homography calibration from pixels to metres
- Single-object position tracking with speed calculation
- Deterministic mock image endpoint for development
- Pytest coverage and Docker deployment files

The detector is intentionally a baseline. It can later be replaced by a tortoise classifier such as YOLO without changing the ingestion or persistence contracts.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/uvicorn turtle_tracker.app:app --reload
```

The API is available at `http://localhost:8000`. Interactive OpenAPI documentation is available at `/docs`.

For an ESP32-CAM on the same LAN, start the backend with the default `0.0.0.0` binding and set the camera's `api_url` to the host computer's LAN address. Do not use `localhost` in the firmware configuration.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health |
| GET | `/api/position` | Latest tracked position |
| GET | `/api/history?limit=1000` | Recent positions |
| GET | `/api/heatmap?grid=0.5` | Binned position counts |
| POST | `/api/frames/{camera_id}` | Ingest a JPEG as multipart field `file` |
| POST | `/api/mock/frame` | Ingest a generated development frame |

## Docker

```bash
cp .env.example .env
docker compose up --build
```

SQLite data is persisted in `./data`.

The backend can later be moved to a dedicated Proxmox container without changing the API or MQTT contract; update the camera `api_url` to the container's fixed address.

## Repository layout

```text
src/turtle_tracker/  Application package
tests/                Unit and API tests
docs/                 Architecture and calibration notes
Dockerfile            Production container
docker-compose.yml    Local deployment
```

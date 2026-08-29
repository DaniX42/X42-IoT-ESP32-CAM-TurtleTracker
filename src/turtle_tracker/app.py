from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile

from .calibration import HomographyCalibration
from .config import Settings, get_settings
from .db import Database, row_to_dict
from .models import HeatmapPoint, IngestResponse, Position
from .mock import mock_jpeg
from .mqtt import MqttPublisher
from .tracking import DoorCrossingTracker, PositionTracker
from .vision import (
    MotionDetector,
    classify_door_detection,
    decode_jpeg,
    draw_door_calibration_overlay,
    draw_enclosure_overlay,
    draw_position_map,
    in_enclosure_polygon,
)

DOOR_CAMERA_ID = "turtle-cam-door"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _prepare_frame(camera_id: str, payload: bytes) -> object:
    image = decode_jpeg(payload)
    if camera_id != DOOR_CAMERA_ID:
        image = draw_enclosure_overlay(image)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = draw_door_calibration_overlay(image)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def create_app(settings: Settings | None = None, database: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = database or Database(settings.database_path)
    database.initialize()
    calibration = HomographyCalibration(
        [[0, 0], [640, 0], [640, 360], [0, 360]],
        settings.enclosure_length_meters,
        settings.enclosure_width_meters,
    )
    detector = MotionDetector()
    tracker = PositionTracker(calibration, settings.enclosure_length_meters, settings.enclosure_width_meters)
    door_tracker = DoorCrossingTracker()
    house_state = {"inside_house": False}
    mqtt_publisher = MqttPublisher(
        settings.mqtt_host,
        settings.mqtt_port,
        settings.mqtt_user,
        settings.mqtt_password,
        settings.mqtt_topic_prefix,
        settings.mqtt_enabled,
    )
    latest_frames: dict[str, bytes] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        yield

    app = FastAPI(title="Turtle Tracker API", version="0.1.0", lifespan=lifespan)

    async def process_frame(camera_id: str, payload: bytes) -> IngestResponse:
        if not camera_id.strip():
            raise HTTPException(status_code=400, detail="camera_id is required")
        if not payload:
            raise HTTPException(status_code=400, detail="JPEG payload is required")
        try:
            image = decode_jpeg(payload)
            detection = detector.detect(image)
        except ValueError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error
        latest_frames[camera_id] = payload
        if detection is None or detection.confidence < settings.min_confidence:
            return IngestResponse(accepted=False, reason="No confident motion detected")
        timestamp = _utc_now()
        if camera_id == DOOR_CAMERA_ID:
            side = classify_door_detection(detection.x_pixel, detection.y_pixel, image.shape[1], image.shape[0])
            crossing = door_tracker.update(side, timestamp)
            if crossing is not None:
                house_state["inside_house"] = crossing.event == "entered_house"
                database.insert_event(timestamp.isoformat(), crossing.event)
                mqtt_publisher.publish_event(crossing.event)
                mqtt_publisher.publish_state(house_state["inside_house"])
            return IngestResponse(accepted=True)
        if not in_enclosure_polygon(detection.x_pixel, detection.y_pixel, image.shape[1], image.shape[0]):
            return IngestResponse(accepted=False, reason="Motion detected outside the enclosure")
        track = tracker.update(detection, timestamp)
        position = Position(
            timestamp=timestamp,
            x=track.x,
            y=track.y,
            inside_house=house_state["inside_house"],
            speed=track.speed,
            confidence=detection.confidence,
        )
        database.insert_position(timestamp.isoformat(), position.x, position.y, position.inside_house, position.speed, position.confidence)
        return IngestResponse(accepted=True, position=position)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/frames/{camera_id}/latest", response_class=Response)
    def latest_frame(camera_id: str) -> Response:
        payload = latest_frames.get(camera_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="No frame received")
        image = _prepare_frame(camera_id, payload)
        success, rotated_payload = cv2.imencode(".jpg", image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode latest frame")
        return Response(content=rotated_payload.tobytes(), media_type="image/jpeg")

    @app.get("/api/frames/turtle-cam-door/calibration", response_class=Response)
    def door_calibration_frame() -> Response:
        payload = latest_frames.get(DOOR_CAMERA_ID)
        if payload is None:
            raise HTTPException(status_code=404, detail="No frame received")
        image = draw_door_calibration_overlay(decode_jpeg(payload))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode calibration frame")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    @app.get("/api/frames/{camera_id}/latest/square", response_class=Response)
    @app.get("/api/frames/{camera_id}/latest/square.jpg", response_class=Response)
    def latest_frame_square(camera_id: str) -> Response:
        payload = latest_frames.get(camera_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="No frame received")
        image = _prepare_frame(camera_id, payload)
        height, width = image.shape[:2]
        size = 500
        scale = min(size / width, size / height)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        background_scale = max(size / width, size / height)
        background = cv2.resize(
            image,
            (round(width * background_scale), round(height * background_scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        background = cv2.GaussianBlur(background, (0, 0), 18)
        background_height, background_width = background.shape[:2]
        background = background[
            (background_height - size) // 2 : (background_height - size) // 2 + size,
            (background_width - size) // 2 : (background_width - size) // 2 + size,
        ]
        x_offset = (size - width) // 2
        y_offset = (size - height) // 2
        background[y_offset : y_offset + height, x_offset : x_offset + width] = image
        square = background
        success, square_payload = cv2.imencode(".jpg", square)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode square latest frame")
        return Response(content=square_payload.tobytes(), media_type="image/jpeg")

    @app.get("/api/position", response_model=Position)
    def current_position() -> Position:
        row = database.latest_position()
        if row is None:
            raise HTTPException(status_code=404, detail="No position recorded")
        return Position(**{**row_to_dict(row), "inside_house": bool(row["inside_house"])})

    @app.get("/api/history", response_model=list[Position])
    def history(limit: int = 1000) -> list[Position]:
        if not 1 <= limit <= 10000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 10000")
        return [Position(**{**row_to_dict(row), "inside_house": bool(row["inside_house"])}) for row in database.positions(limit)]

    @app.get("/api/heatmap", response_model=list[HeatmapPoint])
    def heatmap(grid: float = 0.5) -> list[HeatmapPoint]:
        if grid <= 0:
            raise HTTPException(status_code=400, detail="grid must be positive")
        buckets: dict[tuple[int, int], int] = {}
        for row in database.positions(10000):
            key = (int(row["x"] / grid), int(row["y"] / grid))
            buckets[key] = buckets.get(key, 0) + 1
        return [HeatmapPoint(x=key[0] * grid, y=key[1] * grid, count=count) for key, count in buckets.items()]

    @app.get("/api/position/map", response_class=Response)
    def position_map() -> Response:
        row = database.latest_position()
        x = row["x"] if row is not None else None
        y = row["y"] if row is not None else None
        image = draw_position_map(settings.enclosure_length_meters, settings.enclosure_width_meters, x, y, house_state["inside_house"])
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode position map")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    @app.post("/api/frames/{camera_id}", response_model=IngestResponse)
    async def ingest_frame(camera_id: str, request: Request, file: UploadFile | None = File(None)) -> IngestResponse:
        payload = await file.read() if file is not None else await request.body()
        return await process_frame(camera_id, payload)

    @app.post("/api/mock/frame", response_model=IngestResponse)
    async def ingest_mock_frame() -> IngestResponse:
        if not settings.mock_images_enabled:
            raise HTTPException(status_code=404, detail="Mock images are disabled")
        return await process_frame("mock", mock_jpeg())

    return app


app = create_app()

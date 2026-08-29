from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from turtle_tracker.app import create_app
from turtle_tracker.config import Settings
from turtle_tracker.db import Database
from turtle_tracker.mock import mock_jpeg
from turtle_tracker.vision import decode_jpeg


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(database_path=tmp_path / "test.db")
    return TestClient(create_app(settings=settings, database=Database(settings.database_path)))


def test_health_and_empty_position(tmp_path: Path):
    client = make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/position").status_code == 404


def test_mock_frame_is_persisted(tmp_path: Path):
    client = make_client(tmp_path)
    response = client.post("/api/mock/frame")
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert client.get("/api/position").json()["x"] == body["position"]["x"]
    assert len(client.get("/api/history").json()) == 1


def test_rejects_invalid_jpeg(tmp_path: Path):
    client = make_client(tmp_path)
    response = client.post("/api/frames/outdoor", files={"file": ("frame.jpg", b"invalid", "image/jpeg")})
    assert response.status_code == 415


def test_accepts_raw_jpeg_payload_from_camera(tmp_path: Path):
    client = make_client(tmp_path)
    response = client.post("/api/frames/outdoor", content=mock_jpeg(), headers={"content-type": "image/jpeg"})
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_latest_frame_is_available_as_jpeg(tmp_path: Path):
    client = make_client(tmp_path)
    payload = mock_jpeg()

    response = client.post("/api/frames/outdoor", content=payload, headers={"content-type": "image/jpeg"})

    assert response.status_code == 200
    latest = client.get("/api/frames/outdoor/latest")
    assert latest.status_code == 200
    assert latest.headers["content-type"] == "image/jpeg"
    assert decode_jpeg(latest.content).shape[:2] == (640, 360)


def test_door_latest_frame_is_vga_without_overlay(tmp_path: Path):
    client = make_client(tmp_path)
    image = np.full((480, 640, 3), (40, 80, 160), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    response = client.post("/api/frames/turtle-cam-door", content=encoded.tobytes())

    assert response.status_code == 200
    latest = decode_jpeg(client.get("/api/frames/turtle-cam-door/latest").content)
    assert latest.shape[:2] == (640, 480)
    assert np.mean(np.abs(latest.astype(np.int16) - np.rot90(image, 3).astype(np.int16))) < 3


def test_latest_frame_square_keeps_full_image(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/frames/outdoor", content=mock_jpeg(), headers={"content-type": "image/jpeg"})

    square = client.get("/api/frames/outdoor/latest/square")

    assert square.status_code == 200
    assert square.headers["content-type"] == "image/jpeg"
    image = decode_jpeg(square.content)
    assert image.shape[:2] == (500, 500)


def test_latest_frame_square_jpg_alias_is_jpeg(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/frames/outdoor", content=mock_jpeg(), headers={"content-type": "image/jpeg"})

    response = client.get("/api/frames/outdoor/latest/square.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert decode_jpeg(response.content).shape[:2] == (500, 500)


def test_position_map_without_data_returns_jpeg(tmp_path: Path):
    client = make_client(tmp_path)

    response = client.get("/api/position/map")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_position_map_reflects_latest_position(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/mock/frame")

    response = client.get("/api/position/map")

    assert response.status_code == 200
    assert decode_jpeg(response.content) is not None

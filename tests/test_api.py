from pathlib import Path

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

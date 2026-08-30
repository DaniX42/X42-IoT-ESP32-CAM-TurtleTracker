from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi.testclient import TestClient

from turtle_tracker.app import _save_motion_crop, create_app
from turtle_tracker.config import Settings
from turtle_tracker.db import Database
from turtle_tracker.vision import Detection


def make_client(tmp_path) -> TestClient:
    settings = Settings(database_path=tmp_path / "test.db", motion_crops_path=tmp_path / "motion-crops")
    return TestClient(create_app(settings=settings, database=Database(settings.database_path)))


def test_duplicate_motion_crops_keep_only_one_reference(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    crops_path = tmp_path / "motion-crops"
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    detection = Detection(50, 50, 1.0, 20, 20, 80, 80)

    _save_motion_crop(image, detection, datetime(2026, 1, 1, tzinfo=timezone.utc), "outdoor", crops_path, database)
    _save_motion_crop(image, detection, datetime(2026, 1, 2, tzinfo=timezone.utc), "outdoor", crops_path, database)

    assert len(list(crops_path.glob("*.jpg"))) == 1
    assert len(database.motion_crops()) == 1


def test_negative_label_deletes_crop_unless_kept_for_training(tmp_path):
    client = make_client(tmp_path)
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    success, payload = cv2.imencode(".jpg", image)
    assert success
    detection = Detection(50, 50, 1.0, 20, 20, 80, 80)
    database = Database(tmp_path / "test.db")
    _save_motion_crop(image, detection, datetime(2026, 1, 1, tzinfo=timezone.utc), "outdoor", tmp_path / "motion-crops", database)
    filename = client.get("/api/motion-crops").json()["items"][0]["filename"]

    response = client.post(f"/api/motion-crops/{filename}/label", json={"is_turtle": False})

    assert response.json() == {"deleted": True}
    assert client.get("/api/motion-crops").json()["items"] == []


def test_motion_crop_page_limits_results_and_bulk_labels(tmp_path):
    client = make_client(tmp_path)
    database = Database(tmp_path / "test.db")
    for index in range(51):
        database.insert_motion_crop(f"crop-{index}.jpg", "outdoor", f"2026-01-01T00:00:{index:02d}+00:00", f"{index:016x}")

    first_page = client.get("/api/motion-crops?limit=50&offset=0").json()
    response = client.post(
        "/api/motion-crops/labels",
        json={"items": [{"filename": "crop-0.jpg", "is_turtle": True}, {"filename": "crop-1.jpg", "is_turtle": True}]},
    )

    assert first_page["total"] == 51
    assert len(first_page["items"]) == 50
    assert client.get("/api/motion-crops?limit=50&offset=50").json()["items"][0]["filename"]
    assert response.json() == {"processed": 2, "deleted": 0}
    assert client.get("/api/motion-crops?is_turtle=true").json()["total"] == 2
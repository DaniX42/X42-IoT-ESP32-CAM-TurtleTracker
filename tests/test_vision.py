from datetime import datetime, timezone

from turtle_tracker.tracking import DoorCrossingTracker
from turtle_tracker.vision import DOOR_THRESHOLD_SOURCE, classify_door_detection


def test_classify_door_detection_sides():
    outside_y = DOOR_THRESHOLD_SOURCE[0][1] - 40
    inside_y = DOOR_THRESHOLD_SOURCE[0][1] + 40
    center_x = 300

    assert classify_door_detection(center_x, outside_y) == "outside"
    assert classify_door_detection(center_x, inside_y) == "inside"


def test_classify_door_detection_ignores_points_outside_corridor():
    assert classify_door_detection(10, 10) is None


def test_door_crossing_tracker_reports_entered_then_left():
    tracker = DoorCrossingTracker()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert tracker.update("outside", now) is None
    entered = tracker.update("inside", now)
    assert entered is not None and entered.event == "entered_house"

    left = tracker.update("outside", now)
    assert left is not None and left.event == "left_house"


def test_door_crossing_tracker_ignores_repeated_side():
    tracker = DoorCrossingTracker()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    tracker.update("outside", now)
    assert tracker.update("outside", now) is None

from datetime import datetime, timezone

import numpy as np

from turtle_tracker.tracking import DoorCrossingTracker
from turtle_tracker.vision import (
    ENCLOSURE_POLYGON_SOURCE,
    classify_door_detection,
    door_entrance_lines,
    draw_position_map,
    in_enclosure_polygon,
)


def test_classify_door_detection_sides():
    width, height = 1600, 1200
    center_x = width / 2

    assert classify_door_detection(center_x, height * 0.15, width, height) == "outside"
    assert classify_door_detection(center_x, height * 0.5, width, height) == "buffer"
    assert classify_door_detection(center_x, height * 0.82, width, height) == "inside"


def test_in_enclosure_polygon_accepts_center_and_rejects_far_outside():
    center_x, center_y = ENCLOSURE_POLYGON_SOURCE.mean(axis=0)

    assert in_enclosure_polygon(center_x, center_y, 640, 480) is True
    assert in_enclosure_polygon(5, 5, 640, 480) is False


def test_classify_door_detection_scales_with_resolution():
    assert classify_door_detection(800, 200, 1600, 1200) == classify_door_detection(320, 80, 640, 480)


def test_door_entrance_lines_are_vertical():
    near_line, far_line = door_entrance_lines(1600, 1200)
    assert np.allclose(near_line[:, 0], 0.34 * 1600)
    assert np.allclose(far_line[:, 0], 0.58 * 1600)
    assert np.allclose(near_line[:, 1], [72, 912])
    assert np.allclose(far_line[:, 1], [72, 912])


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


def test_draw_position_map_shape_scales_with_enclosure_size():
    image = draw_position_map(7.0, 2.5, 3.5, 1.25, inside_house=False)
    assert image.shape == (2.5 * 80 + 80, 7.0 * 80 + 80, 3)


def test_draw_position_map_inside_house_ignores_coordinates():
    with_coords = draw_position_map(7.0, 2.5, 3.5, 1.25, inside_house=True)
    without_coords = draw_position_map(7.0, 2.5, None, None, inside_house=True)
    assert (with_coords == without_coords).all()


def test_draw_position_map_clamps_out_of_range_position_into_view():
    image = draw_position_map(7.0, 2.5, 15.97, 2.94, inside_house=False)
    orange = np.array([0, 165, 255])
    assert (image == orange).all(axis=-1).any()

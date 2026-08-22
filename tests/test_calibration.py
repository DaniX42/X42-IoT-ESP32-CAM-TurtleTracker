import pytest

from turtle_tracker.calibration import HomographyCalibration


def test_corners_map_to_enclosure_coordinates():
    calibration = HomographyCalibration([[0, 0], [640, 0], [640, 360], [0, 360]])
    assert calibration.pixel_to_meters(320, 180) == pytest.approx((3.5, 1.25), abs=0.01)

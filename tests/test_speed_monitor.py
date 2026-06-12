import pytest

from src.speed_monitor import SpeedMonitor


def test_no_speed_with_single_observation():
    monitor = SpeedMonitor(pixels_per_meter=8.0, speed_limit_mph=35.0)

    speed, violation = monitor.update(1, (100, 200), 0.0)

    assert speed is None
    assert violation is False


def test_speed_estimation_from_displacement():
    monitor = SpeedMonitor(pixels_per_meter=8.0, speed_limit_mph=100.0, window_seconds=1.0)
    monitor.update(1, (100, 200), 0.0)

    # 80 px / 8 px-per-m = 10 m/s over 1.0s -> ~22.37 mph
    speed, violation = monitor.update(1, (180, 200), 1.0)

    assert speed == pytest.approx(22.37, abs=0.1)
    assert violation is False


def test_speeding_flagged_once_per_track():
    monitor = SpeedMonitor(pixels_per_meter=8.0, speed_limit_mph=15.0, window_seconds=1.0)
    monitor.update(1, (100, 200), 0.0)

    _, first = monitor.update(1, (180, 200), 1.0)
    _, second = monitor.update(1, (260, 200), 2.0)

    assert first is True
    assert second is False


def test_window_drops_old_samples():
    monitor = SpeedMonitor(pixels_per_meter=8.0, speed_limit_mph=100.0, window_seconds=0.5)
    monitor.update(1, (0, 0), 0.0)
    monitor.update(1, (8, 0), 1.0)

    # Only the (1.0 -> 1.4) interval should remain within the 0.5s window
    speed, _ = monitor.update(1, (16, 0), 1.4)

    expected = (8 / 8.0 / 0.4) * SpeedMonitor.MPS_TO_MPH
    assert speed == pytest.approx(expected, rel=0.01)

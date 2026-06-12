import pytest

from src.fps_tracker import FPSTracker


def test_fps_zero_with_no_ticks():
    tracker = FPSTracker()
    assert tracker.fps == 0.0


def test_fps_zero_with_single_tick():
    tracker = FPSTracker()
    tracker.tick()
    assert tracker.fps == 0.0


def test_fps_computed_from_elapsed_time(monkeypatch):
    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr("src.fps_tracker.time.perf_counter", lambda: next(times))

    tracker = FPSTracker()
    tracker.tick()
    tracker.tick()
    tracker.tick()

    # 2 intervals over 0.2s -> 10 fps
    assert tracker.fps == pytest.approx(10.0)

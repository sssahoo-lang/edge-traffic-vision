import numpy as np

from src.preprocessing import resize_frame


def test_resize_frame_changes_dimensions():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = resize_frame(frame, (640, 480))
    assert resized.shape[:2] == (480, 640)


def test_resize_frame_preserves_channels():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_frame(frame, (50, 25))
    assert resized.shape == (25, 50, 3)

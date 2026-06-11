import time
from collections import deque


class FPSTracker:
    """Tracks a rolling-average FPS over the last `window` frames."""

    def __init__(self, window=30):
        self._timestamps = deque(maxlen=window)

    def tick(self):
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

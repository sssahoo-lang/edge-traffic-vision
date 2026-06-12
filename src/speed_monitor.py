import math
from collections import deque


class SpeedMonitor:
    """Estimates vehicle speed from centroid displacement and flags speeding.

    Speed is computed from how far a tracked vehicle's centroid moves (in
    pixels) over `window_seconds`, converted to real-world units via a
    pixels-per-meter calibration constant.

    Calibrating pixels_per_meter: measure a known real-world distance in
    the camera's field of view (e.g. a lane marking interval) and divide
    its length in pixels (in the resized frame) by that distance in
    meters.

    This assumes the camera is roughly perpendicular to the direction of
    travel and that scale is constant across the frame -- accuracy
    degrades under perspective distortion, since objects closer to the
    camera cover more pixels per second than objects further away even
    at the same real-world speed.
    """

    MPS_TO_MPH = 2.23694

    def __init__(self, pixels_per_meter, speed_limit_mph, window_seconds=1.0):
        self.pixels_per_meter = pixels_per_meter
        self.speed_limit_mph = speed_limit_mph
        self.window_seconds = window_seconds
        self._history = {}
        self._flagged_ids = set()

    def update(self, track_id, centroid, timestamp):
        """Record an observation and return (speed_mph, is_new_violation).

        speed_mph is None until there are at least two observations
        spanning a positive time delta. is_new_violation is True the
        first time this track's estimated speed exceeds the configured
        limit.
        """
        history = self._history.setdefault(track_id, deque())
        history.append((timestamp, centroid[0], centroid[1]))

        while history[-1][0] - history[0][0] > self.window_seconds:
            history.popleft()

        if len(history) < 2:
            return None, False

        t0, x0, y0 = history[0]
        t1, x1, y1 = history[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return None, False

        distance_m = math.hypot(x1 - x0, y1 - y0) / self.pixels_per_meter
        speed_mph = (distance_m / elapsed) * self.MPS_TO_MPH

        is_violation = False
        if speed_mph > self.speed_limit_mph and track_id not in self._flagged_ids:
            self._flagged_ids.add(track_id)
            is_violation = True

        return speed_mph, is_violation

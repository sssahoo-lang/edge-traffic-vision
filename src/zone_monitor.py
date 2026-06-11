import cv2
import numpy as np


class ZoneMonitor:
    """Flags vehicles whose centroid enters a restricted polygon zone.

    Each tracked vehicle (by track_id) is only flagged once, so a vehicle
    sitting in the zone doesn't generate repeated violations.
    """

    def __init__(self, zone_fractions):
        self._zone_fractions = zone_fractions
        self._polygon = None
        self._flagged_ids = set()

    def polygon(self, frame_shape):
        if self._polygon is None:
            h, w = frame_shape[:2]
            self._polygon = np.array(
                [(int(x * w), int(y * h)) for x, y in self._zone_fractions],
                dtype=np.int32,
            )
        return self._polygon

    def check(self, detections, frame_shape):
        polygon = self.polygon(frame_shape)
        violations = []
        for det in detections:
            inside = cv2.pointPolygonTest(polygon, det["centroid"], False) >= 0
            if inside and det["track_id"] not in self._flagged_ids:
                self._flagged_ids.add(det["track_id"])
                violations.append(det)
        return violations

import csv
import os
from datetime import datetime

import cv2


class ViolationLogger:
    """Appends zone-violation events to a central CSV log with snapshot crops."""

    FIELDNAMES = ["timestamp", "track_id", "vehicle_class", "confidence", "snapshot"]

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.snapshot_dir = os.path.join(log_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, "violations.csv")
        is_new = not os.path.exists(self.log_path)
        self._file = open(self.log_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if is_new:
            self._writer.writeheader()

        self.violation_count = 0

    def log(self, frame, detection):
        timestamp = datetime.now().isoformat(timespec="seconds")
        snapshot_name = f"{timestamp.replace(':', '-')}_id{detection['track_id']}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)

        x1, y1, x2, y2 = detection["bbox"]
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size > 0:
            cv2.imwrite(snapshot_path, crop)

        self._writer.writerow({
            "timestamp": timestamp,
            "track_id": detection["track_id"],
            "vehicle_class": detection["class_name"],
            "confidence": f"{detection['confidence']:.2f}",
            "snapshot": snapshot_name,
        })
        self._file.flush()
        self.violation_count += 1

    def close(self):
        self._file.close()

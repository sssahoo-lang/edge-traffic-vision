import csv
import os
from datetime import datetime

import cv2


class ViolationLogger:
    """Appends violation events (zone intrusions, speeding) to a central
    CSV log with snapshot crops."""

    FIELDNAMES = [
        "timestamp", "track_id", "vehicle_class", "violation_type",
        "confidence", "speed_mph", "snapshot",
    ]

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

    def log(self, frame, detection, violation_type, speed_mph=None):
        timestamp = datetime.now().isoformat(timespec="seconds")
        snapshot_name = f"{timestamp.replace(':', '-')}_id{detection['track_id']}_{violation_type}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)

        x1, y1, x2, y2 = detection["bbox"]
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size > 0:
            cv2.imwrite(snapshot_path, crop)

        self._writer.writerow({
            "timestamp": timestamp,
            "track_id": detection["track_id"],
            "vehicle_class": detection["class_name"],
            "violation_type": violation_type,
            "confidence": f"{detection['confidence']:.2f}",
            "speed_mph": f"{speed_mph:.1f}" if speed_mph is not None else "",
            "snapshot": snapshot_name,
        })
        self._file.flush()
        self.violation_count += 1

    def close(self):
        self._file.close()

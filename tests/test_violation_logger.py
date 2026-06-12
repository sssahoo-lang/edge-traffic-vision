import csv

import numpy as np

from src.violation_logger import ViolationLogger


def _detection(track_id=1, bbox=(0, 0, 10, 10)):
    return {
        "track_id": track_id,
        "class_name": "car",
        "confidence": 0.9,
        "bbox": bbox,
        "centroid": (5, 5),
    }


def test_log_writes_csv_row_and_snapshot(tmp_path):
    logger = ViolationLogger(str(tmp_path))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    logger.log(frame, _detection(), "zone_intrusion")
    logger.close()

    with open(logger.log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["violation_type"] == "zone_intrusion"
    assert rows[0]["speed_mph"] == ""
    assert logger.violation_count == 1


def test_log_records_speed_for_speeding_violation(tmp_path):
    logger = ViolationLogger(str(tmp_path))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    logger.log(frame, _detection(), "speeding", speed_mph=42.5)
    logger.close()

    with open(logger.log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["violation_type"] == "speeding"
    assert rows[0]["speed_mph"] == "42.5"

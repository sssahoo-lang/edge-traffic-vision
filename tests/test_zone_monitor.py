from src.zone_monitor import ZoneMonitor

ZONE = [(0.1, 0.6), (0.9, 0.6), (0.9, 1.0), (0.1, 1.0)]
FRAME_SHAPE = (480, 640, 3)


def _detection(track_id, centroid):
    return {
        "track_id": track_id,
        "class_name": "car",
        "confidence": 0.9,
        "bbox": (0, 0, 10, 10),
        "centroid": centroid,
    }


def test_flags_vehicle_inside_zone():
    monitor = ZoneMonitor(ZONE)
    inside = _detection(1, (320, 400))
    outside = _detection(2, (320, 50))

    violations = monitor.check([inside, outside], FRAME_SHAPE)

    assert [v["track_id"] for v in violations] == [1]


def test_does_not_reflag_same_track():
    monitor = ZoneMonitor(ZONE)
    inside = _detection(1, (320, 400))

    first = monitor.check([inside], FRAME_SHAPE)
    second = monitor.check([inside], FRAME_SHAPE)

    assert len(first) == 1
    assert second == []


def test_polygon_scales_to_frame_size():
    monitor = ZoneMonitor(ZONE)
    polygon = monitor.polygon(FRAME_SHAPE)

    assert polygon.shape == (4, 2)
    assert tuple(polygon[0]) == (64, 288)  # (0.1 * 640, 0.6 * 480)

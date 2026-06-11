from ultralytics import YOLO


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLO model for tracked vehicle detection."""

    def __init__(self, model_path, confidence, vehicle_classes):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = vehicle_classes

    def track(self, frame):
        """Run detection + tracking on a frame and return a list of detections.

        Each detection is a dict with: track_id, class_name, confidence,
        bbox (x1, y1, x2, y2), and centroid (cx, cy).
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            classes=list(self.vehicle_classes.keys()),
            verbose=False,
        )[0]

        detections = []
        if results.boxes is None or results.boxes.id is None:
            return detections

        for box, track_id, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.id.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            detections.append({
                "track_id": int(track_id),
                "class_name": self.vehicle_classes.get(int(cls), "vehicle"),
                "confidence": float(conf),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            })
        return detections

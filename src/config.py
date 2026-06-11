class Config:
    """Central configuration for the detection and monitoring pipeline."""

    # Ultralytics YOLO weights. yolov8n (nano) is small enough to run
    # in real time on a Raspberry Pi 4.
    MODEL_PATH = "yolov8n.pt"

    CONFIDENCE_THRESHOLD = 0.4

    # (width, height) frames are resized to before inference. Downscaling
    # from a 1920x1080 camera feed to this resolution is what cuts
    # per-frame inference cost on constrained hardware.
    INFERENCE_SIZE = (640, 480)

    # COCO class IDs for the vehicle types we care about.
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # Restricted zone polygon, expressed as fractions of frame
    # (width, height) so it scales to whatever resolution the
    # frame is resized to. Default covers the bottom band of the
    # frame (e.g. a no-stopping zone or restricted lane).
    RESTRICTED_ZONE_FRACTIONS = [
        (0.05, 0.55),
        (0.95, 0.55),
        (0.95, 1.0),
        (0.05, 1.0),
    ]

    LOG_DIR = "logs"

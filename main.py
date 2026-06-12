import argparse
import time

import cv2

from src.config import Config
from src.detector import VehicleDetector
from src.display import draw_detections, draw_fps, draw_violation_alert, draw_zone
from src.fps_tracker import FPSTracker
from src.preprocessing import resize_frame
from src.speed_monitor import SpeedMonitor
from src.violation_logger import ViolationLogger
from src.zone_monitor import ZoneMonitor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time vehicle detection and restricted-zone violation monitor"
    )
    parser.add_argument("--source", default="0",
                         help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default=Config.MODEL_PATH,
                         help="Ultralytics YOLO model weights")
    parser.add_argument("--no-display", action="store_true",
                         help="Run headless, e.g. on a Raspberry Pi without a monitor")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")

    detector = VehicleDetector(args.model, Config.CONFIDENCE_THRESHOLD, Config.VEHICLE_CLASSES)
    zone_monitor = ZoneMonitor(Config.RESTRICTED_ZONE_FRACTIONS)
    speed_monitor = SpeedMonitor(
        Config.PIXELS_PER_METER, Config.SPEED_LIMIT_MPH, Config.SPEED_WINDOW_SECONDS
    )
    logger = ViolationLogger(Config.LOG_DIR)
    fps_tracker = FPSTracker()

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_frame(frame, Config.INFERENCE_SIZE)
            detections = detector.track(frame)
            now = time.perf_counter()

            violations = []
            for det in detections:
                speed_mph, is_speeding = speed_monitor.update(det["track_id"], det["centroid"], now)
                det["speed_mph"] = speed_mph
                if is_speeding:
                    violations.append((det, "speeding", speed_mph))

            for det in zone_monitor.check(detections, frame.shape):
                violations.append((det, "zone_intrusion", det["speed_mph"]))

            for det, violation_type, speed_mph in violations:
                logger.log(frame, det, violation_type, speed_mph)

            fps_tracker.tick()

            if not args.no_display:
                draw_zone(frame, zone_monitor.polygon(frame.shape))
                draw_detections(frame, detections)
                draw_fps(frame, fps_tracker.fps)
                if violations:
                    alert_text = " / ".join(sorted({v[1].upper().replace("_", " ") for v in violations}))
                    draw_violation_alert(frame, alert_text)

                cv2.imshow("Edge Traffic Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print(f"Logged {logger.violation_count} violation(s) to {logger.log_path}")


if __name__ == "__main__":
    main()

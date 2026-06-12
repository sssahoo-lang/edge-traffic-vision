"""Generates a synthetic GIF demonstrating the speed/zone/violation pipeline
without requiring a live camera or external video footage.

This bypasses YOLO detection (see docs/demo_detection.jpg for a real
detection example) and instead feeds hand-built detection dicts -- for a
single "vehicle" moving down the frame -- through the same SpeedMonitor,
ZoneMonitor, and overlay drawing code used by main.py.

Run:
    python demo/generate_demo_gif.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from src.config import Config
from src.display import draw_detections, draw_fps, draw_violation_alert, draw_zone
from src.speed_monitor import SpeedMonitor
from src.zone_monitor import ZoneMonitor

WIDTH, HEIGHT = Config.INFERENCE_SIZE
FRAMES = 24
FPS = 15


def make_background():
    frame = np.full((HEIGHT, WIDTH, 3), (70, 70, 70), dtype=np.uint8)
    for x in range(0, WIDTH, 80):
        cv2.line(frame, (x, 0), (x, HEIGHT), (95, 95, 95), 2)
    return frame


def main():
    zone_monitor = ZoneMonitor(Config.RESTRICTED_ZONE_FRACTIONS)
    speed_monitor = SpeedMonitor(
        Config.PIXELS_PER_METER, Config.SPEED_LIMIT_MPH, Config.SPEED_WINDOW_SECONDS
    )

    frames = []
    alert_text = ""

    for i in range(FRAMES):
        frame = make_background()
        y = 20 + i * 18
        bbox = (270, y, 370, y + 50)
        centroid = (320, y + 25)

        det = {
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.93,
            "bbox": bbox,
            "centroid": centroid,
        }

        speed_mph, is_speeding = speed_monitor.update(1, centroid, i / FPS)
        det["speed_mph"] = speed_mph

        zone_violations = zone_monitor.check([det], frame.shape)

        if is_speeding and "SPEEDING" not in alert_text:
            alert_text = "SPEEDING" if not alert_text else f"{alert_text} / SPEEDING"
        if zone_violations:
            alert_text = "ZONE INTRUSION" if not alert_text else f"{alert_text} / ZONE INTRUSION"

        draw_zone(frame, zone_monitor.polygon(frame.shape))
        draw_detections(frame, [det])
        draw_fps(frame, 28.4)
        if alert_text:
            draw_violation_alert(frame, alert_text)

        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    os.makedirs("docs", exist_ok=True)
    out_path = "docs/demo_pipeline.gif"
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

import cv2


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} #{det['track_id']}"
        speed_mph = det.get("speed_mph")
        label += f" {speed_mph:.0f} mph" if speed_mph is not None else f" {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return frame


def draw_zone(frame, polygon, color=(0, 0, 255)):
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_violation_alert(frame, text="VIOLATION DETECTED"):
    h, w = frame.shape[:2]
    font_scale = 0.7
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    if text_w > w - 20:
        font_scale *= (w - 20) / text_w
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.putText(frame, text, (max((w - text_w) // 2, 10), 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

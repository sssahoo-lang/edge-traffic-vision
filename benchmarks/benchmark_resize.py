"""Compare YOLO inference throughput at native resolution vs. a downscaled
resolution, on the same set of captured frames.

Usage:
    python benchmarks/benchmark_resize.py --source sample/traffic.mp4
    python benchmarks/benchmark_resize.py --source 0 --frames 100
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def capture_frames(source, n):
    cap = cv2.VideoCapture(source)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def benchmark(model, frames, target_size=None):
    times = []
    for frame in frames:
        img = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) if target_size else frame
        start = time.perf_counter()
        model(img, verbose=False)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    return avg, 1.0 / avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or path to a video file")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to benchmark")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Capturing {args.frames} frames from {source!r}...")
    frames = capture_frames(source, args.frames)
    if not frames:
        print("No frames captured — check the --source argument.")
        return

    model = YOLO(args.model)

    h, w = frames[0].shape[:2]
    target = (args.target_width, args.target_height)

    print(f"Native resolution:  {w}x{h}")
    native_avg, native_fps = benchmark(model, frames, target_size=None)

    print(f"Resized resolution: {target[0]}x{target[1]}")
    resized_avg, resized_fps = benchmark(model, frames, target_size=target)

    improvement = (resized_fps - native_fps) / native_fps * 100

    print()
    print(f"Native  ({w}x{h}):           {native_fps:6.2f} FPS  ({native_avg * 1000:.1f} ms/frame)")
    print(f"Resized ({target[0]}x{target[1]}):          {resized_fps:6.2f} FPS  ({resized_avg * 1000:.1f} ms/frame)")
    print(f"Throughput change:        {improvement:+.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate synthetic sample clips so the platform can run without footage.

These clips are real, decodable MP4 files with genuine movement, so the
capture, recording, motion-detection, streaming and event paths all execute
against real video. They are NOT a substitute for real footage when you want
to exercise object detection and face recognition: YOLO and YuNet are trained
on photographic imagery and will find little in synthetic shapes.

    python scripts/make_sample_videos.py

To exercise the full AI pipeline, drop your own clips into
data/sample_videos/ (see the README, "Sample video setup").
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "sample_videos"

WIDTH, HEIGHT, FPS = 960, 540, 20


def _background(width: int, height: int, tone: tuple[int, int, int]) -> np.ndarray:
    """A static scene: ground plane, horizon and some fixed scenery."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.42)
    frame[:horizon] = tone
    frame[horizon:] = (int(tone[0] * 0.45), int(tone[1] * 0.5), int(tone[2] * 0.45))
    cv2.line(frame, (0, horizon), (width, horizon), (90, 90, 95), 2)
    for i in range(6):
        x = int(width * (i + 0.5) / 6)
        cv2.rectangle(frame, (x - 6, horizon - 60), (x + 6, horizon),
                      (70, 70, 80), -1)
    # Light sensor noise keeps the background subtractor realistic.
    noise = np.random.default_rng(7).normal(0, 3, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_walker(frame: np.ndarray, cx: int, ground_y: int, scale: float,
                 colour: tuple[int, int, int], phase: float) -> None:
    """A crude walking figure whose size encodes distance from the camera."""
    height = max(24, int(150 * scale))
    width = max(10, int(height * 0.34))
    head_r = max(4, int(height * 0.13))
    top = ground_y - height

    cv2.rectangle(frame, (cx - width // 2, top + head_r * 2),
                  (cx + width // 2, ground_y - height // 3), colour, -1)
    cv2.circle(frame, (cx, top + head_r), head_r, (198, 176, 160), -1)
    swing = int(math.sin(phase) * width * 0.6)
    for dx in (-swing, swing):
        cv2.line(frame, (cx, ground_y - height // 3), (cx + dx, ground_y),
                 (60, 60, 70), max(2, width // 4))


def _draw_vehicle(frame: np.ndarray, cx: int, cy: int, scale: float,
                  colour: tuple[int, int, int]) -> None:
    w = max(40, int(190 * scale))
    h = max(18, int(78 * scale))
    cv2.rectangle(frame, (cx - w // 2, cy - h), (cx + w // 2, cy), colour, -1)
    cv2.rectangle(frame, (cx - w // 4, cy - int(h * 1.45)),
                  (cx + w // 4, cy - h), (int(colour[0] * 0.7),) * 3, -1)
    for dx in (-w // 3, w // 3):
        cv2.circle(frame, (cx + dx, cy), max(5, h // 4), (25, 25, 25), -1)


def _draw_animal(frame: np.ndarray, cx: int, cy: int, scale: float) -> None:
    w = max(20, int(70 * scale))
    h = max(12, int(38 * scale))
    cv2.ellipse(frame, (cx, cy - h // 2), (w // 2, h // 2), 0, 0, 360,
                (95, 120, 150), -1)
    cv2.circle(frame, (cx + w // 2, cy - h), max(4, h // 3), (95, 120, 150), -1)
    for dx in (-w // 3, 0, w // 3):
        cv2.line(frame, (cx + dx, cy - h // 3), (cx + dx, cy), (80, 100, 130), 3)


def make_clip(path: Path, seconds: int, scenario: str, tone: tuple[int, int, int]) -> Path:
    total = seconds * FPS
    background = _background(WIDTH, HEIGHT, tone)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {path}")

    ground = int(HEIGHT * 0.88)
    try:
        for i in range(total):
            frame = background.copy()
            t = i / total

            if scenario == "approach":
                # One figure walking steadily towards the camera: its box
                # grows, so the proximity estimator sees FAR -> A -> B.
                scale = 0.18 + 0.85 * t
                _draw_walker(frame, WIDTH // 2, ground, scale, (150, 90, 70), i * 0.35)
            elif scenario == "crossing":
                _draw_walker(frame, int(WIDTH * t), ground, 0.55, (70, 110, 160),
                             i * 0.4)
                _draw_walker(frame, int(WIDTH * (1 - t)), ground - 30, 0.38,
                             (160, 120, 80), i * 0.33)
                _draw_animal(frame, int(WIDTH * (0.2 + 0.6 * t)), ground, 0.9)
            else:  # "yard"
                _draw_vehicle(frame, int(WIDTH * (1.15 - 1.3 * t)),
                              int(HEIGHT * 0.78), 0.9, (120, 120, 130))
                if t > 0.35:
                    walk = (t - 0.35) / 0.65
                    _draw_walker(frame, int(WIDTH * (0.15 + 0.5 * walk)), ground,
                                 0.3 + 0.5 * walk, (90, 140, 110), i * 0.36)

            cv2.putText(frame, f"SYNTHETIC TEST CLIP  frame {i:04d}", (14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1,
                        cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return path


CLIPS = [
    ("camera_01.mp4", 25, "approach", (120, 118, 112)),
    ("camera_02.mp4", 25, "crossing", (108, 116, 124)),
    ("camera_03.mp4", 25, "yard", (100, 104, 112)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic sample clips")
    parser.add_argument("--force", action="store_true", help="overwrite existing clips")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Writing synthetic clips to {args.output}")
    for filename, seconds, scenario, tone in CLIPS:
        path = args.output / filename
        if path.exists() and not args.force:
            print(f"  [skip]  {filename} (already exists)")
            continue
        make_clip(path, seconds, scenario, tone)
        size = path.stat().st_size / 1e6
        print(f"  [ok]    {filename}  {seconds}s  {WIDTH}x{HEIGHT}@{FPS}  {size:.1f} MB")

    print(
        "\nThese clips exercise capture, recording, motion detection and streaming.\n"
        "For real object detection and face recognition, add your own footage to\n"
        f"{args.output} and create a source pointing at it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

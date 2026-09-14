#!/usr/bin/env python
"""Build a clip from a real photograph that genuinely exercises the AI.

The synthetic clips from make_sample_videos.py drive capture, recording and
motion detection, but YOLO is trained on photographic imagery and finds
nothing in drawn shapes. This script takes a real photo containing people and
a vehicle and animates a slow dolly-in over it, so the platform sees:

  * real object detections (person, bus, ...)
  * persistent track ids across frames
  * bounding boxes that grow, driving FAR -> Proximity A -> Proximity B
    transitions and the security rules that follow

The photo is the Ultralytics sample asset (AGPL-3.0 assets repo), downloaded
on first run; pass --photo to use your own image instead.

    python scripts/make_detection_clip.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "sample_videos"
PHOTO_URL = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"
DEFAULT_PHOTO = OUTPUT_DIR / "_source_photo.jpg"

WIDTH, HEIGHT, FPS = 960, 540, 20


def fetch_photo(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample photograph -> {path.name}")
    urllib.request.urlretrieve(PHOTO_URL, path)  # noqa: S310
    return path


def dolly_frame(photo: np.ndarray, zoom: float, pan_x: float) -> np.ndarray:
    """Crop a window out of the photo and scale it to the output size.

    A shrinking crop window is a dolly-in: objects grow, which is exactly the
    signal the proximity estimator reads.
    """
    ph, pw = photo.shape[:2]
    crop_w = pw / zoom
    crop_h = crop_w * (HEIGHT / WIDTH)
    if crop_h > ph:
        crop_h = ph
        crop_w = crop_h * (WIDTH / HEIGHT)

    max_x = max(0.0, pw - crop_w)
    max_y = max(0.0, ph - crop_h)
    x = max(0.0, min(max_x, pan_x * max_x))
    y = max(0.0, min(max_y, ph * 0.30))

    window = photo[int(y): int(y + crop_h), int(x): int(x + crop_w)]
    if window.size == 0:
        window = photo
    return cv2.resize(window, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def build(photo_path: Path, output: Path, seconds: int) -> Path:
    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise SystemExit(f"could not read photo: {photo_path}")

    total = seconds * FPS
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {output}")

    rng = np.random.default_rng(11)
    try:
        for i in range(total):
            t = i / max(1, total - 1)
            # Hold wide, dolly in, hold close: gives a clean FAR -> A -> B arc.
            if t < 0.15:
                zoom = 1.0
            elif t < 0.85:
                zoom = 1.0 + 1.45 * ((t - 0.15) / 0.70)
            else:
                zoom = 2.45
            frame = dolly_frame(photo, zoom, 0.5)
            # A little grain so the background subtractor behaves like it
            # would on a real camera.
            noise = rng.normal(0, 2.0, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a real-detection demo clip")
    parser.add_argument("--photo", type=Path, default=None,
                        help="use your own photograph instead of the sample")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "camera_04_people.mp4")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"{args.output.name} already exists (use --force to rebuild)")
        return 0

    photo = args.photo or fetch_photo(DEFAULT_PHOTO)
    build(photo, args.output, args.seconds)
    print(
        f"Wrote {args.output} "
        f"({args.seconds}s, {WIDTH}x{HEIGHT}@{FPS}, "
        f"{args.output.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        "\nRegister it as a source (Monitor -> + Add source) to see real object\n"
        "detection, tracking and proximity transitions in the live overlay."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2

from src.utils import save_json

logger = logging.getLogger(__name__)


def load_video_metadata(video_path: Path) -> dict[str, Any]:
    """
    Extract basic metadata from a video file.

    Args:
        video_path: Path to the input video.

    Returns:
        A dictionary containing video metadata.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS value {fps} for video: {video_path}")

    duration_seconds = total_frames / fps if total_frames > 0 else 0.0

    cap.release()

    metadata = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
    }

    logger.info("Loaded video metadata: %s", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract metadata from an input video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/metadata.json"),
        help="Path to save metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    metadata = load_video_metadata(args.video)
    save_json(metadata, args.output)

    logger.info("Saved metadata to: %s", args.output)


if __name__ == "__main__":
    main()
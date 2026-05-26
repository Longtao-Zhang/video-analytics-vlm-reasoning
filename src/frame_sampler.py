from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def calculate_sampling_interval(video_fps: float, sample_fps: float) -> int:
    """
    Calculate how many frames to skip between sampled frames.

    Example:
        video_fps = 30, sample_fps = 1.0
        interval = 30
    """
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")

    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    return max(int(round(video_fps / sample_fps)), 1)


def sample_frames(
    video_path: Path,
    sample_fps: float,
    output_dir: Path,
    image_format: str = "jpg",
) -> dict[str, Any]:
    """
    Sample frames from a video at a fixed FPS.

    Args:
        video_path: Path to input video.
        sample_fps: Target sampling rate.
        output_dir: Directory to save sampled frames and metadata.
        image_format: Output image format.

    Returns:
        A dictionary containing sampled frame metadata.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS value {video_fps} for video: {video_path}")

    sampling_interval = calculate_sampling_interval(video_fps, sample_fps)

    frames_output_dir = output_dir / "sampled_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    sampled_frames: list[dict[str, Any]] = []
    frame_index = 0
    sampled_index = 0

    logger.info("Sampling video: %s", video_path)
    logger.info("Video FPS: %.2f", video_fps)
    logger.info("Target sampling FPS: %.2f", sample_fps)
    logger.info("Sampling interval: every %d frames", sampling_interval)

    while True:
        # Get the next frame, ret: bool, True if catch successfully.
        ret, frame = cap.read() 

        if not ret:
            break

        if frame_index % sampling_interval == 0:
            timestamp = frame_index / video_fps # Timestep in the original video
            frame_filename = f"frame_{sampled_index:06d}.{image_format}"
            frame_path = frames_output_dir / frame_filename

            success = cv2.imwrite(str(frame_path), frame) # Write in imgs

            if not success:
                cap.release()
                raise IOError(f"Failed to save frame to: {frame_path}")

            sampled_frames.append(
                {
                    "sampled_frame_id": sampled_index,
                    "original_frame_id": frame_index,
                    "timestamp": timestamp,
                    "path": str(frame_path),
                }
            )

            sampled_index += 1

        frame_index += 1

    cap.release()

    result = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "video_fps": video_fps,
        "total_frames": total_frames,
        "sample_fps": sample_fps,
        "num_sampled_frames": len(sampled_frames),
        "sampled_frames": sampled_frames,
    }

    logger.info("Sampled %d frames", len(sampled_frames))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from an input video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. If omitted, use config value.",
    )
    return parser.parse_args()


def main() -> None:
    # Initilize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    start_time = time.time()

    args = parse_args()
    config = load_config(args.config)

    sample_fps = float(config["video"]["sample_fps"])
    image_format = config["sampling"].get("image_format", "jpg")

    if args.output is not None:
        output_dir = args.output
    else:
        output_root = Path(config["video"]["output_dir"])
        output_dir = output_root / args.video.stem

    result = sample_frames(
        video_path=args.video,
        sample_fps=sample_fps,
        output_dir=output_dir,
        image_format=image_format,
    )

    output_json_path = output_dir / "sampled_frames.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Saved sampled frame metadata to: %s", output_json_path)
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()
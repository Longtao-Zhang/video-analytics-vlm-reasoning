"""Run a YOLO (Ultralytics) detector on sampled frames.

Run:
    python -m src.detector --sampled-frames-json <sampled_frames.json> \\
        --config <config.yaml> --output <dir>
Writes detections to ``<output>/detections.json`` and annotated images to
``<output>/visualized_frames/``. Model and thresholds come from the config's
``detection`` section.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)

def load_detector(model_name: str) -> YOLO:
    """Load a YOLO dectector."""
    logger.info("Loading YOLO model: %s", model_name)
    return YOLO(model_name)

def load_sampled_frames(sampled_frames_json: Path) -> dict[str, Any]:
    """Load samoled frame metadata"""
    if not sampled_frames_json.exists():
        raise FileNotFoundError(f"Sampled frames JSON not found: {sampled_frames_json}")
    
    import json

    with sampled_frames_json.open("r", encoding="utf-8") as f:
        return json.load(f)

def run_detection_on_image(
        model: YOLO,
        image_path: Path,
        confidence_threshold: float,
        target_classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run YOLO detection on a single image"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    results = model(str(image_path), conf=confidence_threshold, verbose=False)
    result = results[0]

    # Declare a variable type to return
    detections: list[dict[str, Any]] = []

    if result.boxes is None:
        return detections
    
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        confidence = float(box.conf[0].item())
        bbox = box.xyxy[0].tolist()

        if target_classes is not None and label not in target_classes:
            continue

        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox_xyxy": [float(x) for x in bbox],
            }
        )
        
    return detections

def visualize_detections(
        image_path: Path,
        detections: list[dict[str, Any]],
        output_path: Path,
 ) -> None:
    """Draw detection boxes on an image and save it."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    for det in detections:
        # for bbox, the origin is at the top-left, and down to the bottom
        # is the positive direction for y
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

def run_detection_on_sampled_frames(
        sampled_frames_json: Path,
        output_dir: Path,
        model_name: str,
        confidence_threshold: float,
        target_classes: list[str] | None = None,
        save_visualization: bool = True,
) -> dict[str, Any]:
    """Run YOLO detection on all sampled frames."""
    sampled_data = load_sampled_frames(sampled_frames_json)
    sampled_frames = sampled_data["sampled_frames"]

    model = load_detector(model_name)

    visualized_dir = output_dir / "visualized_frames"
    frame_results: list[dict[str, Any]] = []

    logger.info("Running detection on %d sampled frames", len(sampled_frames))

    for frame_info in sampled_frames:
        image_path = Path(frame_info["path"])

        detections = run_detection_on_image(
            model=model,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            target_classes=target_classes,
        )

        if save_visualization:
            vis_path = visualized_dir / image_path.name
            visualize_detections(
                image_path=image_path,
                detections=detections,
                output_path=vis_path,
            )
        else:
            vis_path = None
        
        frame_results.append(
            {
                "sampled_frame_id": frame_info["sampled_frame_id"],
                "original_frame_id": frame_info["original_frame_id"],
                "timestamp": frame_info["timestamp"],
                "image_path": str(image_path),
                "visualized_path": str(vis_path) if vis_path else None,
                "detections": detections,
            }
        )

    result = {
        "video_name": sampled_data["video_name"],
        "video_path": sampled_data["video_path"],
        "model_name": model_name,
        "confidence_threshold": confidence_threshold,
        "target_classes": target_classes,
        "num_frames": len(sampled_frames),
        "frames": frame_results,
    }

    return result
    
def parse_args() -> argparse.Namespace:
    # For --help
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on sampled video frames."
    )
    parser.add_argument(
        "--sampled-frames-json",
        type=Path,
        required=True,
        help="Path to sampled_frames.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for detection results.",
    )
    return parser.parse_args()

def main() -> None:
    # Initialize output logs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    start_time = time.time()

    args = parse_args()
    config = load_config(args.config)

    detection_config = config["detection"]
    model_name = detection_config["model_name"]
    confidence_threshold = float(detection_config["confidence_threshold"])
    target_classes = detection_config.get("target_classes")

    result = run_detection_on_sampled_frames(
        sampled_frames_json=args.sampled_frames_json,
        output_dir=args.output,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        target_classes=target_classes,
        save_visualization=True,
    )

    output_json_path = args.output / "detections.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Save detections to: %s", output_json_path)
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()



"""Temporal aggregator: turn frame-level detections into per-class time segments.

Purpose
    Read detections.json (frame-level detection results produced by the detector),
    group the scattered per-frame observations by object label, and merge them into
    "time segments" during which that object class is continuously present. Each
    segment carries its evidence frames and confidence stats. The final result is
    written to temporal_segments.json.

Function relationships (top-down call chain)
    main()                              CLI entry: parse args, load config, time, save
      └─ aggregate_temporal_segments()  Orchestrates aggregation per label + metadata
           ├─ load_detections()         Read and validate detections.json
           ├─ collect_label_observations()
           │                            frames -> {label: [time-sorted observations]}
           │                            keeps the highest-confidence box per label/frame
           └─ merge_observations_into_segments()
                                        split observations by max_gap, then filter by
                                        min duration / min evidence frames
                └─ _build_segment()     (private) summarize one segment: time span,
                                        evidence frames, confidence stats

Usage
    Command line:
        python -m src.temporal_aggregator \\
            --detections-json runs/xxx/detections.json \\
            --config configs/default.yaml \\
            --output runs/xxx
        (config must define temporal.max_gap_seconds /
         min_segment_duration_seconds / min_evidence_frames)

    As a library:
        result = aggregate_temporal_segments(
            detections_json=Path("detections.json"),
            max_gap_seconds=2.0,
            min_segment_duration_seconds=1.0,
            min_evidence_frames=2,
        )
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)

def load_detections(detections_json: Path) -> dict[str, Any]:
    """Load frame-level detection results."""
    if not detections_json.exists():
        raise FileNotFoundError(f"Detections JSON not found: {detections_json}")
    
    with detections_json.open("r", encoding="utf-8") as f:
        return json.load(f)
    

def collect_label_observations(
        detections_data: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Collect frame-level observations for each object label.
    
    Return (observations):
        {
            "Person": [
                {
                    "timestamp": 1.0,
                    "sampled_frame_id": 1,
                    "original_frame_id": 30,
                    "confidence": 0.91,
                    "bbox_xyxy": [...]
                }
            ]
        }
    """
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in detections_data.get("frames", []):
        timestamp = float(frame["timestamp"])
        sampled_frame_id = int(frame["sampled_frame_id"])
        original_frame_id = int(frame["original_frame_id"])
        image_path = frame.get("image_path")

        # If multiple objects of the same class appear in one frame,
        # we keep the highest-confidence one for class-level aggregation.
        best_detection_per_label: dict[str, dict[str, Any]] = {}

        for det in frame.get("detections", []):
            label = det["label"]
            confidence = float(det["confidence"])

            if (
                label not in best_detection_per_label 
                or confidence > best_detection_per_label[label]["confidence"]
            ):
                best_detection_per_label[label] = det

        for label, det in best_detection_per_label.items():
            observations[label].append(
                {
                    "timestamp": timestamp,
                    "sampled_frame_id": sampled_frame_id,
                    "original_frame_id": original_frame_id,
                    "image_path": image_path,
                    "confidence": float(det["confidence"]),
                    "bbox_xyxy": det["bbox_xyxy"],
                }
            )
    for label in observations:
        observations[label].sort(key=lambda x: x["timestamp"])
    
    return dict(observations)

def _build_segment(
        label: str,
        segment_index: int,
        observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate information for a specific label from its observations."""
    start_time = float(observations[0]["timestamp"])
    end_time = float(observations[-1]["timestamp"])
    duration = end_time - start_time
    confidences = [float(obs["confidence"]) for obs in observations]

    return {
        "segment_id": f"{label}_segment_{segment_index:03d}",
        "object_label": label,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "evidence_frames":[
            {
                "sampled_frame_id": obs["sampled_frame_id"],
                "original_frame_id": obs["original_frame_id"],
                "timestamp": obs["timestamp"],
                "image_path": obs["image_path"],
                "confidence": obs["confidence"],
                "bbox_xyxy": obs["bbox_xyxy"],
            }
            for obs in observations
        ],
        "num_evidence_frames": len(observations),
        "mean_confidence": sum(confidences) / len(confidences),
        "max_confidence": max(confidences),
    }

def merge_observations_into_segments(
        label: str,
        observations: list[dict[str, Any]],
        max_gap_seconds: float,
        min_segment_duration_seconds: float,
        min_evidence_frames: int,
) -> list[dict[str, Any]]:
    """Merge the whole observations into temporal segments by max-gap"""
    if not observations:
        return []
    
    # To store different segments
    raw_segments: list[list[dict[str, Any]]] = []
    # Current working segments
    current_segment: list[dict[str, Any]] = [observations[0]]

    for obs in observations[1:]:
        previous_timestamp = float(current_segment[-1]["timestamp"])
        current_timestamp = float(obs["timestamp"])
        gap = current_timestamp - previous_timestamp

        if gap <= max_gap_seconds:
            current_segment.append(obs) # counted into the same segment
        else:
            # Save current segment and start a new one
            raw_segments.append(current_segment)
            current_segment = [obs]
    
    raw_segments.append(current_segment)

    final_segments: list[dict[str, Any]] = []

    for idx, segment_obs in enumerate(raw_segments, start=1):
        segment = _build_segment(
            label=label,
            segment_index=idx,
            observations=segment_obs,
        )

        if segment["duration_seconds"] < min_segment_duration_seconds:
            continue

        if segment["num_evidence_frames"] < min_evidence_frames:
            continue

        final_segments.append(segment)

    return final_segments

def aggregate_temporal_segments(
        detections_json: Path,
        max_gap_seconds: float,
        min_segment_duration_seconds: float,
        min_evidence_frames: int,
) -> dict[str, Any]:
    """Aggregate frame-level detections into temporal object presence segments."""
    detections_data = load_detections(detections_json)
    observations_by_label = collect_label_observations(detections_data)

    all_segments: list[dict[str, Any]] = []

    for label, observations in observations_by_label.items():
        segments = merge_observations_into_segments(
            label=label,
            observations=observations,
            max_gap_seconds=max_gap_seconds,
            min_segment_duration_seconds=min_segment_duration_seconds,
            min_evidence_frames=min_evidence_frames,            
        )
        all_segments.extend(segments)
    
    all_segments.sort(key=lambda x: (x["start_time"], x["object_label"]))

    result = {
        "video_name": detections_data.get("video_name"),
        "video_path": detections_data.get("video_path"),
        "aggregation_type": "class_level_temporal_aggregation",
        "parameters":{
            "max_gap_seconds": max_gap_seconds,
            "min_segment_duration_seconds": min_segment_duration_seconds,
            "min_evidence_frames": min_evidence_frames,
        },
        "num_segments": len(all_segments),
        "segments": all_segments,
    } 
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level detections into temporal segments."
    )
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="Path to detections.json.",
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
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    start_time = time.time()

    args = parse_args()
    config = load_config(args.config)

    temporal_config = config["temporal"]

    result = aggregate_temporal_segments(
        detections_json=args.detections_json,
        max_gap_seconds=float(temporal_config["max_gap_seconds"]),
        min_segment_duration_seconds=float(
            temporal_config["min_segment_duration_seconds"],
        ),
        min_evidence_frames=temporal_config["min_evidence_frames"],
    )

    output_json_path = args.output / "temporal_segments.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Saved temporal segments to: %s", output_json_path)
    logger.info("Number of temporal segments: %d", result["num_segments"])
    logger.info("Elapsed time: %.2f seconds", elapsed_time)

if __name__ == "__main__":
    main()
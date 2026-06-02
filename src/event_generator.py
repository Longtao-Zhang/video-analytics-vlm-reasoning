"""Rule-based event candidate generator.

Reads frame-level detections (detections.json) and temporal segments
(temporal_segments.json), then applies configurable rules to produce
candidate events such as person_present, vehicle_present,
multiple_people_present and person_near_vehicle. Results are sorted,
re-numbered and written to event_candidates.json.

CLI usage:
    python -m src.event_generator \
        --detections-json path/to/detections.json \
        --temporal-segments-json path/to/temporal_segments.json \
        --config configs/default.yaml \
        --output path/to/output_dir

Library usage:
    from src.event_generator import generate_event_candidates
    result = generate_event_candidates(
        detections_json=Path("detections.json"),
        temporal_segments_json=Path("temporal_segments.json"),
        config=load_config("configs/default.yaml"),
    )
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)

def load_json(path: Path) -> dict[str, Any]:
    """Load a json file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def euclidean_distance(
        point_a: tuple[float, float],
        point_b: tuple[float, float],
) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
    )

def generate_presence_events(
        temporal_segments_data: dict[str, Any],
        vehicle_classes: list[str],
        min_event_duration_seconds: float,
        min_evidence_frames: int,
) -> list[dict[str, Any]]:
    """
    Generate person_present and vehicle_present events from temporal segments.
    """
    events: list[dict[str, Any]] = []

    for segment in temporal_segments_data.get("segments", []):
        label = segment["object_label"]
        duration = float(segment["duration_seconds"])
        num_evidence_frames = int(segment["num_evidence_frames"])

        if duration < min_event_duration_seconds:
            continue

        if num_evidence_frames < min_evidence_frames:
            continue

        if label == "person":
            event_type = "person_present"
        elif label in vehicle_classes:
            event_type = "vehicle_present"
        else:
            continue

        events.append(
            {
                "event_id": f"event_{len(events) + 1:03d}",
                "event_type": event_type,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration_seconds": segment["duration_seconds"],
                "objects_involved": [label],
                "source_segment_id": segment["segment_id"],
                "evidence_frames": segment["evidence_frames"],
                "num_evidence_frames": segment["num_evidence_frames"],
                "confidence": segment["mean_confidence"],
                "generation_rule": "presence_from_segment",
            }
        )
    return events

def generate_multiple_people_events(
        detections_data: dict[str, Any],
        min_evidence_frames: int,
) -> list[dict[str, Any]]:
    """
    Generate multiple_people_present events from frame-level detections.

    A frame is evidence if it contains at least two person detections.
    Consecutive evidence frames are grouped into one simple event candidate.
    """
    evidence_frames: list[dict[str, Any]] = []

    for frame in detections_data.get("frames", []):
        person_detections = [
            det for det in frame.get("detections", []) if det["label"] == "person"
        ]

        if len(person_detections) >= 2:
            confidences = [float(det["confidence"]) for det in person_detections]

            evidence_frames.append(
                {
                    "sampled_frame_id": frame["sampled_frame_id"],
                    "original_frame_id": frame["original_frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["image_path"],
                    "num_people": len(person_detections),
                    "detections": person_detections,
                    "mean_confidence": sum(confidences) / len(confidences),
                }
            )

    if len(evidence_frames) < min_evidence_frames:
        return []
    
    start_time = float(evidence_frames[0]["timestamp"])
    end_time = float(evidence_frames[-1]["timestamp"])
    confidence = sum(f["mean_confidence"] for f in evidence_frames) / len(
        evidence_frames
    )

    return [
        {
            "event_id": "event_multiple_people_001",
            "event_type": "multiple_people_present",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "objects_involved": ["person"],
            "evidence_frames": evidence_frames,
            "num_evidence_frames": len(evidence_frames),
            "confidence": confidence,
            "generation_rule": "at_least_two_person_detections_in_frame",
        }
    ]

def generate_person_near_vehicle_events(
    detections_data: dict[str, Any],
    vehicle_classes: list[str],
    distance_threshold_pixels: float,
    min_evidence_frames: int,
) -> list[dict[str, Any]]:
    """
    Generate person_near_vehicle events from frame-level detections.

    A frame is evidence if at least one person bbox center is close to
    at least one vehicle bbox center.
    """
    evidence_frames: list[dict[str, Any]] = []

    for frame in detections_data.get("frames", []):
        person_detections = [
            det for det in frame.get("detections", []) if det["label"] == "person"
        ]
        vehicle_detections = [
            det
            for det in frame.get("detections", [])
            if det["label"] in vehicle_classes
        ]

        if not person_detections or not vehicle_detections:
            continue

        matched_pairs: list[dict[str, Any]] = []

        for person_det in person_detections:
            person_center = bbox_center(person_det["bbox_xyxy"])

            for vehicle_det in vehicle_detections:
                vehicle_center = bbox_center(vehicle_det["bbox_xyxy"])
                distance = euclidean_distance(person_center, vehicle_center)

                if distance <= distance_threshold_pixels:
                    matched_pairs.append(
                        {
                            "person_detection": person_det,
                            "vehicle_detection": vehicle_det,
                            "center_distance_pixels": distance,
                        }
                    )

        if matched_pairs:
            pair_confidences = [
                (
                    float(pair["person_detection"]["confidence"])
                    + float(pair["vehicle_detection"]["confidence"])
                )
                / 2.0
                for pair in matched_pairs
            ]

            evidence_frames.append(
                {
                    "sampled_frame_id": frame["sampled_frame_id"],
                    "original_frame_id": frame["original_frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["image_path"],
                    "matched_pairs": matched_pairs,
                    "mean_confidence": sum(pair_confidences)
                    / len(pair_confidences),
                }
            )

    if len(evidence_frames) < min_evidence_frames:
        return []

    start_time = float(evidence_frames[0]["timestamp"])
    end_time = float(evidence_frames[-1]["timestamp"])
    confidence = sum(f["mean_confidence"] for f in evidence_frames) / len(
        evidence_frames
    )

    return [
        {
            "event_id": "event_person_near_vehicle_001",
            "event_type": "person_near_vehicle",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "objects_involved": ["person", "vehicle"],
            "evidence_frames": evidence_frames,
            "num_evidence_frames": len(evidence_frames),
            "confidence": confidence,
            "generation_rule": "bbox_center_distance_below_threshold",
            "parameters": {
                "distance_threshold_pixels": distance_threshold_pixels,
            },
        }
    ]

def assign_event_ids(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign globally consistent event IDs."""
    events = sorted(events, key=lambda x: (x["start_time"], x["event_type"]))

    for idx, event in enumerate(events, start=1):
        event["event_id"] = f"event_{idx:03d}"

    return events

def generate_event_candidates(
    detections_json: Path,
    temporal_segments_json: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate rule-based event candidates."""
    detections_data = load_json(detections_json)
    temporal_segments_data = load_json(temporal_segments_json)

    event_config = config["events"]

    enabled_event_types = set(event_config["enabled_event_types"])
    vehicle_classes = event_config["vehicle_classes"]
    min_event_duration_seconds = float(event_config["min_event_duration_seconds"])
    min_evidence_frames = int(event_config["min_evidence_frames"])
    distance_threshold_pixels = float(
        event_config["person_vehicle_distance_threshold_pixels"]
    )

    events: list[dict[str, Any]] = []

    if "person_present" in enabled_event_types or "vehicle_present" in enabled_event_types:
        presence_events = generate_presence_events(
            temporal_segments_data=temporal_segments_data,
            vehicle_classes=vehicle_classes,
            min_event_duration_seconds=min_event_duration_seconds,
            min_evidence_frames=min_evidence_frames,
        )
        presence_events = [
            e for e in presence_events if e["event_type"] in enabled_event_types
        ]
        events.extend(presence_events)

    if "multiple_people_present" in enabled_event_types:
        events.extend(
            generate_multiple_people_events(
                detections_data=detections_data,
                min_evidence_frames=min_evidence_frames,
            )
        )

    if "person_near_vehicle" in enabled_event_types:
        events.extend(
            generate_person_near_vehicle_events(
                detections_data=detections_data,
                vehicle_classes=vehicle_classes,
                distance_threshold_pixels=distance_threshold_pixels,
                min_evidence_frames=min_evidence_frames,
            )
        )

    events = assign_event_ids(events)

    result = {
        "video_name": detections_data.get("video_name"),
        "video_path": detections_data.get("video_path"),
        "generation_type": "rule_based_event_candidate_generation",
        "enabled_event_types": list(enabled_event_types),
        "num_events": len(events),
        "events": events,
    }

    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate event candidates from detections and temporal segments."
    )
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="Path to detections.json.",
    )
    parser.add_argument(
        "--temporal-segments-json",
        type=Path,
        required=True,
        help="Path to temporal_segments.json.",
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

    result = generate_event_candidates(
        detections_json=args.detections_json,
        temporal_segments_json=args.temporal_segments_json,
        config=config,
    )

    output_json_path = args.output / "event_candidates.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Saved event candidates to: %s", output_json_path)
    logger.info("Number of event candidates: %d", result["num_events"])
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()
"""Rule-based safety event candidate generator.

Reads frame-level detections (detections.json) and temporal segments
(temporal_segments.json), then applies configurable deterministic rules to
produce auditable Level-2 *safety* event candidates. Each candidate carries an
interpretable ``severity_prior`` (with a transparent ``score_breakdown``) and an
``evidence`` block, so the downstream VLM reasoner can explain / verify it rather
than make ungrounded decisions.

Currently implemented safety rule:
    - person_vehicle_proximity (collision_risk)

Scaffolded for later (helpers retained below, not yet registered):
    - crowding_or_gathering            (from multiple-people logic)
    - long_duration_vehicle_presence   (from vehicle temporal segments)

Output: safety_event_candidates.json

CLI usage:
    python -m src.event_generator \\
        --detections-json path/to/detections.json \\
        --temporal-segments-json path/to/temporal_segments.json \\
        --config configs/default.yaml \\
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
from typing import Any, Callable

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

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value into the [low, high] range."""
    return max(low, min(high, value))


def severity_from_score(
        risk_score: float,
        severity_bins: dict[str, float],
) -> str:
    """Map a continuous risk score into a discrete severity prior."""
    if risk_score <= float(severity_bins["low_max"]):
        return "low"
    if risk_score <= float(severity_bins["medium_max"]):
        return "medium"
    return "high"


def compute_risk_score(
        metrics: dict[str, float],
        scoring_config: dict[str, Any],
        distance_threshold_pixels: float,
) -> tuple[float, dict[str, float], str]:
    """Compute an interpretable risk score from a few transparent factors.

    risk_score =
          weights.proximity  * proximity_score
        + weights.duration   * duration_score
        + weights.confidence * confidence_score
        + weights.crowding   * crowding_score

    Returns ``(risk_score, score_breakdown, severity_prior)``. Storing the
    breakdown makes "why is this event high risk?" directly answerable.
    """
    weights = scoring_config["weights"]
    duration_ref = float(scoring_config["duration_ref_seconds"])
    crowd_ref = float(scoring_config["crowd_ref_people"])
    severity_bins = scoring_config["severity_bins"]

    # Closer -> higher; farther (up to the trigger threshold) -> lower.
    proximity_score = clamp(1.0 - metrics["min_distance_px"] / distance_threshold_pixels)
    duration_score = clamp(metrics["duration_seconds"] / duration_ref)
    confidence_score = clamp(metrics["mean_confidence"])
    crowding_score = clamp(metrics["max_people"] / crowd_ref)

    risk_score = (
        float(weights["proximity"]) * proximity_score
        + float(weights["duration"]) * duration_score
        + float(weights["confidence"]) * confidence_score
        + float(weights["crowding"]) * crowding_score
    )

    score_breakdown = {
        "proximity_score": round(proximity_score, 4),
        "duration_score": round(duration_score, 4),
        "confidence_score": round(confidence_score, 4),
        "crowding_score": round(crowding_score, 4),
    }

    severity_prior = severity_from_score(risk_score, severity_bins)
    return round(risk_score, 4), score_breakdown, severity_prior


def generate_person_vehicle_proximity_events(
    detections_data: dict[str, Any],
    vehicle_classes: list[str],
    distance_threshold_pixels: float,
    min_evidence_frames: int,
    scoring_config: dict[str, Any],
    risk_category: str,
) -> list[dict[str, Any]]:
    """Generate person_vehicle_proximity safety events from frame-level detections.

    A frame is evidence if at least one person bbox center is within
    ``distance_threshold_pixels`` of at least one vehicle bbox center. All such
    frames are merged into a single proximity candidate, scored with an
    interpretable severity prior.
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
        persons_near_vehicle: set[tuple[float, ...]] = set()

        for person_det in person_detections:
            person_center = bbox_center(person_det["bbox_xyxy"])
            person_is_close = False

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
                    person_is_close = True

            if person_is_close:
                persons_near_vehicle.add(tuple(person_det["bbox_xyxy"]))

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
                    "num_people_near_vehicle": len(persons_near_vehicle),
                    "matched_pairs": matched_pairs,
                    "mean_confidence": sum(pair_confidences) / len(pair_confidences),
                }
            )

    if len(evidence_frames) < min_evidence_frames:
        return []

    start_time = float(evidence_frames[0]["timestamp"])
    end_time = float(evidence_frames[-1]["timestamp"])
    duration_seconds = end_time - start_time
    mean_confidence = sum(f["mean_confidence"] for f in evidence_frames) / len(
        evidence_frames
    )
    min_distance_px = min(
        float(pair["center_distance_pixels"])
        for f in evidence_frames
        for pair in f["matched_pairs"]
    )
    max_people = max(int(f["num_people_near_vehicle"]) for f in evidence_frames)
    frame_paths = [f["image_path"] for f in evidence_frames]

    metrics = {
        "min_distance_px": min_distance_px,
        "duration_seconds": duration_seconds,
        "mean_confidence": mean_confidence,
        "max_people": float(max_people),
    }
    risk_score, score_breakdown, severity_prior = compute_risk_score(
        metrics=metrics,
        scoring_config=scoring_config,
        distance_threshold_pixels=distance_threshold_pixels,
    )

    return [
        {
            "event_id": "safety_event_person_vehicle_proximity_001",
            "event_type": "person_vehicle_proximity",
            "risk_category": risk_category,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "objects_involved": ["person", "vehicle"],
            "severity_prior": severity_prior,
            "risk_score": risk_score,
            "score_breakdown": score_breakdown,
            "rule_triggered": "person_vehicle_distance_below_threshold",
            "candidate_reason": (
                f"A person and a vehicle were detected in spatial proximity "
                f"(minimum center distance {min_distance_px:.1f}px) across "
                f"{len(evidence_frames)} sampled frames."
            ),
            "evidence": {
                "num_frames": len(evidence_frames),
                "min_distance_px": round(min_distance_px, 2),
                "mean_confidence": round(mean_confidence, 4),
                "frame_paths": frame_paths,
            },
            "evidence_frames": evidence_frames,
            "parameters": {
                "distance_threshold_pixels": distance_threshold_pixels,
            },
            "requires_vlm_review": True,
        }
    ]


# ---------------------------------------------------------------------------
# Retained Level-1 helpers (not yet registered as safety rules). These are the
# building blocks for the scaffolded crowding_or_gathering and
# long_duration_vehicle_presence rules; kept here so they can be wired into the
# registry below with minimal work when enabled in config.
# ---------------------------------------------------------------------------
def generate_presence_events(
        temporal_segments_data: dict[str, Any],
        vehicle_classes: list[str],
        min_event_duration_seconds: float,
        min_evidence_frames: int,
) -> list[dict[str, Any]]:
    """
    Generate person_present and vehicle_present observation events from segments.
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
    Generate multiple_people_present observation events from frame-level detections.

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


# ---------------------------------------------------------------------------
# Safety rule registry
# ---------------------------------------------------------------------------
# Each rule has the uniform signature
#   (detections_data, temporal_segments_data, event_config) -> list[event]
# so new safety event types can be registered with a single line.

def _rule_person_vehicle_proximity(
        detections_data: dict[str, Any],
        temporal_segments_data: dict[str, Any],
        event_config: dict[str, Any],
) -> list[dict[str, Any]]:
    return generate_person_vehicle_proximity_events(
        detections_data=detections_data,
        vehicle_classes=event_config["vehicle_classes"],
        distance_threshold_pixels=float(
            event_config["person_vehicle_distance_threshold_pixels"]
        ),
        min_evidence_frames=int(event_config["min_evidence_frames"]),
        scoring_config=event_config["scoring"],
        risk_category=event_config["risk_categories"]["person_vehicle_proximity"],
    )


SafetyRule = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], list[dict[str, Any]]]

SAFETY_RULES: dict[str, SafetyRule] = {
    "person_vehicle_proximity": _rule_person_vehicle_proximity,
    # "crowding_or_gathering": _rule_crowding_or_gathering,            # future
    # "long_duration_vehicle_presence": _rule_long_duration_presence,  # future
}


def assign_event_ids(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign globally consistent, time-ordered safety event IDs."""
    events = sorted(events, key=lambda x: (x["start_time"], x["event_type"]))

    for idx, event in enumerate(events, start=1):
        event["event_id"] = f"safety_event_{idx:03d}"

    return events

def generate_event_candidates(
    detections_json: Path,
    temporal_segments_json: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate rule-based safety event candidates."""
    detections_data = load_json(detections_json)
    temporal_segments_data = load_json(temporal_segments_json)

    event_config = config["events"]
    enabled_safety_event_types = list(event_config["enabled_safety_event_types"])

    events: list[dict[str, Any]] = []

    for event_type in enabled_safety_event_types:
        rule = SAFETY_RULES.get(event_type)
        if rule is None:
            logger.warning(
                "No rule registered for safety event type: %s (skipping)",
                event_type,
            )
            continue
        events.extend(rule(detections_data, temporal_segments_data, event_config))

    events = assign_event_ids(events)

    result = {
        "video_name": detections_data.get("video_name"),
        "video_path": detections_data.get("video_path"),
        "generation_type": "rule_based_safety_event_candidate_generation",
        "enabled_safety_event_types": enabled_safety_event_types,
        "num_events": len(events),
        "events": events,
    }

    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate safety event candidates from detections and temporal segments."
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

    output_json_path = args.output / "safety_event_candidates.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Saved safety event candidates to: %s", output_json_path)
    logger.info("Number of safety event candidates: %d", result["num_events"])
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()

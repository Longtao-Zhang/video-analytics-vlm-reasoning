"""End-to-end pipeline orchestration (one command: video -> safety report).

Chains every stage of the system in order:

    video
      -> [1] frame sampling          -> sampled_frames.json
      -> [2] object detection        -> detections.json
      -> [3] temporal aggregation    -> temporal_segments.json
      -> [4] safety event generation -> safety_event_candidates.json
      -> [5] VLM-assisted reasoning  -> vlm_event_reasoning.json
      -> [6] safety report           -> safety_report.json + safety_report.md

Design: each stage is a small function that calls the *library* function from the
corresponding module (not a subprocess), saves its JSON output, and returns the
path to that output so the next stage can consume it. ``run_pipeline`` simply wires
them together.

CLI usage:
    python -m src.pipeline \\
        --video examples/input_videos/warehouse_1.mp4 \\
        --config configs/default.yaml \\
        --output examples/outputs/demo/warehouse_1 \\
        --vlm-mode mock
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from src.utils import load_config, save_json
from src.frame_sampler import sample_frames
from src.detector import run_detection_on_sampled_frames
from src.temporal_aggregator import aggregate_temporal_segments
from src.event_generator import generate_event_candidates
from src.vlm_reasoner import reason_about_events
from src.report_generator import build_report, render_markdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — frame sampling  (WORKED EXAMPLE — study this pattern)
# ---------------------------------------------------------------------------
def run_sampling(video_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
    """Sample frames from the video and save sampled_frames.json.

    Returns the path to sampled_frames.json (input to stage 2).
    """
    sampling_cfg = config.get("sampling", {})

    result = sample_frames(
        video_path=video_path,
        sample_fps=float(config["video"]["sample_fps"]),
        output_dir=output_dir,
        image_format=sampling_cfg.get("image_format", "jpg"),
    )

    out_path = output_dir / "sampled_frames.json"
    save_json(result, out_path)
    logger.info("[1/6] Sampling -> %s (%d frames)", out_path, result["num_sampled_frames"])
    return out_path


# ---------------------------------------------------------------------------
# Stage 2 — object detection
# ---------------------------------------------------------------------------
def run_detection(sampled_frames_json: Path, config: dict[str, Any], output_dir: Path) -> Path:
    """Run YOLO detection on the sampled frames and save detections.json.

    Returns the path to detections.json (input to stages 3 and 4).
    """
    detection_cfg = config["detection"]

    result = run_detection_on_sampled_frames(
        sampled_frames_json=sampled_frames_json,
        output_dir=output_dir,
        model_name=detection_cfg["model_name"],
        confidence_threshold=float(detection_cfg["confidence_threshold"]),
        target_classes=detection_cfg.get("target_classes"),
        save_visualization=True,
    )

    out_path = output_dir / "detections.json"
    save_json(result, out_path)
    logger.info("[2/6] Detection -> %s (%d frames)", out_path, result["num_frames"])
    return out_path


# ---------------------------------------------------------------------------
# Stage 3 — temporal aggregation
# ---------------------------------------------------------------------------
def run_aggregation(detections_json: Path, config: dict[str, Any], output_dir: Path) -> Path:
    """Aggregate frame-level detections into temporal segments.

    Returns the path to temporal_segments.json (input to stage 4).
    """
    temporal_cfg = config["temporal"]

    result = aggregate_temporal_segments(
        detections_json=detections_json,
        max_gap_seconds=float(temporal_cfg["max_gap_seconds"]),
        min_segment_duration_seconds=float(temporal_cfg["min_segment_duration_seconds"]),
        min_evidence_frames=int(temporal_cfg["min_evidence_frames"]),
    )

    out_path = output_dir / "temporal_segments.json"
    save_json(result, out_path)
    logger.info("[3/6] Aggregation -> %s (%d segments)", out_path, result["num_segments"])
    return out_path


# ---------------------------------------------------------------------------
# Stage 4 — safety event generation
# ---------------------------------------------------------------------------
def run_event_generation(
        detections_json: Path,
        temporal_segments_json: Path,
        config: dict[str, Any],
        output_dir: Path,
) -> Path:
    """Generate rule-based safety event candidates.

    Returns the path to safety_event_candidates.json (input to stage 5).
    """
    result = generate_event_candidates(
        detections_json=detections_json,
        temporal_segments_json=temporal_segments_json,
        config=config,
    )

    out_path = output_dir / "safety_event_candidates.json"
    save_json(result, out_path)
    logger.info("[4/6] Safety events -> %s (%d events)", out_path, result["num_events"])
    return out_path


# ---------------------------------------------------------------------------
# Stage 5 — VLM-assisted reasoning
# ---------------------------------------------------------------------------
def run_vlm_reasoning(
        safety_events_json: Path,
        config: dict[str, Any],
        output_dir: Path,
        vlm_mode: str | None,
) -> Path:
    """Attach VLM reasoning to each safety event candidate.

    Returns the path to vlm_event_reasoning.json (input to stage 6).
    """
    result = reason_about_events(
        safety_events_json=safety_events_json,
        config=config,
        mode=vlm_mode,
    )

    out_path = output_dir / "vlm_event_reasoning.json"
    save_json(result, out_path)
    logger.info("[5/6] VLM reasoning -> %s (mode=%s)", out_path, result["reasoning_mode"])
    return out_path


# ---------------------------------------------------------------------------
# Stage 6 — video-level safety report
# ---------------------------------------------------------------------------
def run_report(vlm_reasoning_json: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Aggregate reasoning into safety_report.json + safety_report.md.

    Returns the structured report dict (so main() can print a summary).
    """
    report = build_report(vlm_reasoning_json, config)

    save_json(report, output_dir / "safety_report.json")
    (output_dir / "safety_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    logger.info(
        "[6/6] Report -> %s (overall risk: %s)",
        output_dir / "safety_report.md",
        report["overall_risk_level"],
    )
    return report


# ---------------------------------------------------------------------------
# Orchestration  (complete — no need to edit, but read it to see the wiring)
# ---------------------------------------------------------------------------
def run_pipeline(
        video_path: Path,
        config: dict[str, Any],
        output_dir: Path,
        vlm_mode: str | None = None,
) -> dict[str, Any]:
    """Run every stage in order and return the final report dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_frames_json = run_sampling(video_path, config, output_dir)
    detections_json = run_detection(sampled_frames_json, config, output_dir)
    temporal_segments_json = run_aggregation(detections_json, config, output_dir)
    safety_events_json = run_event_generation(
        detections_json, temporal_segments_json, config, output_dir
    )
    vlm_reasoning_json = run_vlm_reasoning(
        safety_events_json, config, output_dir, vlm_mode
    )
    report = run_report(vlm_reasoning_json, config, output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full video -> safety report pipeline."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--vlm-mode",
        choices=["mock", "api"],
        default=None,
        help="VLM reasoning mode; overrides config['vlm']['mode'].",
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

    report = run_pipeline(
        video_path=args.video,
        config=config,
        output_dir=args.output,
        vlm_mode=args.vlm_mode,
    )

    elapsed_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Pipeline complete for: %s", args.video.name)
    logger.info("Overall risk level: %s", report["overall_risk_level"])
    logger.info(
        "Confirmed: %d | Dismissed: %d | Candidates: %d",
        report["num_confirmed_events"],
        report["num_dismissed_events"],
        report["num_candidate_events"],
    )
    logger.info("Report: %s", args.output / "safety_report.md")
    logger.info("Total time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()

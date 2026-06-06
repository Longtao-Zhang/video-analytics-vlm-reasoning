"""Sampling-rate benchmark for the CV front-end of the pipeline.

Runs the deterministic front-end (frame sampling -> object detection -> temporal
aggregation -> rule-based safety event generation) at several sampling rates and
records, per rate: stage runtimes, number of sampled frames, number of
detections, number of temporal segments, and number of safety event candidates.

This quantifies the core trade-off for near-real-time safety monitoring: higher
FPS captures more evidence (and potentially more events) but costs more runtime
and, downstream, more VLM calls. The VLM stage is intentionally excluded here --
this benchmark is about the CV front-end cost, not API cost.

CLI usage:
    python -m src.sampling_benchmark \\
        --video examples/input_videos/warehouse_1.mp4 \\
        --fps 1 2 5 10 \\
        --config configs/default.yaml \\
        --output examples/outputs/benchmarks/warehouse_1

Outputs (under --output):
    <video>_<fps>fps/...        per-rate intermediate artifacts
    benchmark.json              structured results
    benchmark.md                a readable results table
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from src.detector import run_detection_on_sampled_frames
from src.event_generator import generate_event_candidates
from src.frame_sampler import sample_frames
from src.temporal_aggregator import aggregate_temporal_segments
from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def benchmark_single_fps(
        video_path: Path,
        sample_fps: float,
        config: dict[str, Any],
        output_root: Path,
) -> dict[str, Any]:
    """Run the full CV front-end at one sampling rate and time each stage."""
    out_dir = output_root / f"{video_path.stem}_{sample_fps:g}fps"
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_config = config["detection"]
    temporal_config = config["temporal"]
    image_format = config.get("sampling", {}).get("image_format", "jpg")

    # 1. Frame sampling
    t0 = time.time()
    sampled = sample_frames(
        video_path=video_path,
        sample_fps=sample_fps,
        output_dir=out_dir,
        image_format=image_format,
    )
    save_json(sampled, out_dir / "sampled_frames.json")
    sample_time = time.time() - t0
    num_sampled_frames = int(sampled["num_sampled_frames"])

    # 2. Object detection (no visualization, to keep the benchmark lean)
    t0 = time.time()
    detections = run_detection_on_sampled_frames(
        sampled_frames_json=out_dir / "sampled_frames.json",
        output_dir=out_dir,
        model_name=detection_config["model_name"],
        confidence_threshold=float(detection_config["confidence_threshold"]),
        target_classes=detection_config.get("target_classes"),
        save_visualization=False,
    )
    save_json(detections, out_dir / "detections.json")
    detect_time = time.time() - t0
    num_detections = sum(len(f.get("detections", [])) for f in detections["frames"])

    # 3. Temporal aggregation
    t0 = time.time()
    segments = aggregate_temporal_segments(
        detections_json=out_dir / "detections.json",
        max_gap_seconds=float(temporal_config["max_gap_seconds"]),
        min_segment_duration_seconds=float(temporal_config["min_segment_duration_seconds"]),
        min_evidence_frames=int(temporal_config["min_evidence_frames"]),
    )
    save_json(segments, out_dir / "temporal_segments.json")
    aggregate_time = time.time() - t0
    num_segments = int(segments["num_segments"])

    # 4. Rule-based safety event generation
    t0 = time.time()
    events = generate_event_candidates(
        detections_json=out_dir / "detections.json",
        temporal_segments_json=out_dir / "temporal_segments.json",
        config=config,
    )
    save_json(events, out_dir / "safety_event_candidates.json")
    events_time = time.time() - t0
    num_events = int(events["num_events"])

    total_time = sample_time + detect_time + aggregate_time + events_time

    return {
        "sample_fps": sample_fps,
        "num_sampled_frames": num_sampled_frames,
        "num_detections": num_detections,
        "num_segments": num_segments,
        "num_safety_events": num_events,
        "runtime_seconds": {
            "sample": round(sample_time, 3),
            "detect": round(detect_time, 3),
            "aggregate": round(aggregate_time, 3),
            "events": round(events_time, 3),
            "total": round(total_time, 3),
        },
        "frames_per_second_processed": (
            round(num_sampled_frames / total_time, 2) if total_time > 0 else None
        ),
        "output_dir": str(out_dir),
    }


def run_benchmark(
        video_path: Path,
        fps_values: list[float],
        config: dict[str, Any],
        output_root: Path,
) -> dict[str, Any]:
    """Benchmark the pipeline across all requested sampling rates."""
    rows: list[dict[str, Any]] = []
    for fps in fps_values:
        logger.info("Benchmarking %s at %g FPS ...", video_path.name, fps)
        rows.append(benchmark_single_fps(video_path, fps, config, output_root))

    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "model_name": config["detection"]["model_name"],
        "fps_values": fps_values,
        "results": rows,
    }


def render_markdown(benchmark: dict[str, Any]) -> str:
    """Render the benchmark as a readable Markdown table."""
    lines = [
        f"# Sampling-Rate Benchmark — {benchmark['video_name']}",
        "",
        f"Detection model: `{benchmark['model_name']}`",
        "",
        "| FPS | Sampled frames | Detections | Segments | Safety events | "
        "Detect (s) | Total (s) | Frames/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in benchmark["results"]:
        rt = r["runtime_seconds"]
        lines.append(
            f"| {r['sample_fps']:g} | {r['num_sampled_frames']} | "
            f"{r['num_detections']} | {r['num_segments']} | "
            f"{r['num_safety_events']} | {rt['detect']:.2f} | {rt['total']:.2f} | "
            f"{r['frames_per_second_processed']} |"
        )
    lines.append("")
    lines.append(
        "_VLM reasoning is excluded; downstream API cost scales roughly with the "
        "number of safety events requiring review._"
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the CV front-end across sampling rates."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Sampling rates to benchmark (default: 1 2 5 10).",
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
        help="Output directory root for benchmark artifacts.",
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

    benchmark = run_benchmark(
        video_path=args.video,
        fps_values=list(args.fps),
        config=config,
        output_root=args.output,
    )

    json_path = args.output / "benchmark.json"
    md_path = args.output / "benchmark.md"
    save_json(benchmark, json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(benchmark), encoding="utf-8")

    logger.info("Saved benchmark (JSON) to: %s", json_path)
    logger.info("Saved benchmark (Markdown) to: %s", md_path)
    logger.info("Total benchmark wall time: %.2f seconds", time.time() - start_time)
    print()
    print(render_markdown(benchmark))


if __name__ == "__main__":
    main()

"""Video-level safety report generator (business summary).

Aggregates per-event VLM reasoning (vlm_event_reasoning.json) into a single
video-level risk assessment and a business-readable report for non-technical
safety managers. Produces two artifacts:

    safety_report.json  -- structured, machine-readable rollup
    safety_report.md    -- executive summary + key events + recommended actions

Risk rollup (interpretable, "max verified risk"):
    The overall risk level is the highest VLM ``risk_level`` among events the VLM
    did NOT reject. Candidates with ``verified == false`` are treated as likely
    false positives and demoted out of the headline risk (but still listed, for
    auditability). This directly uses the VLM verification signal rather than
    trusting the rule's ``severity_prior`` blindly -- see docs/failure_cases.md
    (FP-001) for why that matters.

CLI usage:
    python -m src.report_generator \\
        --vlm-reasoning-json path/to/vlm_event_reasoning.json \\
        --config configs/default.yaml \\
        --output path/to/output_dir

Library usage:
    from src.report_generator import build_report
    report = build_report(Path("vlm_event_reasoning.json"), config)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import load_config, save_json
from src.vlm_reasoner import format_timestamp

logger = logging.getLogger(__name__)

RISK_ORDER = {"low": 1, "medium": 2, "high": 3}

EVENT_TYPE_LABELS = {
    "person_vehicle_proximity": "Person–Vehicle Proximity",
    "crowding_or_gathering": "Crowding / Gathering",
    "long_duration_vehicle_presence": "Long-Duration Vehicle Presence",
    "person_present": "Person Present",
    "vehicle_present": "Vehicle Present",
    "multiple_people_present": "Multiple People Present",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load a json file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def human_event_type(event_type: str) -> str:
    """Map a machine event type to a business-readable label."""
    return EVENT_TYPE_LABELS.get(event_type, event_type.replace("_", " ").title())


def md_cell(text: Any, max_len: int = 240) -> str:
    """Sanitize text for a single Markdown table cell."""
    s = " ".join(str(text).split())  # collapse whitespace/newlines
    s = s.replace("|", "\\|")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def classify_event(event: dict[str, Any]) -> dict[str, Any]:
    """Derive a report-level view of one event, applying the verification signal.

    status:
        confirmed    -- reviewed and not rejected (counts toward headline risk)
        dismissed    -- VLM verified == false (likely false positive)
        not_reviewed -- no VLM reasoning attached (falls back to rule prior)
    """
    vlm = event.get("vlm_reasoning")
    severity_prior = event.get("severity_prior", "medium")

    start = float(event.get("start_time", 0.0))
    end = float(event.get("end_time", 0.0))
    time_window = f"{format_timestamp(start)}–{format_timestamp(end)}"

    if vlm is None:
        status = "not_reviewed"
        effective_risk = severity_prior
        verified: Any = None
        description = event.get("candidate_reason", "")
        recommended_action = ""
        confidence: Any = None
    else:
        verified = vlm.get("verified")
        confidence = vlm.get("confidence")
        description = vlm.get("scene_description") or vlm.get("safety_risk") or ""
        recommended_action = vlm.get("recommended_action", "")
        if verified is False:
            status = "dismissed"
            effective_risk = "low"
        else:
            status = "confirmed"
            effective_risk = str(vlm.get("risk_level", severity_prior)).lower()

    if effective_risk not in RISK_ORDER:
        effective_risk = "low"

    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "event_type_label": human_event_type(event.get("event_type", "")),
        "risk_category": event.get("risk_category"),
        "start_time": start,
        "end_time": end,
        "time_window": time_window,
        "status": status,
        "severity_prior": severity_prior,
        "vlm_verified": verified,
        "vlm_risk_level": (vlm or {}).get("risk_level") if vlm else None,
        "vlm_confidence": confidence,
        "effective_risk_level": effective_risk,
        "description": description,
        "safety_risk": (vlm or {}).get("safety_risk", "") if vlm else "",
        "recommended_action": recommended_action,
    }


def aggregate_overall_risk(
        classified: list[dict[str, Any]],
) -> tuple[str, str]:
    """Compute the video-level overall risk and a one-line rationale.

    Overall risk = the highest effective risk among confirmed (non-dismissed)
    events. If nothing is confirmed, the video is reported as low risk.
    """
    confirmed = [e for e in classified if e["status"] == "confirmed"]
    if not confirmed:
        return "low", (
            "No safety events were confirmed by VLM reasoning "
            f"({len(classified)} candidate(s) reviewed)."
        )

    top = max(confirmed, key=lambda e: RISK_ORDER[e["effective_risk_level"]])
    overall = top["effective_risk_level"]
    n_top = sum(1 for e in confirmed if e["effective_risk_level"] == overall)
    rationale = (
        f"Driven by {n_top} confirmed {overall}-risk event(s); highest is "
        f"{top['event_type_label']} at {top['time_window']}."
    )
    return overall, rationale


def dedup_actions(classified: list[dict[str, Any]]) -> list[str]:
    """Collect unique recommended actions from confirmed events, in time order."""
    actions: list[str] = []
    for e in sorted(classified, key=lambda x: x["start_time"]):
        if e["status"] != "confirmed":
            continue
        action = (e["recommended_action"] or "").strip()
        if action and action not in actions:
            actions.append(action)
    return actions


def build_report(
        vlm_reasoning_json: Path,
        config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate VLM event reasoning into a structured video-level report."""
    data = load_json(vlm_reasoning_json)
    events = data.get("events", [])

    classified = [classify_event(e) for e in events]
    classified.sort(key=lambda e: e["start_time"])

    overall_risk, rationale = aggregate_overall_risk(classified)

    num_confirmed = sum(1 for e in classified if e["status"] == "confirmed")
    num_dismissed = sum(1 for e in classified if e["status"] == "dismissed")

    contributing_factors = [
        f"{e['event_type_label']} ({e['effective_risk_level']}) at {e['time_window']}"
        for e in classified
        if e["status"] == "confirmed"
    ]

    technical_notes: dict[str, Any] = {
        "reasoning_mode": data.get("reasoning_mode"),
        "pipeline": (
            "frame sampling -> YOLO object detection -> temporal aggregation -> "
            "rule-based safety event generation -> VLM-assisted reasoning"
        ),
    }
    if config is not None:
        technical_notes["detection_model"] = config.get("detection", {}).get("model_name")
        technical_notes["sample_fps"] = config.get("video", {}).get("sample_fps")

    return {
        "video_name": data.get("video_name"),
        "video_path": data.get("video_path"),
        "report_type": "warehouse_safety_video_intelligence_report",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_risk_level": overall_risk,
        "overall_risk_rationale": rationale,
        "num_candidate_events": len(classified),
        "num_confirmed_events": num_confirmed,
        "num_dismissed_events": num_dismissed,
        "contributing_factors": contributing_factors,
        "recommended_actions": dedup_actions(classified),
        "technical_notes": technical_notes,
        "events": classified,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the structured report into a business-readable Markdown document."""
    confirmed = [e for e in report["events"] if e["status"] == "confirmed"]
    dismissed = [e for e in report["events"] if e["status"] == "dismissed"]

    lines: list[str] = []
    lines.append("# Warehouse Safety Video Intelligence Report")
    lines.append("")
    lines.append(f"**Video:** {report.get('video_name', 'unknown')}  ")
    lines.append(f"**Generated:** {report.get('generated_at', '')}  ")
    lines.append(f"**Overall risk level:** {report['overall_risk_level'].upper()}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"The system analyzed `{report.get('video_name', 'the video')}` and generated "
        f"{report['num_candidate_events']} candidate safety event(s). "
        f"{report['num_confirmed_events']} were confirmed by VLM-assisted reasoning "
        f"and {report['num_dismissed_events']} dismissed as likely false positive(s). "
        f"The overall safety risk is assessed as **{report['overall_risk_level']}** "
        f"({report['overall_risk_rationale']})"
    )
    lines.append("")

    # Key safety events
    lines.append("## Key Safety Events")
    lines.append("")
    if confirmed:
        lines.append("| Time Window | Event Type | Risk Level | Description | Recommended Action |")
        lines.append("|---|---|---|---|---|")
        for e in confirmed:
            lines.append(
                f"| {e['time_window']} | {md_cell(e['event_type_label'])} | "
                f"{e['effective_risk_level']} | {md_cell(e['description'])} | "
                f"{md_cell(e['recommended_action'])} |"
            )
    else:
        lines.append("_No safety events were confirmed for this video._")
    lines.append("")

    # Dismissed candidates (false positives caught by the VLM)
    if dismissed:
        lines.append("## Dismissed Candidates (VLM-verified false positives)")
        lines.append("")
        lines.append("| Time Window | Event Type | Rule Severity | Why Dismissed |")
        lines.append("|---|---|---|---|")
        for e in dismissed:
            lines.append(
                f"| {e['time_window']} | {md_cell(e['event_type_label'])} | "
                f"{e['severity_prior']} | {md_cell(e['safety_risk'] or e['description'])} |"
            )
        lines.append("")

    # Risk assessment
    lines.append("## Risk Assessment")
    lines.append("")
    lines.append(f"Overall risk level: **{report['overall_risk_level']}**")
    lines.append("")
    if report["contributing_factors"]:
        lines.append("Main contributing factors:")
        for factor in report["contributing_factors"]:
            lines.append(f"- {factor}")
    else:
        lines.append("No confirmed events contributed to the risk assessment.")
    lines.append("")

    # Recommended actions
    lines.append("## Recommended Actions")
    lines.append("")
    if report["recommended_actions"]:
        for i, action in enumerate(report["recommended_actions"], start=1):
            lines.append(f"{i}. {action}")
    else:
        lines.append("No specific actions recommended; continue routine monitoring.")
    lines.append("")

    # Technical notes
    notes = report["technical_notes"]
    lines.append("## Technical Notes")
    lines.append("")
    lines.append(f"- Pipeline: {notes.get('pipeline')}")
    if notes.get("detection_model"):
        lines.append(f"- Object detection: {notes['detection_model']}")
    if notes.get("sample_fps") is not None:
        lines.append(f"- Frame sampling: {notes['sample_fps']} FPS")
    lines.append(f"- Reasoning: {notes.get('reasoning_mode')}")
    lines.append(
        "- Disclaimer: this is an automated decision-support prototype, not a "
        "certified safety system. Findings should be confirmed by a human reviewer."
    )
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video-level safety report from VLM event reasoning."
    )
    parser.add_argument(
        "--vlm-reasoning-json",
        type=Path,
        required=True,
        help="Path to vlm_event_reasoning.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file (used for technical notes).",
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
    config = load_config(args.config) if args.config.exists() else None

    report = build_report(args.vlm_reasoning_json, config)

    json_path = args.output / "safety_report.json"
    md_path = args.output / "safety_report.md"
    save_json(report, json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    elapsed_time = time.time() - start_time

    logger.info("Saved safety report (JSON) to: %s", json_path)
    logger.info("Saved safety report (Markdown) to: %s", md_path)
    logger.info(
        "Overall risk: %s | confirmed: %d | dismissed: %d",
        report["overall_risk_level"],
        report["num_confirmed_events"],
        report["num_dismissed_events"],
    )
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()

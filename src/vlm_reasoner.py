"""VLM-assisted safety event reasoning.

Reads rule-based safety event candidates (safety_event_candidates.json) and, for
each candidate flagged with ``requires_vlm_review``, produces a business-readable
reasoning block: scene description, safety risk, risk level, reasoning,
recommended action and a confidence score. Results are written to
vlm_event_reasoning.json.

Design principle (see docs/ideas.md):
    Rules generate candidate safety events; the VLM explains, verifies, and
    contextualizes them. The VLM never invents events from scratch -- it reasons
    over evidence the deterministic rules already produced.

Modes
    mock : deterministic templates that echo the rule evidence. No API key, no
           network -- the whole pipeline runs end-to-end for demos / reviewers.
    api  : Alibaba Cloud Model Studio (DashScope) Qwen-VL models via the
           OpenAI-compatible endpoint. Sends evidence frames + event metadata and
           parses a strict-JSON reply. A failed call falls back to the mock block
           for that event so one bad request never breaks the run.

CLI usage:
    python -m src.vlm_reasoner \\
        --safety-events-json path/to/safety_event_candidates.json \\
        --config configs/default.yaml \\
        --output path/to/output_dir \\
        --mode mock          # or: api  (overrides config["vlm"]["mode"])

Library usage:
    from src.vlm_reasoner import reason_about_events
    result = reason_about_events(
        safety_events_json=Path("safety_event_candidates.json"),
        config=load_config("configs/default.yaml"),
        mode="mock",
    )
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)

CONFIDENCE_BY_SEVERITY = {"low": 0.55, "medium": 0.7, "high": 0.85}


def load_json(path: Path) -> dict[str, Any]:
    """Load a json file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds: float) -> str:
    """Format a number of seconds as MM:SS."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    if secs == 60:
        minutes += 1
        secs = 0
    return f"{minutes:02d}:{secs:02d}"


def select_frame_paths(event: dict[str, Any], max_frames: int) -> list[str]:
    """Pick up to ``max_frames`` evidence frames, evenly spaced across the event."""
    frame_paths = event.get("evidence", {}).get("frame_paths", [])
    if max_frames <= 0 or not frame_paths:
        return []
    if len(frame_paths) <= max_frames:
        return list(frame_paths)
    if max_frames == 1:
        return [frame_paths[len(frame_paths) // 2]]

    step = (len(frame_paths) - 1) / (max_frames - 1)
    indices = sorted({round(i * step) for i in range(max_frames)})
    return [frame_paths[i] for i in indices]


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------
MOCK_TEMPLATES: dict[str, dict[str, str]] = {
    "person_vehicle_proximity": {
        "scene_description": (
            "In a {business_context} environment, a person appears in close "
            "proximity to a vehicle across several sampled frames."
        ),
        "safety_risk": (
            "Potential collision or near-miss risk between a worker and a vehicle."
        ),
        "recommended_action": (
            "Review the footage around {start}-{end} and consider enforcing a "
            "minimum separation zone between workers and vehicles."
        ),
    },
}

_GENERIC_TEMPLATE = {
    "scene_description": (
        "In a {business_context} environment, a rule-flagged safety-relevant "
        "situation was observed across several sampled frames."
    ),
    "safety_risk": "A potential safety-relevant condition that warrants review.",
    "recommended_action": (
        "Review the footage around {start}-{end} and assess whether a safety "
        "procedure should be applied."
    ),
}


def mock_reasoning_for_event(
        event: dict[str, Any],
        business_context: str,
) -> dict[str, Any]:
    """Produce a deterministic reasoning block that echoes the rule evidence."""
    event_type = event["event_type"]
    severity = event.get("severity_prior", "medium")
    evidence = event.get("evidence", {})
    start = format_timestamp(event.get("start_time", 0.0))
    end = format_timestamp(event.get("end_time", 0.0))
    num_frames = evidence.get("num_frames", event.get("num_evidence_frames", 0))
    min_distance_px = evidence.get("min_distance_px")

    template = MOCK_TEMPLATES.get(event_type, _GENERIC_TEMPLATE)
    fmt = {"business_context": business_context, "start": start, "end": end}

    reasoning_parts = [
        f"The deterministic rule flagged this event across {num_frames} sampled frames."
    ]
    if min_distance_px is not None:
        reasoning_parts.append(
            f"The minimum person-vehicle center distance was {min_distance_px}px."
        )
    reasoning_parts.append(
        f"The rule-based severity prior is '{severity}'. No clear evidence of "
        "direct contact is visible from the sampled frames alone."
    )

    return {
        "verified": True,
        "scene_description": template["scene_description"].format(**fmt),
        "safety_risk": template["safety_risk"].format(**fmt),
        "risk_level": severity,
        "reasoning": " ".join(reasoning_parts),
        "recommended_action": template["recommended_action"].format(**fmt),
        "confidence": CONFIDENCE_BY_SEVERITY.get(severity, 0.65),
        "reasoning_mode": "mock",
    }


# ---------------------------------------------------------------------------
# API mode (DashScope / Qwen-VL via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a warehouse and manufacturing safety analyst AI. You are given a "
    "candidate safety event detected by a deterministic computer-vision rule "
    "system, together with a few evidence frames sampled from the video. Verify "
    "whether the event is visually plausible, describe the scene, assess the "
    "safety risk, assign a risk level, and recommend a concrete action. Base your "
    "judgement only on what is visible; do not hallucinate motion or contact that "
    "the still frames do not support. Respond with a SINGLE JSON object and "
    "nothing else, using exactly these keys: verified (boolean), scene_description "
    "(string), safety_risk (string), risk_level (one of \"low\", \"medium\", "
    "\"high\"), reasoning (string), recommended_action (string), confidence "
    "(number between 0 and 1)."
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image_to_data_url(image_path: str) -> str:
    """Read an image file and return a base64 data URL."""
    path = Path(image_path)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_api_messages(
        event: dict[str, Any],
        business_context: str,
        image_paths: list[str],
) -> list[dict[str, Any]]:
    """Build chat messages (event metadata + evidence frames) for the VLM."""
    evidence = event.get("evidence", {})
    user_text = (
        f"Business context: {business_context}.\n"
        f"Candidate safety event type: {event.get('event_type')}.\n"
        f"Risk category: {event.get('risk_category')}.\n"
        f"Objects involved: {', '.join(event.get('objects_involved', []))}.\n"
        f"Time window: {format_timestamp(event.get('start_time', 0.0))}"
        f"-{format_timestamp(event.get('end_time', 0.0))}.\n"
        f"Rule-based severity prior: {event.get('severity_prior')}.\n"
        f"Why the rule fired: {event.get('candidate_reason')}\n"
        f"Evidence summary: {evidence.get('num_frames')} frames, minimum "
        f"distance {evidence.get('min_distance_px')}px, mean detection "
        f"confidence {evidence.get('mean_confidence')}.\n"
        f"Below are {len(image_paths)} evidence frames sampled from this event."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image_path)},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_vlm_json(text: str) -> dict[str, Any]:
    """Tolerantly parse a JSON object out of the model reply (handles code fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in VLM response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def normalize_vlm_block(parsed: dict[str, Any], model: str) -> dict[str, Any]:
    """Coerce a parsed VLM reply into the canonical reasoning block."""
    risk_level = str(parsed.get("risk_level", "medium")).lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"

    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    return {
        "verified": bool(parsed.get("verified", True)),
        "scene_description": str(parsed.get("scene_description", "")),
        "safety_risk": str(parsed.get("safety_risk", "")),
        "risk_level": risk_level,
        "reasoning": str(parsed.get("reasoning", "")),
        "recommended_action": str(parsed.get("recommended_action", "")),
        "confidence": round(confidence, 4),
        "reasoning_mode": f"api:{model}",
    }


# DashScope's OpenAI-compatible endpoint is a fixed host that differs by the
# account region -- it is NOT a per-workspace subdomain.
REGION_BASE_URLS = {
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore / international
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "international": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",           # Beijing / China
    "china": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def resolve_base_url(api_config: dict[str, Any]) -> str:
    """Resolve the DashScope compatible-mode base URL.

    Priority: explicit ``base_url`` in config > region lookup (from the env var
    named by ``region_env``) > international default.
    """
    explicit = api_config.get("base_url")
    if explicit:
        return explicit

    region_env = api_config.get("region_env", "DASHSCOPE_REGION")
    region = (os.getenv(region_env) or "").strip().lower()
    if region in REGION_BASE_URLS:
        return REGION_BASE_URLS[region]
    if region:
        logger.warning(
            "Unknown DASHSCOPE region %r; defaulting to the international endpoint. "
            "Set vlm.api.base_url explicitly to override.",
            region,
        )
    return DEFAULT_BASE_URL


def build_api_client(api_config: dict[str, Any]) -> Any:
    """Create an OpenAI-compatible client pointed at the DashScope endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'openai' package is required for VLM api mode. "
            "Install it via: pip install openai"
        ) from exc

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning(
            "python-dotenv is not installed; skipping .env loading. "
            "Set environment variables manually."
        )

    api_key_env = api_config.get("api_key_env", "DASHSCOPE_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set; set it in your shell "
            "or define it in a local .env file."
        )

    base_url = resolve_base_url(api_config)

    # Workspace is optional for compatible mode (the key's default workspace is
    # used). If provided, pass it as a header rather than baking it into the URL.
    workspace_id_env = api_config.get("workspace_id_env", "DASHSCOPE_WORKSPACE_ID")
    workspace_id = os.getenv(workspace_id_env)
    default_headers = {"X-DashScope-WorkSpace": workspace_id} if workspace_id else None

    logger.info("Using DashScope endpoint: %s", base_url)
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)


def api_reasoning_for_event(
        client: Any,
        model: str,
        event: dict[str, Any],
        business_context: str,
        image_paths: list[str],
        api_config: dict[str, Any],
) -> dict[str, Any]:
    """Call the VLM for one event and return a normalized reasoning block."""
    messages = build_api_messages(event, business_context, image_paths)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(api_config.get("temperature", 0.2)),
        max_tokens=int(api_config.get("max_tokens", 800)),
    )
    text = response.choices[0].message.content or ""
    parsed = parse_vlm_json(text)
    return normalize_vlm_block(parsed, model)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def reason_about_events(
        safety_events_json: Path,
        config: dict[str, Any],
        mode: str | None = None,
) -> dict[str, Any]:
    """Attach VLM reasoning to each safety event candidate that requires review."""
    data = load_json(safety_events_json)

    vlm_config = config["vlm"]
    mode = mode or vlm_config.get("mode", "mock")
    if mode not in {"mock", "api"}:
        raise ValueError(f"Unknown VLM mode: {mode!r} (expected 'mock' or 'api').")

    business_context = vlm_config.get("business_context", "safety monitoring")
    max_frames = int(vlm_config.get("max_evidence_frames", 3))
    api_config = vlm_config.get("api", {})
    model = api_config.get("model", "qwen-vl-max")

    # In api mode, set up the client once. If it cannot be created (missing key,
    # missing package), degrade gracefully to mock for every event.
    client: Any = None
    if mode == "api":
        try:
            client = build_api_client(api_config)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Could not initialize VLM API client (%s); falling back to mock "
                "for all events.",
                exc,
            )
            client = None

    reasoned_events: list[dict[str, Any]] = []
    num_reviewed = 0

    for event in data.get("events", []):
        merged = dict(event)

        if not event.get("requires_vlm_review", False):
            merged["vlm_reasoning"] = None
            reasoned_events.append(merged)
            continue

        num_reviewed += 1

        if mode == "mock" or client is None:
            vlm_block = mock_reasoning_for_event(event, business_context)
            if mode == "api":
                # api requested but client unavailable -> graceful fallback
                vlm_block["verified"] = None
                vlm_block["reasoning_mode"] = "api_error_fallback"
        else:
            image_paths = select_frame_paths(event, max_frames)
            try:
                vlm_block = api_reasoning_for_event(
                    client=client,
                    model=model,
                    event=event,
                    business_context=business_context,
                    image_paths=image_paths,
                    api_config=api_config,
                )
            except Exception as exc:  # noqa: BLE001 - one bad call must not break the run
                logger.warning(
                    "VLM API reasoning failed for %s (%s); falling back to mock.",
                    event.get("event_id"),
                    exc,
                )
                vlm_block = mock_reasoning_for_event(event, business_context)
                vlm_block["verified"] = None
                vlm_block["reasoning_mode"] = "api_error_fallback"

        merged["vlm_reasoning"] = {"event_id": event["event_id"], **vlm_block}
        reasoned_events.append(merged)

    reasoning_mode = "mock" if (mode == "mock" or client is None) else f"api:{model}"

    return {
        "video_name": data.get("video_name"),
        "video_path": data.get("video_path"),
        "reasoning_type": "vlm_assisted_safety_event_reasoning",
        "reasoning_mode": reasoning_mode,
        "num_events": len(reasoned_events),
        "num_reviewed_events": num_reviewed,
        "events": reasoned_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-assisted reasoning over safety event candidates."
    )
    parser.add_argument(
        "--safety-events-json",
        type=Path,
        required=True,
        help="Path to safety_event_candidates.json.",
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
    parser.add_argument(
        "--mode",
        choices=["mock", "api"],
        default=None,
        help="Reasoning mode; overrides config['vlm']['mode'].",
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

    result = reason_about_events(
        safety_events_json=args.safety_events_json,
        config=config,
        mode=args.mode,
    )

    output_json_path = args.output / "vlm_event_reasoning.json"
    save_json(result, output_json_path)

    elapsed_time = time.time() - start_time

    logger.info("Saved VLM event reasoning to: %s", output_json_path)
    logger.info("Reasoning mode: %s", result["reasoning_mode"])
    logger.info(
        "Reviewed %d of %d events.",
        result["num_reviewed_events"],
        result["num_events"],
    )
    logger.info("Elapsed time: %.2f seconds", elapsed_time)


if __name__ == "__main__":
    main()

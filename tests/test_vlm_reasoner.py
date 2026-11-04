from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_frame
from src import vlm_reasoner
from src.event_generator import generate_event_candidates
from src.utils import save_json
from src.vlm_reasoner import normalize_vlm_block, parse_verified, reason_about_events


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True), (False, False),
        ("true", True), ("False", False), (" no ", False), ("yes", True),
        (None, None), ("maybe", None), (0.0, None), (1, None),
    ],
)
def test_parse_verified(value, expected):
    assert parse_verified(value) is expected


def test_string_false_is_a_rejection():
    # Regression: bool("false") is True, which used to confirm rejected events.
    assert normalize_vlm_block({"verified": "false"}, "m")["verified"] is False
    assert normalize_vlm_block({}, "m")["verified"] is None


@pytest.fixture
def candidates_json(tmp_path, config) -> Path:
    """Two proximity candidates with real (tiny) evidence image files."""
    frames = []
    for i, t in enumerate([1.0, 1.2, 100.0, 100.2]):
        img = tmp_path / f"frame_{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xd9")
        frames.append(make_frame(i, t, near=True, image_path=str(img)))
    det, seg = tmp_path / "det.json", tmp_path / "seg.json"
    save_json({"video_name": "clip.mp4", "frames": frames}, det)
    save_json({"segments": []}, seg)
    out = tmp_path / "candidates.json"
    save_json(generate_event_candidates(det, seg, config), out)
    return out


def fake_client(replies: list):
    """OpenAI-style client returning the given replies; Exceptions are raised."""
    replies = list(replies)

    def create(**_kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        msg = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_partial_api_failure_is_not_reported_as_clean_api_run(candidates_json, config, monkeypatch):
    ok = '{"verified": true, "risk_level": "medium", "confidence": 0.8}'
    monkeypatch.setattr(
        vlm_reasoner, "build_api_client", lambda _cfg: fake_client([ok, RuntimeError("timeout")])
    )
    result = reason_about_events(candidates_json, config, mode="api")

    assert result["num_api_succeeded"] == 1
    assert result["num_fallback"] == 1
    assert result["reasoning_mode"].endswith("+partial_fallback")
    modes = [e["vlm_reasoning"]["reasoning_mode"] for e in result["events"]]
    assert modes[1] == "api_error_fallback"
    assert result["events"][1]["vlm_reasoning"]["verified"] is None


def test_missing_api_key_falls_back_and_says_so(candidates_json, config, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    result = reason_about_events(candidates_json, config, mode="api")

    assert result["reasoning_mode"] == "api_error_fallback"
    assert result["num_fallback"] == result["num_reviewed_events"] == 2

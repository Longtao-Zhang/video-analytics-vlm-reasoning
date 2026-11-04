from pathlib import Path

from conftest import make_frame
from src.event_generator import generate_event_candidates
from src.utils import save_json


def run_rules(tmp_path: Path, frames: list[dict], config: dict) -> dict:
    det, seg = tmp_path / "detections.json", tmp_path / "segments.json"
    save_json({"video_name": "clip.mp4", "frames": frames}, det)
    save_json({"segments": []}, seg)
    return generate_event_candidates(det, seg, config)


def test_separate_approaches_become_separate_events(tmp_path, config):
    # Two short approaches ~100 s apart, nothing close in between.
    frames = [
        make_frame(0, 1.0, near=True),
        make_frame(1, 1.2, near=True),
        make_frame(2, 50.0, near=False),
        make_frame(3, 100.0, near=True),
        make_frame(4, 100.2, near=True),
    ]
    result = run_rules(tmp_path, frames, config)

    assert result["num_events"] == 2
    first, second = result["events"]
    assert (first["start_time"], first["end_time"]) == (1.0, 1.2)
    assert (second["start_time"], second["end_time"]) == (100.0, 100.2)
    assert [e["event_id"] for e in result["events"]] == ["safety_event_001", "safety_event_002"]
    # The old single merged event had duration 99.2 s, saturating to "high".
    assert all(e["duration_seconds"] < 1.0 for e in result["events"])


def test_continuous_approach_is_one_event(tmp_path, config):
    frames = [make_frame(i, i * 0.2, near=True) for i in range(10)]  # 5 FPS, 1.8 s
    result = run_rules(tmp_path, frames, config)

    assert result["num_events"] == 1
    assert result["events"][0]["evidence"]["num_frames"] == 10


def test_isolated_single_frame_is_ignored(tmp_path, config):
    # min_evidence_frames = 2 in the default config.
    frames = [make_frame(0, 1.0, near=True), make_frame(1, 30.0, near=True)]
    assert run_rules(tmp_path, frames, config)["num_events"] == 0

from src.report_generator import build_report, render_markdown
from src.utils import save_json


def event(event_id, start, verified, risk="high"):
    return {
        "event_id": event_id,
        "event_type": "person_vehicle_proximity",
        "start_time": start,
        "end_time": start + 1,
        "severity_prior": "medium",
        "vlm_reasoning": {"verified": verified, "risk_level": risk, "recommended_action": "Check."},
    }


def report_for(tmp_path, events):
    path = tmp_path / "vlm.json"
    save_json({"video_name": "clip.mp4", "reasoning_mode": "api:m", "events": events}, path)
    return build_report(path)


def test_dismissed_event_does_not_drive_risk(tmp_path):
    report = report_for(tmp_path, [event("e1", 0, False)])
    assert report["overall_risk_level"] == "low"
    assert report["num_dismissed_events"] == 1


def test_unverified_event_still_counts_and_is_flagged(tmp_path):
    report = report_for(tmp_path, [event("e1", 0, None, risk="high"), event("e2", 5, False)])

    assert report["overall_risk_level"] == "high"  # conservative: not silently dropped
    assert report["num_unverified_events"] == 1
    assert report["num_confirmed_events"] == 0
    assert "human review" in report["overall_risk_rationale"]
    assert "(unverified)" in render_markdown(report)

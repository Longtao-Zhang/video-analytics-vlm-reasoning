import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as service

FAKE_REPORT = {
    "video_name": "x", "video_path": "/tmp/secret/x.mp4", "overall_risk_level": "low",
    "num_candidate_events": 0, "num_confirmed_events": 0,
    "num_dismissed_events": 0, "num_unverified_events": 0, "events": [],
}


@pytest.fixture
def calls(monkeypatch):
    """Replace the heavy pipeline; record the paths it was called with."""
    seen: list[dict] = []

    def fake_run_pipeline(video_path, config, output_dir, vlm_mode):
        seen.append({"video_path": Path(video_path), "output_dir": Path(output_dir), "mode": vlm_mode})
        return dict(FAKE_REPORT)

    monkeypatch.setattr(service, "run_pipeline", fake_run_pipeline)
    return seen


client = TestClient(service.app)


def upload(filename, data=b"fake video bytes", **kwargs):
    return client.post("/analyze", files={"video": (filename, io.BytesIO(data), "video/mp4")}, **kwargs)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("filename", ["../../etc/evil.mp4", "/etc/evil.mp4", "..\\..\\evil.mp4"])
def test_client_filename_is_never_used_as_a_path(calls, filename):
    resp = upload(filename)

    assert resp.status_code == 200, resp.text
    saved = calls[0]["video_path"]
    assert saved.parent == calls[0]["output_dir"]  # stays inside the temp work dir
    assert ".." not in saved.name and "evil" not in saved.name
    assert resp.json()["video_name"] == "evil.mp4"
    assert "video_path" not in resp.json()["report"]  # no server path leaked


def test_unsupported_suffix_rejected(calls):
    assert upload("notes.txt").status_code == 400
    assert calls == []


def test_oversized_upload_rejected(calls, monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 10)
    assert upload("clip.mp4", data=b"x" * 11).status_code == 413
    assert calls == []


def test_api_mode_disabled_without_server_token(calls, monkeypatch):
    monkeypatch.delenv(service.API_TOKEN_ENV, raising=False)
    assert upload("clip.mp4", params={"vlm_mode": "api"}).status_code == 403
    assert calls == []


def test_api_mode_requires_matching_bearer_token(calls, monkeypatch):
    monkeypatch.setenv(service.API_TOKEN_ENV, "s3cret")
    assert upload("clip.mp4", params={"vlm_mode": "api"}).status_code == 401
    bad = upload("clip.mp4", params={"vlm_mode": "api"}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    good = upload("clip.mp4", params={"vlm_mode": "api"}, headers={"Authorization": "Bearer s3cret"})
    assert good.status_code == 200
    assert calls[-1]["mode"] == "api"


def test_mock_mode_stays_open(calls, monkeypatch):
    monkeypatch.delenv(service.API_TOKEN_ENV, raising=False)
    assert upload("clip.mp4").status_code == 200

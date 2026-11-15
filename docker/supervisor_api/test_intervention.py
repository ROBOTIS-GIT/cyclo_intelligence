import io
import json
import urllib.error

import pytest
from fastapi import HTTPException

from supervisor_api import intervention


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_state_is_read_from_loopback_controller(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response({"active": False})

    monkeypatch.setattr(intervention.urllib.request, "urlopen", fake_urlopen)

    assert intervention.intervention_state() == {"active": False}
    assert seen == {
        "url": "http://127.0.0.1:8779/api/intervention/state",
        "timeout": 75,
    }


def test_start_forwards_only_supported_leader_model(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _Response({"ok": True})

    monkeypatch.setattr(intervention.urllib.request, "urlopen", fake_urlopen)

    request = intervention.InterventionActionRequest(model="mini")
    assert intervention.intervention_action("start", request) == {"ok": True}
    assert seen["url"].endswith("/api/intervention/start")
    assert seen["body"]["model"] == "mini"


def test_fault_reset_is_forwarded_to_controller(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _Response({"ok": True})

    monkeypatch.setattr(intervention.urllib.request, "urlopen", fake_urlopen)

    request = intervention.InterventionActionRequest(model="a3")
    assert intervention.intervention_action("reset_fault", request) == {"ok": True}
    assert seen["url"].endswith("/api/intervention/reset_fault")
    assert seen["body"]["model"] == "a3"


def test_operator_pose_lists_are_forwarded_to_controller(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _Response({"ok": True})

    monkeypatch.setattr(intervention.urllib.request, "urlopen", fake_urlopen)
    request = intervention.PoseMoveRequest(
        model="f2",
        left_arm=[0.0] * 8,
        right_arm=[0.0] * 8,
        head=[0.0] * 2,
        lift=[-0.2],
        duration=8.0,
    )

    assert intervention.move_to_operator_pose(request) == {"ok": True}
    assert seen["url"].endswith("/api/intervention/pose")
    assert seen["body"] == {
        "model": "f2",
        "left_arm": [0.0] * 8,
        "right_arm": [0.0] * 8,
        "head": [0.0] * 2,
        "lift": [-0.2],
        "duration": 8.0,
    }


def test_controller_http_error_preserves_status_and_detail(monkeypatch):
    def fake_urlopen(_request, timeout):
        del timeout
        raise urllib.error.HTTPError(
            "http://controller/start",
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps({"detail": "Inference must be paused"}).encode()),
        )

    monkeypatch.setattr(intervention.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(HTTPException) as caught:
        intervention.intervention_action(
            "start",
            intervention.InterventionActionRequest(model="a3"),
        )

    assert caught.value.status_code == 409
    assert caught.value.detail == "Inference must be paused"


@pytest.mark.parametrize("model", ["f2", "sg2"])
def test_pose_requires_an_explicit_supported_follower_model(model):
    from pydantic import ValidationError
    targets = dict(left_arm=[0.0] * 8, right_arm=[0.0] * 8,
                   head=[0.0] * 2, lift=[0.0])
    assert intervention.PoseMoveRequest(model=model, **targets).model == model
    for invalid in ({}, {"model": "sh5"}, {"model": "ffw_sg2_rev1"}):
        with pytest.raises(ValidationError):
            intervention.PoseMoveRequest(**targets, **invalid)

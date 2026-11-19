"""Event status is authorized by bounded cache confirmation, not publication age."""

import pytest
from types import SimpleNamespace as Message
from unittest import mock

from test_initial_pose_sync import robot_client_impl as impl


def status(state=1):
    return Message(requested_control_mode=1, active_control_mode=1, state=state, transition_id=0)


@pytest.fixture
def guard():
    with mock.patch.object(impl.time, "monotonic", return_value=10.) as clock:
        value = impl.HandoffGuard({}, {"joints": {"a"}})
        value.update_source(Message(data="model_action"))
        value.update_joints("joints", Message(name=["a"], position=[0.]))
        yield value, clock


def confirm(guard, msg=None):
    return guard.confirm_status(guard.confirmation_token(), msg or status())


def test_live_ready_cannot_authorize_or_refresh_confirmation(guard):
    g, clock = guard
    g.update_status(status())
    assert g.snapshot().state == "yielded"
    assert confirm(g)
    before = g.snapshot()
    for step in range(1, 20):
        clock.return_value = 10. + step * .1
        g.update_source(Message(data="model_action"))
        g.update_joints("joints", Message(name=["a"], position=[0.]))
        assert confirm(g)
        assert g.snapshot() == before
    for _ in range(6):
        clock.return_value += .1
        g.update_source(Message(data="model_action"))
        g.update_joints("joints", Message(name=["a"], position=[0.]))
        g.update_status(status())
    assert g.snapshot().reason == "status_confirmation_stale"
    with pytest.raises(impl.OutputAuthorityError):
        g.arm(g.snapshot().revision)


def test_activation_event_before_source_and_normal_switch_has_no_fault(guard):
    g, _ = guard
    assert confirm(g)
    before = g.snapshot()
    g.update_source(Message(data="switching"))
    g.update_status(status())
    g.update_source(Message(data="model_action"))
    assert g.snapshot().state == "yielded"
    assert confirm(g)
    assert g.snapshot().state == "ready"
    assert g.snapshot().fault_revision == before.fault_revision
    assert g.snapshot().revision > before.revision


def test_query_token_rejects_new_error_and_epoch_roundtrip(guard):
    g, _ = guard
    assert confirm(g)
    token = g.confirmation_token()
    g.update_status(status(5))
    assert not g.confirm_status(token, status())
    assert g.snapshot().reason == "controller_error"
    token = g.confirmation_token()
    g.update_source(Message(data="teleop"))
    g.update_source(Message(data="model_action"))
    assert not g.confirm_status(token, status())


def test_unpolled_source_gap_invalidates_inflight_confirmation_and_lease(guard):
    g, clock = guard
    assert confirm(g)
    g.arm(g.snapshot().revision)
    token = g.confirmation_token()
    clock.return_value += .6
    g.update_source(Message(data="model_action"))
    g.update_joints("joints", Message(name=["a"], position=[0.]))
    assert not g.confirm_status(token, status())
    assert g.snapshot().state != "ready"
    assert confirm(g)
    with pytest.raises(impl.OutputAuthorityError, match="ownership_changed"):
        g.check()


def test_query_failure_revokes_and_requires_manual_rearm(guard):
    g, _ = guard
    assert confirm(g)
    before = g.arm(g.snapshot().revision)
    g.invalidate_confirmation("status_query_failed")
    assert g.snapshot().state == "blocked"
    assert g.snapshot().fault_revision > before.fault_revision
    assert confirm(g)
    with pytest.raises(impl.OutputAuthorityError):
        g.check()


def test_changing_live_ready_events_cannot_postpone_confirmation_expiry(guard):
    g, clock = guard
    assert confirm(g)
    for step in range(1, 7):
        clock.return_value = 10. + step * .1
        g.update_source(Message(data="model_action"))
        g.update_joints("joints", Message(name=["a"], position=[0.]))
        msg = status()
        msg.transition_id = step
        g.update_status(msg)
    assert g.snapshot().reason == "status_confirmation_stale"

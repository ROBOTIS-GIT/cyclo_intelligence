"""Transport ordering and publisher identity, without a ROS process."""

import importlib
import threading
from types import SimpleNamespace as Message
from unittest import mock

import pytest
from test_initial_pose_sync import robot_client_impl as impl
from test_handoff_confirmation import status


def make():
    module = importlib.import_module(impl.__package__ + ".handoff_transport")
    guard = impl.HandoffGuard({}, {"j": {"a"}})
    guard.update_joints("j", Message(name=["a"], position=[0.]))
    return module.CacheConfirmation(guard), guard


def topology(c, node="node", extra=False):
    endpoints = {"source": {("session", node, "10")}, "status": {("session", node, "11")}}
    if extra:
        endpoints["status"].add(("session", "other", "12"))
    c.set_publishers(endpoints)


def sample(stream, seq=1, state=1, gid=None):
    return (gid or stream.encode(), seq,
            Message(data="model_action") if stream == "source" else status(state))


def reconcile(c, callback=None, state=1):
    def query(stream, identity):
        if stream == "status" and callback:
            callback()
        return sample(stream, state=state)
    return c.reconcile(query)


def ready(c, g):
    topology(c)
    reconcile(c)
    c.live("source", "session", *sample("source", 2))
    assert reconcile(c)
    assert g.snapshot().state == "ready"


def test_latejoin_cache_binds_distinct_gids_without_faking_source_heartbeat():
    c, g = make()
    topology(c)
    assert not reconcile(c)
    assert g.snapshot().state != "ready"
    c.live("source", "session", *sample("source", 2))
    assert reconcile(c)
    assert g.snapshot().state == "ready"


@pytest.mark.parametrize("source", ["model_action", "teleop"])
def test_initial_load_waits_for_cache_and_live_source_without_arming(source):
    c, g = make()
    topology(c)
    assert not reconcile(c)
    c.live("source", "session", b"source", 2, Message(data=source))
    # This is the short interval exposed by LOAD followed immediately by START.
    assert g.snapshot().reason == "publisher_changed"
    entered, finished = threading.Event(), threading.Event()
    results = []

    def wait():
        entered.set()
        results.append(g.wait_for_initial_confirmation())
        finished.set()

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        assert entered.wait(1)
        assert not finished.is_set()
        assert c.reconcile(lambda stream, identity: (
            b"source", 2, Message(data=source)) if stream == "source" else sample(stream))
        assert finished.wait(1)
        assert results == [True]
        assert g.snapshot().state == ("ready" if source == "model_action" else "yielded")
        assert g._armed_revision is None
        with pytest.raises(impl.OutputAuthorityError):
            g.check()
    finally:
        thread.join(1)
        assert not thread.is_alive()


def test_initial_wait_is_bounded_by_config_and_does_not_authorize_timeout():
    g = impl.HandoffGuard({"status_confirmation_timeout_s": .04,
                          "status_confirmation_interval_s": .01,
                          "initial_confirmation_timeout_s": .2}, {"j": {"a"}})
    with mock.patch.object(g._initial_confirmation, "wait", return_value=False) as wait:
        assert not g.wait_for_initial_confirmation()
    wait.assert_called_once_with(.2)
    assert g._timeouts['status_confirmation'] == .04
    assert g.snapshot().reason == "source_missing"
    with pytest.raises(impl.OutputAuthorityError):
        g.arm(g.snapshot().revision)


def test_completed_initial_wait_does_not_hide_later_publisher_loss():
    c, g = make()
    ready(c, g)
    assert g.wait_for_initial_confirmation()
    c.set_publishers({"source": set(), "status": set()})
    assert g.wait_for_initial_confirmation()
    assert g.snapshot().state == "blocked"
    with pytest.raises(impl.OutputAuthorityError):
        g.arm(g.snapshot().revision)


def test_missing_cache_never_finishes_initial_confirmation():
    c, g = make()
    topology(c)
    c.live("source", "session", *sample("source", 1))
    def failed_query(*args):
        raise TimeoutError("cache unavailable")
    assert not c.reconcile(failed_query)
    assert not g._initial_confirmation.is_set()


def test_empty_discovery_allows_explicit_local_start_but_partial_publisher_revokes_it():
    c, g = make()
    c.set_publishers({"source": set(), "status": set()})
    assert g.snapshot().reason == "source_missing"
    assert g.request_model_start().reason == "local_model"
    g.arm(g.snapshot().revision)
    c.set_publishers({"source": {("session", "node", "10")}, "status": set()})
    assert g.snapshot().state == "blocked"
    c.set_publishers({"source": set(), "status": set()})
    assert g.request_model_start().state == "blocked"
    with pytest.raises(impl.OutputAuthorityError):
        g.check()


def test_controller_appearing_after_local_start_requires_fresh_external_confirmation():
    c, g = make()
    g.request_model_start()
    old = g.arm(g.snapshot().revision)
    ready(c, g)
    assert g.snapshot().reason == "model_action"
    assert g.snapshot().fault_revision > old.fault_revision
    with pytest.raises(impl.OutputAuthorityError):
        g.check()
    g.arm(g.snapshot().revision)
    g.check()


def test_live_error_wins_over_older_cache_and_query_race():
    c, g = make()
    ready(c, g)
    assert not reconcile(c, lambda: c.live("status", "session", *sample("status", 2, 5)))
    assert g.snapshot().reason == "controller_error"
    assert not reconcile(c)
    assert g.snapshot().reason == "controller_error"
    c.live("status", "session", *sample("status", 1))
    assert g.snapshot().reason == "controller_error"


@pytest.mark.parametrize("change", ["epoch", "restart", "multiple", "close"])
def test_pending_query_cannot_cross_boundaries(change):
    c, g = make()
    ready(c, g)
    def changed():
        if change == "epoch":
            c.live("source", "session", b"source", 3, Message(data="teleop"))
            c.live("source", "session", b"source", 4, Message(data="model_action"))
        elif change == "close":
            c.close()
        else:
            topology(c, node="restart" if change == "restart" else "node", extra=change == "multiple")
    assert not reconcile(c, changed)
    assert g.snapshot().state != "ready"


def test_wrong_session_and_multiple_publishers_fail_closed():
    c, g = make()
    ready(c, g)
    before = g.snapshot()
    c.live("status", "alien", *sample("status", 9, 5))
    assert g.snapshot() == before
    topology(c, extra=True)
    assert not reconcile(c)
    assert g.snapshot().state == "blocked"


def test_duplicate_source_and_cache_cannot_refresh_source_liveness():
    with mock.patch.object(impl.time, "monotonic", return_value=10.) as clock:
        c, g = make()
        ready(c, g)
        for step in range(1, 7):
            clock.return_value = 10. + step * .1
            g.update_joints("j", Message(name=["a"], position=[0.]))
            c.live("source", "session", *sample("source", 2))
            reconcile(c)
        assert g.snapshot().reason == "source_stale"


def test_different_nodes_in_same_session_cannot_authorize():
    c, g = make()
    c.set_publishers({"source": {("session", "a", "10")}, "status": {("session", "b", "11")}})
    assert not reconcile(c)
    assert g.snapshot().state == "blocked"


def test_replacement_publishers_cannot_reuse_previous_live_source():
    c, g = make()
    ready(c, g)
    before = g.snapshot()
    topology(c, node="replacement")
    def query(stream, identity):
        return (b"new_" + stream.encode(), 1,
                Message(data="teleop") if stream == "source" else status())
    assert not c.reconcile(query)
    assert g.snapshot().state != "ready"
    assert g.snapshot().fault_revision > before.fault_revision
    with pytest.raises(impl.OutputAuthorityError):
        g.arm(g.snapshot().revision)


def test_cache_source_watermark_rejects_older_live_without_heartbeat_credit():
    c, g = make()
    ready(c, g)
    before = g.snapshot()
    def query(stream, identity):
        return (b"source", 100, Message(data="teleop")) if stream == "source" else sample(stream)
    assert not c.reconcile(query)
    c.live("source", "session", b"source", 99, Message(data="model_action"))
    assert g.snapshot().state == "yielded"
    assert g.snapshot().fault_revision == before.fault_revision
    assert not c.reconcile(query)
    c.live("source", "session", b"source", 100, Message(data="teleop"))
    assert c.reconcile(query)
    assert g.snapshot().state == "yielded"
    c.live("source", "session", b"source", 101, Message(data="model_action"))
    assert c.reconcile(query)  # Older source cache cannot undo newer live source.
    assert g.snapshot().state == "ready"
    assert g.snapshot().fault_revision == before.fault_revision


def test_unchanged_cache_source_ahead_preserves_ready_revision():
    c, g = make()
    ready(c, g)
    before = g.snapshot()
    def query(stream, identity):
        return sample(stream, 100 if stream == "source" else 1)
    assert c.reconcile(query)
    assert g.snapshot() == before
    c.live("source", "session", b"source", 99, Message(data="teleop"))
    assert g.snapshot() == before
    c.live("source", "session", *sample("source", 100))
    assert g.snapshot() == before


def test_cache_gid_must_match_live_and_persisted_binding():
    c, g = make()
    ready(c, g)
    def query(stream, identity):
        return sample(stream, gid=b"other" if stream == "status" else None)
    assert not c.reconcile(query)
    assert g.snapshot().reason == "status_cache_replay"


def test_failed_query_blocks_without_close():
    c, g = make()
    ready(c, g)
    def query(*args):
        raise TimeoutError()
    assert not c.reconcile(query)
    assert g.snapshot().reason == "status_query_failed"


def test_query_failure_revokes_and_close_does_not_wait_for_query_lock():
    c, g = make()
    ready(c, g)
    entered, release = threading.Event(), threading.Event()
    def query(*args):
        entered.set()
        assert release.wait(2)
        raise TimeoutError()
    thread = threading.Thread(target=lambda: c.reconcile(query))
    thread.start()
    try:
        assert entered.wait(1)
        c.close()
        assert g.snapshot().state == "blocked"
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()


def test_periodic_same_cache_is_not_ownership_change():
    c, g = make()
    ready(c, g)
    before = g.snapshot()
    for _ in range(5):
        assert reconcile(c)
    assert g.snapshot() == before


@pytest.mark.parametrize("replies", [[], ["ok", "ok"], ["wrong_session"], ["wrong_key"], ["error"]])
def test_cache_query_requires_one_reply_with_matching_provenance(replies):
    module = importlib.import_module(impl.__package__ + ".handoff_transport")
    transport = module.HandoffTransport.__new__(module.HandoffTransport)
    transport._stop = threading.Event()
    transport._query_timeout = .1
    transport._subscribers = {"status": Message(keyexpr="173/control/status/type/hash")}
    transport._decode = lambda sub, sample: (b"gid", 1, status())
    values = [Message(ok=None if value == "error" else Message(
        key_expr="other" if value == "wrong_key" else "173/control/status/type/hash"),
        replier_id=Message(zid="other" if value == "wrong_session" else "session")) for value in replies]
    transport._session = mock.Mock()
    transport._session.get.return_value = values
    with pytest.raises(ValueError):
        transport._query("status", ("session", "node"))
    transport._session.get.assert_called_once_with(
        "173/control/status/type/hash/@adv/pub/session/**?_anyke;_max=1", timeout=.1)


def test_transport_close_is_idempotent_and_joins_confirmation_worker():
    module = importlib.import_module(impl.__package__ + ".handoff_transport")
    transport = module.HandoffTransport.__new__(module.HandoffTransport)
    transport.coordinator, guard = make()
    transport._stop = threading.Event()
    transport._query_timeout = .1
    transport._watch = mock.Mock()
    subscriber = mock.Mock()
    transport._subscribers = {"source": subscriber}
    transport._thread = threading.Thread(target=transport._stop.wait)
    transport._thread.start()
    transport.close()
    transport.close()
    assert not transport._thread.is_alive()
    assert guard.snapshot().state == "blocked"
    transport._watch.undeclare.assert_called_once()
    subscriber.close.assert_called_once()

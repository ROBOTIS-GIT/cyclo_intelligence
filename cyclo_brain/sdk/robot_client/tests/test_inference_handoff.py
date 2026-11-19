"""Output authority is independent of observation/model work and fails closed."""

import threading
from types import SimpleNamespace as Message
from unittest import mock

import numpy as np
import pytest

import test_initial_pose_sync as sync_tests
from test_initial_pose_sync import RobotClient, robot_client_impl as impl


@pytest.fixture
def clock():
    with mock.patch.object(impl.time, "monotonic", return_value=10.) as value:
        yield value


@pytest.fixture
def client(clock):
    helper = sync_tests.InitialPoseSyncCommandTest()
    bot = helper._make_client("ffw_sg2_rev1")
    helper._seed_joint_state(bot)
    bot._handoff = impl.HandoffGuard({}, {"joints": {"a", "b"}})
    bot._handoff.update_source(Message(data="model_action"))
    bot._handoff.update_joints("joints", joints())
    confirmed_status(bot._handoff, status())
    return bot


def status(mode=1, state=1, requested=None):
    return Message(active_control_mode=mode, requested_control_mode=mode if requested is None else requested,
                   state=state, transition_id=0)


def confirmed_status(guard, msg):
    guard.update_status(msg)
    guard.confirm_status(guard.confirmation_token(), msg)


def joints(names=("a", "b"), values=(0., 0.)):
    return Message(name=names, position=values)


def arm(bot):
    return bot.arm_handoff(bot.handoff_status().revision)


def action(bot):
    return np.zeros(sync_tests.InitialPoseSyncCommandTest._action_dimension(bot, bot._action_keys))


def count(bot):
    return sum(len(pub.messages) for pub in bot._command_publishers.values())


@pytest.mark.parametrize("robot_type", ["ffw_sg2_rev1", "ffw_bg2_rev4", "w1", "w2"])
def test_robot_profile_uses_common_handoff_timing_defaults(robot_type):
    config = impl.robot_schema.load_robot_section(robot_type)["inference_handoff"]
    assert config == {"source_topic": "/source", "control_status_topic": "/control/status",
                      "allowed_model_modes": [1]}
    guard = impl.HandoffGuard(config, {"joints": {"a", "b"}})
    assert guard.source_topic == "/source"
    assert guard.control_status_topic == "/control/status"
    assert guard._allowed_modes == frozenset({1})
    assert guard._timeouts == {"source": .5, "status_confirmation": .5, "joint_state": .5}
    assert guard._initial_confirmation_timeout == 2.0
    assert guard.confirmation_interval == .1


def test_snapshot_immutable_and_heartbeats_do_not_change_boundary(client, clock):
    before = client.handoff_status()
    assert before.enabled and before.state == "ready" and before.ready_since == 10.
    with pytest.raises((AttributeError, TypeError)):
        before.state = "blocked"
    clock.return_value = 10.1
    client._handoff.update_source(Message(data="model_action"))
    client._handoff.update_joints("joints", joints())
    confirmed_status(client._handoff, status())
    assert client.handoff_status() == before


def test_requires_explicit_arm_and_rejects_stale_arm(client):
    with pytest.raises(impl.OutputAuthorityError, match="unarmed"):
        client.publish_action_with_receipt(action(client))
    initial = arm(client)
    client._handoff.update_source(Message(data="teleop"))
    client._handoff.update_source(Message(data="model_action"))
    confirmed_status(client._handoff, status())
    after = client.handoff_status()
    assert after.state == "ready" and after.revision > initial.revision
    assert after.fault_revision == initial.fault_revision
    with pytest.raises(impl.OutputAuthorityError, match="ownership_changed"):
        client.publish_action_with_receipt(action(client))
    with pytest.raises(impl.OutputAuthorityError):
        client.arm_handoff(initial.revision)
    assert count(client) == 0
    arm(client)
    np.testing.assert_array_equal(client.publish_action_with_receipt(action(client)), action(client))
    assert count(client) == len(client._action_keys)


@pytest.mark.parametrize("state", [1, 4])
def test_holding_and_active_are_ready(client, state):
    confirmed_status(client._handoff, status(state=state))
    assert client.handoff_status().state == "ready"


@pytest.mark.parametrize("state,mode", [(1, 0), (2, 0), (3, 0), (2, 2), (3, 2)])
def test_normal_switch_order_is_yielded_without_fault(client, state, mode):
    before = arm(client)
    confirmed_status(client._handoff, status(mode=mode, state=state, requested=1))
    assert client.handoff_status().state == "yielded"
    assert client.handoff_status().fault_revision == before.fault_revision
    client._handoff.update_source(Message(data="switching"))
    confirmed_status(client._handoff, status())
    client._handoff.update_source(Message(data="model_action"))
    assert client.handoff_status().state == "yielded"  # Wait for current-epoch cache confirmation.
    confirmed_status(client._handoff, status())
    assert client.handoff_status().state == "ready"
    assert client.handoff_status().fault_revision == before.fault_revision


@pytest.mark.parametrize("source", ["teleop", "switching"])
def test_yielded_source_is_not_fault(client, source):
    before = client.handoff_status()
    client._handoff.update_source(Message(data=source))
    confirmed_status(client._handoff, status(mode=3))
    result = client.handoff_status()
    assert result.state == "yielded" and result.fault_revision == before.fault_revision


@pytest.mark.parametrize("fault", ["none", "unknown", "starting", "error", "unsupported", "invalid_state"])
def test_fault_is_sticky_after_recovery(client, fault):
    before = client.handoff_status()
    if fault in ("none", "unknown", "starting"):
        client._handoff.update_source(Message(data=fault))
    else:
        confirmed_status(client._handoff, status(mode=2 if fault == "unsupported" else 1,
                                           state=5 if fault == "error" else 99 if fault == "invalid_state" else 1))
    failed = client.handoff_status()
    assert failed.state == "blocked" and failed.fault_revision > before.fault_revision
    assert client.handoff_status() == failed
    client._handoff.update_source(Message(data="model_action"))
    confirmed_status(client._handoff, status())
    assert client.handoff_status().state == "ready"
    assert client.handoff_status().fault_revision == failed.fault_revision


@pytest.mark.parametrize("stream", ["source", "status_confirmation", "joint_state"])
def test_each_stream_expires_and_repeated_poll_does_not_count_fault_again(client, clock, stream):
    before = arm(client)
    clock.return_value = 10.4
    if stream != "source":
        client._handoff.update_source(Message(data="model_action"))
    if stream != "status_confirmation":
        confirmed_status(client._handoff, status())
    if stream != "joint_state":
        client._handoff.update_joints("joints", joints())
    clock.return_value = 10.51
    failed = client.handoff_status()
    assert failed.state == "blocked" and stream in failed.reason
    assert failed.fault_revision > before.fault_revision
    assert client.handoff_status() == failed


def test_unpolled_stale_gap_is_not_erased_by_new_heartbeats(client, clock):
    before = arm(client)
    clock.return_value = 11.
    client._handoff.update_source(Message(data="model_action"))
    client._handoff.update_joints("joints", joints())
    confirmed_status(client._handoff, status())
    result = client.handoff_status()
    assert result.state == "ready" and result.ready_since == 11.
    assert result.fault_revision > before.fault_revision
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_idle_action()


@pytest.mark.parametrize("names,values", [((), ()), (("a",), (1.,)),
    (("a", "b"), (1.,)), (("a", "a", "b"), (1., 2., 3.)),
    (("a", "b"), (float("nan"), 1.)), (("a", "b"), (1., float("inf")))])
def test_bad_joint_feedback_revokes_authority(client, names, values):
    arm(client)
    client._handoff.update_joints("joints", joints(names, values))
    assert client.handoff_status().state == "blocked"
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_action(action(client))
    assert count(client) == 0


@pytest.mark.parametrize("operation", ["action", "receipt", "zero_receipt", "idle", "sync", "hold", "named"])
def test_all_physical_paths_reject_revoked_lease_even_after_ready_roundtrip(client, operation):
    arm(client)
    client._handoff.update_source(Message(data="teleop"))
    client._handoff.update_source(Message(data="model_action"))
    confirmed_status(client._handoff, status())
    calls = {
        "action": lambda: client.publish_action(action(client)),
        "receipt": lambda: client.publish_action_with_receipt(action(client)),
        "zero_receipt": lambda: client.publish_action_with_receipt(action(client), zero_twist=True),
        "idle": client.publish_idle_action,
        "sync": lambda: client.publish_initial_pose_sync(action(client)),
        "hold": client.publish_current_pose_hold,
        "named": lambda: client.publish_named_pose(dict(client._joint_positions_by_name)),
    }
    with pytest.raises(impl.OutputAuthorityError):
        calls[operation]()
    assert count(client) == 0


@pytest.mark.parametrize("operation", ["receipt", "sync", "hold", "named"])
def test_per_send_gate_aborts_mid_operation_and_never_sends_recovery(client, operation):
    arm(client)
    first = []
    def send(**kwargs):
        first.append(kwargs)
        client._handoff.update_source(Message(data="teleop"))
        client._handoff.update_source(Message(data="model_action"))
        confirmed_status(client._handoff, status())
    for pub in client._command_publishers.values():
        pub.publish = send
    calls = {
        "receipt": lambda: client.publish_action_with_receipt(action(client)),
        "sync": lambda: client.publish_initial_pose_sync(action(client)),
        "hold": client.publish_current_pose_hold,
        "named": lambda: client.publish_named_pose(dict(client._joint_positions_by_name)),
    }
    with pytest.raises(impl.OutputAuthorityError):
        calls[operation]()
    assert len(first) == 1


def test_unarmed_explicit_pose_and_hold_acquire_only_operation_lease(client):
    client.publish_named_pose(dict(client._joint_positions_by_name))
    client.publish_current_pose_hold()
    assert count(client) > 0
    with pytest.raises(impl.OutputAuthorityError, match="unarmed"):
        client.publish_action(action(client))


@pytest.mark.parametrize("key,width", [("mobile", 3), ("arm_left", 8)])
def test_final_gate_catches_roundtrip_during_message_construction(client, key, width):
    arm(client)
    original = impl.get_message_class
    changed = False
    def message_class(name):
        nonlocal changed
        if not changed:
            changed = True
            client._handoff.update_source(Message(data="teleop"))
            client._handoff.update_source(Message(data="model_action"))
            confirmed_status(client._handoff, status())
        return original(name)
    with mock.patch.object(impl, "get_message_class", side_effect=message_class):
        with pytest.raises(impl.OutputAuthorityError, match="ownership_changed"):
            client.publish_action_with_receipt(np.zeros(width), [key])
    assert changed and count(client) == 0


def test_transport_failure_after_handoff_does_not_attempt_recovery_output(client):
    arm(client)
    attempts = []
    def fail(**kwargs):
        attempts.append(kwargs)
        client._handoff.update_source(Message(data="teleop"))
        client._handoff.update_source(Message(data="model_action"))
        confirmed_status(client._handoff, status())
        raise RuntimeError("transport failed")
    for pub in client._command_publishers.values():
        pub.publish = fail
    with pytest.raises(RuntimeError, match="transport failed"):
        client.publish_initial_pose_sync(action(client))
    assert len(attempts) == 1


def test_last_topic_handoff_prevents_successful_receipt(client):
    arm(client)
    sent = []
    def send(**kwargs):
        sent.append(kwargs)
        client._handoff.update_source(Message(data="teleop"))
    client._command_publishers["leader_mobile"].publish = send
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_action_with_receipt(np.zeros(3), ["mobile"])
    assert len(sent) == 1


def test_mode_change_even_between_allowed_modes_revokes_lease(client, clock):
    guard = client._handoff = impl.HandoffGuard({"allowed_model_modes": [1, 2]}, {"joints": {"a", "b"}})
    guard.update_source(Message(data="model_action"))
    guard.update_joints("joints", joints())
    confirmed_status(guard, status())
    before = arm(client)
    clock.return_value = 10.1
    confirmed_status(guard, status(mode=2))
    after = client.handoff_status()
    assert after.state == "ready" and after.revision > before.revision and after.ready_since == 10.1
    assert after.fault_revision == before.fault_revision
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_action(action(client))


def test_all_feedback_topics_required_and_valid_topic_cannot_refresh_another(clock):
    guard = impl.HandoffGuard({}, {"left": {"a"}, "right": {"b"}})
    guard.update_source(Message(data="model_action"))
    confirmed_status(guard, status())
    guard.update_joints("left", joints(["a"], [0.]))
    assert guard.snapshot().state == "blocked"
    guard.update_joints("right", joints(["b"], [0.]))
    confirmed_status(guard, status())
    assert guard.snapshot().state == "ready"
    clock.return_value = 10.4
    guard.update_source(Message(data="model_action"))
    confirmed_status(guard, status())
    guard.update_joints("left", joints(["a"], [0.]))
    clock.return_value = 10.51
    assert guard.snapshot().reason == "joint_state_stale:right"


def test_raw_status_definition_roundtrip_drives_guard(client):
    typesys = pytest.importorskip("rosbags.typesys")
    name = "robotis_interfaces/msg/ControlModeStatus"
    store = typesys.get_typestore(typesys.Stores.EMPTY)
    store.register(typesys.get_types_from_msg(impl.CONTROL_MODE_STATUS_DEF, name))
    values = {line.split()[1]: "none" if line.split()[0] == "string" else 0
              for line in impl.CONTROL_MODE_STATUS_DEF.splitlines() if line.strip() and "=" not in line}
    values.update(state=1, active_control_mode=1, requested_control_mode=1)
    msg = store.types[name](**values)
    decoded = store.deserialize_cdr(store.serialize_cdr(msg, name), name)
    confirmed_status(client._handoff, decoded)
    assert client.handoff_status().state == "ready"


def test_preview_remains_nonphysical_when_authority_missing(client):
    client._preview_publisher = sync_tests.FakePublisher()
    client._handoff.update_source(Message(data="teleop"))
    client.publish_action_preview(action(client))
    assert len(client._preview_publisher.messages) == 1
    assert count(client) == 0


def test_preview_flag_cannot_bypass_a_physical_publisher(client):
    client._preview_publisher = sync_tests.FakePublisher()
    with pytest.raises(impl.OutputAuthorityError):
        client._publish_joint_trajectory(client._command_publishers["leader_arm_left"],
                                         ["a"], np.zeros(1), preview=True)
    assert count(client) == 0


def test_callbacks_do_not_wait_for_observation_lock_or_transport(client):
    arm(client)
    entered, release, updated = threading.Event(), threading.Event(), threading.Event()
    failures = []
    def send(**kwargs):
        entered.set()
        assert release.wait(2)
    for pub in client._command_publishers.values():
        pub.publish = send
    def output():
        try:
            client.publish_action_with_receipt(action(client))
        except impl.OutputAuthorityError:
            failures.append(True)
    def callback():
        client._handoff.update_source(Message(data="teleop"))
        confirmed_status(client._handoff, status())
        client._handoff.update_joints("joints", joints())
        updated.set()
    worker = threading.Thread(target=output)
    callback_worker = threading.Thread(target=callback)
    with client._lock:
        worker.start()
        try:
            assert entered.wait(1)
            callback_worker.start()
            assert updated.wait(1)
        finally:
            release.set()
            worker.join(2)
            if callback_worker.ident:
                callback_worker.join(2)
    assert failures == [True]


@pytest.mark.parametrize("robot_type", ["ffw_sg2_rev1", "ffw_bg2_rev4", "w1", "w2"])
def test_configured_subscriptions_exist_without_observation_subscriptions(clock, robot_type):
    section = impl.robot_schema.load_robot_section(robot_type)
    section["inference_handoff"] = {"source_topic": "/custom_source", "control_status_topic": "/custom_status"}
    with mock.patch.object(impl.robot_schema, "load_robot_section", return_value=section), \
            mock.patch.object(impl, "ROS2Publisher"), \
            mock.patch.object(impl, "HandoffTransport") as transport, \
            mock.patch.object(impl, "ROS2Subscriber") as subscribe:
        bot = RobotClient("test", subscribe_images=False, subscribe_state=False, subscribe_sensors=False,
                          domain_id=7, router_ip="router", router_port=8123, enable_command_publishers=True)
        kwargs = [call.kwargs for call in subscribe.call_args_list]
        feedback = [item for item in kwargs if item["msg_type"] == "sensor_msgs/msg/JointState"]
        assert transport.call_args.args == (bot._handoff, impl.CONTROL_MODE_STATUS_DEF)
        assert bot._handoff.source_topic == "/custom_source"
        assert bot._handoff.control_status_topic == "/custom_status"
        assert transport.call_args.kwargs == dict(domain_id=7, router_ip="router", router_port=8123)
        assert feedback
        for item in kwargs:
            assert item["domain_id"] == 7 and item["router_ip"] == "router" and item["router_port"] == 8123
        bot._handoff.update_source(Message(data="model_action"))
        for item in feedback:
            names = sorted({n for group in bot._config["joint_groups"].values()
                            if group.get("topic") == item["topic"] for n in group["joint_names"]})
            item["callback"](joints(names, [0.] * len(names)))
        confirmed_status(bot._handoff, status())
        assert bot.handoff_status().state == "ready"
        revision = arm(bot).revision
        bot._handoff.update_source(Message(data="teleop"))
        assert bot.handoff_status().state == "yielded"
        with pytest.raises(impl.OutputAuthorityError):
            bot.publish_action_with_receipt(action(bot))
        impl.ROS2Publisher.return_value.publish.assert_not_called()
        bot._handoff.update_source(Message(data="model_action"))
        confirmed_status(bot._handoff, status())
        assert bot.handoff_status().state == "ready"
        assert bot.handoff_status().revision > revision
        with pytest.raises(impl.OutputAuthorityError):
            bot.publish_action_with_receipt(action(bot))
        bot.close()
        transport.return_value.close.assert_called_once()


def test_readonly_worker_keeps_deferred_selected_subscriptions_only():
    section = impl.robot_schema.load_robot_section("ffw_sg2_rev1")
    section["inference_handoff"] = {}
    config = {"cameras": {}, "sensors": {}, "joint_groups": {
        name: {"topic": f"/{name}", "msg_type": "sensor_msgs/msg/JointState", "joint_names": [name]}
        for name in ("left", "right")
    }}
    with mock.patch.object(impl.robot_schema, "load_robot_section", return_value=section), \
            mock.patch.object(impl, "_build_runtime_config", return_value=config), \
            mock.patch.object(impl, "ROS2Subscriber") as subscribe:
        bot = RobotClient("worker", subscribe_images=False, subscribe_sensors=False, defer_subscriptions=True)
        assert not bot.handoff_status().enabled
        subscribe.assert_not_called()
        bot.start_observation_subscriptions(camera_names=[], joint_groups=["left"], sensor_names=[])
        assert [call.kwargs["topic"] for call in subscribe.call_args_list] == ["/left"]
        bot.close()


def test_unconfigured_client_is_unchanged():
    helper = sync_tests.InitialPoseSyncCommandTest()
    bot = helper._make_client("ffw_sg2_rev1")
    assert not bot.handoff_status().enabled
    assert bot.handoff_status().state == "ready"
    bot.arm_handoff(0)
    bot.publish_action_with_receipt(action(bot))
    assert count(bot) > 0


def test_explicit_local_start_needs_feedback_and_arm_but_no_source(client, clock):
    guard = client._handoff = impl.HandoffGuard({}, {"joints": {"a", "b"}})
    assert guard.snapshot().reason == "source_missing"
    assert client.request_model_start().reason == "joint_state_missing:joints"
    guard.update_joints("joints", joints())
    assert guard.snapshot().reason == "local_model"
    assert guard._source is None and guard._status is None
    with pytest.raises(impl.OutputAuthorityError, match="unarmed"):
        client.publish_action_with_receipt(action(client))
    before = arm(client)
    for step in range(1, 31):
        clock.return_value = 10. + step * .1
        guard.update_joints("joints", joints())
        assert guard.snapshot() == before
        client.publish_action_with_receipt(action(client))
    assert count(client) == 30 * len(client._action_keys)


def test_local_feedback_loss_requires_new_start_even_after_feedback_recovers(client, clock):
    guard = client._handoff = impl.HandoffGuard({}, {"joints": {"a", "b"}})
    guard.update_joints("joints", joints())
    client.request_model_start()
    before = arm(client)
    clock.return_value += 1.
    guard.update_joints("joints", joints())
    assert guard.snapshot().reason == "feedback_gap"
    assert guard.snapshot().fault_revision > before.fault_revision
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_action_with_receipt(action(client))
    assert client.request_model_start().reason == "local_model"
    with pytest.raises(impl.OutputAuthorityError, match="ownership_changed"):
        client.publish_action_with_receipt(action(client))
    arm(client)
    client.publish_action_with_receipt(action(client))


@pytest.mark.parametrize("signal", ["source", "cached_source", "status", "discovery"])
def test_external_control_revokes_local_mode_and_start_cannot_override_it(client, signal):
    guard = client._handoff = impl.HandoffGuard({}, {"joints": {"a", "b"}})
    guard.update_joints("joints", joints())
    client.request_model_start()
    arm(client)
    calls = {
        "source": lambda: guard.update_source(Message(data="teleop")),
        "cached_source": lambda: guard.observe_cached_source(Message(data="teleop")),
        "status": lambda: guard.update_status(status(state=5)),
        "discovery": guard.observe_external_control,
    }
    calls[signal]()
    assert client.request_model_start().state != "ready"
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_current_pose_hold()
    guard.invalidate_confirmation("publisher_unavailable", reset_source=True)
    assert client.request_model_start().reason == "source_missing"
    assert count(client) == 0


@pytest.mark.parametrize("reason", ["status_query_failed", "status_metadata_invalid", "handoff_closed"])
def test_local_start_cannot_override_transport_failure_or_close(client, reason):
    guard = client._handoff = impl.HandoffGuard({}, {"joints": {"a", "b"}})
    guard.update_joints("joints", joints())
    client.request_model_start()
    arm(client)
    guard.invalidate_confirmation(reason)
    assert client.request_model_start().state == "blocked"
    with pytest.raises(impl.OutputAuthorityError):
        client.publish_action_with_receipt(action(client))
    assert count(client) == 0


@pytest.mark.parametrize("config", [{"source_timeout_s": 0}, {"joint_state_timeout_s": float("nan")},
    {"status_confirmation_timeout_s": -1}, {"allowed_model_modes": []}, {"allowed_model_modes": [0]},
    {"allowed_model_modes": [1.5]}, {"source_topic": ""}, {"control_status_topic": None}])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        impl.HandoffGuard(config, {"joints": {"a"}})

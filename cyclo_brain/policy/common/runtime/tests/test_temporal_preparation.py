"""Opt-in history warmup is not action latency or permission to ignore Stop."""

import threading
from unittest import mock

import pytest

from .test_control_loop_feedback import rig, fill, flush
from .test_control_loop import control_loop_module
from engine_process.protocol import CMD_GET_ACTION
from engine_process.protocol import CMD_LOAD_POLICY, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import ExecutionContract, LoadedExecution
from inference_context.execution import ExecutionContext
from inference_context.timing import action_latency, encode_observation_wait


@pytest.fixture
def temporal_rig(rig):
    loop, robot, engine, requests = rig
    loop._requester._get_action_timeout_s = 7.
    contract = ExecutionContract(observation_warmup_timeout_s=3.)
    with mock.patch.object(control_loop_module, "RobotClient", return_value=robot):
        loop.configure("test", publish_to_robot=True, execution_context=ExecutionContext("test-session", 0, 0, "ready"),
                       execution_contract=contract)
    yield loop, robot, engine


def test_warmup_contract_requires_context_and_preserves_non_temporal_defaults():
    contract = ExecutionContract(observation_warmup_timeout_s=3.)
    with pytest.raises(ValueError, match="execution session"):
        LoadedExecution(contract)
    loaded = LoadedExecution(contract, ExecutionContext("s", 0, 0, "ready"))
    assert LoadedExecution.from_json(loaded.to_json()) == loaded
    assert LoadedExecution.from_json("").contract.observation_warmup_timeout_s is None


def test_worker_allocates_context_from_temporal_contract_without_model_name_or_extra_flag():
    contract = ExecutionContract(observation_warmup_timeout_s=3.)
    engine = mock.Mock()
    engine.load_policy.return_value = {"success": True, "execution_contract": contract}
    response = EngineWorker(engine).handle(EngineCommandRequest(command=CMD_LOAD_POLICY, seq_id=1))
    assert response.success, response.message
    loaded = LoadedExecution.from_json(response.capabilities_json)
    assert loaded.contract == contract and loaded.context is not None
    engine.update_execution_context.assert_called_once_with(loaded.context)


@pytest.mark.parametrize("value", [True, "3", 0., -1., float("nan"), float("inf"), 121.])
def test_warmup_contract_rejects_unbounded_or_invalid_values(value):
    with pytest.raises(ValueError, match="warmup timeout"):
        ExecutionContract(observation_warmup_timeout_s=value)


def test_temporal_chunk_prepares_per_generation_without_dropping_wait_time_from_chunk(temporal_rig):
    loop, _, engine = temporal_rig
    clock = [100.]
    original = engine.get_action_chunk
    budgets = []
    call = loop._requester._client.call

    def predict(request):
        result = original(request)
        clock[0] += 2.25
        return {**result, "observation_wait_s": 2.}

    def transport(request, timeout_s):
        if request.command == CMD_GET_ACTION:
            budgets.append(timeout_s)
        return call(request, timeout_s)

    engine.get_action_chunk = predict
    with mock.patch.object(control_loop_module.time, "monotonic", side_effect=lambda: clock[0]), \
            mock.patch.object(loop._requester._client, "call", side_effect=transport):
        loop.start()
        assert loop.preparing()
        loop._latency_warmup_remaining = 0
        with mock.patch.object(loop._processor, "enqueue", wraps=loop._processor.enqueue) as enqueue:
            loop._request_and_buffer("move", loop._generation, "async")
        assert enqueue.call_args.args[2] == pytest.approx(.25)
        assert loop._request_latency_ema_s == pytest.approx(.25)
        assert not loop.preparing()
        fill(loop)
        assert loop.pause()
        flush(loop)
        loop.start()
        assert loop.preparing()
        fill(loop)
    assert budgets == [10., 7., 10.]


@pytest.mark.parametrize("pose_sync", [False, True])
def test_stop_during_temporal_preparation_never_publishes_late_result(temporal_rig, pose_sync):
    loop, robot, engine = temporal_rig
    loop._initial_pose_sync_enabled = pose_sync
    entered, release = threading.Event(), threading.Event()
    original = engine.get_action_chunk

    def wait_for_input(request):
        entered.set()
        assert release.wait(3)
        return {**original(request), "observation_wait_s": 0.}

    engine.get_action_chunk = wait_for_input
    try:
        loop.start()
        if not pose_sync:
            loop.tick()
        assert entered.wait(2)
        assert loop.preparing()
        assert loop.stop()
        assert robot.holds and not robot.commands
        with pytest.raises(RuntimeError, match="still finishing"):
            loop.start()
    finally:
        release.set()
        loop._request_thread.join(3)
        assert not loop._request_thread.is_alive()
        flush(loop)
    assert not robot.commands and not robot.sync_targets
    assert loop._processor.buffer_size == 0


@pytest.mark.parametrize("pose_sync", [False, True])
def test_missing_negotiated_timing_fails_closed_instead_of_trimming_unknown_latency(temporal_rig, pose_sync):
    loop, robot, _ = temporal_rig
    loop._initial_pose_sync_enabled = pose_sync
    loop.start()
    if pose_sync:
        loop._request_thread.join(3)
        assert not loop._request_thread.is_alive()
    else:
        fill(loop)
    flush(loop)
    assert not loop._running and loop._processor.buffer_size == 0
    assert robot.holds and not robot.commands
    assert not robot.sync_targets


@pytest.mark.parametrize("raw", [None, "{}", "[]", "bad", " " * 257,
    '{"observation_wait_s":true}', '{"observation_wait_s":-1}',
    '{"observation_wait_s":121}', '{"observation_wait_s":NaN}',
    '{"observation_wait_s":3}', '{"observation_wait_s":0,"extra":1}',
])
def test_invalid_wait_metadata_cannot_reduce_action_latency(raw):
    with pytest.raises(ValueError):
        action_latency(raw, 2.5)


def test_wait_metadata_roundtrip_keeps_model_and_transport_time():
    assert action_latency(encode_observation_wait(2.), 2.5) == .5

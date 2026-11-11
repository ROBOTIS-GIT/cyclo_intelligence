"""Opt-in two-process EngineCommand transport test in a network-none container.

No model allocation, robot subscription or command publication. The probe checks
the SDK transport separately from context validation using payload digests.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


def configure(server):
    os.environ["ROS_DOMAIN_ID"] = "175"
    os.environ["ZENOH_CONFIG_OVERRIDE"] = (
        'transport/shared_memory/enabled=false;scouting/multicast/enabled=false;'
        + ('mode="peer";connect/endpoints=[];listen/endpoints=["tcp/127.0.0.1:7447"]'
           if server else 'mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];listen/endpoints=[]')
    )


def serve(ready):
    from zenoh_ros2_sdk import ROS2ServiceServer
    from engine_process.protocol import (ENGINE_COMMAND_REQUEST_DEF, ENGINE_COMMAND_RESPONSE_DEF,
                                         EngineCommandResponse, response_to_message_kwargs)
    from engine_process.worker import EngineWorker

    class Engine:
        calls = 0
        context = None

        def load_policy(self, request):
            return {"success": True}

        def update_execution_context(self, context):
            self.context = context

        def get_action_chunk(self, request):
            self.calls += 1
            if request.task_instruction.startswith("control-loop"):
                if request.task_instruction == "control-loop-fail":
                    raise RuntimeError("injected model prediction failure")
                if request.task_instruction == "control-loop-slow":
                    Path(ready).with_name("model_busy").touch()
                    time.sleep(.5)
                return {"success": True, "chunk_size": 16, "action_dim": 3,
                        "action_chunk": [1., 2., 3.] * 16}
            return {"success": True, "chunk_size": 1, "action_dim": 3,
                    "action_chunk": [self.context.latest_event_id, len(self.context.actions), self.calls]}

    engine = Engine()
    worker = EngineWorker(engine)

    def callback(request):
        if request.task_instruction == "transport-probe":
            digest = hashlib.sha256(request.execution_context_json.encode()).hexdigest()
            response = EngineCommandResponse(success=True, seq_id=request.seq_id, message=digest)
        else:
            response = worker.handle(request)
            if response.success and engine.context is not None:
                response.message = hashlib.sha256(engine.context.to_json().encode()).hexdigest()
        return service.response_msg_class(**response_to_message_kwargs(response))

    service = ROS2ServiceServer(service_name="/cyclo_test/context", srv_type="interfaces/srv/EngineCommand",
                               request_definition=ENGINE_COMMAND_REQUEST_DEF,
                               response_definition=ENGINE_COMMAND_RESPONSE_DEF, callback=callback)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        Path(ready).touch()
        stop.wait(120)
    finally:
        service.close()


def probe_control_loop(client, directory):
    """Real ControlLoop/processor/SDK/Worker, with only weights and robot I/O replaced."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from engine_process.protocol import CMD_GET_ACTION, request_to_message_kwargs
    from inference_context.execution import ExecutionContext
    from main_runtime import control_loop as loop_module
    from main_runtime.inference_requester import InferenceRequester

    class Sink:
        action_keys = ["test_arm"]

        def __init__(self):
            self.commands = []
            self.holds = 0
            self.hold_failures = 0

        def set_state_subscription(self, enabled):
            pass

        def publish_action_with_receipt(self, action, keys, **kwargs):
            emitted = np.array(action, copy=True)
            emitted[0] = 0.  # Verify emitted values, not just original predictions.
            self.commands.append(tuple(emitted))
            return emitted

        def publish_current_pose_hold(self, keys, duration_s):
            if self.hold_failures:
                self.hold_failures -= 1
                raise RuntimeError("injected stale joint state")
            self.holds += 1

        def publish_idle_action(self, keys):
            pass

        def close(self):
            pass

    class Bridge:
        def __init__(self):
            self.contexts = []

        def call(self, request, timeout_s):
            client.timeout = timeout_s
            response = client.call(**request_to_message_kwargs(request))
            if request.command == CMD_GET_ACTION and response is not None and response.success:
                context = ExecutionContext.from_json(request.execution_context_json)
                assert response.message == hashlib.sha256(context.to_json().encode()).hexdigest()
                self.contexts.append(context)
            return response

    def wait_for(predicate, label, timeout=8.):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(label)
            time.sleep(.01)

    reports = []
    for mode in ("sync", "async"):
        bridge = Bridge()
        requester = InferenceRequester(bridge)
        context = ExecutionContext(f"loop-{mode}", 0, 0, "ready")
        assert requester.load_policy(SimpleNamespace(execution_context_json=context.to_json())).success
        sink = Sink()
        loop = loop_module.ControlLoop(requester)
        faults = []
        loop.set_fault_callback(lambda *args: faults.append(args))
        try:
            with patch.object(loop_module, "RobotClient", return_value=sink):
                loop.configure("test", task_instruction="control-loop", publish_to_robot=True,
                               action_request_mode=mode, control_hz=100, inference_hz=15,
                               execution_context=context)
            loop.start()
            loop.run_background()
            wait_for(lambda: len(bridge.contexts) >= 3, f"{mode} refills")
            assert not faults, faults
            published = [record for ctx in bridge.contexts for record in ctx.actions
                         if record.status == "published"]
            assert published and all(record.values[0] == 0. for record in published)
            assert any(record.planned_values[0] != 0. for record in published)
            assert all(value[0] == 0. for value in sink.commands)
            assert loop._processor.buffer_size > 0

            # Wait for the separate Worker to enter a deliberately slow model
            # call, then stop locally without waiting for its response.
            busy = Path(directory) / "model_busy"
            busy.unlink(missing_ok=True)
            loop.set_task_instruction("control-loop-slow")
            wait_for(busy.exists, f"{mode} slow model entered")
            started = time.monotonic()
            assert loop.pause()
            pause_s = time.monotonic() - started
            assert pause_s < .25, pause_s
            stopped_count = len(sink.commands)
            for thread in (loop._request_thread, loop._feedback_thread):
                if thread is not None:
                    thread.join(3.)
                    assert not thread.is_alive()
            time.sleep(.05)
            assert len(sink.commands) == stopped_count
            assert loop._processor.buffer_size == 0
            assert loop._feedback.phase == "paused" and sink.holds == 1
            assert not faults, faults

            loop.set_task_instruction("control-loop")
            loop.start()
            wait_for(lambda: len(sink.commands) > stopped_count + 5, f"{mode} resume")
            sink.hold_failures = 1
            loop.set_task_instruction("control-loop-fail")
            wait_for(lambda: bool(faults), f"{mode} prediction failure")
            assert len(faults) == 1 and faults[0][1] is False, faults
            assert "injected model prediction failure" in faults[0][0]
            assert loop._feedback.phase == "error" and loop._processor.buffer_size == 0
            failed_count = len(sink.commands)
            assert loop.initial_pose_sync_hold_required()
            try:
                loop.start()
            except RuntimeError as error:
                assert "hold" in str(error)
            else:
                raise AssertionError("START accepted while safe hold was pending")
            time.sleep(.05)
            assert len(sink.commands) == failed_count
            assert loop.stop()
            for thread in (loop._request_thread, loop._feedback_thread):
                if thread is not None:
                    thread.join(3.)
                    assert not thread.is_alive()
            assert not loop.initial_pose_sync_hold_required()
            assert loop._feedback.phase == "stopped"
            reports.append({"mode": mode, "recorded_commands": len(sink.commands),
                            "action_requests": len(bridge.contexts), "pause_s": pause_s,
                            "late_response_discarded": True, "resumed": True,
                            "model_failure_stopped": True, "hold_retry_required": True})
        finally:
            loop.shutdown()
            for thread in (loop._request_thread, loop._feedback_thread):
                if thread is not None:
                    thread.join(6.)
                    assert not thread.is_alive()
    return reports


def probe():
    from zenoh_ros2_sdk import ROS2ServiceClient
    from engine_process.protocol import (ENGINE_COMMAND_REQUEST_DEF, ENGINE_COMMAND_RESPONSE_DEF,
                                         EngineCommandRequest, CMD_UPDATE_CONTEXT, request_to_message_kwargs)
    from inference_context.execution import ExecutionContext
    from main_runtime.execution_feedback import ExecutionFeedback
    from main_runtime.inference_requester import InferenceRequester
    from types import SimpleNamespace
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        ready = Path(directory) / "ready"
        server = subprocess.Popen([sys.executable, __file__, "--server", str(ready)])
        client = None
        try:
            deadline = time.monotonic() + 20
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"test server exited: {server.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("test server startup")
                time.sleep(.05)
            client = ROS2ServiceClient(service_name="/cyclo_test/context", srv_type="interfaces/srv/EngineCommand",
                                       request_definition=ENGINE_COMMAND_REQUEST_DEF,
                                       response_definition=ENGINE_COMMAND_RESPONSE_DEF, timeout=5.)
            time.sleep(.5)
            results = []
            for seq, size in enumerate((256 * 1024, 2 * 1024 * 1024, 8 * 1024 * 1024), start=1):
                payload = os.urandom(size // 2).hex()
                request = EngineCommandRequest(command=CMD_UPDATE_CONTEXT, seq_id=seq,
                                               execution_context_json=payload, task_instruction="transport-probe")
                start = time.monotonic()
                reply = client.call(**request_to_message_kwargs(request))
                elapsed = time.monotonic() - start
                assert reply is not None and reply.success and reply.seq_id == seq
                assert reply.message == hashlib.sha256(payload.encode()).hexdigest()
                results.append({"bytes": size, "roundtrip_s": elapsed})

            class Bridge:
                last = None

                def call(self, request, timeout_s):
                    self.last = request
                    client.timeout = timeout_s
                    return client.call(**request_to_message_kwargs(request))

            bridge = Bridge()
            requester = InferenceRequester(bridge)
            initial = ExecutionContext("transport-test", 0, 0, "ready")
            assert requester.load_policy(SimpleNamespace(execution_context_json=initial.to_json())).success
            ledger = ExecutionFeedback(initial, pending_command_count=3, postprocess=False)
            ledger.phase = "running"
            ledger.buffer.enqueue(1, np.random.default_rng(42).normal(size=(2048, 22)))
            for _ in range(1024):
                command = ledger.buffer.take()
                ledger.buffer.finish(command.command_id, status="published", emitted_values=command.values)
            context = ledger.project(ledger.capture())
            assert len(context.to_json().encode()) > 65536
            start = time.monotonic()
            reply = requester.get_action("test", context=context)
            action_s = time.monotonic() - start
            assert reply.success, reply.message
            assert reply.action_list == [1025., 1027., 1.]
            assert reply.message == hashlib.sha256(context.to_json().encode()).hexdigest()
            retry = bridge.call(bridge.last, 5.)
            assert retry.success and list(retry.action_list) == reply.action_list
            ledger.acknowledge(context)
            ledger.reset("stop", "paused")
            stopped = ledger.project(ledger.capture())
            assert len(stopped.actions) == 1024 and len(stopped.resets) == 1
            assert len(stopped.to_json().encode()) > 65536
            start = time.monotonic()
            stopped_reply = requester.update_context(stopped)
            stop_s = time.monotonic() - start
            assert stopped_reply.success, stopped_reply.message
            assert stopped_reply.message == hashlib.sha256(stopped.to_json().encode()).hexdigest()
            ledger.acknowledge(stopped)
            assert not ledger.capture().snapshot.events
            assert not requester.get_action("test", context=stopped).success
            control_loop_results = probe_control_loop(client, directory)
            print(json.dumps({"transport": "two-process Zenoh TCP/CDR", "probe": results,
                              "worker": {"published": 1024, "discarded": 1024, "model_calls": 1,
                                         "action_roundtrip_s": action_s, "stop_roundtrip_s": stop_s,
                                         "action_bytes": len(context.to_json().encode()),
                                         "stop_bytes": len(stopped.to_json().encode())},
                              "control_loop": control_loop_results,
                              "robot_commands": 0}), flush=True)
        finally:
            if client is not None:
                client.close()
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server")
    args = parser.parse_args()
    configure(bool(args.server))
    serve(args.server) if args.server else probe()

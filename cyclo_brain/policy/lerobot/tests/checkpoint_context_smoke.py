"""Opt-in real-weight Engine/Worker/reset smoke, without robot I/O.

Run inside a pinned Worker image with --network none, offline assets and memory
limits. Frames must be RGB uint8 arrays plus state, exported from training data.
Robot mapping is a test double; neural inference and saved processors are real.
Transport is in-process or two-process Zenoh TCP/CDR inside the isolated container.
This does not measure task success or ROS/rmw interoperability.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from engine_process.worker import EngineWorker
from inference_context.contract import LoadedExecution
from main_runtime.execution_feedback import ExecutionFeedback
from main_runtime.inference_requester import InferenceRequester
from lerobot_engine.engine import LeRobotEngine
import lerobot_engine.image_preprocessing as images


class ReplayRobot:
    def __init__(self, frames, cameras):
        self._config = {"cameras": {}}
        self._frames, self._cameras = frames, cameras
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = None
        self._thread = threading.Thread(target=self._receive)
        self._thread.start()

    def _receive(self):
        index = 0
        while not self._stop.is_set():
            stamp = time.monotonic()
            snapshot = {
                "images": {name: self._frames[name][index] for name in self._cameras},
                "joint_positions": {"follower_arm": self._frames["state"][index]}, "sensors": {},
                "reception_monotonic_timestamps": {
                    **{f"camera:{name}": stamp for name in self._cameras}, "joint:follower_arm": stamp,
                },
            }
            with self._lock:
                self._snapshot = snapshot
            index = (index + 1) % len(self._frames["state"])
            self._stop.wait(1 / 30)

    def get_input_snapshot(self):
        with self._lock:
            if self._snapshot is None:
                raise ValueError("replay has not received a frame")
            return self._snapshot

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        assert not self._thread.is_alive()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--image-configs", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--transport", choices=("in-process", "zenoh"), default="in-process")
    args = parser.parse_args()
    assert 1 <= args.steps <= 200
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.85)
    config = json.loads((args.model / "config.json").read_text())
    with np.load(args.frames, allow_pickle=False) as archive:
        frames = {key: archive[key] for key in archive.files}
    cameras = {key.rsplit(".", 1)[-1]: key for key in config["input_features"]
               if key.startswith("observation.images.")}
    assert frames["state"].shape[1] == config["input_features"]["observation.state"]["shape"][0]
    for name in cameras:
        assert frames[name].dtype == np.uint8 and frames[name].shape[-1] == 3
    engine = LeRobotEngine()

    def init_robot(_robot_type):
        engine._robot = ReplayRobot(frames, cameras)
        engine._cameras = cameras
        engine._state_modalities = ["arm"]
        engine._action_keys = ["arm"]

    engine._init_robot = init_robot
    worker = EngineWorker(engine)

    class Transport:
        def call(self, request, timeout_s):
            started = time.monotonic()
            response = worker.handle(request)
            elapsed = time.monotonic() - started
            if elapsed > timeout_s:
                raise TimeoutError(f"command {request.command} took {elapsed:.3f}s > {timeout_s}s")
            return response

    transport = Transport()
    if args.transport == "zenoh":
        from zenoh_checkpoint_transport import ZenohCheckpointTransport
        transport = ZenohCheckpointTransport(worker)
    requester = InferenceRequester(transport)
    report = {"passed": False, "policy_type": config["type"], "model": str(args.model),
              "neural_compute_mocked": False, "transport": args.transport, "robot_commands": 0,
              "observation_source": "30 Hz dataset replay thread", "cycles": []}
    started = time.monotonic()
    try:
        from engine_process.protocol import ENGINE_PROTOCOL_VERSION
        descriptor = requester.describe(timeout_s=10.)
        assert descriptor.success and descriptor.protocol_version == ENGINE_PROTOCOL_VERSION
        report["protocol_version"] = descriptor.protocol_version
        policy_identity = None
        for cycle in range(2):
            with patch.object(images, "CONFIG_DIR", args.image_configs):
                loaded = requester.load_policy(SimpleNamespace(model_path=str(args.model), robot_type="test",
                                                                task_instruction=args.instruction))
            assert loaded.success, loaded.message
            if policy_identity is not None:
                assert id(engine._policy) == policy_identity, "cached LOAD reconstructed weights"
            policy_identity = id(engine._policy)
            execution = LoadedExecution.from_json(loaded.capabilities_json)
            feedback = ExecutionFeedback(execution.context, postprocess=False,
                                         pending_command_count=execution.contract.pending_command_count) if execution.context else None
            if feedback:
                feedback.phase = "running"
            latencies = []

            def predict(first=False):
                context = feedback.project(feedback.capture()) if feedback else None
                budget = ((execution.contract.initial_action_timeout_s or requester.get_action_timeout_s)
                          + (execution.contract.observation_warmup_timeout_s or 0)) if first else requester.get_action_timeout_s
                tick = time.monotonic()
                result = requester.get_action(args.instruction, context=context, timeout_s=budget)
                assert result.success, result.message
                action = np.asarray(result.action_list).reshape(result.chunk_size, result.action_dim)
                assert action.shape[1] == config["output_features"]["action"]["shape"][0]
                assert np.isfinite(action).all() and len(action)
                if feedback:
                    feedback.acknowledge(context)
                    feedback.buffer.enqueue(result.seq_id, action, align=False)
                    while feedback.buffer.buffer_size:
                        command = feedback.buffer.take()
                        feedback.buffer.finish(command.command_id, status="published", emitted_values=command.values)
                latencies.append(time.monotonic() - tick)

            for step in range(args.steps):
                predict(first=step == 0)
            if feedback:
                feedback.reset("test pause", "paused")
                paused = feedback.project(feedback.capture())
                response = requester.update_context(paused)
                assert response.success, response.message
                feedback.acknowledge(paused)
                feedback.phase = "running"
                predict(first=True)
            report["cycles"].append({"cycle": cycle, "predictions": len(latencies),
                                      "first_s": latencies[0], "max_subsequent_s": max(latencies[1:args.steps], default=0),
                                      "after_reset_s": latencies[-1] if feedback else None,
                                      "execution_contract": asdict(execution.contract),
                                      "cached_weights": cycle == 1})
            print(json.dumps(report["cycles"][-1]), flush=True)
        if feedback:
            engine._robot.close()
            time.sleep(1.05)
            stale = requester.get_action(args.instruction, context=feedback.project(feedback.capture()))
            assert not stale.success and "observations are not ready" in stale.message, stale.message
            report["stale_observation_rejected"] = stale.message
        assert requester.unload_policy().success
        assert not engine.is_ready and engine._policy is None
        report.update(passed=True, elapsed_s=time.monotonic() - started,
                      peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if args.transport == "zenoh":
            transport.close()
        engine.cleanup()
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

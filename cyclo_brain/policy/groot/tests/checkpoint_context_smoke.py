"""Opt-in real GR00T checkpoint/Worker smoke with synthetic robot observations.

Run in a network-none Worker container with read-only checkpoint/cache mounts and
explicit memory limits. This tests neural inference, not task quality or robot
safety. Only robot I/O is replaced; no command publishers are constructed.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from engine_process.worker import EngineWorker
from groot_engine import create_engine
from main_runtime.inference_requester import InferenceRequester


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    assert 1 <= args.steps <= 20
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.55)
    engine = create_engine()
    robots = []

    class Robot:
        def __init__(self, observation):
            self.closed = False
            self.observation = observation

        def wait_for_ready(self, **kwargs):
            return True

        def get_images(self, **kwargs):
            return {key: value[0, 0] for key, value in self.observation["video"].items()}

        def get_joint_positions(self):
            return {key: value[0, 0] for key, value in self.observation["state"].items()}

        def close(self):
            self.closed = True

    def init_robot(_):
        for modality in ("video", "state", "language"):
            assert engine._modality_horizon(modality) == 1, "Fixture only supports current observations"
        robot = Robot(engine.build_synthetic_observation("Smoke test"))
        engine.robot = robot
        robots.append(robot)
        engine.robot_info = {
            "cameras": list(robot.observation["video"]),
            "camera_sources": {key: key for key in robot.observation["video"]},
            "camera_rotations": {},
            "joints": {key: key for key in robot.observation["state"]},
            "sensor_states": {},
        }

    engine.init_robot_info = init_robot
    worker = EngineWorker(engine)

    class Transport:
        def call(self, request, timeout_s):
            started = time.monotonic()
            response = worker.handle(request)
            elapsed = time.monotonic() - started
            if elapsed > timeout_s:
                raise TimeoutError(f"Command {request.command}: {elapsed:.3f}s > {timeout_s}s")
            return response

    requester = InferenceRequester(Transport(), load_policy_timeout_s=600.)
    report = {"passed": False, "model": str(args.model), "neural_compute_mocked": False,
              "transport": "in-process", "robot_commands": 0,
              "observation_source": "checkpoint-schema synthetic images/state", "cycles": []}
    started = time.monotonic()
    try:
        identity = None
        for cycle in range(2):
            loaded = requester.load_policy(SimpleNamespace(model_path=str(args.model), robot_type="test",
                                                          acceleration_mode="pytorch"))
            assert loaded.success, loaded.message
            if identity is not None:
                assert engine.policy is identity, "Cached LOAD reconstructed weights"
            identity = engine.policy
            latencies = []
            for _ in range(args.steps):
                tick = time.monotonic()
                result = requester.get_action("Smoke test")
                assert result.success, result.message
                values = np.asarray(result.action_list).reshape(result.chunk_size, result.action_dim)
                assert values.size and np.isfinite(values).all()
                latencies.append(time.monotonic() - tick)
            report["cycles"].append({"cycle": cycle, "predictions": len(latencies),
                                      "latencies_s": latencies, "shape": list(values.shape)})
        identity = None
        unloaded = requester.unload_policy()
        assert unloaded.success, unloaded.message
        assert engine.policy is None and engine.robot is None
        assert all(robot.closed for robot in robots)
        report.update(passed=True, elapsed_s=time.monotonic() - started,
                      peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        identity = None
        engine.cleanup()
        report["final_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

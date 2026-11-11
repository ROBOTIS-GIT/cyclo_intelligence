"""Opt-in real CUDA allocation cleanup test; no GR00T weights or robot commands.

Run in the GR00T image with network disabled and explicit memory/GPU limits.
Actual Cyclo GR00T adapter/Worker are used; policy and robot are test doubles.
"""

import json
import weakref
from unittest.mock import patch

import torch

from engine_process.protocol import CMD_LOAD_POLICY, CMD_UNLOAD_POLICY, EngineCommandRequest
from engine_process.worker import EngineWorker
from groot_engine import create_engine
import runtime.inference_engine as implementation


def main():
    torch.cuda.init()
    baseline = torch.cuda.memory_allocated()
    references = []
    robots = []
    failure = None

    class Policy:
        def __init__(self):
            self.weights = torch.empty((4096, 4096), device="cuda")
            self.cycle = self  # Requires GC, just like cyclic model components.
            references.append(weakref.ref(self))
            if failure == "constructor":
                raise RuntimeError("synthetic constructor failure")

    class Robot:
        closed = False

        def close(self):
            self.closed = True

        def wait_for_ready(self, **kwargs):
            return True

    engine = create_engine()
    worker = EngineWorker(engine)

    def init_robot(_):
        engine.robot = Robot()
        robots.append(engine.robot)
        if failure == "robot":
            raise RuntimeError("synthetic robot setup failure")

    engine.init_robot_info = init_robot
    engine.init_policy_info = lambda: None
    seq = 0

    def command(kind, path="/synthetic"):
        nonlocal seq
        seq += 1
        return worker.handle(EngineCommandRequest(command=kind, seq_id=seq,
                             model_path=path, robot_type="test", acceleration_mode="pytorch"))

    try:
        with patch.object(implementation, "Gr00tPolicy", new=lambda **kwargs: Policy()), \
                patch.object(engine, "_sync_hf_token_for_gated_backbones"):
            assert command(CMD_LOAD_POLICY).success
            first = references[-1]
            allocated = torch.cuda.memory_allocated() - baseline
            assert allocated >= 64 * 1024 * 1024
            assert command(CMD_LOAD_POLICY).success
            assert len(references) == 1 and engine.policy is first()
            assert robots[0].closed
            assert command(CMD_LOAD_POLICY, "/replacement").success
            assert first() is None and len(references) == 2
            assert torch.cuda.memory_allocated() == baseline + allocated
            assert command(CMD_UNLOAD_POLICY).success
            assert first() is None and engine.policy is None and robots[-1].closed
            assert torch.cuda.memory_allocated() == baseline
            assert torch.cuda.memory_reserved() == 0
            for failure in ("constructor", "robot"):
                result = command(CMD_LOAD_POLICY)
                assert not result.success and "synthetic" in result.message
                assert all(reference() is None for reference in references)
                assert engine.policy is None and engine.robot is None
                assert torch.cuda.memory_allocated() == baseline
            failure = None
            assert command(CMD_LOAD_POLICY).success
            assert command(CMD_UNLOAD_POLICY).success
            assert all(reference() is None for reference in references)
            assert all(robot.closed for robot in robots)
            assert torch.cuda.memory_allocated() == baseline
            print(json.dumps({"passed": True, "allocated_bytes": allocated,
                              "final_allocated_bytes": torch.cuda.memory_allocated(),
                              "final_reserved_bytes": torch.cuda.memory_reserved(),
                              "baseline_allocated_bytes": baseline, "robot_commands": 0,
                              "actual_gr00t_weights": False}), flush=True)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()

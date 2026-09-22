"""Compare latest-input assembly with the previous numerical path. No robot I/O."""

import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "runtime"))

import numpy as np
import torch
import yaml

from test_preprocessing import Preprocessor, image_preprocessing
from lerobot_engine.input_pipeline import InputPipelineConfig, CONFIG_DIR


def main():
    torch.set_num_threads(1)
    engine = Preprocessor(list(range(22)), 22)
    dimensions = {"head": (376, 672), "left": (424, 240), "right": (424, 240)}
    frames = {name: np.random.default_rng(i).integers(0, 256, (*shape, 3), dtype=np.uint8)
              for i, (name, shape) in enumerate(dimensions.items())}
    engine._cameras = {name: f"observation.images.{name}" for name in dimensions}
    engine._robot.get_images = lambda format: frames
    config = yaml.safe_load((CONFIG_DIR / "act.yaml").read_text())
    features = {engine._cameras[name]: {"shape": [3, *shape]} for name, shape in dimensions.items()}
    engine._input_pipeline_config = InputPipelineConfig.from_user_config(config, {"input_features": features}, CONFIG_DIR / "act.yaml")
    previous = image_preprocessing.ImagePreprocessing(
        {"backend": "torch", "operations": [{"type": "identity"}]}, features)

    def baseline():
        images = engine._robot.get_images("rgb")
        joints = engine._robot.get_joint_positions()
        return {**{key: previous.apply(images[name], key).to(engine._device) for name, key in engine._cameras.items()},
                "observation.state": engine._transform_state((joints["follower_cyclo_input_0"],)), "task": ["pick"]}

    def current():
        result = engine._build_observation("pick")
        engine._input_evaluation = None
        return result

    old, new = baseline(), current()
    for key in old:
        if isinstance(old[key], torch.Tensor):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    timings = {"baseline": [], "graph": []}
    for i in range(260):
        order = (("baseline", baseline), ("graph", current)) if i % 2 else (("graph", current), ("baseline", baseline))
        for name, function in order:
            started = time.perf_counter_ns()
            function()
            if i >= 20:
                timings[name].append((time.perf_counter_ns() - started) / 1e6)
    result = {name + "_p95_ms": float(np.percentile(values, 95)) for name, values in timings.items()}
    result["limit_ms"] = result["baseline_p95_ms"] * 1.1 + .2
    result["passed"] = result["graph_p95_ms"] <= result["limit_ms"]
    result["max_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["retained_session_bytes"] = engine._input_pipeline_config.budget.used
    profiles = {}
    for name, function in (("baseline", baseline), ("graph", current)):
        tracemalloc.start()
        value = function()
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del value
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True) as profile:
            value = function()
        events = profile.key_averages()
        profiles[name] = {
            "torch_copy_calls": sum(e.count for e in events if e.key == "aten::copy_"),
            "torch_positive_self_allocated_bytes": sum(max(0, e.self_cpu_memory_usage) for e in events),
            "tracemalloc_peak_bytes": python_peak,
        }
        del value
    result["copy_allocation_profiles"] = profiles
    result["allocation_scope"] = "Torch copy_ calls and summed positive self allocations; tracemalloc excludes Torch storage; neither is GPU peak"
    result["scope"] = "CPU latest-input assembly, three cameras; not control jitter or GPU peak memory"
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

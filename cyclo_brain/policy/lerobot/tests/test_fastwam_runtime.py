#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
from unittest import mock

import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
PACKAGE_NAME = "lerobot_engine_fastwam_test"

package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}",
        ENGINE_DIR / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_module("image_preprocessing")

lerobot = types.ModuleType("lerobot")
lerobot.__path__ = []
lerobot_configs = types.ModuleType("lerobot.configs")
lerobot_configs.__path__ = []
lerobot_config_policies = types.ModuleType("lerobot.configs.policies")
lerobot_policies = types.ModuleType("lerobot.policies")
lerobot_policies.__path__ = []
lerobot_pretrained = types.ModuleType("lerobot.policies.pretrained")
lerobot_config_policies.PreTrainedConfig = object
lerobot_policies.get_policy_class = mock.Mock()
lerobot_policies.make_pre_post_processors = mock.Mock()
lerobot_pretrained.PreTrainedPolicy = object

with mock.patch.dict(
    sys.modules,
    {
        "lerobot": lerobot,
        "lerobot.configs": lerobot_configs,
        "lerobot.configs.policies": lerobot_config_policies,
        "lerobot.policies": lerobot_policies,
        "lerobot.policies.pretrained": lerobot_pretrained,
    },
):
    loading = load_module("loading")

optimization = load_module("optimization")

engine_base = types.ModuleType("engine")
engine_base.InferenceEngine = type("InferenceEngine", (), {})
robot_stub = types.ModuleType("robot_client")
robot_stub.RobotClient = object
camera_stub = types.ModuleType("robot_client.camera_mapping")
camera_stub.resolve_camera_mappings = mock.Mock()
with mock.patch.dict(sys.modules, {
    "engine": engine_base,
    "robot_client": robot_stub,
    "robot_client.camera_mapping": camera_stub,
    "lerobot.policies.pretrained": lerobot_pretrained,
    f"{PACKAGE_NAME}.loading": loading,
}):
    engine_module = load_module("engine")


class EngineModelLifecycleTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(engine_module, "load_image_preprocessing", return_value=object())
        self.image_loader = patcher.start()
        self.addCleanup(patcher.stop)

    def make_engine(self):
        engine = engine_module.LeRobotEngine()
        engine._resolve_model_dir = lambda path: path
        engine._apply_policy_optimization = mock.Mock()
        engine._init_robot = mock.Mock()
        return engine

    @staticmethod
    def request(path):
        return types.SimpleNamespace(model_path=path, robot_type="robot")

    def test_previous_policy_is_collected_before_replacement_is_constructed(self):
        engine = self.make_engine()
        old = FakePolicy(types.SimpleNamespace(type="act"))
        old.cycle = old
        reference = weakref.ref(old)
        engine._policy = old
        engine._loaded_model_path = "/old"
        del old

        def load(path, device):
            self.assertIsNone(reference(), "old policy still lives during replacement LOAD")
            return FakePolicy(types.SimpleNamespace(type="act")), object(), object()

        engine._load_policy_assets = mock.Mock(side_effect=load)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertTrue(engine.load_policy(self.request("/new"))["success"])
            engine.cleanup()

    def test_same_checkpoint_reuses_weights_without_loading_again(self):
        engine = self.make_engine()
        engine._policy = FakePolicy(types.SimpleNamespace(type="act"))
        engine._loaded_model_path = "/same"
        engine._load_policy_assets = mock.Mock(side_effect=AssertionError("unexpected reload"))
        self.assertTrue(engine.load_policy(self.request("/same"))["success"])
        engine._load_policy_assets.assert_not_called()
        self.image_loader.assert_called_once_with("/same")
        self.assertIs(engine._image_preprocessing, self.image_loader.return_value)

    def test_invalid_yaml_blocks_load_before_weights(self):
        engine = self.make_engine()
        engine._load_policy_assets = mock.Mock()
        self.image_loader.side_effect = ValueError("Invalid image preprocessing config")
        result = engine.load_policy(self.request("/new"))
        self.assertFalse(result["success"])
        engine._load_policy_assets.assert_not_called()
        self.assertIsNone(engine._image_preprocessing)

    def test_cached_load_reads_new_snapshot_and_cleanup_clears_it(self):
        engine = self.make_engine()
        engine._policy = FakePolicy(types.SimpleNamespace(type="act"))
        engine._loaded_model_path = "/same"
        first, second = object(), object()
        self.image_loader.side_effect = [first, second]
        self.assertTrue(engine.load_policy(self.request("/same"))["success"])
        self.assertIs(engine._image_preprocessing, first)
        self.assertTrue(engine.load_policy(self.request("/same"))["success"])
        self.assertIs(engine._image_preprocessing, second)
        engine.cleanup()
        self.assertIsNone(engine._image_preprocessing)

    def test_missing_observations_cannot_report_successful_load(self):
        engine = self.make_engine()
        engine._load_policy_assets = mock.Mock(return_value=(
            FakePolicy(types.SimpleNamespace(type="act")), object(), object(),
        ))
        engine._init_robot.side_effect = RuntimeError("Required observations: joint:arm")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            result = engine.load_policy(self.request("/new"))
        self.assertFalse(result["success"])
        self.assertIn("joint:arm", result["message"])
        self.assertIsNone(engine._policy)

    def test_saved_processor_precedes_shape_validation_and_prediction(self):
        engine = self.make_engine()
        engine._policy = FakePolicy(types.SimpleNamespace(type="act"))
        engine._robot = object()
        engine._image_preprocessing = object()
        raw, processed = {"raw": True}, {"processed": True}
        action = torch.zeros(1, 2, 3)
        engine._build_observation = mock.Mock(return_value=raw)
        engine._preprocessor = mock.Mock(return_value=processed)
        engine._validate_camera_shapes = mock.Mock()
        engine._predict_chunk = mock.Mock(return_value=action)
        engine._postprocessor = mock.Mock(side_effect=lambda value: value)
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction="pick"))
        self.assertTrue(result["success"])
        engine._preprocessor.assert_called_once_with(raw)
        engine._validate_camera_shapes.assert_called_once_with(processed)
        engine._predict_chunk.assert_called_once_with(processed)
        engine._postprocessor.assert_called_once_with(action)

        engine._validate_camera_shapes.side_effect = ValueError("bad camera sizes")
        engine._predict_chunk.reset_mock()
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction="pick"))
        self.assertFalse(result["success"])
        engine._predict_chunk.assert_not_called()
        self.assertNotIn("action_chunk", result)


class FakePolicy:
    from_pretrained_calls = []

    def __init__(self, config):
        self.config = config
        self.to_calls = []
        self.eval_calls = 0

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.from_pretrained_calls.append((model_path, kwargs))
        config = kwargs.get("config") or types.SimpleNamespace(type="act")
        return cls(config)

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class FakeConfigLoader:
    loaded = []

    @classmethod
    def from_pretrained(cls, model_path):
        config = types.SimpleNamespace(type="fastwam", device="cuda")
        cls.loaded.append((model_path, config))
        return config


class FastWamLoadingTest(unittest.TestCase):
    def setUp(self):
        FakePolicy.from_pretrained_calls.clear()
        FakeConfigLoader.loaded.clear()
        loading.get_policy_class = mock.Mock(return_value=FakePolicy)
        loading.make_pre_post_processors = mock.Mock(return_value=("pre", "post"))
        loading.PreTrainedConfig = FakeConfigLoader

    def write_config(self, directory: str, policy_type: str):
        Path(directory, "config.json").write_text(json.dumps({"type": policy_type}))

    def test_fastwam_is_initially_loaded_on_cpu_without_full_gpu_move(self):
        with tempfile.TemporaryDirectory() as model_path:
            self.write_config(model_path, "fastwam")
            policy, preprocessor, postprocessor = (
                loading.LoadingMixin._load_policy_assets(
                    model_path, torch.device("cuda")
                )
            )

        self.assertEqual(FakeConfigLoader.loaded[0][1].device, "cpu")
        self.assertIs(
            FakePolicy.from_pretrained_calls[0][1]["config"],
            FakeConfigLoader.loaded[0][1],
        )
        self.assertEqual(policy.to_calls, [])
        self.assertEqual(policy.eval_calls, 1)
        self.assertEqual((preprocessor, postprocessor), ("pre", "post"))
        self.assertEqual(
            loading.make_pre_post_processors.call_args.kwargs["preprocessor_overrides"],
            {"device_processor": {"device": "cuda"}},
        )

    def test_other_policies_keep_the_existing_runtime_device_move(self):
        with tempfile.TemporaryDirectory() as model_path:
            self.write_config(model_path, "act")
            policy, _, _ = loading.LoadingMixin._load_policy_assets(
                model_path, torch.device("cuda")
            )

        self.assertEqual(FakeConfigLoader.loaded, [])
        self.assertEqual(FakePolicy.from_pretrained_calls[0][1], {})
        self.assertEqual(policy.to_calls, ["cuda"])
        self.assertEqual(policy.eval_calls, 1)

    def test_pi0_fast_checkpoint_still_selects_its_own_class(self):
        with tempfile.TemporaryDirectory() as model_path:
            self.write_config(model_path, "pi0_fast")
            loading.LoadingMixin._load_policy_assets(model_path, torch.device("cpu"))
        loading.get_policy_class.assert_called_with("pi0_fast")

    def test_invalid_wall_x_is_rejected_before_constructing_model(self):
        loading.get_policy_class.reset_mock()
        with tempfile.TemporaryDirectory() as model_path:
            (Path(model_path) / "config.json").write_text(json.dumps({
                "type": "wall_x",
                "input_features": {"observation.state": {"shape": [22]}},
                "output_features": {"action": {"shape": [22]}},
            }))
            with self.assertRaisesRegex(ValueError, "WALL-X"):
                loading.LoadingMixin._load_policy_assets(model_path, torch.device("cpu"))
        loading.get_policy_class.assert_not_called()


class FakeChild:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self


class FakeFastWamModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.text_encoder = FakeChild()
        self.core = FakeChild()
        self.vae = FakeChild()
        self.encoded = []

    def named_children(self):
        return (
            ("text_encoder", self.text_encoder),
            ("core", self.core),
            ("vae", self.vae),
        )

    def encode_prompt(self, prompts):
        self.encoded.append((list(prompts), str(self.device)))
        value = float(len(self.encoded))
        return torch.tensor([[value]]), torch.ones((1, 1), dtype=torch.bool)


class FastWamContextTest(unittest.TestCase):
    def make_runtime(self):
        model = FakeFastWamModel()
        config = types.SimpleNamespace(
            type="fastwam",
            device="cpu",
            prompt_template="do {task}",
            proprio_dim=4,
        )
        captured_batches = []

        def predict_action_chunk(batch, *args, **kwargs):
            captured_batches.append(batch)
            return batch

        policy = types.SimpleNamespace(
            model=model,
            config=config,
            predict_action_chunk=predict_action_chunk,
        )
        runtime = optimization.OptimizationMixin()
        runtime._policy = policy
        runtime._device = torch.device("cpu")
        return runtime, policy, model, captured_batches

    def test_context_is_reused_then_refreshed_when_instruction_changes(self):
        runtime, policy, model, captured_batches = self.make_runtime()
        runtime._offload_fastwam(types.SimpleNamespace(task_instruction="pick"))

        policy.predict_action_chunk(
            {"task": ["pick"], "observation.state": torch.tensor([[1.0, 2.0]])}
        )
        self.assertEqual(model.encoded, [(["do pick"], "cpu")])

        policy.predict_action_chunk(
            {"task": ["place"], "observation.state": torch.tensor([[1.0, 2.0]])}
        )

        self.assertEqual(
            model.encoded,
            [(["do pick"], "cpu"), (["do place"], "cpu")],
        )
        latest_batch = captured_batches[-1]
        self.assertNotIn("task", latest_batch)
        self.assertNotIn("prompt", latest_batch)
        self.assertEqual(latest_batch["context"].item(), 2.0)
        self.assertEqual(tuple(latest_batch["proprio"].shape), (1, 4))
        self.assertEqual(model.text_encoder.to_calls, [])
        self.assertTrue(model.core.to_calls)

    def test_initial_instruction_is_still_required(self):
        runtime, _, _, _ = self.make_runtime()

        with self.assertRaisesRegex(RuntimeError, "task instruction"):
            runtime._offload_fastwam(types.SimpleNamespace(task_instruction=""))


if __name__ == "__main__":
    unittest.main()

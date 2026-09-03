#!/usr/bin/env python3
"""Tests for the GR00T Engine-process factory contract."""

from __future__ import annotations

import importlib.util
import json
import numpy as np
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
INFERENCE_ENGINE = ROOT / "cyclo_brain/policy/groot/runtime/inference_engine.py"


def _install_stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _write_tt_rtc_manifest(root: Path) -> None:
    (root / "tt_rtc_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cyclo.training-time-rtc/v1",
                "training_time_rtc": {
                    "trained": True,
                    "action_horizon": 16,
                    "action_dimension": 19,
                    "action_hz": 15.0,
                    "max_delay_steps": 6,
                    "delay_sampling": {
                        "type": "uniform_integer",
                        "min_inclusive": 0,
                        "max_inclusive": 6,
                    },
                    "prefix_input": "ground_truth_clean_action",
                    "loss_region": "postfix_only",
                    "per_action_timestep": True,
                    "flow_convention": {
                        "noise_endpoint": 0.0,
                        "clean_endpoint": 1.0,
                        "velocity_target": "action_minus_noise",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


class _Robot:
    def get_images(self, **_kwargs):
        return {"camera": object()}

    def get_joint_positions(self):
        return {"joint": object()}


def _tt_request(*, action_policy_mode: str = "base") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        task_instruction="pick",
        action_policy_mode=action_policy_mode,
        action_request_mode="tt_rtc",
        rtc_delay_steps=6,
        rtc_action_dim=19,
        rtc_prefix_action_list=[0.1] * (6 * 19),
    )


class GR00TEngineFactoryTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = dict(sys.modules)

        _install_stub(
            "cv2",
            ROTATE_90_CLOCKWISE=0,
            ROTATE_180=1,
            ROTATE_90_COUNTERCLOCKWISE=2,
        )
        _install_stub("torch", inference_mode=lambda: None)
        _install_stub("gr00t")
        _install_stub("gr00t.model")
        _install_stub("gr00t.data")
        _install_stub(
            "gr00t.data.embodiment_tags",
            EmbodimentTag=types.SimpleNamespace(NEW_EMBODIMENT="new_embodiment"),
        )
        _install_stub("gr00t.policy")
        _install_stub("gr00t.policy.gr00t_policy", Gr00tPolicy=object)
        _install_stub("robot_client", RobotClient=object)
        _install_stub(
            "robot_client.camera_mapping",
            resolve_camera_feature_sources=lambda *_args, **_kwargs: {},
        )
        _install_stub("scripts")
        _install_stub("scripts.deployment")
        _install_stub(
            "scripts.deployment.standalone_inference_script",
            replace_dit_with_tensorrt=lambda *_args, **_kwargs: None,
        )
        _install_stub(
            "scripts.deployment.export_onnx_n1d7",
            DiTInputCapture=object,
            export_dit_to_onnx=lambda *_args, **_kwargs: None,
        )

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._saved_modules)

    def test_runtime_module_exposes_create_engine_factory(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        engine = module.create_engine()

        self.assertIsInstance(engine, module.GR00TInference)

    def test_acceleration_request_resolves_model_local_engine_path(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        engine = module.create_engine()

        mode, engine_path, strict = engine._resolve_acceleration_request(
            types.SimpleNamespace(
                acceleration_mode="tensorrt",
                acceleration_engine_path="custom.trt",
            ),
            "/models/policy",
        )

        self.assertEqual(mode, "tensorrt_dit")
        self.assertEqual(engine_path, "/models/policy/custom.trt")
        self.assertTrue(strict)

    def test_synthetic_observation_uses_model_schema(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        def modality(keys, deltas):
            return types.SimpleNamespace(
                modality_keys=keys,
                delta_indices=deltas,
            )

        state_action_processor = types.SimpleNamespace(
            norm_params={
                "new_embodiment": {
                    "state": {
                        "arm": {
                            "dim": np.array(2),
                            "mean": np.array([0.25, -0.5], dtype=np.float32),
                        }
                    }
                }
            }
        )
        processor = types.SimpleNamespace(
            image_target_size=[12, 16],
            processor=types.SimpleNamespace(
                image_processor=types.SimpleNamespace(
                    image_mean=[0.5, 0.5, 0.5],
                )
            ),
            state_action_processor=state_action_processor,
        )
        policy = types.SimpleNamespace(
            embodiment_tag=types.SimpleNamespace(value="new_embodiment"),
            processor=processor,
            modality_configs={
                "video": modality(["cam"], [0, 1]),
                "state": modality(["arm"], [0]),
                "action": modality(["arm"], [0, 1, 2]),
                "language": modality(["task"], [0]),
            },
        )

        engine = module.create_engine()
        engine.policy = policy
        engine.init_policy_info()

        observation = engine.build_synthetic_observation("pick")

        self.assertEqual(observation["video"]["cam"].shape, (1, 2, 12, 16, 3))
        self.assertEqual(observation["video"]["cam"].dtype, np.uint8)
        self.assertEqual(observation["state"]["arm"].shape, (1, 1, 2))
        np.testing.assert_allclose(
            observation["state"]["arm"][0, 0],
            np.array([0.25, -0.5], dtype=np.float32),
        )
        self.assertEqual(observation["language"]["task"], [["pick"]])

    def test_tt_rtc_fails_closed_without_the_upstream_model_entry_point(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            _write_tt_rtc_manifest(model_root)
            engine = module.create_engine()
            engine.policy = types.SimpleNamespace()
            engine.robot = _Robot()
            engine._loaded_model_path = str(model_root)
            engine.preprocess = lambda *_args: {"observation": True}

            result = engine.get_action_chunk(_tt_request())

        self.assertFalse(result["success"])
        self.assertIn("get_action_tt_rtc", result["message"])
        self.assertIn("legacy inference-only RTC", result["message"])

    def test_tt_rtc_start_fails_before_loading_a_model_without_runtime_support(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            _write_tt_rtc_manifest(model_root)
            engine = module.create_engine()

            result = engine.load_policy(
                types.SimpleNamespace(
                    model_path=str(model_root),
                    robot_type="ffw_sg2_rev1",
                    action_request_mode="tt_rtc",
                    acceleration_mode="pytorch",
                    acceleration_engine_path="",
                    rlt_enabled=False,
                    rlt_bundle_path="",
                )
            )

        self.assertFalse(result["success"])
        self.assertIn("get_action_tt_rtc", result["message"])
        self.assertIsNone(engine.policy)

    def test_tt_rtc_rejects_tensorrt_before_model_execution(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            _write_tt_rtc_manifest(model_root)
            engine = module.create_engine()
            engine.policy = types.SimpleNamespace()
            engine.robot = _Robot()
            engine._loaded_model_path = str(model_root)
            engine._loaded_acceleration_mode = "tensorrt_dit"

            result = engine.get_action_chunk(_tt_request())

        self.assertFalse(result["success"])
        self.assertIn("unavailable with TensorRT", result["message"])

    def test_tt_rtc_routes_vla_to_an_explicit_policy_api(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class _TTRTCPolicy:
            def __init__(self):
                self.prefix = None

            def get_action_tt_rtc(self, _observation, **kwargs):
                self.prefix = kwargs["committed_action_prefix"]
                return {
                    "action": np.zeros((1, 16, 19), dtype=np.float32)
                }, {}

        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            _write_tt_rtc_manifest(model_root)
            engine = module.create_engine()
            engine.policy = _TTRTCPolicy()
            engine.robot = _Robot()
            engine.policy_info["action"] = ["action"]
            engine._loaded_model_path = str(model_root)
            engine.preprocess = lambda *_args: {"observation": True}

            result = engine.get_action_chunk(_tt_request())

        self.assertTrue(result["success"])
        self.assertEqual(result["chunk_size"], 16)
        self.assertEqual(engine.policy.prefix.shape, (1, 6, 19))

    def test_tt_rtc_routes_mlp_to_the_rlt_adapter_not_the_vla_policy(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test",
            INFERENCE_ENGINE,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class _Policy:
            def get_action_tt_rtc(self, *_args, **_kwargs):
                raise AssertionError("VLA route must not run for RLT MLP mode")

        class _Adapter:
            def __init__(self):
                self.validated = False

            def require_tt_rtc_capability(self):
                self.validated = True

            def get_action_tt_rtc(self, _observation, **_kwargs):
                return {"action": np.zeros((1, 10, 19), dtype=np.float32)}

        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            _write_tt_rtc_manifest(model_root)
            engine = module.create_engine()
            engine.policy = _Policy()
            engine.robot = _Robot()
            engine._rlt_adapter = _Adapter()
            engine.policy_info["action"] = ["action"]
            engine._loaded_model_path = str(model_root)
            engine.preprocess = lambda *_args: {"observation": True}

            result = engine.get_action_chunk(
                _tt_request(action_policy_mode="rlt")
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["chunk_size"], 10)
        self.assertTrue(engine._rlt_adapter.validated)


if __name__ == "__main__":
    unittest.main()

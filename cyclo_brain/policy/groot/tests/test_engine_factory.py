#!/usr/bin/env python3
"""Tests for the GR00T Engine-process factory contract."""

from __future__ import annotations

import importlib.util
import numpy as np
import sys
import types
import unittest
from unittest.mock import Mock, patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
INFERENCE_ENGINE = ROOT / "cyclo_brain/policy/groot/runtime/inference_engine.py"


def _install_stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class GR00TEngineFactoryTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = dict(sys.modules)

        _install_stub(
            "cv2",
            ROTATE_90_CLOCKWISE=0,
            ROTATE_180=1,
            ROTATE_90_COUNTERCLOCKWISE=2,
        )
        _install_stub("torch", inference_mode=lambda: None,
                      cuda=types.SimpleNamespace(is_initialized=lambda: False, empty_cache=Mock()))
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

    def load_module(self):
        spec = importlib.util.spec_from_file_location("groot_lifecycle_under_test", INFERENCE_ENGINE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cleanup_releases_weights_cache_and_robot_even_when_close_fails(self):
        for fail_close in (False, True):
            with self.subTest(fail_close=fail_close):
                module = self.load_module()
                engine = module.create_engine()
                robot = Mock()
                if fail_close:
                    robot.close.side_effect = RuntimeError("close failed")
                engine.policy, engine.robot = object(), robot
                engine._loaded_model_path = '/model'
                if fail_close:
                    with self.assertRaisesRegex(RuntimeError, 'close failed'):
                        engine.cleanup()
                else:
                    engine.cleanup()
                self.assertIsNone(engine.policy)
                self.assertIsNone(engine.robot)
                self.assertIsNone(engine._loaded_model_path)
                self.assertFalse(engine.is_ready)
                engine.cleanup()
                robot.close.assert_called_once()

    def test_repeated_load_reuses_weights_but_load_after_cleanup_does_not(self):
        module = self.load_module()
        engine = module.create_engine()
        robots = []
        def init_robot(_):
            robot = Mock()
            robots.append(robot)
            engine.robot = robot
        engine.init_robot_info = init_robot
        engine.init_policy_info = Mock()
        request = types.SimpleNamespace(model_path='/model', robot_type='test', acceleration_mode='pytorch')
        with patch.object(module, 'Gr00tPolicy', side_effect=lambda **kwargs: object()) as factory, \
                patch.object(engine, '_sync_hf_token_for_gated_backbones'):
            self.assertTrue(engine.load_policy(request)['success'])
            policy = engine.policy
            self.assertTrue(engine.load_policy(request)['success'])
            self.assertIs(engine.policy, policy)
            self.assertEqual(factory.call_count, 1)
            robots[0].close.assert_called_once()
            engine.cleanup()
            self.assertTrue(engine.load_policy(request)['success'])
            self.assertEqual(factory.call_count, 2)
            self.assertIsNot(engine.policy, policy)
            engine.cleanup()

    def test_failed_robot_setup_does_not_retain_loaded_weights(self):
        module = self.load_module()
        engine = module.create_engine()
        robot = Mock()
        def fail_robot(_):
            engine.robot = robot
            raise RuntimeError('invalid camera layout')
        engine.init_robot_info = fail_robot
        engine.init_policy_info = Mock()
        request = types.SimpleNamespace(model_path='/model', robot_type='test', acceleration_mode='pytorch')
        with patch.object(module, 'Gr00tPolicy', return_value=object()), \
                patch.object(engine, '_sync_hf_token_for_gated_backbones'):
            response = engine.load_policy(request)
        self.assertFalse(response['success'])
        self.assertIn('invalid camera layout', response['message'])
        self.assertIsNone(engine.policy)
        self.assertIsNone(engine.robot)
        self.assertIsNone(engine._loaded_model_path)
        robot.close.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for the GR00T Engine-process factory contract."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
import numpy as np
import sys
import types
import tempfile
import unittest
import threading
import time
from unittest.mock import patch
from unittest import mock
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
    def test_output_action_config_selects_right_decoded_chunk_without_changing_model(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        action_config = engine.policy.modality_configs["action"]
        action_config.modality_keys = ["arm_left", "arm_right"]
        with tempfile.TemporaryDirectory() as root:
            Path(root, "inference_config.json").write_text(
                json.dumps({"output_action_keys": ["arm_right"]})
            )
            engine.init_policy_info(root)
            self.assertEqual(engine.policy_info["action"], ["arm_right"])
            self.assertEqual(action_config.modality_keys, ["arm_left", "arm_right"])
            self.assertEqual(engine.policy_info["state"], ["arm_left", "arm_right"])
            result = engine.postprocess_action({
                "arm_left": np.full((1, 24, 8), -5.0),
                "arm_right": np.full((1, 24, 8), 0.7),
            })
            self.assertTrue(result["success"])
            self.assertEqual(result["action_dim"], 8)
            np.testing.assert_array_equal(result["action_chunk"], np.full(24 * 8, 0.7))
            # Reloading a checkpoint without an override restores its output.
            Path(root, "inference_config.json").unlink()
            engine.init_policy_info(root)
            self.assertEqual(engine.policy_info["action"], ["arm_left", "arm_right"])

    def test_invalid_output_action_config_is_rejected(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        with tempfile.TemporaryDirectory() as root:
            for keys in ([], ["arm_left"], ["arm_right", "arm_right"], "arm_right", [None]):
                with self.subTest(keys=keys):
                    Path(root, "inference_config.json").write_text(
                        json.dumps({"output_action_keys": keys})
                    )
                    with self.assertRaisesRegex(ValueError, "output_action_keys"):
                        engine.init_policy_info(root)

    def test_new_subtask_waits_for_camera_and_joint_updates(self):
        module, engine, _images, _joints = self._right_arm_engine()
        engine._subtask_choices = [['phase 1'], ['phase 2']]
        clock = [10.0]
        engine.robot.get_image_timestamp = lambda _key: clock[0]
        engine.robot.get_joint_timestamp = lambda _key: clock[0]
        def advance(_delay):
            clock[0] += 0.1
        with mock.patch.object(module.time, 'time', side_effect=lambda: clock[0]), \
             mock.patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(module.time, 'sleep', side_effect=advance) as sleep:
            engine._wait_for_subtask_observation('phase 2')
        sleep.assert_called_once()
        engine._last_subtask_observation_instruction = 'phase 2'
        engine.robot.get_image_timestamp = mock.Mock(side_effect=AssertionError('must not wait again'))
        engine._wait_for_subtask_observation('phase 2')

    def test_stale_subtask_observations_are_rejected(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        engine._subtask_choices = [['phase 1'], ['phase 2']]
        engine.robot.get_image_timestamp = lambda _key: 0.0
        engine.robot.get_joint_timestamp = lambda _key: 0.0
        with self.assertRaisesRegex(RuntimeError, 'Fresh camera/joint'):
            engine._wait_for_subtask_observation('phase 2', timeout=0)

    def test_whole_task_models_skip_subtask_freshness_gate(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        engine.robot.get_image_timestamp = mock.Mock(side_effect=AssertionError('legacy path'))
        engine._wait_for_subtask_observation('whole task')

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
            TensorRTDiTWrapper=object,
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

    def _right_arm_engine(self):
        spec = importlib.util.spec_from_file_location(
            "groot_runtime_inference_engine_under_test", INFERENCE_ENGINE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.create_engine()
        keys = {
            "video": ["cam_left_head", "cam_left_wrist", "cam_right_wrist"],
            "state": ["arm_left", "arm_right"],
            "action": ["arm_right"],
            "language": ["sub_task"],
        }
        engine.policy = types.SimpleNamespace(
            embodiment_tag=types.SimpleNamespace(value="new_embodiment"),
            modality_configs={
                name: types.SimpleNamespace(
                    modality_keys=value,
                    delta_indices=list(range(24)) if name == "action" else [0],
                )
                for name, value in keys.items()
            },
            processor=types.SimpleNamespace(
                state_action_processor=types.SimpleNamespace(norm_params={
                    "new_embodiment": {
                        "state": {key: {"dim": np.array(8)} for key in keys["state"]},
                        "action": {"arm_right": {"dim": np.array(8)}},
                    }
                }),
            ),
        )
        engine.init_policy_info()
        engine.robot_info = {
            "cameras": keys["video"],
            "camera_sources": {key: key for key in keys["video"]},
            "camera_rotations": {},
            "joints": {key: "follower_" + key for key in keys["state"]},
            "sensor_states": {},
        }
        engine.robot = types.SimpleNamespace(
            action_keys=["arm_left", "arm_right", "head", "lift", "mobile"],
            _action_groups={
                "arm_right": {
                    "msg_type": "trajectory_msgs/msg/JointTrajectory",
                    "joint_names": ["right_%s" % i for i in range(8)],
                }
            },
            _resolve_action_key=lambda key: key,
            get_joint_names=lambda group: list(range(8)),
        )
        images = {key: np.zeros((4, 6, 3), dtype=np.uint8) for key in keys["video"]}
        joints = {
            "follower_arm_left": np.arange(8, dtype=np.float32),
            "follower_arm_right": np.arange(8, dtype=np.float32) + 10,
        }
        return module, engine, images, joints

    def test_right_arm_only_keeps_left_state_and_three_cameras(self):
        _module, engine, images, joints = self._right_arm_engine()
        observation = engine.preprocess(images, joints, "place the roll")
        self.assertEqual(list(observation["video"]), engine.policy_info["video"])
        self.assertEqual(list(observation["state"]), ["arm_left", "arm_right"])
        np.testing.assert_array_equal(observation["state"]["arm_left"][0, 0], joints["follower_arm_left"])
        result = engine.postprocess_action({
            "arm_right": np.ones((1, 24, 8), dtype=np.float32),
            "arm_left": np.zeros((1, 24, 8), dtype=np.float32),
        })
        self.assertTrue(result["success"])
        self.assertEqual((result["chunk_size"], result["action_dim"]), (24, 8))
        self.assertEqual(result["action_chunk"].dtype, np.float64)

    def test_missing_left_wrist_fails_before_policy_call(self):
        _module, engine, images, joints = self._right_arm_engine()
        del images["cam_left_wrist"]
        self.assertFalse(engine.preprocess(images, joints, "place")["success"])

    def test_unmapped_left_state_is_rejected_at_load(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        del engine.robot_info["joints"]["arm_left"]
        with self.assertRaisesRegex(RuntimeError, "arm_left"):
            engine._validate_robot_contract()

    def test_wrong_robot_action_width_is_rejected_at_load(self):
        _module, engine, _images, _joints = self._right_arm_engine()
        engine.robot._action_groups["arm_right"]["joint_names"].pop()
        with self.assertRaisesRegex(RuntimeError, "arm_right.*8.*7"):
            engine._validate_robot_contract()

    def test_wrong_or_nonfinite_state_is_rejected(self):
        for positions in (np.zeros(7), np.full(8, np.nan)):
            with self.subTest(positions=positions):
                _module, engine, images, joints = self._right_arm_engine()
                joints["follower_arm_left"] = positions
                result = engine.preprocess(images, joints, "place")
                self.assertFalse(result.get("success", True))
                self.assertIn("arm_left", result["message"])

    def test_invalid_right_arm_action_is_rejected(self):
        for action in ({}, {"arm_right": np.zeros((1, 24, 7))},
                       {"arm_right": np.zeros((1, 40, 8))},
                       {"arm_right": np.full((1, 24, 8), np.nan)},
                       {"arm_right": np.zeros((2, 24, 8))}):
            with self.subTest(action=action):
                _module, engine, _images, _joints = self._right_arm_engine()
                self.assertFalse(engine.postprocess_action(action)["success"])

    def test_readiness_requires_only_model_sensors(self):
        _module, engine, images, joints = self._right_arm_engine()
        engine.robot.get_images = mock.Mock(return_value=images)
        engine.robot.get_joint_positions = mock.Mock(return_value=joints)
        engine.robot.wait_for_ready = mock.Mock(return_value=False)
        engine._wait_for_model_observation(timeout=0.01)
        engine.robot.wait_for_ready.assert_not_called()

    def test_missing_model_sensor_is_a_load_failure(self):
        _module, engine, images, joints = self._right_arm_engine()
        del images["cam_left_wrist"]
        engine.robot.get_images = lambda **_kwargs: images
        engine.robot.get_joint_positions = lambda: joints
        with self.assertRaisesRegex(RuntimeError, "cam_left_wrist"):
            engine._wait_for_model_observation(timeout=0)

    def test_dit_verification_rejects_nonfinite_or_divergent_output(self):
        module, _engine, _images, _joints = self._right_arm_engine()
        reference = np.array([[1., 2., 3.]], dtype=np.float32)
        result = module.validate_dit_outputs(reference, reference.copy())
        self.assertAlmostEqual(result["cosine_similarity"], 1.)
        for output in (-reference, np.full_like(reference, np.inf), reference[:, :2]):
            with self.subTest(shape=output.shape):
                with self.assertRaises(RuntimeError):
                    module.validate_dit_outputs(reference, output)

    def test_failed_dit_verification_preserves_previous_engine(self):
        module, _engine, _images, _joints = self._right_arm_engine()

        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)
                self.shape = self.values.shape

            def to(self, *_args): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.values

        capture = types.SimpleNamespace(
            captured=True, hook_fn=lambda *_args: None,
            sa_embs=Tensor([[1., 2.]]), vl_embs=Tensor([[1., 2.]]),
            timestep=Tensor([1]), image_mask=None, backbone_attention_mask=None,
        )
        hook = mock.Mock()
        model = mock.Mock(return_value=Tensor([[1., 2.]]))
        model.register_forward_pre_hook.return_value = hook
        policy = types.SimpleNamespace(
            model=types.SimpleNamespace(action_head=types.SimpleNamespace(model=model)),
            get_action=lambda _obs: None,
        )
        module.torch.inference_mode = nullcontext
        module.torch.onnx = types.SimpleNamespace(export=mock.Mock())
        _install_stub(
            "scripts.deployment.build_tensorrt_engine",
            build_engine=lambda **kwargs: Path(kwargs['engine_path']).write_bytes(b'new'),
            derive_shapes_with_hint=lambda *_args, **_kwargs: ({}, {}, {}),
        )
        module.TensorRTDiTWrapper = (
            lambda _path: lambda **_kwargs: Tensor([[-1., -2.]])
        )
        with tempfile.TemporaryDirectory() as tempdir:
            engine_path = Path(tempdir) / 'existing.trt'
            engine_path.write_bytes(b'previous verified engine')
            with mock.patch.object(module, 'DiTInputCapture', return_value=capture):
                with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                    module.build_trt_engine(policy, {}, str(engine_path))
            self.assertEqual(engine_path.read_bytes(), b'previous verified engine')
            self.assertEqual(list(Path(tempdir).iterdir()), [engine_path])
        hook.remove.assert_called_once()

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

    def _load_engine_module(self):
        spec = importlib.util.spec_from_file_location("groot_runtime_engine_extra_test", INFERENCE_ENGINE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module




    def test_ordinary_inference_uses_original_sensor_api(self):
        module = self._load_engine_module()
        engine = module.create_engine()
        from unittest.mock import Mock
        engine.robot = types.SimpleNamespace(get_images=Mock(return_value={"cam": "rgb"}),
                                             get_joint_positions=Mock(return_value={"arm": "state"}))
        engine.policy = types.SimpleNamespace(get_action=Mock(return_value=({"arm": "actions"}, {})))
        engine.preprocess = Mock(return_value={"observation": True})
        engine.postprocess_action = Mock(return_value={"success": True})
        before = time.monotonic()
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction="pick"))
        after = time.monotonic()
        self.assertTrue(result["success"])
        self.assertLessEqual(before, result['timing']['sensor_snapshot_monotonic_s'])
        self.assertLessEqual(result['timing']['sensor_snapshot_monotonic_s'], after)
        self.assertEqual(result['timing']['timestamp_source'], 'cache_read_fallback')
        self.assertNotIn('sensor_age_s', result['timing'])
        engine.robot.get_images.assert_called_once_with(format="rgb")
        engine.preprocess.assert_called_once_with({"cam": "rgb"}, {"arm": "state"}, "pick")
        engine.policy.get_action.assert_called_once_with({"observation": True})

    def _atomic_snapshot_engine(self):
        module, engine, images, joints = self._right_arm_engine()
        # Check the SDK receives robot camera sources, including aliases.
        engine.robot_info['camera_sources']['cam_left_head'] = 'head_rgb'
        images['head_rgb'] = images.pop('cam_left_head')
        now = time.monotonic()
        received = {**{'image:' + name: now - .1 for name in images},
                    **{'joint:' + name: now - .05 for name in joints}}
        snapshot = {
            'images': images, 'joints': joints,
            'timing': {
                'sensor_snapshot_monotonic_s': now - .05,
                'snapshot_read_monotonic_s': now,
                'sensor_received_monotonic_s': received,
                'sensor_age_s': {name: now - stamp for name, stamp in received.items()},
                'sensor_skew_s': .05,
                'timestamp_source': 'local_monotonic_receive',
            },
        }
        engine.robot.get_observation_snapshot = mock.Mock(return_value=snapshot)
        engine.robot.get_images = mock.Mock(side_effect=AssertionError('must use atomic snapshot'))
        engine.robot.get_joint_positions = mock.Mock(side_effect=AssertionError('must use atomic snapshot'))
        engine.policy.get_action = mock.Mock(return_value=({'arm_right': np.zeros((1, 24, 8))}, {}))
        return module, engine, snapshot

    def test_atomic_snapshot_uses_selected_sensors_and_receive_time_before_request(self):
        _module, engine, snapshot = self._atomic_snapshot_engine()
        request_started = time.monotonic()
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction='pick'))
        self.assertTrue(result['success'], result)
        self.assertLess(result['timing']['sensor_snapshot_monotonic_s'], request_started)
        self.assertEqual(result['timing']['timestamp_source'], 'local_monotonic_receive')
        self.assertEqual(result['timing']['sensor_age_s'], snapshot['timing']['sensor_age_s'])
        self.assertEqual(result['timing']['sensor_skew_s'], .05)
        engine.robot.get_observation_snapshot.assert_called_once_with(
            format='rgb', camera_names=['head_rgb', 'cam_left_wrist', 'cam_right_wrist'],
            joint_groups=['follower_arm_left', 'follower_arm_right'])
        engine.robot.get_images.assert_not_called()
        engine.robot.get_joint_positions.assert_not_called()
        observed = engine.policy.get_action.call_args.args[0]
        np.testing.assert_array_equal(
            observed['video']['cam_left_head'][0, 0], snapshot['images']['head_rgb'])

    def test_stale_atomic_sensor_snapshot_is_rejected_before_inference(self):
        _module, engine, snapshot = self._atomic_snapshot_engine()
        sensor = 'image:cam_left_wrist'
        snapshot['timing']['sensor_received_monotonic_s'][sensor] -= 2
        snapshot['timing']['sensor_age_s'][sensor] += 2
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction='pick'))
        self.assertFalse(result['success'])
        self.assertIn('Stale sensor observation: image:cam_left_wrist', result['message'])
        self.assertEqual(result['timing']['sensor_age_s'][sensor], snapshot['timing']['sensor_age_s'][sensor])
        engine.policy.get_action.assert_not_called()

    def test_atomic_snapshot_errors_do_not_fall_back_to_unchecked_cached_data(self):
        _module, engine, _snapshot = self._atomic_snapshot_engine()
        engine.robot.get_observation_snapshot.side_effect = RuntimeError('Missing joint feedback')
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction='pick'))
        self.assertFalse(result['success'])
        self.assertIn('Missing joint feedback', result['message'])
        engine.robot.get_images.assert_not_called()
        engine.policy.get_action.assert_not_called()

    def test_invalid_atomic_sensor_timestamps_are_rejected_before_inference(self):
        for invalid in (None, True, float('nan'), float('inf'), -1):
            with self.subTest(invalid=invalid):
                _module, engine, snapshot = self._atomic_snapshot_engine()
                snapshot['timing']['sensor_received_monotonic_s']['joint:follower_arm_left'] = invalid
                result = engine.get_action_chunk(types.SimpleNamespace(task_instruction='pick'))
                self.assertFalse(result['success'])
                self.assertIn('Invalid sensor observation timestamp', result['message'])
                engine.policy.get_action.assert_not_called()

    def test_sensor_age_and_optional_skew_thresholds_are_configurable(self):
        with mock.patch.dict('os.environ', {'GROOT_SENSOR_MAX_AGE_S': '3', 'GROOT_SENSOR_MAX_SKEW_S': '.01'}):
            _module, engine, _snapshot = self._atomic_snapshot_engine()
        self.assertEqual(engine._sensor_max_age_s, 3)
        result = engine.get_action_chunk(types.SimpleNamespace(task_instruction='pick'))
        self.assertFalse(result['success'])
        self.assertIn('GROOT_SENSOR_MAX_SKEW_S=0.01', result['message'])
        engine.policy.get_action.assert_not_called()
        for invalid in ('nan', 'inf', '0', '-1', 'not-a-number'):
            with self.subTest(invalid=invalid), mock.patch.dict('os.environ', {
                    'GROOT_SENSOR_MAX_AGE_S': invalid, 'GROOT_SENSOR_MAX_SKEW_S': invalid}):
                engine = self._load_engine_module().create_engine()
                self.assertEqual(engine._sensor_max_age_s, 1)
                self.assertIsNone(engine._sensor_max_skew_s)


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

"""Parity with the prior numerical path, without loading policy weights."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sdk" / "action_chunk_processing"))

import numpy as np
import pytest
import torch
import yaml

from test_preprocessing import Preprocessor, image_preprocessing
from lerobot_engine.input_pipeline import InputPipelineConfig, CONFIG_DIR


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
@pytest.mark.parametrize("rotation", [0, 270])
def test_all_model_defaults_match_previous_numerics(path, rotation):
    config = yaml.safe_load(path.read_text())
    key = "observation.images.head"
    features = {key: {"shape": [3, 8, 8]}, "observation.state": {"shape": [2]}}
    engine = Preprocessor([1., 2.], 2)
    engine._cameras = {"head": key}
    engine._robot._config = {"cameras": {"head": {"rotation_deg": rotation}}}
    image = np.random.default_rng(5).integers(0, 256, (12, 20, 3), dtype=np.uint8)
    engine._robot.get_images = lambda format: {"head": image}
    engine._input_pipeline_config = InputPipelineConfig.from_user_config(config, {"input_features": features}, path)
    # Frozen pre-migration defaults, deliberately independent of the new YAML.
    prior = {"backend": "torch", "operations": [{"type": "identity"}]}
    if path.stem == "diffusion":
        prior = {"backend": "opencv", "operations": [
            {"type": "resize", "size": "checkpoint", "interpolation": "bilinear"}]}
    elif path.stem == "multi_task_dit":
        prior = {"backend": "torch", "operations": [
            {"type": "resize", "size": [224, 224], "interpolation": "bilinear", "antialias": True}]}
    previous = image_preprocessing.ImagePreprocessing(prior, features)
    batch = engine._build_observation("pick")
    torch.testing.assert_close(batch[key], previous.apply(image, key, rotation), rtol=0, atol=0)
    torch.testing.assert_close(batch["observation.state"], torch.tensor([[1., 2.]]))
    assert batch["task"] == ["pick"]
    assert engine._input_evaluation.run("after", batch) == batch


def test_yaml_edit_only_applies_to_new_load(tmp_path):
    from lerobot_engine.input_pipeline import load_input_pipeline
    path = tmp_path / "act.yaml"
    path.write_text((CONFIG_DIR / "act.yaml").read_text())
    (tmp_path / "config.json").write_text(json.dumps({"type": "act", "input_features": {}}))
    first = load_input_pipeline(tmp_path, tmp_path)
    config = yaml.safe_load(path.read_text())
    config["preprocessing"] = {"images": [{"resize": {"size": [10, 10], "backend": "torch", "interpolation": "area"}}]}
    path.write_text(yaml.safe_dump(config))
    second = load_input_pipeline(tmp_path, tmp_path)
    assert first.config["nodes"]["prepared"]["options"]["images"] == "identity"
    assert second.config["nodes"]["prepared"]["options"]["images"][0]["resize"]["size"] == [10, 10]


def test_real_feature_encoder_worker_feedback_and_public_action_memory():
    from types import SimpleNamespace
    from unittest import mock
    import time
    from test_fastwam_runtime import engine_module
    from engine_process.worker import EngineWorker
    from engine_process.protocol import EngineCommandRequest, CMD_LOAD_POLICY, CMD_GET_ACTION, CMD_UNLOAD_POLICY, CMD_UPDATE_CONTEXT
    from inference_context.contract import LoadedExecution
    from main_runtime.execution_feedback import ExecutionFeedback

    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 4), torch.nn.Tanh()).eval()
    encoded = []
    def extensions(registry, bindings, engine):
        def compile_encoder(options, context):
            def run(values):
                assert not torch.is_grad_enabled()
                image = torch.from_numpy(values[0].value.copy()).float().reshape(1, 12)
                feature = encoder(image)
                encoded.append(feature.clone())
                return values[0].derived(feature)
            return run
        registry.register("tiny_encoder", compile_encoder, cacheable=True)

    recipe = {"sources": {"image": {"source": "camera:head"}, "instruction": {"source": "instruction"}},
              "memory": {"max_bytes": 4096, "slots": {"feature": {"event": "plan_terminal"}, "action": {}}},
              "execution": {"request_after": {"event": "plan_terminal"}},
              "nodes": {
                  "encoded": {"op": "tiny_encoder", "inputs": ["image"], "cache": {"dependencies": ["image", "instruction"]}},
                  "previous": {"op": "memory_read", "inputs": ["encoded"], "options": {"slot": "feature", "initial": "input"}},
                  "joined": {"op": "tensor_concat", "inputs": ["previous", "encoded"], "options": {"axis": 1}},
                  "save_feature": {"op": "memory_write", "inputs": ["encoded"], "options": {"slot": "feature"}},
                  "save_action": {"op": "memory_write", "stage": "result", "inputs": ["postprocessed_action"], "options": {"slot": "action"}},
              }, "outputs": {"before": {"features": "joined"}, "after": {"*": "processed"}}}
    def feature_memory(graph, options):
        assert options == {"combine": "concat"}
        return recipe
    graph_config = InputPipelineConfig.from_user_config(
        yaml.safe_load("preprocessing:\n  custom:\n    handler: feature_memory\n    options: {combine: concat}\n"),
        {}, Path("test.yaml"), handlers={"feature_memory": feature_memory})
    seen = []
    policy = SimpleNamespace(config=SimpleNamespace(type="act"), reset=lambda: None)
    def predict(batch):
        seen.append(batch["features"].clone())
        return batch["features"][:, :2].unsqueeze(1).repeat(1, 2, 1)
    policy.predict_action_chunk = predict
    engine = engine_module.LeRobotEngine()
    engine._resolve_model_dir = lambda path: path
    engine._load_policy_assets = lambda *a: (policy, lambda x: x, lambda x: x * 2)
    class Robot:
        _config = {}
        value = 1
        def get_input_snapshot(self):
            return {"images": {"head": np.full((2, 2, 3), self.value, np.uint8)},
                    "joint_positions": {}, "sensors": {}, "reception_monotonic_timestamps": {"camera:head": time.monotonic()}}
        def close(self):
            pass
    robot = Robot()
    def init_robot(_):
        engine._robot = robot
        engine._cameras = {"head": "image"}
        engine._state_modalities = []
        engine._action_keys = ["arm"]
    engine._init_robot = init_robot
    definition = engine_module.resolve_adapter("act").__class__(input_extensions=extensions)
    worker = EngineWorker(engine)
    with mock.patch.object(engine_module, "load_input_pipeline", return_value=graph_config), \
            mock.patch.object(engine_module, "resolve_adapter", return_value=definition), \
            mock.patch.object(torch.cuda, "is_available", return_value=False):
        try:
            response = worker.handle(EngineCommandRequest(command=CMD_LOAD_POLICY, model_path="/test", robot_type="test", seq_id=1))
            assert response.success, response.message
            loaded = LoadedExecution.from_json(response.capabilities_json)
            assert loaded.contract.request_after == "plan_terminal" and loaded.context.feedback_schema == 2
            ledger = ExecutionFeedback(loaded.context, postprocess=False)
            ledger.phase = "running"
            for sequence in (2, 3):
                context = ledger.project(ledger.capture())
                response = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=sequence,
                                  task_instruction="pick", execution_context_json=context.to_json()))
                assert response.success, response.message
                assert not encoder[1].weight.grad
                ledger.acknowledge(context)
                actions = np.asarray(response.action_list).reshape(response.chunk_size, response.action_dim)
                ledger.buffer.enqueue(sequence, actions)
                while ledger.buffer.buffer_size:
                    command = ledger.buffer.take()
                    ledger.buffer.finish(command.command_id, status="published", emitted_values=command.values)
                robot.value += 1
            torch.testing.assert_close(seen[0][:, :4], seen[0][:, 4:])
            torch.testing.assert_close(seen[1][:, :4], encoded[0])
            torch.testing.assert_close(seen[1][:, 4:], encoded[1])
            assert graph_config.memory.read("action").semantics == (("space", "postprocessed_action"),)
            ledger.reset("stop", "paused")
            context = ledger.project(ledger.capture())
            assert worker.handle(EngineCommandRequest(command=CMD_UPDATE_CONTEXT, seq_id=4,
                                 execution_context_json=context.to_json())).success
            assert graph_config.budget.used == 0
            assert worker.handle(EngineCommandRequest(command=CMD_UNLOAD_POLICY, seq_id=5)).success
        finally:
            engine.cleanup()

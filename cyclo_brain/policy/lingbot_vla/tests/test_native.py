"""Real upstream preprocessing/unprocessing; tiny tokenizer and action-model doubles."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from lingbot_vla_engine.bundle import CheckpointBundle
from lingbot_vla_engine.native import reset_native


def test_native_chunk_path_restores_relative_actions_and_resizes_each_camera(bundle):
    if importlib.util.find_spec("lingbotvla") is None:
        pytest.skip("Run in the LingBot-VLA image to exercise upstream transforms")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Upstream imports query GPU capability; use a --gpus test container")
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.data.vla_data.utils import FeatureTransform

    bundle = CheckpointBundle(bundle)
    data = SimpleNamespace(**bundle.training["data"], img_size=8)
    config = SimpleNamespace(**bundle.training["train"], tokenizer_max_length=8)
    seen_images = []

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert "pick" in prompts[0]
            return {"input_ids": torch.ones((1, 8), dtype=torch.long),
                    "attention_mask": torch.ones((1, 8), dtype=torch.bool)}

    def image_processor(image):
        seen_images.append(image.clone())
        return {"pixel_values": image.float() / 255}

    processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=image_processor)
    transform = FeatureTransform(str(bundle.robot_config), data, config, processor,
                                 chunk_size=2, norm_stats_path=str(bundle.norm_stats))

    def sample(batch, *args, **kwargs):
        assert batch["images"].shape == (1, 3, 3, 8, 8)
        torch.testing.assert_close(batch["state"], torch.tensor([[2., 1., 0., 0.]]))
        return torch.ones((1, 2, 4))

    # Avoid model allocation/download/CUDA while exercising the official infer path.
    server = LingbotVLAv2Server.__new__(LingbotVLAv2Server)
    server.vla = SimpleNamespace(feature_transform=transform, sample_actions_batch=sample)
    server.data_config = data
    server.config = config
    server.use_bf16 = server.use_compile = False
    server.chunk_ret, server.use_length = True, -1
    server.sample_actions_fn = None
    server.action_key = transform.org_features["actions"]
    reset_native(server)
    observation = {"observation.state": np.array([2., 1.], dtype=np.float32), "task": "pick"}
    for i, camera in enumerate(bundle.metadata["cameras"]):
        observation[camera] = np.full((6 + i, 10 - i, 3), 30 * (i + 1), dtype=np.uint8)
    for _ in range(2):
        actions = server.infer(observation)
        np.testing.assert_allclose(actions["action"], [[3., 2.], [3., 2.]], atol=1e-5)
    assert len(seen_images) == 6
    np.testing.assert_allclose(sorted(float(image.mean()) for image in seen_images),
                               [30., 30., 60., 60., 90., 90.], atol=1e-4)
    assert observation[bundle.metadata["cameras"][0]].shape == (6, 10, 3)
    np.testing.assert_array_equal(observation["observation.state"], [2., 1.])

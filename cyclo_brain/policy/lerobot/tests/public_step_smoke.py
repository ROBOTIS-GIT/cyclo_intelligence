"""Run in a pinned Worker with no downloads or model allocation.

docker exec -i -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 lerobot_server \
    /lerobot/.venv/bin/python - < cyclo_brain/policy/lerobot/tests/public_step_smoke.py

Public queue/reset logic is real. Neural generation, encoders and KV computation
are mocked. This is not a checkpoint, performance or robot-execution test.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch


def check_multi_task_dit():
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

    policy = MultiTaskDiTPolicy.__new__(MultiTaskDiTPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_obs_steps=2, n_action_steps=3, image_features={})
    policy.reset()
    policy._generate_actions = Mock(return_value=torch.arange(6, dtype=torch.float32).reshape(1, 3, 2))
    outputs = [policy.select_action({"observation.state": torch.full((1, 2), float(i))}) for i in range(4)]
    assert policy._generate_actions.call_count == 2
    histories = [call.args[0]["observation.state"][0, :, 0].tolist()
                 for call in policy._generate_actions.call_args_list]
    assert histories == [[0., 0.], [2., 3.]]
    assert [a.tolist() for a in outputs] == [[[0., 1.]], [[2., 3.]], [[4., 5.]], [[0., 1.]]]
    policy.reset()
    assert all(len(q) == 0 for q in policy._queues.values())
    print("Multi-Task DiT: public history/action queues and reset PASS")


def check_lingbot_va():
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy

    policy = LingBotVAPolicy.__new__(LingBotVAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = LingBotVAConfig(device="cpu")
    policy._frozen = {}
    policy.reset()
    policy._ensure_frozen_modules = Mock()
    policy._maybe_init_prompt = Mock()
    policy._extract_raw_obs = lambda batch: batch.copy()
    policy._encode_frames = Mock(return_value=torch.zeros(1))
    policy._init_streaming_cache = Mock()
    policy._compute_kv_cache = Mock()
    actions = torch.arange(30 * 4 * 4, dtype=torch.float32).reshape(1, 30, 4, 4, 1)
    policy._infer = Mock(return_value=(actions, torch.zeros(1)))
    # Frame zero is conditioning, so the initial chunk returns 12, not 16 steps.
    outputs = [policy.select_action({"frame": i}) for i in range(13)]
    assert all(action.shape == (1, 7) for action in outputs)
    assert policy._infer.call_count == 2
    policy._compute_kv_cache.assert_called_once()
    frames, previous_actions = policy._compute_kv_cache.call_args.args
    assert [frame["frame"] for frame in frames] == list(range(1, 13))
    assert previous_actions is actions
    policy.reset()
    assert not policy._action_queue and not policy._obs_buffer
    assert policy._executed_actions is None and not policy._started
    print("LingBot-VA: public keyframe/action feedback sequence and reset PASS")


if __name__ == "__main__":
    check_multi_task_dit()
    check_lingbot_va()

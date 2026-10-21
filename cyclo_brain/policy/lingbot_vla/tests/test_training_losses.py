"""Boundary masking must preserve native objectives on valid samples."""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingbot_vla_training.losses import MaskedTarget, install_boundary_losses, prepare_loss_inputs


class Flow:
    use_future_video_cls = False

    def _emb_loss(self, predictions, target):
        return F.smooth_l1_loss(predictions.float(), target.float().detach())

    def _video_emb_loss(self, predictions, target, metric_prefix="future_video"):
        loss = F.mse_loss(predictions, target.detach())
        return loss, {metric_prefix: loss.detach()}


@pytest.mark.parametrize("valid", [[True, True], [True, False], [False, False]])
@pytest.mark.parametrize("method", ["_emb_loss", "_video_emb_loss"])
def test_masked_gradients_and_native_parity(valid, method):
    flow = Flow()
    predictions = torch.ones(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(predictions)
    expected = getattr(flow, method)(predictions, target)
    install_boundary_losses(flow)
    result = getattr(flow, method)(predictions, MaskedTarget(target, torch.tensor(valid)))
    loss = result[0] if isinstance(result, tuple) else result
    baseline = expected[0] if isinstance(expected, tuple) else expected
    assert loss == (baseline if any(valid) else 0)
    loss.backward()
    for index, active in enumerate(valid):
        assert bool(predictions.grad[index].abs().sum() > 0) == active


def test_current_targets_are_not_masked_and_double_install_is_rejected():
    flow = Flow()
    install_boundary_losses(flow)
    predictions = torch.ones(2, 3, 4, requires_grad=True)
    loss, metrics = flow._video_emb_loss(predictions, torch.zeros_like(predictions), "current_video")
    assert metrics == {"current_video": 1}
    loss.backward()
    assert predictions.grad.ne(0).all()
    with pytest.raises(ValueError, match="already"):
        install_boundary_losses(flow)


def inputs():
    batch = {
        "actions": torch.zeros(2, 3, 4),
        "joint_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, False], [False, True, True]]),
        "future_valid": torch.tensor([True, False]),
    }
    targets = torch.zeros(2, 4, 5)
    return batch, dict(depth=targets, future_depth=targets, video={"patch": targets, "current_patch": targets})


def test_action_time_and_channel_masks_are_combined_without_mutation():
    batch, teachers = inputs()
    batch["joint_mask"][..., -1] = False
    result = prepare_loss_inputs(batch, **teachers)
    assert result["joint_mask"][0].sum() == 9
    assert result["joint_mask"][1].sum() == 3
    assert batch["joint_mask"][1].sum() == 9
    assert result["future_depth_targets"].valid.tolist() == [True, False]
    assert result["future_video_current_patch"] is teachers["video"]["current_patch"]


@pytest.mark.parametrize("failure", ["mask_type", "all_pad", "grad", "nan", "video_missing"])
def test_invalid_loss_inputs_fail(failure):
    batch, teachers = inputs()
    if failure == "mask_type":
        batch["action_is_pad"] = batch["action_is_pad"].float()
    elif failure == "all_pad":
        batch["action_is_pad"][0] = True
    elif failure == "grad":
        teachers["depth"] = teachers["depth"].requires_grad_()
    elif failure == "nan":
        teachers["depth"][0, 0, 0] = float("nan")
    else:
        del teachers["video"]["current_patch"]
    with pytest.raises(ValueError):
        prepare_loss_inputs(batch, **teachers)

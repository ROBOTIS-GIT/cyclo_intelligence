"""TT-RTC DiT export/installation; ordinary GR00T TensorRT stays unchanged."""

import torch


class _DiTExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask):
        return self.model(
            hidden_states=sa_embs, encoder_hidden_states=vl_embs,
            timestep=timestep, image_mask=image_mask,
            encoder_attention_mask=backbone_attention_mask,
            backbone_attention_mask=backbone_attention_mask,
            return_all_hidden_states=False,
        )


def export_tt_rtc_dit(policy, captured_inputs, output_path, use_bf16=True):
    from scripts.deployment.export_onnx_n1d7 import (
        _consolidate_external_data, verify_onnx_export,
    )

    capture = captured_inputs
    if capture.timestep.shape != capture.sa_embs.shape[:2]:
        raise ValueError("TT-RTC export requires one timestep per state/action token")
    names = ["sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"]
    values = [getattr(capture, name) for name in names]
    if any(value is None for value in values):
        raise ValueError("TT-RTC N1.7 export requires both vision/language masks")
    inputs = tuple(value.cuda() for value in values)
    wrapper = _DiTExport(policy.model.action_head.model).eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, inputs, output_path, input_names=names, output_names=["output"],
            opset_version=19, dynamo=False, export_params=True,
            dynamic_axes={name: {1: "vl_seq_len"}
                          for name in ("vl_embs", "image_mask", "backbone_attention_mask")},
        )
    _consolidate_external_data(output_path)
    verify_onnx_export(output_path)


def install_tt_rtc_dit(policy, engine_path):
    from scripts.deployment.standalone_inference_script import TensorRTDiTWrapper

    runner = TensorRTDiTWrapper(engine_path)
    timestep_shape = tuple(runner.engine.get_tensor_shape("timestep"))
    sa_shape = tuple(runner.engine.get_tensor_shape("sa_embs"))
    if len(timestep_shape) != 2 or timestep_shape != sa_shape[:2]:
        raise ValueError("Not a TT-RTC TensorRT engine: expected per-token timestep input")

    def forward(hidden_states, encoder_hidden_states, timestep,
                encoder_attention_mask=None, return_all_hidden_states=False,
                image_mask=None, backbone_attention_mask=None):
        if return_all_hidden_states:
            raise ValueError("TT-RTC TensorRT exports only the final DiT output")
        if tuple(timestep.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError("TT-RTC TensorRT requires per-token timesteps")
        return runner(hidden_states, encoder_hidden_states, timestep,
                      image_mask=image_mask, backbone_attention_mask=backbone_attention_mask)

    policy.model.action_head.model.forward = forward

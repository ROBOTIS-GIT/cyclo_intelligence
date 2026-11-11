"""Construct upstream modules without downloading redundant VLM weights."""

from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from ABot.model.framework.ABot_M0 import ABot_M0
from ABot.model.modules.action_model.AML_ActionHeader import get_action_model
from ABot.model.modules.vlm.QWen3 import _QWen3_VL_Interface

from .initialization import load_pretrained


class TrainingQwen(_QWen3_VL_Interface):
    def __init__(self, config, qwen_config, processor_path):
        torch.nn.Module.__init__(self)
        self.config = config
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration._from_config(
            Qwen3VLConfig.from_dict(qwen_config), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.config.hidden_size = self.model.config.text_config.hidden_size


class TrainingPolicy(ABot_M0):
    """Only module construction differs; forward and predict_action are upstream."""

    def __init__(self, model_config, processor_path):
        torch.nn.Module.__init__(self)
        self.config = OmegaConf.create(deepcopy(model_config))
        if self.config.framework.name != "ABot_M0" or self.config.framework.use_vggt is not False:
            raise ValueError("This recipe requires ABot_M0 without VGGT")
        self.qwen_vl_interface = TrainingQwen(
            self.config, model_config["cyclo_qwen_config"], str(processor_path),
        )
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.text_config.hidden_size
        )
        self.action_model = get_action_model(self.config).to(dtype=torch.bfloat16)
        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.use_vggt = False
        self.spatial_model = self.spatial_projector = self.fuser = None

    def saved_config(self):
        config = OmegaConf.to_container(self.config, resolve=True)
        config["cyclo_qwen_config"] = self.qwen_vl_interface.model.config.to_dict()
        return config


def initialize(model_config, processor_path, checkpoint):
    model = TrainingPolicy(model_config, processor_path)
    weights = torch.load(Path(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    report = load_pretrained(model, weights, restore_vocabulary=True)
    del weights
    return model, report


def reload_model(directory, model_config, receipt):
    directory = Path(directory)
    model = TrainingPolicy(model_config, directory / "processor")
    weights = torch.load(directory / receipt["weights"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(weights, strict=True)
    del weights
    return model

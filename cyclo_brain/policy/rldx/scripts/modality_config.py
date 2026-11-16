"""RLDX registration using the prepared dataset, without robot-specific names."""

import json
import os
from pathlib import Path

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

metadata = json.loads((Path(os.environ["RLDX_DATASET_PATH"]) / "meta/modality.json").read_text())
register_modality_config({
    "video": ModalityConfig(delta_indices=[0], modality_keys=list(metadata["video"])),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_position"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)), modality_keys=["joint_position"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                     type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
}, EmbodimentTag.GENERAL_EMBODIMENT)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import FINETUNE_ONLY_TAGS, POSTTRAIN_TAGS, EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _sim_language_batch_to_sequence(value: Any) -> Any:
    """Normalize sim language batches while preserving validation semantics."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, str):
        return [value]
    return value


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type.
                Accepts an EmbodimentTag enum or a string (resolved case-insensitively).
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation.
        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root.  Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            # Map raw checkpoint tag values to user-friendly enum names where possible.
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
                else:
                    supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model. "
                    f"See the example READMEs for how to finetune and download checkpoints."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots. "
                    f"Use it with launch_finetune.py, not with the base model directly."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Check that the expected state key exists in the observation
            # (must happen before indexing — see video validation above)
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Check that the expected language key exists in the observation
            # (must happen before indexing — see video validation above)
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional standard inference parameters.

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}


    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}

    @property
    def training_time_rtc_enabled(self) -> bool:
        """Whether the loaded model config declares TT-RTC training support."""

        return bool(getattr(self.model.config, "training_time_rtc_enabled", False))

    @property
    def training_time_rtc_max_delay_steps(self) -> int:
        """Maximum trained clean-prefix length declared by the model config."""

        value = getattr(self.model.config, "training_time_rtc_max_delay_steps", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                "Invalid model.config.training_time_rtc_max_delay_steps: "
                f"{value!r}"
            )
        return value

    def get_tt_rtc_model_contract(self) -> dict[str, int | float | bool]:
        """Expose loaded config/dimensions for manifest cross-validation."""

        action_hz = getattr(self.model.config, "training_time_rtc_action_hz", 15.0)
        if isinstance(action_hz, bool) or not isinstance(action_hz, (int, float)):
            raise RuntimeError(
                f"Invalid model.config.training_time_rtc_action_hz: {action_hz!r}"
            )
        return {
            "enabled": self.training_time_rtc_enabled,
            "max_delay_steps": self.training_time_rtc_max_delay_steps,
            "action_hz": float(action_hz),
            "action_horizon": len(self.modality_configs["action"].delta_indices),
            "action_dimension": sum(
                dimension for _key, dimension in self._action_group_dimensions()
            ),
            "model_action_horizon": int(self.model.config.action_horizon),
            "model_action_dimension": int(self.model.config.max_action_dim),
            "processor_action_horizon": int(self.processor.max_action_horizon),
            "processor_action_dimension": int(self.processor.max_action_dim),
            "num_inference_timesteps": int(self.model.config.num_inference_timesteps),
            "num_timestep_buckets": int(self.model.config.num_timestep_buckets),
        }

    def _action_group_dimensions(self) -> list[tuple[str, int]]:
        """Return physical action groups in checkpoint serialization order."""

        processor = self.processor.state_action_processor
        tag = self.embodiment_tag.value
        groups = []
        for key in self.modality_configs["action"].modality_keys:
            params = processor.norm_params[tag]["action"][key]
            dim = params.get("dim")
            if dim is None:
                for statistic_name in ("mean", "min", "max"):
                    statistic = params.get(statistic_name)
                    if statistic is not None:
                        dim = np.asarray(statistic).size
                        break
            try:
                value = int(np.asarray(dim).item())
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Cannot determine action dimension for modality {key!r}"
                ) from error
            if value <= 0:
                raise RuntimeError(f"Invalid action dimension for modality {key!r}: {value}")
            groups.append((key, value))
        return groups

    def _normalize_tt_rtc_prefix(
        self,
        committed_action_prefix: np.ndarray,
        raw_states: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """Normalize a physical prefix and pad it to GR00T's model action width."""

        prefix = np.asarray(committed_action_prefix, dtype=np.float32)
        if prefix.ndim != 3:
            raise ValueError(
                "TT-RTC committed_action_prefix must have shape (B, d, A), got "
                f"{prefix.shape}"
            )
        if prefix.shape[0] != len(raw_states):
            raise ValueError("TT-RTC prefix batch size does not match observation batch size")
        if not np.isfinite(prefix).all():
            raise ValueError("TT-RTC committed_action_prefix must be finite")

        groups = self._action_group_dimensions()
        logical_action_dim = sum(dim for _key, dim in groups)
        if prefix.shape[2] != logical_action_dim:
            raise ValueError(
                "TT-RTC physical action dimension mismatch: expected "
                f"{logical_action_dim}, got {prefix.shape[2]}"
            )
        model_action_dim = int(self.model.config.max_action_dim)
        if logical_action_dim > model_action_dim:
            raise RuntimeError("TT-RTC logical action exceeds GR00T model action width")

        normalized_batches = []
        for batch_index, state in enumerate(raw_states):
            offset = 0
            physical_groups = {}
            for key, dim in groups:
                physical_groups[key] = prefix[
                    batch_index,
                    :,
                    offset : offset + dim,
                ]
                offset += dim
            if prefix.shape[1] == 0:
                normalized = np.zeros((0, logical_action_dim), dtype=np.float32)
            else:
                normalized_groups = self.processor.state_action_processor.apply_action(
                    physical_groups,
                    self.embodiment_tag.value,
                    state=state,
                    clip_outliers=False,
                )
                normalized = np.concatenate(
                    [normalized_groups[key] for key, _dim in groups],
                    axis=-1,
                ).astype(np.float32, copy=False)
            padded = np.zeros((prefix.shape[1], model_action_dim), dtype=np.float32)
            padded[:, :logical_action_dim] = normalized
            normalized_batches.append(padded)
        return torch.from_numpy(np.stack(normalized_batches, axis=0))

    @torch.inference_mode()
    def get_action_tt_rtc(
        self,
        observation: dict[str, Any],
        *,
        committed_action_prefix: np.ndarray,
        delay_steps: int,
        action_horizon: int,
        temporal_overlap_steps: int = 0,
        return_rlt_context: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a TT-RTC chunk from a physical committed action prefix.

        This entry point deliberately bypasses the legacy inference-time RTC
        ``options`` contract.  It normalizes the physical prefix with the same
        checkpoint processor used for training and exposes an optional RLT
        context from this exact backbone/action-head forward.
        """

        if self.strict:
            self.check_observation(observation)
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int):
            raise ValueError("TT-RTC delay_steps must be an integer")
        logical_horizon = len(self.modality_configs["action"].delta_indices)
        if action_horizon != logical_horizon:
            raise ValueError(
                f"TT-RTC action_horizon must be {logical_horizon}, got {action_horizon}"
            )
        if not self.training_time_rtc_enabled:
            raise RuntimeError("GR00T checkpoint is not configured for Training-Time RTC")
        max_delay = self.training_time_rtc_max_delay_steps
        if (
            isinstance(max_delay, bool)
            or not isinstance(max_delay, int)
            or not 0 <= delay_steps <= max_delay
        ):
            raise ValueError(
                f"TT-RTC delay_steps must be in [0, {max_delay}], got {delay_steps}"
            )
        if delay_steps >= logical_horizon:
            raise ValueError("TT-RTC delay must leave a non-empty logical postfix")
        if (
            isinstance(temporal_overlap_steps, bool)
            or not isinstance(temporal_overlap_steps, int)
            or not 0 <= temporal_overlap_steps
            <= logical_horizon - delay_steps
        ):
            raise ValueError(
                "TT-RTC temporal_overlap_steps must fit in the logical postfix"
            )

        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        raw_states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            raw_states.append(vla_step_data.states)
            processed_inputs.append(
                self.processor(
                    [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
                )
            )
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        if "inputs" not in collated_inputs:
            raise RuntimeError("TT-RTC expected collator output containing 'inputs'")

        prefix = np.asarray(committed_action_prefix, dtype=np.float32)
        expected_prefix_shape = (
            len(unbatched_observations),
            delay_steps,
            sum(dim for _key, dim in self._action_group_dimensions()),
        )
        if prefix.shape != expected_prefix_shape:
            raise ValueError(
                "TT-RTC committed_action_prefix must have shape "
                f"{expected_prefix_shape}, got {prefix.shape}"
            )
        normalized_prefix = self._normalize_tt_rtc_prefix(prefix, raw_states)

        backbone_inputs, action_inputs = self.model.prepare_input(collated_inputs["inputs"])
        backbone_output = self.model.backbone(backbone_inputs)
        # RLT Stage 1 consumes the unprojected backbone representation.  Clone
        # it before action_head.process_backbone_output mutates the container.
        raw_tokens = backbone_output["backbone_features"].detach().clone()
        raw_token_valid = backbone_output["backbone_attention_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        raw_image_token = backbone_output["image_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        normalized_state = action_inputs["state"].detach().clone()

        model_pred = self.model.action_head.get_action_tt_rtc(
            backbone_output,
            action_inputs,
            normalized_action_prefix=normalized_prefix,
            delay_steps=delay_steps,
            logical_action_horizon=logical_horizon,
            temporal_overlap_steps=temporal_overlap_steps,
        )
        normalized_action = model_pred["action_pred"].float()

        batched_states = {
            key: np.stack([state[key] for state in raw_states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        # The sampler conditions in the model's compute dtype (normally
        # BF16), while the control queue owns float32 physical commands.  A
        # decode round-trip can therefore move a committed prefix by a few
        # ULPs.  Return the queue-owned physical prefix verbatim: those rows
        # have already executed or are irrevocably committed, and the common
        # runtime requires an exact handoff before accepting the postfix.
        offset = 0
        for key, dimension in self._action_group_dimensions():
            group_action = casted_action.get(key)
            expected_group_shape = (
                prefix.shape[0],
                logical_horizon,
                dimension,
            )
            if group_action is None or tuple(group_action.shape) != expected_group_shape:
                raise RuntimeError(
                    f"TT-RTC decoded action group {key!r} must have shape "
                    f"{expected_group_shape}, got "
                    f"{None if group_action is None else tuple(group_action.shape)}"
                )
            group_action[:, :delay_steps, :] = prefix[
                :, :delay_steps, offset : offset + dimension
            ]
            offset += dimension
        if offset != expected_prefix_shape[2]:
            raise RuntimeError("TT-RTC decoded action groups did not consume the full prefix")
        if self.strict:
            self.check_action(casted_action)

        info: dict[str, Any] = {}
        if return_rlt_context:
            logical_action_dim = expected_prefix_shape[2]
            reference = normalized_action[
                :, :logical_horizon, :logical_action_dim
            ].detach().clone()
            normalized_committed_prefix = model_pred["normalized_action_prefix"][
                :, :, :logical_action_dim
            ].float().detach().clone()
            if not torch.equal(
                reference[:, :delay_steps],
                normalized_committed_prefix.to(
                    device=reference.device,
                    dtype=reference.dtype,
                ),
            ):
                raise RuntimeError("TT-RTC sampler did not preserve the exact clean prefix")
            info["tt_rtc_rlt_context"] = {
                "schema": "cyclo.groot.tt-rtc-rlt-context/v1",
                "delay_steps": delay_steps,
                "tokens": raw_tokens,
                "token_valid": raw_token_valid,
                "image_token": raw_image_token,
                "proprio": normalized_state[
                    :, -1, :logical_action_dim
                ].float().detach().clone(),
                "reference_actions": reference,
                "normalized_committed_prefix": normalized_committed_prefix,
                "batched_states": batched_states,
            }
        return casted_action, info

    def reset_tt_rtc_noise_cache(self) -> None:
        """Reset cached initial noise when trajectory continuity is broken."""

        reset = getattr(self.model.action_head, "reset_tt_rtc_noise_cache", None)
        if callable(reset):
            reset()


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language = _sim_language_batch_to_sequence(observation[language_key])

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple, list, or numpy array. "
                f"Got {type(observation[language_key])}"
            )
            assert batched_language, f"Language key '{language_key}' must not be empty"

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    arr = _sim_language_batch_to_sequence(arr)
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()

#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GR00T N1.7 to PI-RLT inference adapter.

The adapter keeps the upstream GR00T package untouched.  It extracts the raw
Qwen3 token sequence before the action head mutates it, samples the frozen
GR00T reference chunk once, and decodes the RLT Action MLP result with the
same checkpoint processor used by normal GR00T inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from cyclo_brain.algorithm.common import file_sha256

from cyclo_brain.model.common import (
    RLT_ACTION_DIM,
    RLT_ACTION_HORIZON,
)

from .rlt_provenance import (
    build_groot_rlt_provenance,
    validate_stage1_encoder_provenance,
    validate_stage2_bundle_provenance,
)


EXPECTED_CHUNK_LENGTH = RLT_ACTION_HORIZON
EXPECTED_ACTION_DIM = RLT_ACTION_DIM
SIMULATION_ONLY_QUALIFICATION = "training_only_not_deployment_validated"
DEPLOYMENT_QUALIFICATION = "deployment_qualified"


def is_deployment_qualified(qualification: object) -> bool:
    """Fail closed unless the actor carries the exact deployment marker."""

    return str(qualification or "").strip().lower() == DEPLOYMENT_QUALIFICATION


@dataclass(frozen=True)
class RLTBundlePaths:
    root: Path
    encoder: Path
    actor: Path


def _regular_files(root: Path, name: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(name)
        if path.is_file() and not path.is_symlink()
    )


def _manifest_fingerprint(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("artifact", {}).get("artifact_fingerprint")
    return value if isinstance(value, str) else None


def _actor_encoder_fingerprint(actor_root: Path) -> str | None:
    manifests = _regular_files(actor_root, "rlt_stage2.pt.run.json")
    if len(manifests) != 1:
        return None
    try:
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = (
        payload.get("base_contract", {})
        .get("learner", {})
        .get("contract", {})
        .get("spec", {})
        .get("rl_token_artifact_fingerprint")
    )
    return value if isinstance(value, str) else None


def resolve_rlt_bundle(bundle_path: str | os.PathLike[str]) -> RLTBundlePaths:
    """Resolve one Stage-2 actor and its exact Stage-1 encoder.

    A normal UI selection points at a Stage-2 output directory.  Stage-1 and
    Stage-2 are sibling training runs, so the resolver uses their persisted
    artifact fingerprint rather than guessing from directory names.
    """

    text = os.fspath(bundle_path).strip()
    if not text:
        raise ValueError("RLT bundle path is empty")
    root = Path(os.path.abspath(Path(text).expanduser()))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"RLT bundle must be a non-symlink directory: {root}")

    actors = _regular_files(root, "rlt_actor.pt")
    if len(actors) != 1:
        raise ValueError(
            f"RLT bundle must contain exactly one rlt_actor.pt; found {len(actors)}"
        )
    actor = actors[0]
    actor_root = actor.parent.parent if actor.parent.name == "artifacts" else root

    local_encoders = _regular_files(root, "rl_token_encoder.pt")
    search_root = root.parent
    encoders = local_encoders or _regular_files(search_root, "rl_token_encoder.pt")
    if not encoders:
        raise ValueError(
            "RLT bundle has no compatible rl_token_encoder.pt in the bundle "
            "or its sibling training runs"
        )

    expected = _actor_encoder_fingerprint(actor_root)
    if expected:
        matched: list[Path] = []
        for encoder in encoders:
            encoder_root = (
                encoder.parent.parent
                if encoder.parent.name == "artifacts"
                else encoder.parent
            )
            manifests = _regular_files(encoder_root, "rlt_stage1.pt.run.json")
            if len(manifests) == 1 and _manifest_fingerprint(manifests[0]) == expected:
                matched.append(encoder)
        encoders = matched

    unique = sorted(set(encoders))
    if len(unique) != 1:
        raise ValueError(
            "RLT bundle encoder resolution is ambiguous; expected exactly one "
            f"compatible encoder, found {len(unique)}"
        )
    return RLTBundlePaths(root=root, encoder=unique[0], actor=actor)


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, Tensor) and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_rec_to_dtype(item, dtype) for item in value)
    return value


def _checkpoint_weight_fingerprint(checkpoint: Path) -> str:
    """Reproduce the Stage-1 path-independent GR00T weight identity."""

    return build_groot_rlt_provenance(checkpoint).weight_fingerprint


def _policy_request(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._policy_request_lock:
            self.apply_pending_policy()
            return method(self, *args, **kwargs)
    return wrapped


class GR00TRLTInferenceAdapter:
    """Preloaded RLT policy that shares one frozen GR00T instance."""

    def __init__(self, policy: Any, shadow_policy: Any, bundle: RLTBundlePaths):
        self.policy = policy
        self.shadow_policy = shadow_policy
        self.bundle = bundle
        self.spec = shadow_policy.spec
        self.recording_context = None
        self.recording_identity = {}
        self._groot_checkpoint_fingerprint = None
        self._policy_updates = None
        self._policy_request_lock = threading.RLock()
        self._async_training = None
        self._async_run = None
        from .rlt_async_checkpoint import RLTAsyncCheckpoint
        self._async_checkpoint = RLTAsyncCheckpoint(self)
        self._async_prepare = None
        self._async_cancel = threading.Event()
        self._async_error = None
        self._async_enabled = False
        self._async_max_updates = 1000
        self._async_replay = None
        self._async_replay_builder = None
        self._async_selected_paths = ()
        self._async_selection_lock = threading.Lock()
        if (self.spec.reference_horizon, self.spec.action_dim) not in {(16, 19), (32, 16)}:
            raise ValueError(
                "RLT reference must be 16x19 or 32x16"
            )
        if (
            self.spec.chunk_length != EXPECTED_CHUNK_LENGTH
            or self.spec.proprio_dim != self.spec.action_dim
        ):
            raise ValueError(
                "RLT runtime requires 10 actions and matching proprio/action dimensions"
            )
        contract_getter = getattr(policy, "get_tt_rtc_model_contract", None)
        if callable(contract_getter):
            contract = contract_getter()
            if (contract["action_horizon"], contract["action_dimension"]) != (
                self.spec.reference_horizon, self.spec.action_dim
            ):
                raise ValueError("RLT bundle reference shape disagrees with the loaded GR00T model")

    @classmethod
    def load(
        cls,
        policy: Any,
        bundle_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
    ) -> "GR00TRLTInferenceAdapter":
        from cyclo_brain.algorithm.rl.rlt import load_groot_rlt_shadow_policy

        bundle = resolve_rlt_bundle(bundle_path)
        contract_getter = getattr(policy, "get_tt_rtc_model_contract", None)
        action_dim = (contract_getter()["action_dimension"]
                      if callable(contract_getter) else EXPECTED_ACTION_DIM)
        shadow = load_groot_rlt_shadow_policy(
            bundle.encoder,
            bundle.actor,
            device=policy.model.device,
            dtype=torch.float32,
            expected_chunk_length=EXPECTED_CHUNK_LENGTH,
            expected_action_dim=action_dim,
        )

        provenance = build_groot_rlt_provenance(model_path)
        validate_stage1_encoder_provenance(
            shadow.encoder.representation_contract,
            provenance,
        )
        actor_root = (
            bundle.actor.parent.parent
            if bundle.actor.parent.name == "artifacts"
            else bundle.root
        )
        validate_stage2_bundle_provenance(
            actor_root,
            spec=shadow.spec,
            provenance=provenance,
        )
        adapter = cls(policy, shadow, bundle)
        adapter._groot_checkpoint_fingerprint = provenance.checkpoint_fingerprint
        adapter.recording_identity = {
            'actor_sha256': file_sha256(bundle.actor),
            'groot_sha256': provenance.weight_fingerprint,
            'rl_token_artifact_fingerprint': shadow.spec.rl_token_artifact_fingerprint,
        }
        return adapter

    @property
    def qualification(self) -> str:
        return str(self.shadow_policy.actor_qualification)

    @property
    def policy_updates(self):
        if self._policy_updates is not None:
            return self._policy_updates
        with self._policy_request_lock:
            if self._policy_updates is None:
                from cyclo_brain.algorithm.rl.rlt.policy_update import RLTPolicyUpdate
                self._policy_updates = RLTPolicyUpdate(self.shadow_policy.actor, self.spec)
            return self._policy_updates

    def apply_pending_policy(self):
        if self._policy_updates is None:
            return
        actor, identity = self._policy_updates.apply_pending(self.shadow_policy.actor)
        if identity is not None:
            self.shadow_policy.actor = actor
            self.shadow_policy.actor_qualification = 'training_only_not_deployment_validated'
            # The original bundle's file hash no longer describes the live MLP.
            original = self.recording_identity.pop('actor_sha256', None)
            if original is not None:
                self.recording_identity['base_actor_sha256'] = original
            self.recording_identity.update(identity)

    @contextmanager
    def inference_slot(self, *, background=False):
        # Also used for base VLA requests: training cannot overlap either route.
        if self._async_training is None:
            with self._policy_request_lock:
                yield
        else:
            with self._async_training.inference(background=background):
                with self._policy_request_lock:
                    yield

    def set_async_training(self, enabled, *, max_updates=None):
        """Keep learner state; selected datasets, not bundled replay, supply batches."""
        if max_updates is not None and (
            isinstance(max_updates, bool) or not isinstance(max_updates, int)
            or not 1 <= max_updates <= 4294967295
        ):
            raise ValueError('Async RL max_updates must be a positive uint32')
        if enabled and self._async_training is None:
            if not all((self.bundle.root / name).is_file() for name in (
                'manifest.json', 'training_state/rlt_stage2.pt',
            )):
                raise ValueError('Async RL requires a completed Stage-2 bundle with feature replay and optimizer state')
        if enabled and not self._async_enabled and max_updates is not None:
            self._async_max_updates = max_updates
        self._async_enabled = enabled
        if self._async_training is not None:
            self._async_replay.set_enabled(enabled)
            self._async_training.set_enabled(enabled, max_updates=self._async_max_updates)
            return
        if not enabled or (self._async_prepare and self._async_prepare.is_alive()):
            return
        self._async_error = None
        self._async_cancel.clear()

        def prepare():
            try:
                from cyclo_brain.algorithm.rl.rlt import RLTAsyncLearner, RLTStage2Run
                from .rlt_async_replay import RLTAsyncReplay, RLTAsyncReplayBuilder
                with self._policy_request_lock:
                    if self._async_cancel.is_set():
                        return
                    run = RLTStage2Run.resume(
                        self.bundle.root, device=self.policy.model.device,
                        expected_groot_checkpoint_fingerprint=self._groot_checkpoint_fingerprint,
                    )
                    if run.learner.spec != self.spec:
                        raise ValueError('Async RL learner differs from the loaded RLT spec')
                    self._async_run = run
                    updates = self.policy_updates
                    batch_size = run.training_round['optimization']['batch_size']
                    def completed(update):
                        if update.actor_updated and update.completed_actor_updates % 10 == 0:
                            updates.publish(run.learner)
                        if self._async_training.status()['limit_reached']:
                            self._async_enabled = False
                            self._async_replay.set_enabled(False)

                    self._async_training = RLTAsyncLearner(
                        run.learner, None, after_update=completed,
                    )
                    self._async_replay_builder = RLTAsyncReplayBuilder(self)
                    self._async_replay = RLTAsyncReplay(
                        self._async_training, self._async_replay_builder, batch_size,
                        run.learner.random_seed + run.learner.completed_critic_updates,
                    )
                    if run.async_state is not None:
                        from .rlt_stage2_dataset import RLTStage2FeatureReplay
                        self._async_replay.generator.set_state(run.async_state['sampling_generator'])
                        self._async_replay.seed = run.async_state['history'][-1]['sampling_seed']
                        self._async_training.replay_history = list(run.async_state['history'])
                        self._async_replay_builder.restored = RLTStage2FeatureReplay.load(
                            run.replay_root, expected_spec=self.spec)
                    with self._async_selection_lock:
                        self._async_replay.select(self._async_selected_paths)
                    self._async_replay.set_enabled(self._async_enabled)
                    self._async_training.set_enabled(self._async_enabled, max_updates=self._async_max_updates)
            except Exception as error:
                self._async_error = f'{type(error).__name__}: {error}'
                self._async_enabled = False

        self._async_prepare = threading.Thread(target=prepare, name='rlt-async-prepare', daemon=True)
        self._async_prepare.start()

    def select_async_datasets(self, paths):
        if not isinstance(paths, (list, tuple)) or any(
            not isinstance(path, str) or not Path(path).is_absolute() for path in paths
        ):
            raise ValueError('Async RL requires absolute LeRobot dataset paths')
        # No filesystem/model work on the command handler; preparation is background.
        with self._async_selection_lock:
            self._async_selected_paths = tuple(dict.fromkeys(paths))
            if self._async_replay is not None:
                self._async_replay.select(self._async_selected_paths)

    def async_training_status(self):
        worker = self._async_training.status() if self._async_training else None
        replay = self._async_replay.status() if self._async_replay else {
            'selected_paths': list(self._async_selected_paths), 'replay_error': None,
        }
        return {
            **replay,
            **self._async_checkpoint.status(),
            'async_enabled': worker['enabled'] if worker else self._async_enabled,
            'max_updates': self._async_max_updates,
            'updates_this_run': worker['updates_this_run'] if worker else 0,
            'limit_reached': worker['limit_reached'] if worker else False,
            'total_critic_updates': self._async_run.learner.completed_critic_updates if self._async_run else None,
            'total_actor_updates': self._async_run.learner.completed_actor_updates if self._async_run else None,
            'preparing': bool(self._async_prepare and self._async_prepare.is_alive()) or replay.get('preparing', False),
            'training_error': worker['error'] if worker else self._async_error,
            'training_phase': worker['training_phase'] if worker else None,
            'publish_interval': 10,
            'last_update': worker['last_update'] if worker else None,
            'replay_source': {
                'kind': 'selected_datasets',
                **(worker['replay'] if worker else {'paths': [], 'transitions': 0}),
            },
            'waiting_data': worker['waiting_data'] if worker else True,
        }

    def close_async_training(self, *, dispose=False):
        # A save owns a CPU snapshot but still needs the model/encoder for its
        # short verification slot. Finish it before disposing those resources.
        self._async_checkpoint.close()
        self._async_enabled = False
        self._async_cancel.set()
        if self._async_prepare is not None:
            self._async_prepare.join()
        if self._async_replay is not None:
            if dispose:
                self._async_replay.close()
                self._async_replay_builder.close()
            else:
                self._async_replay.set_enabled(False)
        if self._async_training is not None:
            if dispose:
                self._async_training.close()
            elif not self._async_training.status()['closed']:
                self._async_training.set_enabled(False)
                with self._async_training.inference():
                    pass

    @property
    def deployment_qualified(self) -> bool:
        return is_deployment_qualified(self.qualification)

    def require_tt_rtc_capability(self):
        """Validate the Stage-2 bundle's explicit delayed-reference contract.

        Existing RLT bundles remain valid for ordinary inference, but are not
        inferred to support TT-RTC.  A TT-RTC-trained bundle must carry the
        combined manifest documented in ``docs/TRAINING_TIME_RTC_RLT.md``.
        """

        from runtime.tt_rtc import load_tt_rtc_capability

        capability = load_tt_rtc_capability(self.bundle.root, require_rlt=True)
        rtc = capability.payload["training_time_rtc"]
        if (rtc["action_horizon"], rtc["action_dimension"]) != (
            self.spec.reference_horizon, self.spec.action_dim
        ):
            raise ValueError("TT-RTC manifest disagrees with RLT bundle reference shape")
        return capability

    def _prepare(self, observation: Mapping[str, object]):
        if getattr(self.policy, "strict", False):
            self.policy.check_observation(observation)
        unbatched = self.policy._unbatch_observation(observation)
        processed = []
        raw_states = []
        for item in unbatched:
            vla_step = self.policy._to_vla_step_data(item)
            raw_states.append(vla_step.states)
            processed.append(
                self.policy.processor(
                    [{"type": "episode_step", "content": vla_step}]
                )
            )
        collated = self.policy.collate_fn(processed)
        collated = _rec_to_dtype(
            collated,
            dtype=getattr(self.policy.model, "dtype", torch.bfloat16),
        )
        if not isinstance(collated, Mapping) or "inputs" not in collated:
            raise RuntimeError("GR00T RLT expected collator output with inputs")
        batched_states = {
            key: np.stack([state[key] for state in raw_states], axis=0)
            for key in self.policy.modality_configs["state"].modality_keys
        }
        return collated, batched_states

    def _decode_candidate(
        self,
        candidate: Tensor,
        batched_states: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not bool(torch.isfinite(candidate).all()):
            raise FloatingPointError("RLT Action MLP returned non-finite actions")

        decoded = self.policy.processor.decode_action(
            candidate.float().cpu().numpy(),
            self.policy.embodiment_tag,
            batched_states,
        )
        physical = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in decoded.items()
        }
        expected_keys = list(self.policy.modality_configs["action"].modality_keys)
        if list(physical) != expected_keys:
            raise RuntimeError(
                f"RLT decoder returned {list(physical)}, expected {expected_keys}"
            )
        if any(not np.isfinite(value).all() for value in physical.values()):
            raise FloatingPointError("Decoded RLT action contains non-finite values")
        return physical

    def _validate_tt_rtc_rlt_context(
        self,
        context: object,
        *,
        delay_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Mapping[str, np.ndarray]]:
        from runtime.tt_rtc import TT_RTC_RLT_CONTEXT_SCHEMA

        if not isinstance(context, Mapping):
            raise RuntimeError("TT-RTC RLT context must be a mapping")
        if context.get("schema") != TT_RTC_RLT_CONTEXT_SCHEMA:
            raise RuntimeError(
                "TT-RTC RLT context schema must be "
                f"{TT_RTC_RLT_CONTEXT_SCHEMA!r}"
            )
        context_delay = context.get("delay_steps")
        if (
            isinstance(context_delay, bool)
            or not isinstance(context_delay, int)
            or context_delay != delay_steps
        ):
            raise RuntimeError("TT-RTC RLT context delay_steps disagrees with request")

        tokens = context.get("tokens")
        token_valid = context.get("token_valid")
        image_token = context.get("image_token")
        proprio = context.get("proprio")
        reference = context.get("reference_actions")
        normalized_prefix = context.get("normalized_committed_prefix")
        batched_states = context.get("batched_states")

        if not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1:
            raise RuntimeError("TT-RTC RLT tokens must have shape (1, M, K)")
        token_shape = tuple(tokens.shape[:2])
        for mask, name in (
            (token_valid, "token_valid"),
            (image_token, "image_token"),
        ):
            if (
                not isinstance(mask, Tensor)
                or tuple(mask.shape) != token_shape
                or mask.dtype != torch.bool
            ):
                raise RuntimeError(
                    f"TT-RTC RLT {name} must be a bool tensor with shape {token_shape}"
                )
        if not bool(token_valid.any(dim=1).all()):
            raise RuntimeError("TT-RTC RLT token sequence is empty")
        if not bool((token_valid & image_token).any(dim=1).all()):
            raise RuntimeError("TT-RTC RLT context has no valid image token")

        expected_proprio = (1, self.spec.proprio_dim)
        if not isinstance(proprio, Tensor) or tuple(proprio.shape) != expected_proprio:
            raise RuntimeError(
                f"TT-RTC RLT proprio must have shape {expected_proprio}"
            )
        expected_reference = (
            1,
            self.spec.reference_horizon,
            self.spec.action_dim,
        )
        if (
            not isinstance(reference, Tensor)
            or tuple(reference.shape) != expected_reference
        ):
            raise RuntimeError(
                f"TT-RTC RLT reference_actions must have shape {expected_reference}"
            )
        expected_prefix = (1, delay_steps, self.spec.action_dim)
        if (
            not isinstance(normalized_prefix, Tensor)
            or tuple(normalized_prefix.shape) != expected_prefix
        ):
            raise RuntimeError(
                "TT-RTC RLT normalized_committed_prefix must have shape "
                f"{expected_prefix}"
            )
        for tensor, name in (
            (tokens, "tokens"),
            (proprio, "proprio"),
            (reference, "reference_actions"),
            (normalized_prefix, "normalized_committed_prefix"),
        ):
            if tensor.requires_grad:
                raise RuntimeError(f"TT-RTC RLT {name} must be detached")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"TT-RTC RLT {name} must be finite and floating")

        expected_clean_prefix = normalized_prefix.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if not torch.equal(reference[:, :delay_steps], expected_clean_prefix):
            raise RuntimeError(
                "TT-RTC GR00T reference does not preserve the exact normalized "
                "committed prefix"
            )
        if not isinstance(batched_states, Mapping):
            raise RuntimeError("TT-RTC RLT batched_states must be a mapping")
        expected_state_keys = list(
            self.policy.modality_configs["state"].modality_keys
        )
        if any(key not in batched_states for key in expected_state_keys):
            raise RuntimeError("TT-RTC RLT batched_states are incomplete")

        return (
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
            batched_states,
        )

    @torch.inference_mode()
    @_policy_request
    def get_action(
        self,
        observation: Mapping[str, object],
        *,
        reference_offset_steps: int = 0,
        capture_context: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return an RLT action from a normalized GR00T reference.

        Normal RLT inference keeps the legacy ``reference[:10]`` behavior.
        A Training-Time RTC caller must pass the already selected request delay
        so the actor consumes ``reference[d:d+10]``.  This offset does not make
        a legacy GR00T reference prefix-conditioned; that capability is gated
        separately by the engine.
        """

        self.recording_context = None
        collated, batched_states = self._prepare(observation)
        model_inputs = collated["inputs"]
        backbone_inputs, action_inputs = self.policy.model.prepare_input(model_inputs)
        backbone_output = self.policy.model.backbone(backbone_inputs)

        tokens = backbone_output["backbone_features"].detach().clone()
        token_valid = backbone_output["backbone_attention_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        image_token = backbone_output["image_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        normalized_state = action_inputs["state"]
        proprio = normalized_state[:, -1, : self.spec.proprio_dim].float().detach()

        model_pred = self.policy.model.action_head.get_action(
            backbone_output,
            action_inputs,
            collated.get("options"),
        )
        reference = model_pred["action_pred"][
            :, : self.spec.reference_horizon, : self.spec.action_dim
        ].float().detach()
        expected_reference = (
            tokens.shape[0],
            self.spec.reference_horizon,
            self.spec.action_dim,
        )
        if tuple(reference.shape) != expected_reference:
            raise RuntimeError(
                f"GR00T RLT reference shape {tuple(reference.shape)} does not "
                f"match {expected_reference}"
            )

        output = self.shadow_policy(
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
            reference_offset_steps=reference_offset_steps,
        )
        if capture_context:
            self._capture_context(output, proprio, reference)
        return self._decode_candidate(output.action_mean, batched_states)

    def _capture_context(self, output, proprio, reference):
        # Small detached numeric tensors only; never retain VLM graphs/images.
        self.recording_context = {
            name: value.detach().float().cpu().numpy().copy()
            for name, value in {
                'z_rl': output.z_rl, 'proprio': proprio,
                'reference_actions': reference,
                'mlp_reference': output.reference_prefix,
                'action_mean': output.action_mean,
            }.items()
        }

    @torch.inference_mode()
    @_policy_request
    def get_action_tt_rtc(
        self,
        observation: Mapping[str, object],
        *,
        committed_action_prefix: np.ndarray,
        delay_steps: int,
        action_horizon: int,
        capture_context: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run RLT from an explicitly prefix-conditioned GR00T context.

        The upstream policy owns physical-to-normalized conversion and the
        paper-faithful flow sampler.  It must return the exact frozen features
        and normalized reference from that same forward in
        ``info['tt_rtc_rlt_context']``.  This adapter never reconstructs or
        guesses those tensors from the decoded VLA action.
        """

        self.recording_context = None
        from runtime.tt_rtc import (
            TT_RTC_MAX_DELAY_STEPS,
        )

        if action_horizon != self.spec.reference_horizon:
            raise RuntimeError(
                f"TT-RTC RLT action_horizon must be {self.spec.reference_horizon}"
            )
        if (
            isinstance(delay_steps, bool)
            or not isinstance(delay_steps, int)
            or not 0 <= delay_steps <= TT_RTC_MAX_DELAY_STEPS
        ):
            raise RuntimeError(
                f"TT-RTC RLT delay_steps must be in 0..{TT_RTC_MAX_DELAY_STEPS}"
            )
        prefix = np.asarray(committed_action_prefix)
        expected_prefix = (1, delay_steps, self.spec.action_dim)
        if prefix.shape != expected_prefix or not np.isfinite(prefix).all():
            raise RuntimeError(
                f"TT-RTC RLT committed_action_prefix must have shape {expected_prefix}"
            )

        self.require_tt_rtc_capability()
        runner = getattr(self.policy, "get_action_tt_rtc", None)
        if not callable(runner):
            raise RuntimeError(
                "TT-RTC RLT requires Gr00tPolicy.get_action_tt_rtc with an "
                "explicit normalized RLT context"
            )
        result = runner(
            observation,
            committed_action_prefix=prefix.astype(np.float32, copy=False),
            delay_steps=delay_steps,
            action_horizon=action_horizon,
            return_rlt_context=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "TT-RTC GR00T policy must return (action, info) for RLT"
            )
        _base_action, info = result
        if not isinstance(info, Mapping):
            raise RuntimeError("TT-RTC GR00T policy returned invalid RLT info")
        context = info.get("tt_rtc_rlt_context")
        (
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
            batched_states,
        ) = self._validate_tt_rtc_rlt_context(
            context,
            delay_steps=delay_steps,
        )

        output = self.shadow_policy(
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
            reference_offset_steps=delay_steps,
        )
        if capture_context:
            self._capture_context(output, proprio, reference)
        return self._decode_candidate(output.action_mean, batched_states)


class GR00TRLTTokenExtractor:
    """Read detached backbone tokens without running the GR00T action head.

    PI RLT Stage 1 reconstructs the frozen VLA representation.  Keeping that
    extraction boundary next to the inference adapter ensures training and
    deployment use exactly the same processor, collator and backbone output.
    """

    def __init__(self, policy: Any):
        self.policy = policy
        self.policy.model.eval()
        self.policy.model.requires_grad_(False)
        processor_eval = getattr(self.policy.processor, "eval", None)
        if callable(processor_eval):
            processor_eval()

    def _prepare(self, observation: Mapping[str, object]) -> Mapping[str, Any]:
        if getattr(self.policy, "strict", False):
            self.policy.check_observation(observation)
        processed = []
        for item in self.policy._unbatch_observation(observation):
            vla_step = self.policy._to_vla_step_data(item)
            processed.append(
                self.policy.processor(
                    [{"type": "episode_step", "content": vla_step}]
                )
            )
        collated = self.policy.collate_fn(processed)
        collated = _rec_to_dtype(
            collated,
            dtype=getattr(self.policy.model, "dtype", torch.bfloat16),
        )
        if not isinstance(collated, Mapping) or "inputs" not in collated:
            raise RuntimeError("GR00T RLT expected collator output with inputs")
        return collated

    @torch.inference_mode()
    def extract(self, observation: Mapping[str, object]) -> dict[str, Tensor]:
        collated = self._prepare(observation)
        backbone_inputs, _action_inputs = self.policy.model.prepare_input(
            collated["inputs"]
        )
        backbone_output = self.policy.model.backbone(backbone_inputs)
        tokens = backbone_output["backbone_features"].detach().clone()
        token_valid = backbone_output["backbone_attention_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        image_token = backbone_output["image_mask"].to(
            dtype=torch.bool
        ).detach().clone()
        if tokens.ndim != 3:
            raise RuntimeError("GR00T RLT backbone tokens must have shape (B, M, D)")
        expected_mask = tuple(tokens.shape[:2])
        if (
            tuple(token_valid.shape) != expected_mask
            or tuple(image_token.shape) != expected_mask
        ):
            raise RuntimeError("GR00T RLT backbone masks disagree with token shape")
        if not bool(token_valid.any(dim=1).all()):
            raise RuntimeError("GR00T RLT backbone returned an empty token sequence")
        if not bool((token_valid & image_token).any(dim=1).all()):
            raise RuntimeError("GR00T RLT backbone returned no valid image tokens")
        return {
            "tokens": tokens,
            "token_valid": token_valid,
            "image_token": image_token,
        }


__all__ = [
    "DEPLOYMENT_QUALIFICATION",
    "EXPECTED_ACTION_DIM",
    "EXPECTED_CHUNK_LENGTH",
    "GR00TRLTInferenceAdapter",
    "GR00TRLTTokenExtractor",
    "RLTBundlePaths",
    "SIMULATION_ONLY_QUALIFICATION",
    "is_deployment_qualified",
    "resolve_rlt_bundle",
]

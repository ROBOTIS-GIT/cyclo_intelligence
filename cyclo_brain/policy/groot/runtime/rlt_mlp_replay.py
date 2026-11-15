"""Build MLP coordinates from immutable raw frames and saved frozen features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cyclo_brain.algorithm.rl.rlt.normalization import (
    fit_normalization,
    normalize,
    reference_to_mlp,
    validate_normalization,
)

from .rlt_provenance import rlt_action_group_names
from .rlt_stage2_dataset import open_rlt_stage2_source, RLTStage2FeatureReplay
from .rlt_tt_rtc_actor_replay import TERMINAL_HOLD_POLICY


def _vla_action_bounds(checkpoint: str | Path, action_dim: int) -> tuple[list[float], list[float]]:
    """Return the exact physical bounds used by the frozen GR00T processor."""

    root = Path(checkpoint).expanduser().absolute()
    processor = json.loads((root / "processor_config.json").read_text(encoding="utf-8"))
    statistics = json.loads((root / "statistics.json").read_text(encoding="utf-8"))
    kwargs = processor.get("processor_kwargs", {})
    embodiment = kwargs.get("modality_configs", {}).get("new_embodiment", {})
    action_config = embodiment.get("action", {})
    configs = action_config.get("action_configs")
    groups = tuple(rlt_action_group_names(action_dim))
    if (
        action_config.get("modality_keys") != list(groups)
        or not isinstance(configs, list)
        or len(configs) != len(groups)
        or any(
            not isinstance(config, dict)
            or config.get("rep") != "ABSOLUTE"
            or config.get("type") != "NON_EEF"
            for config in configs
        )
        or kwargs.get("use_mean_std") is not False
    ):
        raise ValueError(
            "Independent MLP codec currently requires absolute minmax GR00T actions"
        )
    statistic = "q01" if kwargs.get("use_percentiles") is True else "min"
    upper_statistic = "q99" if statistic == "q01" else "max"
    action_statistics = statistics.get("new_embodiment", {}).get("action", {})
    lower = [
        float(value)
        for group in groups
        for value in action_statistics.get(group, {}).get(statistic, [])
    ]
    upper = [
        float(value)
        for group in groups
        for value in action_statistics.get(group, {}).get(upper_statistic, [])
    ]
    if (
        len(lower) != action_dim
        or len(upper) != action_dim
        or not np.isfinite([lower, upper]).all()
        or np.any(np.asarray(upper) <= np.asarray(lower))
    ):
        raise ValueError("GR00T action minmax statistics are invalid")
    return lower, upper


def prepare_mlp_replay(
    replay: RLTStage2FeatureReplay,
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    expected_normalization: dict | None = None,
) -> tuple[RLTStage2FeatureReplay, dict]:
    """Rebuild raw state/targets and convert an Actor replay to MLP coordinates."""

    if (
        not replay.actor_only
        or replay.metadata.get("terminal_hold_policy") != TERMINAL_HOLD_POLICY
    ):
        raise ValueError(
            "Prepare Actor replay again: MLP warmup requires successful terminal-hold windows"
        )

    from .rlt_data_validation import verify_snapshots

    verify_snapshots(replay)
    saved_normalization = replay.metadata.get("mlp_normalization")
    if saved_normalization is not None:
        saved_normalization = validate_normalization(
            saved_normalization, replay.spec.action_dim
        )
        if (
            expected_normalization is not None
            and validate_normalization(
                expected_normalization, replay.spec.action_dim
            )
            != saved_normalization
        ):
            raise ValueError(
                "Selected replay MLP normalization disagrees with the Resume bundle"
            )
        replay.save(output_root)
        return (
            RLTStage2FeatureReplay.load(output_root, expected_spec=replay.spec),
            saved_normalization,
        )
    groups = rlt_action_group_names(replay.spec.action_dim)
    episodes: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, bool]] = {}
    successful_states: list[np.ndarray] = []
    successful_actions: list[np.ndarray] = []
    for dataset_root in replay.metadata.get("dataset_roots", []):
        source = open_rlt_stage2_source(
            dataset_root,
            expected_fps=replay.spec.action_hz,
            action_dim=replay.spec.action_dim,
        )
        for episode in source.iter_episodes():
            state = np.concatenate(
                [episode.state_groups[group] for group in groups], axis=-1
            ).astype(np.float32, copy=False)
            action = np.concatenate(
                [episode.action_groups[group] for group in groups], axis=-1
            ).astype(np.float32, copy=False)
            episodes[(str(source.root), episode.episode_index)] = (
                state,
                action,
                episode.successful,
            )
            if episode.successful:
                successful_states.append(state)
                successful_actions.append(action)
    if not successful_states:
        raise ValueError("MLP normalization requires explicit successful episodes")

    vla_lower, vla_upper = _vla_action_bounds(checkpoint, replay.spec.action_dim)
    normalization = fit_normalization(
        np.concatenate(successful_states),
        np.concatenate(successful_actions),
        joint_names=[name for names in groups.values() for name in names],
        vla_lower=vla_lower,
        vla_upper=vla_upper,
        source_fingerprint=replay.metadata["dataset_snapshot_fingerprint"],
    )
    if expected_normalization is not None:
        expected = validate_normalization(expected_normalization, replay.spec.action_dim)
        if expected != normalization:
            raise ValueError(
                "Selected datasets produce a different MLP normalization; "
                "start a new Actor warmup instead of Resume"
            )
        normalization = expected

    rows = replay.metadata.get("row_provenance", [])
    if len(rows) != len(replay):
        raise ValueError("RLT Stage 2 row provenance is incomplete")
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row in rows:
        key = (str(Path(row["dataset_root"]).expanduser().absolute()), row["episode_index"])
        if key not in episodes:
            raise ValueError("RLT Stage 2 row provenance dataset/episode is missing")
        state, action, successful = episodes[key]
        t = row.get("observation_frame_index", row.get("start_frame_index"))
        begin = row.get("action_start_frame_index")
        end = row.get("action_stop_frame_index")
        if (
            type(t) is not int
            or type(begin) is not int
            or type(end) is not int
            or t < 0
            or t >= len(state)
            or begin < 0
            or end - begin != replay.spec.chunk_length
            or (
                end > len(action)
                and not (successful and row.get("synthetic_hold") is True)
            )
        ):
            raise ValueError("MLP replay raw frame provenance is invalid")
        indices = np.minimum(np.arange(begin, end), len(action) - 1)
        states.append(state[t])
        targets.append(action[indices])

    tensors = {key: value.clone() for key, value in replay.tensors.items()}
    tensors["proprio"] = torch.from_numpy(
        normalize(np.stack(states), normalization, kind="state")
    )
    tensors["executed_actions"] = torch.from_numpy(
        normalize(np.stack(targets), normalization)
    )
    tensors["reference_actions"] = torch.from_numpy(
        reference_to_mlp(tensors["reference_actions"].numpy(), normalization)
    )
    metadata = dict(replay.metadata)
    metadata["mlp_normalization"] = normalization
    metadata["mlp_normalization_id"] = normalization["fingerprint"]
    prepared = RLTStage2FeatureReplay(replay.spec, tensors, metadata=metadata)
    prepared.save(output_root)
    verify_snapshots(prepared)
    return RLTStage2FeatureReplay.load(output_root, expected_spec=replay.spec), normalization


def prepare_mlp_transition_replay(
    replay: RLTStage2FeatureReplay,
    *,
    output_root: str | Path,
    expected_normalization: dict,
) -> tuple[RLTStage2FeatureReplay, dict]:
    """Convert real TT-RTC transitions to one resumed Actor's MLP coordinates."""

    normalization = validate_normalization(
        expected_normalization, replay.spec.action_dim
    )
    if replay.actor_only or replay.metadata.get("replay_role") != "tt_rtc_transition/v1":
        raise ValueError(
            "MLP Critic/RL requires a real TT-RTC transition replay"
        )
    if replay.metadata.get("terminal_hold_policy") is not None:
        raise ValueError(
            "Synthetic terminal-hold Actor targets cannot train Critic/RL"
        )

    from .rlt_data_validation import verify_snapshots

    verify_snapshots(replay)
    saved_normalization = replay.metadata.get("mlp_normalization")
    if saved_normalization is not None:
        saved_normalization = validate_normalization(
            saved_normalization, replay.spec.action_dim
        )
        if saved_normalization != normalization:
            raise ValueError(
                "Selected Critic/RL replay MLP normalization disagrees with the Resume bundle"
            )
        replay.save(output_root)
        return (
            RLTStage2FeatureReplay.load(output_root, expected_spec=replay.spec),
            normalization,
        )

    groups = rlt_action_group_names(replay.spec.action_dim)
    episodes: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset_root in replay.metadata.get("dataset_roots", []):
        source = open_rlt_stage2_source(
            dataset_root,
            expected_fps=replay.spec.action_hz,
            action_dim=replay.spec.action_dim,
        )
        for episode in source.iter_episodes():
            state = np.concatenate(
                [episode.state_groups[group] for group in groups], axis=-1
            ).astype(np.float32, copy=False)
            action = np.concatenate(
                [episode.action_groups[group] for group in groups], axis=-1
            ).astype(np.float32, copy=False)
            episodes[(str(source.root), episode.episode_index)] = (state, action)

    rows = replay.metadata.get("row_provenance", [])
    if len(rows) != len(replay):
        raise ValueError("RLT Stage 2 transition provenance is incomplete")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray | None] = []
    for row in rows:
        key = (
            str(Path(row["dataset_root"]).expanduser().absolute()),
            row["episode_index"],
        )
        if key not in episodes:
            raise ValueError("RLT Stage 2 transition dataset/episode is missing")
        state, action = episodes[key]
        t = row.get("observation_frame_index", row.get("start_frame_index"))
        begin = row.get("action_start_frame_index")
        end = row.get("action_stop_frame_index")
        next_t = row.get("next_observation_frame_index")
        if (
            type(t) is not int
            or type(begin) is not int
            or type(end) is not int
            or t < 0
            or t >= len(state)
            or begin < 0
            or end - begin != replay.spec.chunk_length
            or end > len(action)
            or (next_t is not None and (type(next_t) is not int or next_t < 0 or next_t >= len(state)))
        ):
            raise ValueError("RLT Stage 2 real transition provenance is invalid")
        has_bootstrap = float(replay.tensors["bootstrap_discount"][len(states), 0]) > 0.0
        if has_bootstrap != (next_t is not None):
            raise ValueError("RLT Stage 2 next-state provenance disagrees with bootstrap")
        states.append(state[t])
        actions.append(action[begin:end])
        next_states.append(state[next_t] if next_t is not None else None)

    tensors = {key: value.clone() for key, value in replay.tensors.items()}
    tensors["proprio"] = torch.from_numpy(
        normalize(np.stack(states), normalization, kind="state")
    )
    tensors["executed_actions"] = torch.from_numpy(
        normalize(np.stack(actions), normalization)
    )
    tensors["reference_actions"] = torch.from_numpy(
        reference_to_mlp(tensors["reference_actions"].numpy(), normalization)
    )
    tensors["committed_prefix"] = torch.from_numpy(
        normalize(tensors["committed_prefix"].numpy(), normalization)
    )

    nonterminal = np.asarray(
        [value is not None for value in next_states], dtype=np.bool_
    )
    tensors["next_proprio"] = torch.zeros_like(tensors["next_proprio"])
    tensors["next_reference_actions"] = torch.zeros_like(
        tensors["next_reference_actions"]
    )
    tensors["next_committed_prefix"] = torch.zeros_like(
        tensors["next_committed_prefix"]
    )
    if bool(nonterminal.any()):
        indices = np.flatnonzero(nonterminal)
        stacked_next_states = np.stack(
            [next_states[index] for index in indices]
        )
        tensors["next_proprio"][indices] = torch.from_numpy(
            normalize(stacked_next_states, normalization, kind="state")
        )
        tensors["next_reference_actions"][indices] = torch.from_numpy(
            reference_to_mlp(
                replay.tensors["next_reference_actions"][indices].numpy(),
                normalization,
            )
        )
        tensors["next_committed_prefix"][indices] = torch.from_numpy(
            normalize(
                replay.tensors["next_committed_prefix"][indices].numpy(),
                normalization,
            )
        )

    metadata = dict(replay.metadata)
    metadata["mlp_normalization"] = normalization
    metadata["mlp_normalization_id"] = normalization["fingerprint"]
    metadata["mlp_coordinate_contract"] = (
        "state_action_reference_and_prefix/v1"
    )
    prepared = RLTStage2FeatureReplay(replay.spec, tensors, metadata=metadata)
    prepared.save(output_root)
    verify_snapshots(prepared)
    return (
        RLTStage2FeatureReplay.load(output_root, expected_spec=replay.spec),
        normalization,
    )

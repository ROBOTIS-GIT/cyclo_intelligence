"""Train only from immutable saved inputs; never run the VLA or encoder."""

from pathlib import Path
import torch

from cyclo_brain.algorithm.common import canonical_json_sha256
from .rlt_stage2_dataset import RLTStage2FeatureReplay


def tt_replay_cache(root, training_phase):
    hold = Path(root) / "rlt/tt_rtc_actor_hold_v1/replay_cache"
    if training_phase == "actor_warmup" and (hold / "manifest.json").is_file():
        return hold
    subdir = "tt_rtc_actor_v1" if training_phase == "actor_warmup" else "tt_rtc_transition_v1"
    return Path(root) / "rlt" / subdir / "replay_cache"


def has_fresh_tt_replay(roots, training_phase):
    return bool(roots) and all((tt_replay_cache(root, training_phase) / "manifest.json").is_file() for root in roots)


def load_saved_replay(
    roots,
    *,
    spec,
    output_root,
    replay_bundle=None,
    training_phase=None,
    allow_stale_replay=False,
):
    roots = tuple(Path(p).resolve() for p in roots)
    parts = []
    remaining = roots
    if replay_bundle:
        parent = RLTStage2FeatureReplay.load(Path(replay_bundle) / "replay_cache", expected_spec=spec)
        recorded = tuple(Path(p).resolve() for p in parent.metadata.get("dataset_roots", []))
        if not recorded or roots[:len(recorded)] != recorded:
            raise ValueError("Selected datasets must retain the saved replay datasets in order. VLA inference is disabled.")
        parts.append(parent)
        remaining = roots[len(recorded):]
    for root in remaining:
        if parts:
            role = parts[0].metadata.get("replay_role", "transition")
            subdir = {"transition": "", "tt_rtc_actor/v1": "tt_rtc_actor_v1",
                      "tt_rtc_transition/v1": "tt_rtc_transition_v1"}[role]
            cache = (tt_replay_cache(root, "actor_warmup") if role == "tt_rtc_actor/v1"
                     else root / "rlt" / subdir / "replay_cache")
        else:
            preferred = tt_replay_cache(root, training_phase)
            candidates = [preferred, root / "rlt/tt_rtc_transition_v1/replay_cache",
                          root / "rlt/tt_rtc_actor_v1/replay_cache", root / "rlt/replay_cache"]
            cache = next((p for p in candidates if p.is_dir()), candidates[-1])
        if not (cache / "manifest.json").is_file():
            raise ValueError(f"No saved RLT replay in {cache}. Supply recorded replay inputs or select an existing replay bundle; VLA inference is disabled.")
        part = RLTStage2FeatureReplay.load(cache, expected_spec=spec)
        if tuple(Path(p).resolve() for p in part.metadata.get("dataset_roots", [])) != (root,):
            raise ValueError("Saved replay dataset identity disagrees")
        parts.append(part)
    if not parts:
        raise ValueError("Select saved RLT replay data")
    # Validated copies remain tied to the exact data verified by the button.
    for index, part in enumerate(parts):
        if part.tt_rtc or part.metadata.get("data_validation"):
            if index == 0 and replay_bundle and allow_stale_replay:
                continue
            from .rlt_data_validation import verify_snapshots
            verify_snapshots(part)
    metadata = dict(parts[0].metadata)
    for part in parts[1:]:
        if part._TENSOR_NAMES != parts[0]._TENSOR_NAMES:
            raise ValueError("Cannot mix TT-RTC Actor replay and legacy transitions")
        for key in ("reward_contract", "reference_extraction", "config", "action_codec", "replay_role", "prefix_length", "timing_contract", "terminal_hold_policy"):
            if part.metadata.get(key) != metadata.get(key):
                raise ValueError(f"Cannot mix saved replay with different {key}")
    snapshots = [snapshot for p in parts for snapshot in p.metadata["dataset_snapshots"]]
    metadata.update(
        dataset_roots=[str(p) for p in roots],
        dataset_snapshots=snapshots,
        dataset_snapshot_fingerprint=canonical_json_sha256({
            "format": "cyclo.groot.rlt.dataset_collection/v1",
            "ordered_content_fingerprints": [s["content_fingerprint"] for s in snapshots],
        }),
        transition_count=sum(len(p) for p in parts),
        row_provenance=[r for p in parts for r in p.metadata.get("row_provenance", [])],
    )
    for key in ("episode_count", "usable_episode_count", "success_episode_count", "failure_episode_count", "skipped_short_episode_count", "feature_frame_count", "terminal_return_count", "synthetic_hold_count"):
        metadata[key] = sum(p.metadata.get(key, 0) for p in parts)
    replay = RLTStage2FeatureReplay(spec, {
        k: torch.cat([p.tensors[k] for p in parts]) for k in parts[0]._TENSOR_NAMES
    }, metadata=metadata)
    replay.save(output_root)
    return RLTStage2FeatureReplay.load(output_root, expected_spec=spec)

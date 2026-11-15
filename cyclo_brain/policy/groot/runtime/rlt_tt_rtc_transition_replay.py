"""Build TT-RTC TD samples from saved inputs, without a model forward.

A decision observes the robot while six commands remain queued. Its new ten
commands execute next. The next decision follows four of those new commands.
Near the recording's terminal boundary, use the observed discounted return
rather than inventing an incomplete next ten-action behavior chunk.
"""
from collections import defaultdict
import math
import torch

from .rlt_stage2_dataset import RLTStage2FeatureReplay
from .rlt_tt_rtc_actor_replay import ROLE as ACTOR_ROLE, PREFIX

ROLE = "tt_rtc_transition/v1"
CACHE_SUBDIR = "tt_rtc_transition_v1"
REWARD_CONTRACT = "tt_rtc_decision_td_with_observed_terminal_return/v1"


def build_tt_rtc_transition_replay(actor_replay, *, output_root, discount=0.99):
    if actor_replay.metadata.get("replay_role") != ACTOR_ROLE:
        raise ValueError("TT-RTC transitions require saved prefix-conditioned Actor inputs")
    if isinstance(discount, bool) or not math.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("TT-RTC discount must be in (0, 1]")
    if actor_replay.metadata.get("terminal_hold_policy"):
        raise ValueError("Terminal-hold Actor replay contains synthetic targets; prepare a separate real-transition RL replay")
    if actor_replay.metadata.get("mlp_normalization"):
        raise ValueError("MLP-normalized Actor replay cannot be converted to legacy TD transitions")
    spec = actor_replay.spec
    if spec.chunk_length != 10 or actor_replay.metadata.get("prefix_length") != PREFIX:
        raise ValueError("TT-RTC transitions require ten actions and six queued commands")
    rows = actor_replay.metadata.get("row_provenance", [])
    if len(rows) != len(actor_replay):
        raise ValueError("TT-RTC observation/action provenance is incomplete")
    episodes = defaultdict(dict)
    for i, row in enumerate(rows):
        t = row.get("observation_frame_index")
        if (type(t) is not int or t < 0 or row.get("action_start_frame_index") != t + PREFIX
                or row.get("action_stop_frame_index") != t + PREFIX + spec.chunk_length
                or type(row.get("episode_success")) is not bool):
            raise ValueError("TT-RTC observation/action offsets or outcomes disagree")
        key = (row["dataset_root"], row["episode_index"])
        if t in episodes[key]:
            raise ValueError("Duplicate TT-RTC observation frame")
        episodes[key][t] = i
    lengths = {}
    for key, indices in episodes.items():
        if sorted(indices) != list(range(max(indices) + 1)):
            raise ValueError("TT-RTC stride-one replay has missing observation frames")
        length = max(indices) + PREFIX + spec.chunk_length
        if len({rows[i]["episode_success"] for i in indices.values()}) != 1:
            raise ValueError("TT-RTC episode outcomes disagree")
        lengths[key] = length

    tensors = {k: v.clone() for k, v in actor_replay.tensors.items()}
    n = len(actor_replay)
    tensors.update(
        reward=torch.zeros(n, 1), bootstrap_discount=torch.zeros(n, 1),
        next_z_rl=torch.zeros_like(tensors["z_rl"]),
        next_proprio=torch.zeros_like(tensors["proprio"]),
        next_reference_actions=torch.zeros_like(tensors["reference_actions"]),
        next_committed_prefix=torch.zeros_like(tensors["committed_prefix"]),
    )
    provenance = []
    terminal_returns = 0
    for i, row in enumerate(rows):
        key = (row["dataset_root"], row["episode_index"])
        t = row["observation_frame_index"]
        # Prefix 6 + first 4 new commands = ten control steps between requests.
        next_t = t + spec.chunk_length
        next_i = episodes[key].get(next_t)
        if next_i is not None:
            for current, following in (("z_rl", "next_z_rl"), ("proprio", "next_proprio"),
                                       ("reference_actions", "next_reference_actions"),
                                       ("committed_prefix", "next_committed_prefix")):
                tensors[following][i].copy_(tensors[current][next_i])
            elapsed = spec.chunk_length
            tensors["bootstrap_discount"][i, 0] = discount ** elapsed
            target_end = next_t
        else:
            # All ten current actions were recorded. Only the NEXT behavior
            # chunk is short. Use the real episode's remaining return, including
            # its terminal reward. Do not pad actions or pretend the next request
            # itself was terminal. This is an observed behavior-return backup.
            elapsed = lengths[key] - t
            target_end = lengths[key]
            tensors["reward"][i, 0] = discount ** (elapsed - 1) if row["episode_success"] else 0.0
            terminal_returns += 1
        provenance.append({**row,
            "episode_length": lengths[key],
            "next_request_after_action_steps": spec.chunk_length - PREFIX,
            "next_request_frame_index": next_t,
            "next_observation_frame_index": next_t if next_i is not None else None,
            "target_end_frame_index": target_end,
            "elapsed_steps": elapsed,
            "target_kind": "decision_td" if next_i is not None else "observed_terminal_return",
            "terminal_return": next_i is None,
        })
    metadata = {**actor_replay.metadata,
        "replay_role": ROLE,
        "supported_training_phases": ["actor_warmup", "critic_warmup", "rl"],
        "config": {**actor_replay.metadata.get("config", {}), "discount": float(discount),
                   "decision_interval_steps": spec.chunk_length},
        "timing_contract": "reconstructed_fixed_prefix_6_output_10/v1",
        "reward_contract": REWARD_CONTRACT,
        "terminal_return_count": terminal_returns,
        "row_provenance": provenance,
    }
    # Validation reports refer to the old row/tensor artifact; revalidate the new one.
    metadata.pop("data_validation", None)
    replay = RLTStage2FeatureReplay(spec, tensors, metadata=metadata)
    replay.save(output_root)
    return RLTStage2FeatureReplay.load(output_root, expected_spec=spec)

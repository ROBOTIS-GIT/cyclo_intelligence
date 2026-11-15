"""Deterministic diagnostics for an offline Flow-SDE PPO value warm-up.

The metrics in this module deliberately operate on every chunk boundary once.
They do not reuse the stochastic warm-up sampler.  This makes the result a
reproducible *training-set diagnostic*, not an estimate of generalization.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead

from .value_warmup import ChunkBoundaryRecord


VALUE_WARMUP_EVALUATION_FORMAT = "cyclo.flow_sde_ppo.value_warmup.evaluation.v1"
CURRENT_VALUE_HEAD_HIDDEN_DIMS = (512, 256)


@dataclass(frozen=True)
class ValueWarmupEvaluationSample:
    """One deterministic prediction at a policy chunk boundary."""

    dataset_index: int
    episode_index: int
    chunk_index: int
    chunk_count: int
    successful: bool
    target_return: float
    prediction: float


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def samples_from_records(
    records: Sequence[ChunkBoundaryRecord], predictions: Tensor | Sequence[float]
) -> tuple[ValueWarmupEvaluationSample, ...]:
    """Bind ordered predictions to the exact warm-up dataset records."""

    if isinstance(predictions, Tensor):
        if predictions.ndim != 1:
            raise ValueError("value predictions must be one-dimensional")
        prediction_values: Sequence[Any] = predictions.detach().cpu().tolist()
    else:
        prediction_values = predictions
    if len(records) != len(prediction_values) or not records:
        raise ValueError("records and predictions must have the same non-zero length")
    result: list[ValueWarmupEvaluationSample] = []
    for record, prediction in zip(records, prediction_values, strict=True):
        result.append(
            ValueWarmupEvaluationSample(
                dataset_index=record.dataset_index,
                episode_index=record.episode_index,
                chunk_index=record.chunk_index,
                chunk_count=record.chunk_count,
                successful=record.successful,
                target_return=_finite_float(record.target_return, name="target return"),
                prediction=_finite_float(prediction, name="value prediction"),
            )
        )
    return tuple(result)


def _episode_groups(
    samples: Sequence[ValueWarmupEvaluationSample],
) -> dict[tuple[int, int], tuple[ValueWarmupEvaluationSample, ...]]:
    grouped: dict[tuple[int, int], list[ValueWarmupEvaluationSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.dataset_index, sample.episode_index)].append(sample)
    result: dict[tuple[int, int], tuple[ValueWarmupEvaluationSample, ...]] = {}
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda item: item.chunk_index))
        if [item.chunk_index for item in ordered] != list(range(len(ordered))):
            raise ValueError(f"episode {key!r} has non-contiguous chunk indices")
        if any(item.chunk_count != len(ordered) for item in ordered):
            raise ValueError(f"episode {key!r} has an inconsistent chunk_count")
        if len({item.successful for item in ordered}) != 1:
            raise ValueError(f"episode {key!r} has mixed success labels")
        result[key] = ordered
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sequence")
    return float(statistics.median(values))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted values and weights must have the same non-zero length")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("metric weights must have a finite positive sum")
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)) / total)


def _weighted_auc(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    positive_weights: Sequence[float] | None = None,
    negative_weights: Sequence[float] | None = None,
) -> float:
    """Compute ROC AUC as a weighted positive/negative pair probability."""

    if not positive_scores or not negative_scores:
        raise ValueError("ROC AUC requires both positive and negative samples")
    positive_weights = positive_weights or [1.0] * len(positive_scores)
    negative_weights = negative_weights or [1.0] * len(negative_scores)
    if len(positive_scores) != len(positive_weights) or len(negative_scores) != len(
        negative_weights
    ):
        raise ValueError("ROC AUC score and weight lengths differ")
    numerator = 0.0
    denominator = 0.0
    for positive, positive_weight in zip(positive_scores, positive_weights, strict=True):
        for negative, negative_weight in zip(negative_scores, negative_weights, strict=True):
            weight = positive_weight * negative_weight
            denominator += weight
            numerator += weight * (1.0 if positive > negative else 0.5 if positive == negative else 0.0)
    if denominator <= 0.0:
        raise ValueError("ROC AUC weights must have positive mass")
    return float(numerator / denominator)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1)
        for slot in order[cursor:end]:
            ranks[slot] = rank
        cursor = end
    return ranks


def _spearman_with_chunk_index(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("Spearman correlation requires at least two values")
    ranks = _average_ranks(values)
    center = (len(values) - 1) / 2.0
    numerator = sum((index - center) * (rank - center) for index, rank in enumerate(ranks))
    index_norm = sum((index - center) ** 2 for index in range(len(values)))
    rank_norm = sum((rank - center) ** 2 for rank in ranks)
    if rank_norm == 0.0:
        return 0.0
    return float(numerator / math.sqrt(index_norm * rank_norm))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    mean = _mean(values)
    variance = _mean([(value - mean) ** 2 for value in values])
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def evaluate_value_predictions(
    samples: Sequence[ValueWarmupEvaluationSample],
) -> dict[str, Any]:
    """Return raw and episode-balanced diagnostics for all chunk boundaries."""

    if not samples:
        raise ValueError("value evaluation requires at least one sample")
    episodes = _episode_groups(samples)
    success_episodes = [values for values in episodes.values() if values[0].successful]
    failure_episodes = [values for values in episodes.values() if not values[0].successful]
    if not success_episodes or not failure_episodes:
        raise ValueError("value evaluation requires both success and failure episodes")

    ordered_samples = tuple(samples)
    predictions = [sample.prediction for sample in ordered_samples]
    targets = [sample.target_return for sample in ordered_samples]
    squared_errors = [(prediction - target) ** 2 for prediction, target in zip(predictions, targets)]
    absolute_errors = [abs(prediction - target) for prediction, target in zip(predictions, targets)]

    sample_weights: list[float] = []
    for sample in ordered_samples:
        outcome_episodes = success_episodes if sample.successful else failure_episodes
        episode = episodes[(sample.dataset_index, sample.episode_index)]
        sample_weights.append(0.5 / (len(outcome_episodes) * len(episode)))
    if not math.isclose(sum(sample_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("episode-balanced metric weights do not sum to one")

    balanced_mse = _weighted_mean(squared_errors, sample_weights)
    balanced_mae = _weighted_mean(absolute_errors, sample_weights)
    zero_squared_errors = [target**2 for target in targets]
    zero_absolute_errors = [abs(target) for target in targets]
    constant = _weighted_mean(targets, sample_weights)
    constant_squared_errors = [(constant - target) ** 2 for target in targets]
    constant_absolute_errors = [abs(constant - target) for target in targets]
    constant_mse = _weighted_mean(constant_squared_errors, sample_weights)

    success_scores = [sample.prediction for sample in ordered_samples if sample.successful]
    failure_scores = [sample.prediction for sample in ordered_samples if not sample.successful]
    success_weights = [
        1.0 / (len(success_episodes) * len(episodes[(sample.dataset_index, sample.episode_index)]))
        for sample in ordered_samples
        if sample.successful
    ]
    failure_weights = [
        1.0 / (len(failure_episodes) * len(episodes[(sample.dataset_index, sample.episode_index)]))
        for sample in ordered_samples
        if not sample.successful
    ]

    success_episode_scores = [_mean([sample.prediction for sample in episode]) for episode in success_episodes]
    failure_episode_scores = [_mean([sample.prediction for sample in episode]) for episode in failure_episodes]
    success_terminal = [episode[-1].prediction for episode in success_episodes]
    failure_terminal = [episode[-1].prediction for episode in failure_episodes]

    success_temporal = []
    for episode in success_episodes:
        if len(episode) < 2:
            continue
        values = [sample.prediction for sample in episode]
        success_temporal.append(
            {
                "spearman": _spearman_with_chunk_index(values),
                "adjacent_nondecrease_rate": _mean(
                    [
                        float(right >= left)
                        for left, right in zip(values[:-1], values[1:], strict=True)
                    ]
                ),
                "endpoint_delta": values[-1] - values[0],
            }
        )
    temporal_summary: dict[str, Any] = {"eligible_episode_count": len(success_temporal)}
    for key in ("spearman", "adjacent_nondecrease_rate", "endpoint_delta"):
        values = [item[key] for item in success_temporal]
        temporal_summary[key] = (
            {
                "mean": _mean(values),
                "median": _median(values),
                "positive_episode_fraction": _mean([float(value > 0.0) for value in values]),
            }
            if values
            else None
        )

    success_outcome = [sample for sample in ordered_samples if sample.successful]
    failure_outcome = [sample for sample in ordered_samples if not sample.successful]

    def outcome_calibration(outcome: Sequence[ValueWarmupEvaluationSample]) -> dict[str, float]:
        outcome_predictions = [sample.prediction for sample in outcome]
        outcome_targets = [sample.target_return for sample in outcome]
        return {
            "prediction_mean": _mean(outcome_predictions),
            "target_mean": _mean(outcome_targets),
            "mean_bias": _mean(
                [prediction - target for prediction, target in zip(outcome_predictions, outcome_targets)]
            ),
            "mse": _mean(
                [(prediction - target) ** 2 for prediction, target in zip(outcome_predictions, outcome_targets)]
            ),
            "mae": _mean(
                [abs(prediction - target) for prediction, target in zip(outcome_predictions, outcome_targets)]
            ),
        }

    return {
        "scope": "training_dataset_diagnostic",
        "counts": {
            "chunk_boundaries": len(ordered_samples),
            "episodes": len(episodes),
            "success_episodes": len(success_episodes),
            "failure_episodes": len(failure_episodes),
            "success_chunks": len(success_outcome),
            "failure_chunks": len(failure_outcome),
        },
        "raw_chunk_metrics": {
            "mse": _mean(squared_errors),
            "mae": _mean(absolute_errors),
        },
        "episode_balanced_metrics": {
            "mse": balanced_mse,
            "mae": balanced_mae,
        },
        "baselines": {
            "zero": {
                "prediction": 0.0,
                "episode_balanced_mse": _weighted_mean(zero_squared_errors, sample_weights),
                "episode_balanced_mae": _weighted_mean(zero_absolute_errors, sample_weights),
            },
            "best_constant_for_mse": {
                "prediction": constant,
                "episode_balanced_mse": constant_mse,
                "episode_balanced_mae": _weighted_mean(constant_absolute_errors, sample_weights),
            },
            "mse_skill_over_best_constant": (
                1.0 - balanced_mse / constant_mse if constant_mse > 0.0 else None
            ),
        },
        "discrimination": {
            "episode_mean_roc_auc": _weighted_auc(success_episode_scores, failure_episode_scores),
            "episode_balanced_chunk_roc_auc": _weighted_auc(
                success_scores,
                failure_scores,
                positive_weights=success_weights,
                negative_weights=failure_weights,
            ),
            "episode_mean_margin": _mean(success_episode_scores) - _mean(failure_episode_scores),
            "terminal_chunk_margin": _mean(success_terminal) - _mean(failure_terminal),
            "success_episode_score_mean": _mean(success_episode_scores),
            "failure_episode_score_mean": _mean(failure_episode_scores),
            "success_terminal_score_mean": _mean(success_terminal),
            "failure_terminal_score_mean": _mean(failure_terminal),
        },
        "calibration": {
            "success_chunks": outcome_calibration(success_outcome),
            "failure_chunks": outcome_calibration(failure_outcome),
            "prediction_distribution": _distribution(predictions),
            "target_distribution": _distribution(targets),
            "prediction_below_zero_fraction": _mean([float(value < 0.0) for value in predictions]),
            "prediction_above_one_fraction": _mean([float(value > 1.0) for value in predictions]),
        },
        "success_temporal": temporal_summary,
        "interpretation": {
            "generalization_estimate": False,
            "notes": [
                "Metrics use the same labelled episodes used for warm-up.",
                "Success and failure roots may contain collection-session cues.",
                "Temporal monotonicity is diagnostic, not a required value-function invariant.",
            ],
        },
    }


def validate_current_value_head_state_dict(
    state_dict: Mapping[str, Tensor], *, conditioning_dim: int
) -> dict[str, Any]:
    """Fail closed against the current implicit default value-head layout."""

    reference = MultiTaskDiTValueHead(
        conditioning_dim, hidden_dims=CURRENT_VALUE_HEAD_HIDDEN_DIMS
    ).state_dict()
    if set(state_dict) != set(reference):
        raise ValueError(
            "value-head checkpoint keys do not match the current default architecture: "
            f"expected={sorted(reference)!r}, actual={sorted(state_dict)!r}"
        )
    for name, expected in reference.items():
        actual = state_dict[name]
        if not isinstance(actual, Tensor):
            raise TypeError(f"value-head state {name!r} is not a tensor")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"value-head state {name!r} contract mismatch: "
                f"expected shape={tuple(expected.shape)} dtype={expected.dtype}, "
                f"got shape={tuple(actual.shape)} dtype={actual.dtype}"
            )
        if not bool(torch.isfinite(actual).all()):
            raise ValueError(f"value-head state {name!r} contains non-finite values")
    return {
        "conditioning_dim": conditioning_dim,
        "hidden_dims": list(CURRENT_VALUE_HEAD_HIDDEN_DIMS),
        "architecture_serialized_in_checkpoint": False,
        "technical_debt": (
            "The v1 warm-up checkpoint does not serialize a value-head architecture contract; "
            "this evaluator strictly validates the current default (512, 256) tensor layout."
        ),
    }


def assert_exact_value_head_reload(
    state_dict: Mapping[str, Tensor],
    conditioning_batches: Sequence[Tensor],
    reference_predictions: Sequence[Tensor],
    *,
    conditioning_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    """Strict-load a fresh head and reproduce predictions on cached features."""

    if len(conditioning_batches) != len(reference_predictions) or not conditioning_batches:
        raise ValueError("reload validation batches must have the same non-zero length")
    contract = validate_current_value_head_state_dict(
        state_dict, conditioning_dim=conditioning_dim
    )
    reloaded = MultiTaskDiTValueHead(
        conditioning_dim, hidden_dims=CURRENT_VALUE_HEAD_HIDDEN_DIMS
    ).to(device)
    reloaded.load_state_dict(state_dict, strict=True)
    reloaded.eval()
    maximum_error = 0.0
    exact = True
    with torch.no_grad():
        for conditioning, reference in zip(
            conditioning_batches, reference_predictions, strict=True
        ):
            actual = reloaded(conditioning.to(device)).detach().cpu()
            expected = reference.detach().cpu()
            if actual.shape != expected.shape:
                raise AssertionError("reloaded value-head prediction shape changed")
            exact = exact and torch.equal(actual, expected)
            maximum_error = max(maximum_error, float((actual - expected).abs().max().item()))
    if not exact:
        raise AssertionError(
            f"reloaded value head did not reproduce exact predictions (max error={maximum_error})"
        )
    return {**contract, "exact_prediction_match": True, "prediction_max_abs_error": maximum_error}


__all__ = [
    "VALUE_WARMUP_EVALUATION_FORMAT",
    "CURRENT_VALUE_HEAD_HIDDEN_DIMS",
    "ValueWarmupEvaluationSample",
    "samples_from_records",
    "evaluate_value_predictions",
    "validate_current_value_head_state_dict",
    "assert_exact_value_head_reload",
]

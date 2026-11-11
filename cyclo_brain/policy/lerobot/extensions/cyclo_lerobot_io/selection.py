"""Read-only channel selection between dataset sampling and policy processing."""

from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .mapping import FILENAME, feature_dim, mapping_from_features, validate_mapping


VECTORS = {"observation.state": "state_names", "action": "action_names"}
CHANNEL_STATS = {"min", "max", "mean", "std", "q01", "q05", "q10", "q50", "q90", "q95", "q99"}


def select_stats(stats, indices, source_dim, label):
    result = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = deepcopy(value)
            continue
        if key not in CHANNEL_STATS:
            raise ValueError(f"Unsupported statistic {label}.{key}; channel axis is unknown")
        array = np.asarray(value)
        if array.ndim != 1 or array.shape != (source_dim,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid channel statistics {label}.{key}: expected finite ({source_dim},)")
        if key == "std" and (array < 0).any():
            raise ValueError(f"Negative standard deviation: {label}")
        result[key] = array[indices].copy()
    return result


class SelectedDataset(Dataset):
    """Expose selected samples and metadata without forwarding private readers."""

    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.meta = copy(dataset.meta)
        self.meta.info = deepcopy(dataset.meta.info)
        self.meta.stats = deepcopy(dataset.meta.stats)
        self.indices = {}
        self.source_dims = {}
        for feature, key in VECTORS.items():
            source = dataset.meta.features[feature]
            names = source["names"]
            selected = mapping[key]
            indices = [names.index(name) for name in selected]
            self.indices[feature] = torch.tensor(indices, dtype=torch.long)
            self.source_dims[feature] = len(names)
            self.meta.features[feature]["names"] = list(selected)
            self.meta.features[feature]["shape"] = (len(selected),)
            self.meta.stats[feature] = select_stats(
                dataset.meta.stats.get(feature, {}), indices, len(names), feature
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        for feature, indices in self.indices.items():
            value = sample[feature]
            if value.shape[-1] != self.source_dims[feature]:
                raise ValueError(f"Sample dimension differs from source metadata: {feature}")
            sample[feature] = (
                value.index_select(-1, indices)
                if isinstance(value, torch.Tensor)
                else np.take(value, indices.numpy(), axis=-1)
            )
            if not np.isfinite(np.asarray(sample[feature])).all():
                raise ValueError(f"Non-finite selected training channel: {feature}")
        return sample

    @property
    def features(self):
        return self.meta.features

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    @property
    def episodes(self):
        return self.dataset.episodes

    @property
    def absolute_to_relative_idx(self):
        return self.dataset.absolute_to_relative_idx

    @property
    def hf_dataset(self):
        # The trainer uses this only to select held-out task indices. Never
        # expose unselected state/action columns through an alternate path.
        return self.dataset.hf_dataset.select_columns(["task_index"])


class ChannelSelectionAdapter:
    def validate_config(self, cfg):
        unsupported = []
        if cfg.dataset.streaming:
            unsupported.append("streaming")
        if not isinstance(cfg.dataset.repo_id, str):
            unsupported.append("multiple datasets")
        if cfg.is_reward_model_training:
            unsupported.append("reward training")
        if cfg.env is not None:
            unsupported.append("environment rollout evaluation")
        if cfg.job.is_remote:
            unsupported.append("HF Jobs")
        if getattr(cfg, "sample_weighting", None) is not None:
            unsupported.append("sample weighting with independent dataset readers")
        if cfg.policy is not None and cfg.policy.type == "groot" and cfg.policy.use_relative_actions:
            unsupported.append("GR00T relative-action statistics from the original dataset")
        if any(k in VECTORS or v in VECTORS for k, v in cfg.rename_map.items()):
            unsupported.append("state/action feature renaming")
        if unsupported:
            raise ValueError("Channel selection does not support: " + ", ".join(unsupported))

    def prepare(self, train_dataset, eval_dataset, cfg):
        from lerobot.common.training_adapter import AdaptedTrainingData
        from lerobot.utils.feature_utils import dataset_to_policy_features

        source = mapping_from_features(
            train_dataset.meta.features, repo_id=cfg.dataset.repo_id, revision=cfg.dataset.revision
        )
        if source is None:
            raise ValueError("Channel selection requires dataset state/action names")
        spec = cfg.training_adapter
        requested = None
        if spec.config_path is not None and (not cfg.resume or Path(spec.config_path).exists()):
            requested = self._read_selection(spec.config_path, source)
        if cfg.resume:
            saved = spec.resolved
            if not saved or saved.get("source") != source:
                raise ValueError("Source channel definitions or dataset identity changed during resume")
            selected = saved["selected"]
            validate_mapping(selected)
            if requested is not None and requested != selected:
                raise ValueError("Channel selection changed during resume; start a new run")
        else:
            if requested is None:
                raise ValueError("Channel selection requires training_adapter.config_path")
            selected = requested
        for key in VECTORS.values():
            if not set(selected[key]) <= set(source[key]):
                raise ValueError(f"Unknown channels in {key}")
        self.mapping = selected
        self.resolved = {"source": source, "selected": selected}
        train_view = SelectedDataset(train_dataset, selected)
        eval_view = None
        if eval_dataset is not None:
            eval_source = mapping_from_features(
                eval_dataset.meta.features, repo_id=cfg.dataset.repo_id, revision=cfg.dataset.revision
            )
            if eval_source != source:
                raise ValueError("Train/eval channel definitions differ")
            eval_view = SelectedDataset(eval_dataset, selected)

        features = dataset_to_policy_features(train_view.meta.features)
        # Populate all features when the policy has none, preserving image
        # discovery. Existing checkpoint visual features remain unchanged.
        if not cfg.policy.input_features:
            cfg.policy.input_features = {k: v for k, v in features.items() if k != "action"}
        else:
            cfg.policy.input_features = {
                **cfg.policy.input_features,
                "observation.state": features["observation.state"],
            }
        cfg.policy.output_features = {**cfg.policy.output_features, "action": features["action"]}
        if hasattr(cfg.policy, "action_feature_names"):
            cfg.policy.action_feature_names = list(selected["action_names"])
        self._validate_relative(cfg.policy)
        cfg.policy.validate_features()
        return AdaptedTrainingData(train_view, eval_view, self.resolved, {FILENAME: selected})

    @staticmethod
    def _read_selection(path, source):
        with Path(path).open() as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or set(data) != set(VECTORS.values()):
            raise ValueError("Selection YAML requires exactly state_names and action_names")
        mapping = {"version": 1, "dataset": deepcopy(source["dataset"])}
        for key in VECTORS.values():
            mapping[key] = list(source[key]) if data[key] == "all" else data[key]
        validate_mapping(mapping)
        for key in VECTORS.values():
            missing = set(mapping[key]) - set(source[key])
            if missing:
                raise ValueError(f"Unknown {key}: {sorted(missing)}")
        return mapping

    def _validate_relative(self, config):
        if not getattr(config, "use_relative_actions", False):
            return
        self._validate_relative_order(getattr(config, "relative_exclude_joints", []))

    def _validate_relative_order(self, excludes):
        states, actions = self.mapping["state_names"], self.mapping["action_names"]
        excludes = [name.lower() for name in excludes if name]
        if len(states) < len(actions):
            raise ValueError("Relative actions require a state prefix covering all actions")
        for index, name in enumerate(actions):
            if not any(token in name.lower() for token in excludes) and states[index] != name:
                raise ValueError(f"Relative state/action prefix mismatch at {index}: {name}")

    def validate_policy(self, cfg, preprocessor, postprocessor):
        validate_mapping(
            self.mapping,
            state_dim=feature_dim(cfg.policy.input_features, "observation.state"),
            action_dim=feature_dim(cfg.policy.output_features, "action"),
        )
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        from lerobot.processor.relative_action_processor import (
            AbsoluteActionsProcessorStep,
            RelativeActionsProcessorStep,
        )

        relative_steps = [
            step
            for step in preprocessor.steps
            if isinstance(step, RelativeActionsProcessorStep) and step.enabled
        ]
        for step in relative_steps:
            if step.action_names is not None and list(step.action_names) != self.mapping["action_names"]:
                raise ValueError("Relative processor action names differ from selected channels")
            self._validate_relative_order(step.exclude_joints if step.action_names is not None else [])
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.enabled:
                if not any(step.relative_step is relative for relative in relative_steps):
                    raise ValueError("Absolute processor has no matching enabled relative processor")
        if bool(relative_steps) != any(
            isinstance(step, AbsoluteActionsProcessorStep) and step.enabled for step in postprocessor.steps
        ):
            raise ValueError("Relative and absolute processors must be enabled together")
        for pipeline in (preprocessor, postprocessor):
            for step in pipeline.steps:
                if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                    for feature, key in VECTORS.items():
                        if feature not in step.features:
                            continue
                        dim = len(self.mapping[key])
                        if feature_dim(step.features, feature) != dim:
                            raise ValueError(f"Processor feature mismatch: {feature}")
                        select_stats(step.stats.get(feature, {}), list(range(dim)), dim, feature)
                        mode = step.norm_map.get(step.features[feature].type)
                        required = {
                            "MEAN_STD": {"mean", "std"},
                            "MIN_MAX": {"min", "max"},
                            "QUANTILES": {"q01", "q99"},
                            "QUANTILE10": {"q10", "q90"},
                        }
                        if (
                            not required.get(getattr(mode, "value", mode), set())
                            <= step.stats.get(feature, {}).keys()
                        ):
                            raise ValueError(f"Missing normalization statistics: {feature}")
        self._validate_relative(cfg.policy)

"""Read prepared v2.1 episodes without modifying or converting the dataset."""

from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from abot_engine.bundle import Normalization
from .channels import resolve_channels, resolve_dimensions
from .profile import canonical_hash, inside, read_json, validate_profile


class PreparedDataset:
    def __init__(self, root, profile, *, horizon, image_size, statistics=None, channel_selection=None):
        self.root = Path(root).resolve(strict=True)
        self.info = read_json(inside(self.root, "meta/info.json"))
        self.profile = validate_profile(profile, self.info, channel_selection=channel_selection)
        self.channels = resolve_channels(self.info, channel_selection)
        self.selected = self.channels["selected"]
        self.model_dimensions = resolve_dimensions(profile["padding"], self.selected)
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if (not isinstance(image_size, list) or len(image_size) != 2
                or any(type(n) is not int or n < 1 for n in image_size)):
            raise ValueError("image_size must be [width, height], matching ABot")
        self.horizon = horizon
        self.image_size = list(image_size)
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset fps must be positive and finite")
        if type(self.info["chunks_size"]) is not int or self.info["chunks_size"] < 1:
            raise ValueError("Invalid chunks_size")
        self.episodes = self._jsonl("meta/episodes.jsonl")
        ids = [e["episode_index"] for e in self.episodes]
        if (any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids))
                or len(ids) != self.info["total_episodes"]
                or any(type(e.get("length")) is not int or e["length"] < 1 for e in self.episodes)
                or sum(e["length"] for e in self.episodes) != self.info["total_frames"]):
            raise ValueError("Episode metadata is inconsistent")
        # Do not silently normalize on validation/test episodes.
        if self.info.get("splits") != {"train": f"0:{len(ids)}"} or ids != list(range(len(ids))):
            raise ValueError("This recipe requires a contiguous, train-only prepared dataset")
        self.tasks = {}
        for task in self._jsonl("meta/tasks.jsonl"):
            index, text = task["task_index"], task.get("task")
            if type(index) is not int or index in self.tasks or not isinstance(text, str) or not text.strip():
                raise ValueError("Expected unique task IDs with textual instructions")
            self.tasks[index] = text
        self.ends = np.cumsum([max(0, e["length"] - horizon + 1) for e in self.episodes]).tolist()
        if not self.ends or not self.ends[-1]:
            raise ValueError("No complete within-episode action chunks")
        self._cache = OrderedDict()
        for episode in self.episodes:
            self._path(episode, "data_path")
            for camera in profile["cameras"].values():
                self._path(episode, "video_path", camera)
        statistics_contract = {
            "version": 1, "channels": self.channels, "normalization": profile["normalization"],
            "dataset_info_sha256": canonical_hash(self.info),
        }
        if statistics is None:
            self.statistics = {**deepcopy(statistics_contract), **self._statistics()}
        else:
            if any(statistics.get(k) != v for k, v in statistics_contract.items()):
                raise ValueError("Saved statistics channel order, normalization or dataset metadata differs")
            self.statistics = deepcopy(statistics)
        mode = {"quantile": "q99"}.get(profile["normalization"], profile["normalization"])
        self.normalizers = {
            key: Normalization([mode] * len(self.selected[f"{key}_names"]),
                               self.statistics[key], len(self.selected[f"{key}_names"]))
            for key in ("state", "action")
        }

    def _jsonl(self, path):
        with inside(self.root, path).open() as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def _path(self, episode, template, camera=None):
        index = episode["episode_index"]
        relative = self.info[template].format(
            episode_chunk=index // self.info["chunks_size"], episode_index=index, video_key=camera,
        )
        return inside(self.root, relative)

    def _table(self, episode):
        import pyarrow.parquet as pq

        index = episode["episode_index"]
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        columns = ["observation.state", "action", "timestamp", "episode_index", "task_index"]
        table = pq.read_table(self._path(episode, "data_path"), columns=columns).to_pydict()
        length = episode["length"]
        if (len(table["timestamp"]) != length
                or any(i != index for i in table["episode_index"])
                or any(type(i) is not int or i not in self.tasks for i in table["task_index"])):
            raise ValueError(f"Invalid episode or task index in episode {index}")
        for kind, column in (("state", "observation.state"), ("action", "action")):
            vector = np.asarray(table[column], dtype=np.float32)
            if vector.shape != (length, len(self.profile[f"{kind}_names"])):
                raise ValueError(f"Invalid {kind} vector shape in episode {index}")
            selected = vector[:, self.channels["indices"][column]]
            if not np.isfinite(selected).all():
                raise ValueError(f"Invalid selected {kind} values in episode {index}")
            table[column] = selected
        stamps = np.asarray(table["timestamp"], dtype=np.float64)
        if (not np.isfinite(stamps).all()
                or not np.allclose(stamps, np.arange(length) / self.fps, atol=1e-4, rtol=0)):
            raise ValueError(f"Expected episode-relative, constant-fps timestamps in episode {index}")
        self._cache[index] = table
        if len(self._cache) > 2:
            self._cache.popitem(last=False)
        return table

    def _statistics(self):
        result = {}
        for kind, column in (("state", "observation.state"), ("action", "action")):
            count, mean, m2, low, high = 0, None, None, None, None
            quantiles = []
            for episode in self.episodes:
                values = self._table(episode)[column].astype(np.float64)
                n, avg = len(values), values.mean(axis=0)
                variance_sum = ((values - avg) ** 2).sum(axis=0)
                if count:
                    delta = avg - mean
                    m2 += variance_sum + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    low, high = np.minimum(low, values.min(axis=0)), np.maximum(high, values.max(axis=0))
                else:
                    mean, m2 = avg, variance_sum
                    low, high = values.min(axis=0), values.max(axis=0)
                count += n
                if self.profile["normalization"] == "quantile":
                    quantiles.append(values)
            stats = {"count": count, "mean": mean.tolist(), "std": np.sqrt(m2 / count).tolist(),
                     "min": low.tolist(), "max": high.tolist()}
            if quantiles:
                q01, q99 = np.quantile(np.concatenate(quantiles), [0.01, 0.99], axis=0)
                stats.update(q01=q01.tolist(), q99=q99.tolist())
            result[kind] = stats
        return result

    def __len__(self):
        return self.ends[-1]

    def _image(self, episode, camera, timestamp):
        import av

        with av.open(str(self._path(episode, "video_path", camera))) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
            nearest, distance = None, float("inf")
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError("Video frame has no presentation timestamp")
                diff = abs(frame.time - timestamp)
                if diff < distance:
                    nearest, distance = frame, diff
                if frame.time >= timestamp:
                    break
            if nearest is None or distance > 0.5 / self.fps:
                raise ValueError(f"Video timestamp mismatch: episode={episode['episode_index']} camera={camera}")
            image = nearest.to_image().convert("RGB")
        return image.resize(tuple(self.image_size), Image.Resampling.BICUBIC)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_pos = bisect_right(self.ends, index)
        episode = self.episodes[episode_pos]
        offset = index - (self.ends[episode_pos - 1] if episode_pos else 0)
        table = self._table(episode)
        state = np.zeros((1, self.model_dimensions["state_dim"]), dtype=np.float32)
        action = np.zeros((self.horizon, self.model_dimensions["action_dim"]), dtype=np.float32)
        state[0, :len(self.selected["state_names"])] = self.normalizers["state"].apply(table["observation.state"][offset])
        raw = table["action"][offset:offset + self.horizon]
        action[:, :raw.shape[1]] = self.normalizers["action"].apply(raw)
        mask = np.arange(action.shape[1]) < raw.shape[1]
        return {
            "state": state, "action": action, "action_mask": mask,
            "image": [self._image(episode, self.profile["cameras"][role], table["timestamp"][offset])
                      for role in self.profile["camera_order"]],
            "lang": self.tasks[table["task_index"][offset]],
        }

    def configure_model(self, config):
        """Resolve dimensions before model construction or pretrained loading."""
        result = deepcopy(config)
        framework = result["framework"]
        if framework["name"] != "ABot_M0":
            raise ValueError("Expected an ABot_M0 model config")
        framework["use_vggt"] = False
        framework["action_model"].update(
            **self.model_dimensions, action_horizon=self.horizon,
            future_action_window_size=self.horizon - 1, past_action_window_size=0,
        )
        result.setdefault("datasets", {}).setdefault("vla_data", {}).update(
            include_state=True, action_mode="abs", image_size=list(self.image_size),
        )
        return result

    def checkpoint_metadata(self, run_config, model_config, initialization):
        """Freeze the raw/selected contract and initialization report together."""
        if self.configure_model(model_config) != model_config:
            raise ValueError("Model config differs from selected training dimensions or transforms")
        if ("channel_selection" in run_config
                and resolve_channels(self.info, run_config["channel_selection"]) != self.channels):
            raise ValueError("Run channel selection differs from the training dataset")
        source = deepcopy(self.channels["source"])
        selected = deepcopy(self.selected)
        dataset = run_config.get("dataset", {})
        if dataset:
            identity = {"repo_id": dataset.get("repo_id"), "revision": dataset.get("revision")}
            source["dataset"] = deepcopy(identity)
            selected["dataset"] = deepcopy(identity)
        train_config = deepcopy(run_config)
        train_config.update(
            channel_selection={key: list(self.selected[key]) for key in ("state_names", "action_names")},
            training_adapter={"name": "cyclo_channels", "resolved": {"source": source, "selected": selected}},
            model_dimensions=deepcopy(self.model_dimensions), initialization=deepcopy(initialization),
        )
        return {
            "profile.json": deepcopy(self.profile), "train_config.json": train_config,
            "model_config.json": deepcopy(model_config), "dataset_statistics.json": deepcopy(self.statistics),
            "cyclo_io_mapping.json": deepcopy(selected), "initialization.json": deepcopy(initialization),
        }

    @classmethod
    def from_checkpoint(cls, root, checkpoint):
        from .checkpoint import verify_checkpoint

        verify_checkpoint(checkpoint)
        directory = Path(checkpoint)
        metadata = {name: read_json(directory / name) for name in (
            "profile.json", "train_config.json", "model_config.json", "dataset_statistics.json",
            "cyclo_io_mapping.json", "initialization.json",
        )}
        model_config = metadata["model_config.json"]
        mapping = metadata["cyclo_io_mapping.json"]
        instance = cls(
            root, metadata["profile.json"],
            horizon=model_config["framework"]["action_model"]["action_horizon"],
            image_size=model_config["datasets"]["vla_data"]["image_size"],
            channel_selection={key: mapping[key] for key in ("state_names", "action_names")},
            statistics=metadata["dataset_statistics.json"],
        )
        expected = instance.checkpoint_metadata(
            metadata["train_config.json"], model_config, metadata["initialization.json"],
        )
        if expected != metadata:
            raise ValueError("Checkpoint training config, I/O mapping or initialization report disagree")
        return instance

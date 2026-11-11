"""Read prepared LeRobot v2.1 episodes; never convert or rewrite datasets."""

import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .inputs import InputContract, InputTransform, plan_frames


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contained_file(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"File escapes prepared dataset: {name}")
    return path


class PreparedEpisodes:
    def __init__(self, root, profile):
        self.root = Path(root).resolve(strict=True)
        self.info = json.loads(contained_file(self.root, "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError("Expected prepared v2.1 data; format conversion belongs to Bench")
        self.contract = InputContract.from_profile(profile, self.info)
        fps = self.info.get("fps")
        if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Dataset FPS must be finite and positive")
        split = self.info.get("splits", {}).get("train", "")
        if not isinstance(split, str) or not re.fullmatch(r"[0-9]+:[0-9]+", split):
            raise ValueError("A contiguous explicit train episode split is required")
        first, end = map(int, split.split(":"))
        if not 0 <= first < end <= self.info["total_episodes"]:
            raise ValueError("Invalid train split")
        chunks = self.info.get("chunks_size")
        if type(chunks) is not int or chunks <= 0:
            raise ValueError("Invalid v2.1 chunks_size")
        tasks = _jsonl(contained_file(self.root, "meta/tasks.jsonl"))
        self.tasks = {item["task_index"]: item["task"] for item in tasks}
        if len(self.tasks) != len(tasks) or any(not isinstance(t, str) or not t for t in self.tasks.values()):
            raise ValueError("Invalid task metadata")
        episodes = _jsonl(contained_file(self.root, "meta/episodes.jsonl"))
        by_index = {item["episode_index"]: item for item in episodes}
        if len(by_index) != len(episodes):
            raise ValueError("Duplicate episode index")
        self.episodes = []
        for index in range(first, end):
            if index not in by_index:
                raise ValueError(f"Missing train episode {index}")
            fmt = dict(episode_index=index, episode_chunk=index // chunks)
            path = contained_file(self.root, self.info["data_path"].format(**fmt))
            table = pq.read_table(path, columns=["observation.state", "action", "timestamp", "episode_index", "task_index"]).to_pydict()
            count = len(table["action"])
            if count != by_index[index]["length"] or count < 1 or set(table["episode_index"]) != {index}:
                raise ValueError(f"Episode metadata/Parquet mismatch: {index}")
            stamps = np.asarray(table["timestamp"], dtype=np.float64).reshape(-1)
            fps = self.info["fps"]
            if (
                not np.isfinite(stamps).all() or stamps.shape != (count,)
                or stamps[0] < 0 or not np.allclose(np.diff(stamps), 1 / fps, atol=1e-4, rtol=1e-4)
            ):
                raise ValueError(f"Episode {index} does not have the declared fixed time axis")
            item = {"episode_index": index, "timestamps": stamps, "tasks": table["task_index"], "videos": {}}
            if any(task not in self.tasks for task in item["tasks"]):
                raise ValueError(f"Episode {index} references an unknown task")
            for kind, column in (("state", "observation.state"), ("action", "action")):
                values = np.asarray(table[column], dtype=np.float32)
                layout = getattr(self.contract, kind)
                if values.shape != (count, len(layout.names)) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {kind} values in episode {index}")
                item[kind] = values
            for _, key in self.contract.cameras:
                item["videos"][key] = contained_file(self.root, self.info["video_path"].format(**fmt, video_key=key))
            self.episodes.append(item)
        self.ends = np.cumsum([len(ep["action"]) for ep in self.episodes])

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        local = index - (int(self.ends[episode - 1]) if episode else 0)
        return self.episodes[episode], local

    def statistics(self):
        # Parallel variance combination avoids cancellation on nearly constant joints
        # and never samples overlapping future windows or validation episodes.
        result = {}
        for kind in ("state", "action"):
            count, mean, m2 = 0, 0.0, 0.0
            for episode in self.episodes:
                values = episode[kind].astype(np.float64)
                n, batch_mean = len(values), values.mean(axis=0)
                delta = batch_mean - mean
                m2 = m2 + ((values - batch_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                mean = mean + delta * n / (count + n)
                count += n
            result[kind] = {"names": list(getattr(self.contract, kind).names), "count": count,
                            "mean": mean.tolist(), "std": np.sqrt(m2 / count).tolist()}
        return result


class TrainingDataset(Dataset):
    def __init__(self, episodes, stats, processor, model_config, *, chunk_size, image_size, warmup, image_augment=False):
        self.episodes = episodes
        self.transform = InputTransform(episodes.contract, stats)
        self.processor, self.config = processor, model_config
        self.chunk_size, self.image_size, self.warmup = chunk_size, image_size, warmup
        self.image_augment = image_augment

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        from torchvision.transforms.v2 import Resize
        from lingbotvla.data.vla_data.transform import prepare_images, prepare_language
        from lingbotvla.data.vla_data.video_utils import decode_video_frames

        episode, local = self.episodes.locate(index)
        plan = plan_frames(frame_index=local, episode_length=len(episode["action"]),
                           chunk_size=self.chunk_size, fps=self.episodes.info["fps"], use_warmup_frame=self.warmup)
        values = self.transform.prepare(episode["state"][local], episode["action"][list(plan.action_indices)],
                                        np.asarray(plan.action_is_pad, dtype=bool))
        state = torch.from_numpy(values["state"])
        current, future = {}, {}
        resize = Resize((self.image_size, self.image_size), antialias=True)
        for _, key in self.episodes.contract.cameras:
            timestamps = [float(episode["timestamps"][local]), float(episode["timestamps"][plan.action_indices[-1]])]
            frames = decode_video_frames(episode["videos"][key], timestamps, 1 / self.episodes.info["fps"] / 2, backend="pyav")
            current[key], future[key] = resize(frames[0]), resize(frames[1])
        keys = list(current)
        images, img_masks, pil, grid, augment = prepare_images(
            self.processor.image_processor, {"state": state, "image": current}, image_keys=keys,
            train=self.image_augment, use_depth_align=True, return_image_grid_thw=True, return_augment_params=True,
        )
        _, _, future_pil, _ = prepare_images(
            self.processor.image_processor, {"state": state, "image": future}, image_keys=keys,
            train=self.image_augment, use_depth_align=True, augment_params=augment,
        )
        tokens, masks = prepare_language(self.config, self.processor.tokenizer,
                                         {"state": state, "prompt": [self.episodes.tasks[episode["tasks"][local]]]})
        return {"state": state, "actions": torch.from_numpy(values["action"]),
                "joint_mask": torch.from_numpy(values["action_loss_mask"]),
                "action_is_pad": torch.from_numpy(values["action_is_pad"]),
                "images": images, "img_masks": img_masks, "image_grid_thw": grid,
                "lang_tokens": tokens, "lang_masks": masks,
                "pil_images": torch.as_tensor(pil), "future_pil_images": torch.as_tensor(future_pil),
                "future_valid": torch.tensor(plan.future_valid), "effective_fps": torch.tensor(plan.effective_fps)}

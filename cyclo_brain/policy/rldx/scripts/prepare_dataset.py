"""Export a LeRobot v3 dataset to RLDX's per-episode v2.1 input layout.

The source is read-only. Videos are frame-trimmed and losslessly re-encoded;
there is no resize, rotation, joint remapping, or instruction substitution.
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def task_map(frame):
    if "task_index" not in frame:
        raise ValueError("tasks.parquet has no task_index")
    result = {}
    for index, row in frame.iterrows():
        text = row.get("task", index)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Missing textual task instruction; integer row index is not a task")
        key = int(row.task_index)
        if key in result:
            raise ValueError(f"Duplicate task_index {key}")
        result[key] = text
    return result


def feature_names(info, key):
    feature = info["features"][key]
    names = feature.get("names")
    if (not isinstance(names, list) or len(names) != feature["shape"][0]
            or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{key}: ordered, unique channel names are required")
    return names


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] == "v2.1":
        return prepare_v21(source, destination, info)
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected a LeRobot v2.1 or v3.0 dataset")
    names = {k: feature_names(info, k) for k in ("observation.state", "action")}
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    tasks = task_map(pd.read_parquet(source / "meta/tasks.parquet"))
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / "meta/episodes").rglob("*.parquet"))])
    episodes = episodes.sort_values("episode_index")
    if len(episodes) != info["total_episodes"] or episodes.episode_index.duplicated().any():
        raise ValueError("Episode metadata count/IDs are inconsistent")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "INCOMPLETE").touch()
    metadata, vectors = [], {k: [] for k in names}
    fps = float(info["fps"])
    for _, episode in episodes.iterrows():
        eid, length = int(episode.episode_index), int(episode.length)
        data_file = source / info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"]))
        frame = pd.read_parquet(data_file)
        frame = frame.loc[frame.episode_index == eid].copy()
        if len(frame) != length or not np.array_equal(frame.frame_index.to_numpy(), np.arange(length)):
            raise ValueError(f"Episode {eid}: frame count/order mismatch")
        if set(map(int, frame.task_index)) - tasks.keys():
            raise ValueError(f"Episode {eid}: unknown task index")
        if not np.allclose(frame.timestamp, np.arange(length) / fps, atol=1e-4):
            raise ValueError(f"Episode {eid}: timestamps are not aligned to dataset FPS")
        for key, channels in names.items():
            values = np.stack(frame[key])
            if values.shape != (length, len(channels)) or not np.isfinite(values).all():
                raise ValueError(f"Episode {eid}: invalid {key}")
            vectors[key].append(values)
        chunk = eid // 1000
        output = destination / f"data/chunk-{chunk:03d}/episode_{eid:06d}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        for camera in cameras:
            prefix = f"videos/{camera}"
            start = float(episode[f"{prefix}/from_timestamp"]) * fps
            end = float(episode[f"{prefix}/to_timestamp"]) * fps
            if abs(start - round(start)) > 1e-3 or abs(end - start - length) > 1e-3:
                raise ValueError(f"Episode {eid}: ambiguous video frame boundary for {camera}")
            video = source / info["video_path"].format(
                video_key=camera, chunk_index=int(episode[f"{prefix}/chunk_index"]),
                file_index=int(episode[f"{prefix}/file_index"]))
            out = destination / f"videos/chunk-{chunk:03d}/{camera}/episode_{eid:06d}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(video),
                "-vf", f"trim=start_frame={round(start)}:end_frame={round(start) + length},setpts=PTS-STARTPTS",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-threads", "2", str(out),
            ], check=True)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(out),
            ]))["streams"][0]
            expected = info["features"][camera]["info"]
            if (int(probe["nb_read_frames"]) != length or probe["height"] != expected["video.height"]
                    or probe["width"] != expected["video.width"]):
                raise ValueError(f"Episode {eid}: exported video mismatch for {camera}")
        metadata.append({"episode_index": eid, "length": length,
                         "tasks": [tasks[i] for i in sorted(set(map(int, frame.task_index)))]})
        print(f"Exported episode {eid}: {length} frames", flush=True)
    if sum(m["length"] for m in metadata) != info["total_frames"]:
        raise ValueError("Exported total frame count mismatch")
    stats = {}
    for key, values in vectors.items():
        data = np.concatenate(values).astype(np.float64)
        stats[key] = {name: value.tolist() for name, value in {
            "min": data.min(0), "max": data.max(0), "mean": data.mean(0), "std": data.std(0),
            "q01": np.quantile(data, .01, axis=0), "q99": np.quantile(data, .99, axis=0),
        }.items()}
    write_json(destination / "meta/stats.json", stats)
    info.update(codebase_version="v2.1", chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    write_json(destination / "meta/info.json", info)
    write_json(destination / "meta/modality.json", {
        "state": {"joint_position": {"start": 0, "end": len(names["observation.state"])}},
        "action": {"joint_position": {"start": 0, "end": len(names["action"])}},
        "video": {k.removeprefix("observation.images.rgb."): {"original_key": k} for k in cameras},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    })
    for name, rows in (("episodes", metadata), ("tasks", [
        {"task_index": i, "task": t} for i, t in tasks.items()])):
        (destination / f"meta/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(destination / "cyclo_input_metadata.json", {
        "robot_type": info["robot_type"], "fps": fps,
        "state_names": names["observation.state"], "action_names": names["action"],
        "cameras": {k.removeprefix("observation.images.rgb."): k for k in cameras},
    })
    (destination / "INCOMPLETE").unlink()


def prepare_v21(source, destination, info):
    """Validate a per-episode v2.1 dataset and add Run-local Cyclo metadata.

    The original Parquet/video tree is referenced read-only; metadata and stats
    are validated and rewritten under the Run so the source is never modified.
    """
    import numpy as np
    import pandas as pd

    source, destination = Path(source).resolve(), Path(destination)
    names = {key: feature_names(info, key) for key in ("observation.state", "action")}
    cameras = [key for key, feature in info["features"].items()
               if feature.get("dtype") == "video"]
    if not cameras:
        raise ValueError("RLDX requires at least one video camera")
    tasks_path = source / "meta/tasks.jsonl"
    episodes_path = source / "meta/episodes.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    task_text = {}
    for task in tasks:
        index, instruction = task.get("task_index"), task.get("task")
        if (type(index) is not int or index in task_text or not isinstance(instruction, str)
                or not instruction.strip()):
            raise ValueError("v2.1 task instructions are missing or duplicated")
        task_text[index] = instruction
    if not task_text or len(episodes) != info.get("total_episodes"):
        raise ValueError("v2.1 task or episode metadata is inconsistent")
    fps = float(info["fps"])
    chunks = int(info["chunks_size"])
    if not np.isfinite(fps) or fps <= 0 or chunks <= 0:
        raise ValueError("v2.1 FPS or chunk size is invalid")
    vectors = {key: [] for key in names}
    seen = set()
    total = 0
    for episode in episodes:
        eid, length = episode.get("episode_index"), episode.get("length")
        if (type(eid) is not int or eid < 0 or eid in seen or type(length) is not int
                or length < 1):
            raise ValueError("v2.1 episode index or length is invalid")
        seen.add(eid)
        total += length
        if (not isinstance(episode.get("tasks"), list) or not episode["tasks"]
                or any(task not in task_text.values() for task in episode["tasks"])):
            raise ValueError(f"Episode {eid}: textual task instructions are invalid")
        format_values = {"episode_chunk": eid // chunks, "episode_index": eid}
        data_path = source / info["data_path"].format(**format_values)
        if not data_path.resolve().is_relative_to(source) or not data_path.is_file():
            raise ValueError(f"Episode {eid}: Parquet path is invalid")
        frame = pd.read_parquet(data_path)
        if len(frame) != length or "episode_index" not in frame or not np.all(
            frame["episode_index"].to_numpy() == eid
        ):
            raise ValueError(f"Episode {eid}: frame count/episode mismatch")
        if "frame_index" in frame:
            ordered = np.array_equal(frame["frame_index"].to_numpy(), np.arange(length))
        elif "index" in frame:
            # Standard LeRobot v2.1 uses one global index across episodes.
            indices = frame["index"].to_numpy()
            ordered = (np.issubdtype(indices.dtype, np.integer)
                       and np.array_equal(indices, np.arange(int(indices[0]), int(indices[0]) + length)))
        else:
            ordered = False
        if not ordered or "timestamp" not in frame or not np.allclose(
            frame["timestamp"].to_numpy(), np.arange(length) / fps, atol=1e-4,
        ):
            raise ValueError(f"Episode {eid}: frame order or timestamps mismatch")
        if "task_index" not in frame or set(map(int, frame["task_index"])) - task_text.keys():
            raise ValueError(f"Episode {eid}: unknown task index")
        for key, channels in names.items():
            values = np.stack(frame[key])
            if values.shape != (length, len(channels)) or not np.isfinite(values).all():
                raise ValueError(f"Episode {eid}: invalid {key}")
            vectors[key].append(values)
        for camera in cameras:
            video_path = source / info["video_path"].format(
                **format_values, video_key=camera,
            )
            if not video_path.resolve().is_relative_to(source) or not video_path.is_file():
                raise ValueError(f"Episode {eid}: video path is invalid for {camera}")
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(video_path),
            ]))["streams"][0]
            expected = info["features"][camera]["info"]
            if (int(probe["nb_read_frames"]) != length
                    or int(probe["height"]) != expected["video.height"]
                    or int(probe["width"]) != expected["video.width"]):
                raise ValueError(f"Episode {eid}: video frame count or dimensions differ for {camera}")
    if total != info.get("total_frames"):
        raise ValueError("v2.1 total frame count differs from meta/info.json")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "INCOMPLETE").touch()
    (destination / "data").symlink_to(source / "data", target_is_directory=True)
    (destination / "videos").symlink_to(source / "videos", target_is_directory=True)
    write_json(destination / "meta/info.json", info)
    for name in ("tasks.jsonl", "episodes.jsonl"):
        (destination / "meta" / name).write_bytes((source / "meta" / name).read_bytes())
    stats = {}
    for key, arrays in vectors.items():
        data = np.concatenate(arrays).astype(np.float64)
        stats[key] = {name: value.tolist() for name, value in {
            "min": data.min(0), "max": data.max(0), "mean": data.mean(0), "std": data.std(0),
            "q01": np.quantile(data, .01, axis=0), "q99": np.quantile(data, .99, axis=0),
        }.items()}
    write_json(destination / "meta/stats.json", stats)
    write_json(destination / "meta/modality.json", {
        "state": {"joint_position": {"start": 0, "end": len(names["observation.state"])}},
        "action": {"joint_position": {"start": 0, "end": len(names["action"])}},
        "video": {key.removeprefix("observation.images.rgb."): {"original_key": key}
                  for key in cameras},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    })
    write_json(destination / "cyclo_input_metadata.json", {
        "robot_type": info["robot_type"], "fps": fps,
        "state_names": names["observation.state"], "action_names": names["action"],
        "cameras": {key.removeprefix("observation.images.rgb."): key for key in cameras},
    })
    (destination / "INCOMPLETE").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)

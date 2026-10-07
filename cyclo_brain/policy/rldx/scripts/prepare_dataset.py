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
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected a LeRobot v3.0 dataset")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)

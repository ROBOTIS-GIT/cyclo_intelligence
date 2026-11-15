"""Analyze and materialize intervention-focused rosbag2 dataset copies.

The source MCAP is never rewritten.  A curated copy keeps byte-identical MCAP
files while its camera MP4 files and timestamp sidecars are physically sliced
to the accepted start/end window.  The module deliberately has no ROS service
dependencies so its safety and frame-alignment rules can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Callable, Iterable, Optional
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cyclo_data.converter.base_converter import ConversionConfig
from cyclo_data.converter.pipeline_worker import _convert_single_episode_worker
from cyclo_data.converter.to_lerobot_v21 import RosbagToLerobotConverter


SCHEMA_VERSION = 1
DEFAULT_INTERVENTION_TOPIC = "/leader/intervention_mode"
DEFAULT_MOTION_THRESHOLD_RAD = 5e-4
DEFAULT_ACTION_PAIR_TOLERANCE_MS = 50.0
DEFAULT_TRAILING_STATIONARY_SECONDS = 2.0
DEFAULT_FPS = 15
DEFAULT_ACTION_TOPICS = [
    "/command_source_arbiter/executed/arm_left",
    "/command_source_arbiter/executed/arm_right",
]

DEFAULT_SYNC_THRESHOLDS = {
    "camera_joint_nearest_max_ms": 20.0,
    "camera_camera_nearest_p99_ms": 40.0,
    "camera_camera_nearest_max_ms": 133.334,
    "left_right_action_pair_max_ms": 50.0,
    "joint_state_gap_max_ms": 25.0,
    "camera_to_grid_abs_max_ms": 75.0,
    "selected_camera_pair_max_ms": 133.334,
}


class CurationCancelled(RuntimeError):
    """Raised at safe checkpoints when the user cancels a curation job."""


ProgressCallback = Callable[[float, str, str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_digest(path: Path) -> str:
    """Return one stable digest for every file in an episode directory."""
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _verify_mcap_files_unchanged(source: Path, destination: Path) -> None:
    """Verify every MCAP in a copied episode is present and byte-identical."""
    source_files = {
        item.relative_to(source): _sha256_file(item)
        for item in source.rglob("*.mcap")
        if item.is_file()
    }
    destination_files = {
        item.relative_to(destination): _sha256_file(item)
        for item in destination.rglob("*.mcap")
        if item.is_file()
    }
    if not source_files:
        raise RuntimeError(f"source episode has no MCAP file: {source}")
    if source_files != destination_files:
        raise RuntimeError(f"curated MCAP files changed: {destination}")


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _timestamp_column(table: pa.Table) -> str:
    for candidate in ("header_stamp_ns", "recv_ns"):
        if candidate in table.column_names:
            return candidate
    raise RuntimeError("timestamp sidecar has no header_stamp_ns or recv_ns column")


def _timestamp_values(table: pa.Table) -> np.ndarray:
    values = np.asarray(table[_timestamp_column(table)].to_numpy(), dtype=np.int64)
    if values.size == 0:
        raise RuntimeError("timestamp sidecar is empty")
    if np.any(np.diff(values) <= 0):
        raise RuntimeError("camera timestamps are not strictly increasing")
    return values


def _slice_bounds(stamps: np.ndarray, start_ns: int, end_ns: int) -> tuple[int, int]:
    start = int(np.searchsorted(stamps, int(start_ns), side="left"))
    stop = int(np.searchsorted(stamps, int(end_ns), side="right"))
    if start >= len(stamps) or stop <= start:
        raise RuntimeError(
            f"trim window has no camera frames: start={start_ns}, end={end_ns}"
        )
    return start, min(stop, len(stamps))


def _nearest_abs_ns(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.asarray([], dtype=np.int64)
    positions = np.searchsorted(reference, query)
    lower = np.clip(positions - 1, 0, len(reference) - 1)
    upper = np.clip(positions, 0, len(reference) - 1)
    return np.minimum(
        np.abs(query - reference[lower]),
        np.abs(query - reference[upper]),
    )


def _stats_ms(values: Iterable[int] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                       dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "p50_ms": 0.0, "p95_ms": 0.0,
            "p99_ms": 0.0, "max_ms": 0.0,
        }
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50) / 1e6),
        "p95_ms": float(np.percentile(array, 95) / 1e6),
        "p99_ms": float(np.percentile(array, 99) / 1e6),
        "max_ms": float(np.max(array) / 1e6),
    }


def _last_motion_frame(values: np.ndarray, threshold: float) -> Optional[int]:
    if values.ndim != 2 or len(values) < 2:
        return None
    deltas = np.max(np.abs(np.diff(values.astype(np.float64), axis=0)), axis=1)
    moving = np.flatnonzero((~np.isfinite(deltas)) | (deltas > threshold))
    return int(moving[-1]) + 1 if moving.size else None


def _selected_state_matrix(episode: Any) -> np.ndarray:
    states = np.asarray(episode.observation_state, dtype=np.float64)
    if states.ndim != 2:
        return states
    state_names = list(getattr(episode, "observation_state_names", []) or [])
    action_names = set(getattr(episode, "action_names", []) or [])
    if state_names and action_names:
        indices = [index for index, name in enumerate(state_names) if name in action_names]
        if indices:
            return states[:, indices]
    if state_names:
        indices = [
            index for index, name in enumerate(state_names)
            if str(name).startswith(("arm_", "gripper_"))
        ]
        if indices:
            return states[:, indices]
    return states


class InterventionAnalysisConverter(RosbagToLerobotConverter):
    """Pair two arm command streams tolerantly and expose pairing metrics."""

    def __init__(self, config: ConversionConfig, pair_tolerance_sec: float):
        super().__init__(config)
        self.pair_tolerance_sec = float(pair_tolerance_sec)
        self.last_action_pairing: dict[str, Any] = {}
        self.raw_state_times_sec = np.asarray([], dtype=np.float64)

    def _resample_to_fps(
        self,
        episode,
        state_messages,
        action_messages,
        start_time,
    ):
        # Retain raw, header-derived joint timestamps for the curation sync
        # audit. The regular EpisodeData intentionally only exposes the
        # resampled target grid.
        self.raw_state_times_sec = np.asarray(
            [timestamp for timestamp, _ in state_messages], dtype=np.float64
        )
        return super()._resample_to_fps(
            episode, state_messages, action_messages, start_time
        )

    def _merge_action_messages(self, action_messages_by_topic, action_joint_names_by_topic):
        if len(action_messages_by_topic) != 2:
            return super()._merge_action_messages(
                action_messages_by_topic, action_joint_names_by_topic
            )
        topics = [
            topic for topic in self.config.action_topics
            if topic in action_messages_by_topic
        ]
        if len(topics) != 2:
            topics = sorted(action_messages_by_topic)
        if any(not action_messages_by_topic.get(topic) for topic in topics):
            raise ValueError("configured action topic produced no data")

        filters: dict[str, Optional[np.ndarray]] = {}
        combined_names: list[str] = []
        for topic in topics:
            group = self._get_topic_group_key(topic, "action")
            target_names = self._joint_order_by_group.get(group, [])
            message_names = action_joint_names_by_topic.get(topic, [])
            if target_names:
                indices = self._joint_order_index_array(message_names, target_names)
                if indices is None:
                    raise ValueError(f"cannot order action joints for {topic}")
                filters[topic] = indices
                combined_names.extend(target_names)
            else:
                filters[topic] = None
                combined_names.extend(message_names)
        self._action_joint_names = combined_names

        left, right = (action_messages_by_topic[topic] for topic in topics)
        li = ri = dropped_left = dropped_right = 0
        max_skew = 0.0
        paired = []
        while li < len(left) and ri < len(right):
            lt, lv = left[li]
            rt, rv = right[ri]
            skew = float(lt - rt)
            if abs(skew) <= self.pair_tolerance_sec:
                lf, rf = filters[topics[0]], filters[topics[1]]
                if lf is not None:
                    lv = lv[lf]
                if rf is not None:
                    rv = rv[rf]
                paired.append((max(lt, rt), np.concatenate([lv, rv]).astype(np.float32)))
                max_skew = max(max_skew, abs(skew))
                li += 1
                ri += 1
            elif lt < rt:
                dropped_left += 1
                li += 1
            else:
                dropped_right += 1
                ri += 1
        dropped_left += len(left) - li
        dropped_right += len(right) - ri
        if not paired:
            raise ValueError("no paired action commands remained")
        self.last_action_pairing = {
            "paired_commands": len(paired),
            "dropped_unpaired_left": dropped_left,
            "dropped_unpaired_right": dropped_right,
            "max_pair_skew_ms": max_skew * 1000.0,
            "pair_tolerance_ms": self.pair_tolerance_sec * 1000.0,
        }
        return paired


@dataclass
class _Settings:
    fps: int
    intervention_topic: str
    motion_threshold_rad: float
    action_pair_tolerance_ms: float
    minimum_trailing_stationary_seconds: float
    robot_config_path: str
    action_topics: list[str]
    sync_thresholds: dict[str, float]

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "_Settings":
        raw = request.get("settings") or {}
        thresholds = dict(DEFAULT_SYNC_THRESHOLDS)
        thresholds.update({
            str(key): float(value)
            for key, value in (raw.get("sync_thresholds") or {}).items()
            if key in thresholds
        })
        fps = int(raw.get("fps", DEFAULT_FPS))
        threshold = float(raw.get("motion_threshold_rad", DEFAULT_MOTION_THRESHOLD_RAD))
        stationary = float(raw.get(
            "minimum_trailing_stationary_seconds",
            DEFAULT_TRAILING_STATIONARY_SECONDS,
        ))
        pair_ms = float(raw.get(
            "action_pair_tolerance_ms", DEFAULT_ACTION_PAIR_TOLERANCE_MS
        ))
        if fps <= 0 or threshold <= 0 or stationary < 0 or pair_ms <= 0:
            raise ValueError("fps and thresholds must be positive")
        return cls(
            fps=fps,
            intervention_topic=str(
                raw.get("intervention_topic") or DEFAULT_INTERVENTION_TOPIC
            ),
            motion_threshold_rad=threshold,
            action_pair_tolerance_ms=pair_ms,
            minimum_trailing_stationary_seconds=stationary,
            robot_config_path=str(raw.get("robot_config_path") or ""),
            action_topics=[
                str(item)
                for item in (raw.get("action_topics") or DEFAULT_ACTION_TOPICS)
                if item
            ],
            sync_thresholds=thresholds,
        )


class CurationEngine:
    """Two-phase, source-preserving dataset curation engine."""

    def __init__(
        self,
        *,
        rosbag_root: Path = Path("/workspace/rosbag2"),
        work_root: Path = Path("/workspace/.cyclo_intelligence/curation_jobs"),
    ) -> None:
        self.rosbag_root = Path(rosbag_root).resolve(strict=False)
        self.work_root = Path(work_root).resolve(strict=False)

    def _safe_task_path(self, raw: str, *, must_exist: bool) -> Path:
        path = Path(str(raw or "")).expanduser()
        if not path.is_absolute():
            path = self.rosbag_root / path
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.rosbag_root)
        except ValueError as exc:
            raise ValueError(f"dataset path must remain under {self.rosbag_root}") from exc
        if must_exist and not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return resolved

    @staticmethod
    def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CurationCancelled("Dataset curation was cancelled")

    @staticmethod
    def _progress(callback: Optional[ProgressCallback], value: float, stage: str,
                  message: str = "") -> None:
        if callback is not None:
            callback(float(value), stage, message)

    def _source_specs(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        if int(request.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        sources = request.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("at least one source dataset is required")
        result = []
        for raw_source in sources:
            source = self._safe_task_path(raw_source.get("task_dir", ""), must_exist=True)
            output_raw = raw_source.get("output_task_dir") or (
                source.parent / f"{source.name}_intervention_motion_trimmed"
            )
            output = self._safe_task_path(str(output_raw), must_exist=False)
            if (
                output == source
                or output.is_relative_to(source)
                or source.is_relative_to(output)
            ):
                raise ValueError(
                    "source and curated output paths must be separate sibling trees"
                )
            episodes = raw_source.get("episodes") or []
            normalized = []
            for item in episodes:
                index = int(item["index"] if isinstance(item, dict) else item)
                normalized.append({
                    "index": index,
                    "episode_success": (
                        item.get("episode_success") if isinstance(item, dict) else None
                    ),
                })
            if not normalized:
                raise ValueError(f"no episodes selected from {source}")
            result.append({
                "task_dir": source,
                "output_task_dir": output,
                "episodes": sorted(normalized, key=lambda item: item["index"]),
            })
        output_paths = [source["output_task_dir"] for source in result]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("curated output paths must be unique")
        source_paths = [source["task_dir"] for source in result]
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("source datasets must be unique")
        return result

    def analyze(
        self,
        request: dict[str, Any],
        *,
        job_id: str,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        settings = _Settings.from_request(request)
        sources = self._source_specs(request)
        total = sum(len(source["episodes"]) for source in sources)
        completed = 0
        stage_root = self.work_root / job_id / "analysis"
        if stage_root.exists():
            shutil.rmtree(stage_root)
        stage_root.mkdir(parents=True, exist_ok=True)
        source_reports = []
        try:
            for source_position, source in enumerate(sources):
                rows = []
                signatures = {}
                for selection in source["episodes"]:
                    self._check_cancel(cancel_event)
                    index = int(selection["index"])
                    episode_dir = source["task_dir"] / str(index)
                    row = {
                        "source_task_dir": str(source["task_dir"]),
                        "source_episode_index": index,
                        "episode_success": selection.get("episode_success"),
                        "status": "needs_review",
                    }
                    try:
                        if not episode_dir.is_dir():
                            raise FileNotFoundError(episode_dir)
                        info_path = episode_dir / "episode_info.json"
                        info = json.loads(info_path.read_text(encoding="utf-8"))
                        if row["episode_success"] is None:
                            row["episode_success"] = info.get("episode_success")
                        robot_type = str(info.get("robot_type") or "unknown")
                        staged = stage_root / f"source_{source_position}" / f"{index}_converted"
                        _, success, _ = _convert_single_episode_worker(
                            str(episode_dir), str(staged), settings.fps, False, True
                        )
                        if not success:
                            raise RuntimeError("failed to prepare episode for analysis")
                        config = ConversionConfig(
                            repo_id="local/curation-analysis",
                            output_dir=stage_root / "unused",
                            fps=settings.fps,
                            robot_type=robot_type,
                            robot_config_path=settings.robot_config_path or None,
                            selected_action_topics=list(settings.action_topics),
                            use_videos=False,
                            apply_trim=False,
                            apply_exclude_regions=False,
                            trim_stationary_edges=False,
                        )
                        converter = InterventionAnalysisConverter(
                            config,
                            pair_tolerance_sec=settings.action_pair_tolerance_ms / 1000.0,
                        )
                        # Keep the configured topic overridable without changing the
                        # converter's public constant for all other callers.
                        import cyclo_data.converter.base_converter as base_converter
                        previous_topic = base_converter._INTERVENTION_MODE_TOPIC
                        base_converter._INTERVENTION_MODE_TOPIC = settings.intervention_topic
                        try:
                            episode = converter.convert_single_rosbag(staged, index)
                        finally:
                            base_converter._INTERVENTION_MODE_TOPIC = previous_topic
                        if episode is None or episode.length <= 1:
                            raise RuntimeError("episode produced no 15 Hz joint grid")
                        signatures[str(index)] = _episode_digest(episode_dir)
                        actions = np.asarray(episode.action, dtype=np.float64)
                        labels = list(episode.action_sources)
                        grid_ns = np.rint(
                            np.asarray(episode.grid_log_times_sec, dtype=np.float64) * 1e9
                        ).astype(np.int64)
                        row.update({
                            "robot_type": robot_type,
                            "fps": settings.fps,
                            "original_frames_at_target_fps": int(episode.length),
                            "grid_origin_time_ns": int(grid_ns[0]),
                            "original_duration_seconds": float(
                                grid_ns[-1] - grid_ns[0]
                            ) / 1e9,
                            "cameras": self._camera_inventory(episode_dir),
                        })
                        if len(labels) != int(episode.length):
                            raise RuntimeError("intervention signal does not cover the frame grid")
                        human = next((i for i, label in enumerate(labels) if label == "human"), None)
                        if human is None:
                            raise RuntimeError("intervention=true was not found")
                        deltas = np.max(np.abs(np.diff(actions, axis=0)), axis=1)
                        moving = np.flatnonzero(
                            ((~np.isfinite(deltas)) | (deltas > settings.motion_threshold_rad))
                            & (np.arange(len(deltas)) >= human)
                        )
                        if moving.size == 0:
                            raise RuntimeError("no meaningful action motion after intervention")
                        start_frame = int(moving[0]) + 1
                        action_last = _last_motion_frame(actions, settings.motion_threshold_rad)
                        state_last = _last_motion_frame(
                            _selected_state_matrix(episode), settings.motion_threshold_rad
                        )
                        last_motion = max(
                            value for value in (action_last, state_last) if value is not None
                        )
                        stationary_seconds = max(
                            0.0, float(grid_ns[-1] - grid_ns[last_motion]) / 1e9
                        )
                        should_trim_end = (
                            stationary_seconds
                            >= settings.minimum_trailing_stationary_seconds
                        )
                        end_frame = last_motion if should_trim_end else int(episode.length) - 1
                        start_ns = int(grid_ns[start_frame])
                        end_ns = int(grid_ns[end_frame])
                        camera_rows, sync = self._analyze_cameras(
                            episode_dir,
                            grid_ns=grid_ns[start_frame:end_frame + 1],
                            joint_state_ns=np.rint(
                                converter.raw_state_times_sec * 1e9
                            ).astype(np.int64),
                            start_ns=start_ns,
                            end_ns=end_ns,
                            action_pairing=converter.last_action_pairing,
                            thresholds=settings.sync_thresholds,
                        )
                        row.update({
                            "status": "ready" if all(sync["checks"].values()) else "needs_review",
                            "review_reason": "" if all(sync["checks"].values()) else "timestamp synchronization check failed",
                            "intervention_frame": int(human),
                            "intervention_time_ns": int(grid_ns[human]),
                            "start_frame": start_frame,
                            "start_time_ns": start_ns,
                            "start_offset_seconds": float(start_ns - grid_ns[0]) / 1e9,
                            "end_frame": end_frame,
                            "end_time_ns": end_ns,
                            "end_offset_seconds": float(end_ns - grid_ns[0]) / 1e9,
                            "removed_prefix_frames": start_frame,
                            "removed_suffix_frames": int(episode.length) - end_frame - 1,
                            "trailing_stationary_seconds": stationary_seconds,
                            "trailing_trim_applied": should_trim_end,
                            "action_pairing": dict(converter.last_action_pairing),
                            "cameras": camera_rows,
                            "sync_validation": sync,
                        })
                    except Exception as exc:  # noqa: BLE001 - report per episode
                        row["review_reason"] = str(exc)
                    rows.append(row)
                    completed += 1
                    self._progress(
                        progress,
                        completed * 100.0 / max(total, 1),
                        "analyzing",
                        f"Analyzed {source['task_dir'].name}/{index}",
                    )
                source_reports.append({
                    "task_dir": str(source["task_dir"]),
                    "output_task_dir": str(source["output_task_dir"]),
                    "episode_signatures": signatures,
                    "episodes": rows,
                })
        finally:
            if stage_root.exists():
                shutil.rmtree(stage_root, ignore_errors=True)

        ready = sum(
            row["status"] == "ready"
            for source in source_reports for row in source["episodes"]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "analysis",
            "job_id": job_id,
            "created_at": _utc_now(),
            "settings": {
                "fps": settings.fps,
                "intervention_topic": settings.intervention_topic,
                "motion_threshold_rad": settings.motion_threshold_rad,
                "action_pair_tolerance_ms": settings.action_pair_tolerance_ms,
                "minimum_trailing_stationary_seconds": (
                    settings.minimum_trailing_stationary_seconds
                ),
                "robot_config_path": settings.robot_config_path,
                "action_topics": list(settings.action_topics),
                "sync_thresholds": settings.sync_thresholds,
            },
            "sources": source_reports,
            "summary": {
                "episodes": total,
                "ready": ready,
                "needs_review": total - ready,
            },
        }

    @staticmethod
    def _camera_inventory(episode_dir: Path) -> list[dict[str, Any]]:
        """Return validated source-camera metadata for manual review previews."""
        rows = []
        for sidecar in sorted((episode_dir / "videos").rglob("*_timestamps.parquet")):
            camera = sidecar.stem.removesuffix("_timestamps")
            video = sidecar.with_name(f"{camera}.mp4")
            if not video.is_file():
                raise FileNotFoundError(video)
            table = pq.read_table(sidecar)
            stamps = _timestamp_values(table)
            frames = _video_frame_count(video)
            if frames != table.num_rows:
                raise RuntimeError(
                    f"{camera}: MP4/sidecar frame mismatch ({frames}/{table.num_rows})"
                )
            rows.append({
                "camera": camera,
                "video": str(video.relative_to(episode_dir)),
                "sidecar": str(sidecar.relative_to(episode_dir)),
                "input_frames": frames,
                "start_frame": 0,
                "stop_frame_exclusive": frames,
                "output_frames": frames,
                "first_kept_stamp_ns": int(stamps[0]),
                "last_kept_stamp_ns": int(stamps[-1]),
                "source_video_offset_seconds": 0.0,
            })
        if not rows:
            raise RuntimeError("no camera timestamp sidecars found")
        return rows

    def _analyze_cameras(
        self,
        episode_dir: Path,
        *,
        grid_ns: np.ndarray,
        joint_state_ns: np.ndarray,
        start_ns: int,
        end_ns: int,
        action_pairing: dict[str, Any],
        thresholds: dict[str, float],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        camera_rows = []
        selected_stamps = []
        camera_to_grid = []
        camera_to_joint = []
        active_joint_state_ns = joint_state_ns[
            (joint_state_ns >= start_ns) & (joint_state_ns <= end_ns)
        ]
        if active_joint_state_ns.size < 2:
            raise RuntimeError("fewer than two joint-state timestamps in trim window")
        for sidecar in sorted((episode_dir / "videos").rglob("*_timestamps.parquet")):
            camera = sidecar.stem.removesuffix("_timestamps")
            video = sidecar.with_name(f"{camera}.mp4")
            if not video.is_file():
                raise FileNotFoundError(video)
            table = pq.read_table(sidecar)
            stamps = _timestamp_values(table)
            frames = _video_frame_count(video)
            if frames != table.num_rows:
                raise RuntimeError(
                    f"{camera}: MP4/sidecar frame mismatch ({frames}/{table.num_rows})"
                )
            start, stop = _slice_bounds(stamps, start_ns, end_ns)
            active = stamps[start:stop]
            selected_stamps.append(active)
            camera_to_joint.extend(
                _nearest_abs_ns(active, active_joint_state_ns).tolist()
            )
            positions = np.searchsorted(active, grid_ns, side="right") - 1
            np.clip(positions, 0, len(active) - 1, out=positions)
            selected = active[positions]
            camera_to_grid.extend(np.abs(grid_ns - selected).tolist())
            camera_rows.append({
                "camera": camera,
                "video": str(video.relative_to(episode_dir)),
                "sidecar": str(sidecar.relative_to(episode_dir)),
                "input_frames": frames,
                "start_frame": start,
                "stop_frame_exclusive": stop,
                "output_frames": stop - start,
                "first_kept_stamp_ns": int(active[0]),
                "last_kept_stamp_ns": int(active[-1]),
                "source_video_offset_seconds": float(active[0] - stamps[0]) / 1e9,
            })
        if not camera_rows:
            raise RuntimeError("no camera timestamp sidecars found")

        camera_camera = []
        selected_pairs = []
        for left in range(len(selected_stamps)):
            for right in range(left + 1, len(selected_stamps)):
                camera_camera.extend(
                    _nearest_abs_ns(selected_stamps[left], selected_stamps[right]).tolist()
                )
                left_pos = np.searchsorted(selected_stamps[left], grid_ns, side="right") - 1
                right_pos = np.searchsorted(selected_stamps[right], grid_ns, side="right") - 1
                np.clip(left_pos, 0, len(selected_stamps[left]) - 1, out=left_pos)
                np.clip(right_pos, 0, len(selected_stamps[right]) - 1, out=right_pos)
                selected_pairs.extend(np.abs(
                    selected_stamps[left][left_pos] - selected_stamps[right][right_pos]
                ).tolist())

        joint_state_gaps = np.diff(active_joint_state_ns)
        metrics = {
            "camera_joint_nearest": _stats_ms(camera_to_joint),
            "camera_camera_nearest": _stats_ms(camera_camera),
            "left_right_action_pairs": {
                "count": int(action_pairing.get("paired_commands", 0)),
                "max_ms": float(action_pairing.get("max_pair_skew_ms", 0.0)),
                "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            },
            "joint_state_gaps": _stats_ms(joint_state_gaps),
            "camera_to_grid": _stats_ms(camera_to_grid),
            "selected_camera_pairs": _stats_ms(selected_pairs),
        }
        checks = {
            "camera_joint_nearest": metrics["camera_joint_nearest"]["max_ms"]
            <= thresholds["camera_joint_nearest_max_ms"],
            "camera_camera_nearest_p99": metrics["camera_camera_nearest"]["p99_ms"]
            <= thresholds["camera_camera_nearest_p99_ms"],
            "camera_camera_nearest_max": metrics["camera_camera_nearest"]["max_ms"]
            <= thresholds["camera_camera_nearest_max_ms"],
            "left_right_action_pair": metrics["left_right_action_pairs"]["max_ms"]
            <= thresholds["left_right_action_pair_max_ms"],
            "joint_state_gap": metrics["joint_state_gaps"]["max_ms"]
            <= thresholds["joint_state_gap_max_ms"],
            "camera_to_grid": metrics["camera_to_grid"]["max_ms"]
            <= thresholds["camera_to_grid_abs_max_ms"],
            "selected_camera_pair": metrics["selected_camera_pairs"]["max_ms"]
            <= thresholds["selected_camera_pair_max_ms"],
        }
        return camera_rows, {
            "status": "passed" if all(checks.values()) else "failed",
            "thresholds": thresholds,
            "checks": checks,
            "metrics": metrics,
        }

    def _validate_reviewed_window(
        self,
        episode_dir: Path,
        *,
        source_episode_index: int,
        robot_type: str,
        start_ns: int,
        end_ns: int,
        analysis_settings: dict[str, Any],
        job_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Recompute sync metrics when an operator supplies manual boundaries."""
        settings = _Settings.from_request({"settings": analysis_settings})
        staged = (
            self.work_root / job_id / "manual_validation"
            / f"{episode_dir.parent.name}_{source_episode_index}"
        )
        if staged.exists():
            shutil.rmtree(staged)
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            _, success, _ = _convert_single_episode_worker(
                str(episode_dir), str(staged), settings.fps, False, True
            )
            if not success:
                raise RuntimeError("failed to prepare manually reviewed episode")
            config = ConversionConfig(
                repo_id="local/curation-manual-validation",
                output_dir=self.work_root / job_id / "unused",
                fps=settings.fps,
                robot_type=robot_type,
                robot_config_path=settings.robot_config_path or None,
                selected_action_topics=list(settings.action_topics),
                use_videos=False,
                apply_trim=False,
                apply_exclude_regions=False,
                trim_stationary_edges=False,
            )
            converter = InterventionAnalysisConverter(
                config,
                pair_tolerance_sec=settings.action_pair_tolerance_ms / 1000.0,
            )
            import cyclo_data.converter.base_converter as base_converter
            previous_topic = base_converter._INTERVENTION_MODE_TOPIC
            base_converter._INTERVENTION_MODE_TOPIC = settings.intervention_topic
            try:
                episode = converter.convert_single_rosbag(
                    staged, source_episode_index
                )
            finally:
                base_converter._INTERVENTION_MODE_TOPIC = previous_topic
            if episode is None or episode.length <= 1:
                raise RuntimeError("manual window validation produced no joint grid")
            all_grid_ns = np.rint(
                np.asarray(episode.grid_log_times_sec, dtype=np.float64) * 1e9
            ).astype(np.int64)
            grid_ns = all_grid_ns[
                (all_grid_ns >= start_ns) & (all_grid_ns <= end_ns)
            ]
            if grid_ns.size < 2:
                raise RuntimeError("manual trim window has fewer than two joint frames")
            camera_rows, sync = self._analyze_cameras(
                episode_dir,
                grid_ns=grid_ns,
                joint_state_ns=np.rint(
                    converter.raw_state_times_sec * 1e9
                ).astype(np.int64),
                start_ns=start_ns,
                end_ns=end_ns,
                action_pairing=converter.last_action_pairing,
                thresholds=settings.sync_thresholds,
            )
            if not all(sync["checks"].values()):
                failed = [name for name, passed in sync["checks"].items() if not passed]
                raise RuntimeError(
                    "manual trim window failed sync validation: " + ", ".join(failed)
                )
            return camera_rows, sync
        finally:
            if staged.exists():
                shutil.rmtree(staged, ignore_errors=True)

    def apply(
        self,
        analysis: dict[str, Any],
        request: dict[str, Any],
        *,
        job_id: str,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        if analysis.get("kind") != "analysis" or analysis.get("job_id") != job_id:
            raise ValueError("analysis result does not match the curation job")
        overrides = {
            (str(item["source_task_dir"]), int(item["source_episode_index"])): item
            for item in (request.get("overrides") or [])
        }
        output_by_source = {
            str(item["task_dir"]): self._safe_task_path(
                str(item.get("output_task_dir") or ""), must_exist=False
            )
            for item in (request.get("sources") or [])
        }
        total = sum(len(source["episodes"]) for source in analysis["sources"])
        staged_outputs: list[tuple[Path, Path]] = []
        published_outputs: list[tuple[Path, Path]] = []
        result_sources = []
        completed = 0
        try:
            for source_report in analysis["sources"]:
                self._check_cancel(cancel_event)
                source = self._safe_task_path(source_report["task_dir"], must_exist=True)
                output = output_by_source.get(
                    str(source), self._safe_task_path(source_report["output_task_dir"], must_exist=False)
                )
                if output.exists() or output.is_symlink():
                    raise FileExistsError(f"curated output already exists: {output}")
                stage = output.parent / f".{output.name}.curation-{job_id}"
                if stage.exists() or stage.is_symlink():
                    raise FileExistsError(stage)
                stage.mkdir(parents=True)
                staged_outputs.append((stage, output))
                for root_file in source.iterdir():
                    if root_file.is_file():
                        shutil.copy2(root_file, stage / root_file.name)

                source_manifest_rows = []
                for row in source_report["episodes"]:
                    self._check_cancel(cancel_event)
                    key = (str(source), int(row["source_episode_index"]))
                    override = overrides.get(key, {})
                    include = bool(override.get("include", row.get("status") == "ready"))
                    if not include:
                        completed += 1
                        continue
                    if row.get("status") != "ready" and not {
                        "start_time_ns", "end_time_ns"
                    }.issubset(override):
                        raise ValueError(
                            f"episode {key[1]} needs manual start/end review"
                        )
                    source_episode = source / str(key[1])
                    expected_signature = source_report.get("episode_signatures", {}).get(str(key[1]))
                    if expected_signature and _episode_digest(source_episode) != expected_signature:
                        raise RuntimeError(f"source episode changed after analysis: {source_episode}")
                    destination = stage / str(key[1])
                    shutil.copytree(source_episode, destination, copy_function=shutil.copy2)
                    start_value = (
                        override["start_time_ns"]
                        if "start_time_ns" in override else row.get("start_time_ns")
                    )
                    end_value = (
                        override["end_time_ns"]
                        if "end_time_ns" in override else row.get("end_time_ns")
                    )
                    if start_value is None or end_value is None:
                        raise ValueError(
                            f"episode {key[1]} needs explicit manual start/end timestamps"
                        )
                    start_ns = int(start_value)
                    end_ns = int(end_value)
                    if end_ns <= start_ns:
                        raise ValueError(f"episode {key[1]} has an invalid trim window")
                    effective_row = dict(row)
                    window_changed = (
                        row.get("status") != "ready"
                        or start_ns != row.get("start_time_ns")
                        or end_ns != row.get("end_time_ns")
                    )
                    if window_changed:
                        reviewed_cameras, reviewed_sync = self._validate_reviewed_window(
                            source_episode,
                            source_episode_index=key[1],
                            robot_type=str(row.get("robot_type") or "unknown"),
                            start_ns=start_ns,
                            end_ns=end_ns,
                            analysis_settings=dict(analysis.get("settings") or {}),
                            job_id=job_id,
                        )
                        effective_row["cameras"] = reviewed_cameras
                        effective_row["sync_validation"] = reviewed_sync
                    success_label = override.get("episode_success", row.get("episode_success"))
                    camera_results = self._materialize_episode_media(
                        destination, start_ns=start_ns, end_ns=end_ns
                    )
                    self._patch_episode_metadata(
                        destination,
                        source_episode_index=key[1],
                        start_ns=start_ns,
                        end_ns=end_ns,
                        episode_success=success_label,
                        row=effective_row,
                    )
                    _verify_mcap_files_unchanged(source_episode, destination)
                    source_manifest_rows.append({
                        **{
                            k: v for k, v in effective_row.items()
                            if k not in {"cameras"}
                        },
                        "start_time_ns": start_ns,
                        "end_time_ns": end_ns,
                        "episode_success": success_label,
                        "cameras": camera_results,
                    })
                    completed += 1
                    self._progress(
                        progress,
                        completed * 70.0 / max(total, 1),
                        "materializing",
                        f"Trimmed {source.name}/{key[1]}",
                    )
                if not source_manifest_rows:
                    raise ValueError(f"no reviewed episodes selected from {source}")
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "format": "cyclo.intervention_motion_curation/v2",
                    "created_at": _utc_now(),
                    "job_id": job_id,
                    "source": str(source),
                    "curated_rosbag": str(output),
                    "source_untouched": True,
                    "mcap_rewritten": False,
                    "settings": analysis["settings"],
                    "episodes": source_manifest_rows,
                }
                validation = {
                    "status": "passed",
                    "validated_at": _utc_now(),
                    "episodes": len(source_manifest_rows),
                    "source_untouched": True,
                    "curated_mcap_sha256_unchanged": True,
                    "video_sidecar_frame_counts_match": True,
                    "timestamp_sync": {
                        "status": "passed",
                        "checks": [item["sync_validation"]["checks"] for item in source_manifest_rows],
                    },
                }
                _write_json(stage / "curation_manifest.json", manifest)
                _write_json(stage / "validation_report.json", validation)
                _write_json(stage / "sync_validation_report.json", {
                    "status": "passed",
                    "episodes": [item["sync_validation"] for item in source_manifest_rows],
                })
                self._normalize_permissions(stage, output.parent)
                result_sources.append({
                    "source_task_dir": str(source),
                    "output_task_dir": str(output),
                    "episodes": len(source_manifest_rows),
                    "episode_indices": [
                        int(item["source_episode_index"]) for item in source_manifest_rows
                    ],
                })

            self._check_cancel(cancel_event)
            # Publish only after every source-specific copy has passed validation.
            for stage, output in staged_outputs:
                os.replace(stage, output)
                published_outputs.append((output, stage))
            self._progress(progress, 75.0, "curation_complete", "Curated copies validated")
        except Exception:
            for output, stage in reversed(published_outputs):
                try:
                    if output.exists() and not stage.exists():
                        os.replace(output, stage)
                except OSError:
                    pass
            raise
        finally:
            for stage, _ in staged_outputs:
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "materialization",
            "job_id": job_id,
            "completed_at": _utc_now(),
            "status": "passed",
            "sources": result_sources,
        }

    def _materialize_episode_media(
        self, episode_dir: Path, *, start_ns: int, end_ns: int
    ) -> list[dict[str, Any]]:
        results = []
        for sidecar in sorted((episode_dir / "videos").rglob("*_timestamps.parquet")):
            camera = sidecar.stem.removesuffix("_timestamps")
            video = sidecar.with_name(f"{camera}.mp4")
            table = pq.read_table(sidecar)
            stamps = _timestamp_values(table)
            input_frames = _video_frame_count(video)
            if input_frames != table.num_rows:
                raise RuntimeError(f"{video}: MP4/sidecar frame mismatch")
            start, stop = _slice_bounds(stamps, start_ns, end_ns)
            output_frames = stop - start
            if start != 0 or stop != input_frames:
                temporary_video = video.with_name(f".{video.stem}.curation.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(video), "-map", "0:v:0",
                        "-vf", f"trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS",
                        "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-an", str(temporary_video),
                    ],
                    check=True,
                )
                os.replace(temporary_video, video)
            trimmed = table.slice(start, output_frames)
            frame_type = trimmed.schema.field("frame_index").type
            indices = pa.array(np.arange(output_frames), type=frame_type)
            column = trimmed.schema.get_field_index("frame_index")
            trimmed = trimmed.set_column(column, "frame_index", indices)
            temporary_sidecar = sidecar.with_name(f".{sidecar.name}.curation.tmp")
            pq.write_table(trimmed, temporary_sidecar, compression="snappy")
            os.replace(temporary_sidecar, sidecar)
            if _video_frame_count(video) != output_frames:
                raise RuntimeError(f"{video}: trimmed frame count mismatch")
            checked = pq.read_table(sidecar)
            if checked.num_rows != output_frames or checked["frame_index"].to_pylist() != list(
                range(output_frames)
            ):
                raise RuntimeError(f"{sidecar}: trimmed sidecar validation failed")
            results.append({
                "camera": camera,
                "input_frames": input_frames,
                "start_frame": start,
                "stop_frame_exclusive": stop,
                "output_video_frames": output_frames,
                "output_sidecar_rows": output_frames,
                "first_kept_stamp_ns": int(stamps[start]),
                "last_kept_stamp_ns": int(stamps[stop - 1]),
            })
        if not results:
            raise RuntimeError(f"no timestamp sidecars found in {episode_dir}")
        return results

    @staticmethod
    def _patch_episode_metadata(
        episode_dir: Path,
        *,
        source_episode_index: int,
        start_ns: int,
        end_ns: int,
        episode_success: Any,
        row: dict[str, Any],
    ) -> None:
        info_path = episode_dir / "episode_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(episode_success, bool):
            info["episode_success"] = episode_success
        _write_json(info_path, info)

        metadata_path = episode_dir / "metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        bag_info = metadata.get("rosbag2_bagfile_information") or {}
        bag_start_ns = int(
            (bag_info.get("starting_time") or {}).get("nanoseconds_since_epoch", 0)
        )
        config_path = episode_dir / "robot_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        config = config or {}
        fps = int(row.get("fps", DEFAULT_FPS))
        grid_origin_ns = int(row.get("grid_origin_time_ns", start_ns))
        start_frame = max(0, int(round((start_ns - grid_origin_ns) * fps / 1e9)))
        end_frame = max(start_frame, int(round((end_ns - grid_origin_ns) * fps / 1e9)))
        config["trim_points"] = {
            "start": {
                "frame": start_frame,
                "time": start_ns / 1e9,
                "relative_time": (start_ns - bag_start_ns) / 1e9,
            },
            "end": {
                "frame": end_frame,
                "time": end_ns / 1e9,
                "relative_time": (end_ns - bag_start_ns) / 1e9,
            },
        }
        config["curation"] = {
            "format": "cyclo.intervention_motion_trim/v2",
            "source_episode_index": int(source_episode_index),
            "target_fps": fps,
            "source_mcap_modified": False,
            "start_rule": "first meaningful action motion after intervention=true",
            "end_rule": "last meaningful action/state motion when trailing idle is long enough",
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    @staticmethod
    def _normalize_permissions(root: Path, parent: Path) -> None:
        try:
            parent_stat = parent.stat()
        except OSError:
            parent_stat = None
        for path in [root, *root.rglob("*")]:
            try:
                if parent_stat is not None:
                    os.chown(path, parent_stat.st_uid, parent_stat.st_gid)
                os.chmod(path, 0o775 if path.is_dir() else 0o664)
            except (OSError, PermissionError):
                continue

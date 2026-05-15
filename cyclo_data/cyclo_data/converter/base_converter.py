#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

"""
Shared infrastructure for ROSbag → LeRobot converters (v2.1, v3.0).

Holds the rosbag extraction, causal-sync resampling, video discovery, feature
building, and per-episode statistics — i.e. everything that is identical
between LeRobot v2.1 and v3.0. Format-specific writers live in
``to_lerobot_v21.py`` and ``to_lerobot_v30.py``.
"""

import bisect
import json
import os
import shutil  # noqa: F401  (re-exported transitively for legacy callers)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa  # noqa: F401  (re-exported transitively for legacy callers)
import pyarrow.parquet as pq  # noqa: F401

from cyclo_data.reader.bag_reader import BagReader
from cyclo_data.reader.metadata_manager import MetadataManager
from cyclo_data.reader.video_metadata_extractor import VideoMetadataExtractor
from shared.robot_configs import schema as robot_schema


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FPS = 30

# Floor applied to per-joint observation.state / action std when writing
# stats. A joint that was not actuated during a recording has true std=0,
# which makes downstream (x - mean) / std normalization explode to
# ±Inf / NaN and crashes training. Flooring also tames "noise-only"
# joints (std well below sensor noise): after normalization their
# contribution stays bounded instead of being amplified into huge inputs.
# Joints with real motion sit well above 1e-3 rad so they are unaffected.
STATS_STD_FLOOR = 1e-3


def _convert_rosbag_worker(bag_path_str, episode_index, config_dict):
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Creates a fresh base converter in each worker and parses a single
    rosbag episode. Worker only invokes extraction methods (all defined
    on the base class), so the base instance is sufficient regardless
    of which subclass triggered the parallel run.
    """
    config = ConversionConfig(**config_dict)
    converter = RosbagToLerobotConverterBase(config, logger=None)
    result = converter.convert_single_rosbag(Path(bag_path_str), episode_index)
    return episode_index, result


# Hard ceiling so a 32-core dev host doesn't spawn 16 workers and over-
# subscribe ffmpeg subprocesses. Conversion is IO + ffmpeg heavy and
# 8 concurrent episodes already saturate disk + camera ffmpeg threads.
_CONVERSION_WORKER_CEILING = 8


def _resolve_conversion_worker_count(work_units: int) -> int:
    """Pick the worker count for conversion ProcessPoolExecutor pools.

    Defaults to **half of the visible CPUs** so the other half stays
    free for the ROS control loop, camera drivers, and any inference
    container that's running alongside. Without this cap a Jetson Orin
    (8 cores) would launch 8 episode workers × 4 camera ffmpegs = 32
    CPU-bound processes and starve the 100 Hz control loop.

    Override with ``CYCLO_CONVERSION_MAX_WORKERS`` when an operator
    explicitly wants to dedicate the machine to conversion (e.g. CI,
    benchmark host) or further reduce parallelism on a thermally
    constrained system.
    """
    if work_units <= 1:
        return 1
    env_override = os.environ.get('CYCLO_CONVERSION_MAX_WORKERS')
    if env_override:
        try:
            override = max(1, int(env_override))
            return max(1, min(override, work_units))
        except ValueError:
            pass
    cpu_total = os.cpu_count() or 4
    half = max(1, cpu_total // 2)
    return max(1, min(half, work_units, _CONVERSION_WORKER_CEILING))


def _conversion_worker_init() -> None:
    """ProcessPoolExecutor initializer — yield CPU to higher-priority work.

    niceness +10 (default 0; range -20..+19) makes the worker a
    "background" citizen so the kernel scheduler prefers the ROS
    control loop and inference container when CPUs are saturated.
    Best-effort: silently no-op on platforms / containers that block
    the syscall.
    """
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass


@dataclass
class StalenessMetrics:
    """Metrics for tracking data staleness during causal sync resampling."""

    topic: str
    total_samples: int = 0
    stale_warning_count: int = 0
    stale_error_count: int = 0
    max_staleness_ms: float = 0.0
    mean_staleness_ms: float = 0.0
    stale_samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def warning_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.stale_warning_count / self.total_samples

    @property
    def error_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.stale_error_count / self.total_samples

    @property
    def status(self) -> str:
        if self.stale_error_count > 0:
            return "ERROR"
        if self.stale_warning_count > 0:
            return "WARNING"
        return "GOOD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "total_samples": self.total_samples,
            "staleness": {
                "warning_count": self.stale_warning_count,
                "error_count": self.stale_error_count,
                "warning_ratio": round(self.warning_ratio * 100, 2),
                "error_ratio": round(self.error_ratio * 100, 2),
                "max_ms": round(self.max_staleness_ms, 2),
                "mean_ms": round(self.mean_staleness_ms, 2),
            },
            "status": self.status,
            "stale_samples": self.stale_samples[:20],  # Limit to first 20
        }


@dataclass
class ConversionConfig:
    """Configuration for ROSbag to LeRobot conversion."""

    repo_id: str
    output_dir: Path
    fps: int = DEFAULT_FPS
    robot_type: str = "unknown"
    use_videos: bool = True
    chunks_size: int = DEFAULT_CHUNK_SIZE

    # Robot config file path (e.g., ffw_sg2_rev1_config.yaml)
    robot_config_path: Optional[str] = None

    # Topic mappings (populated from robot config or auto-detected)
    state_topics: List[str] = field(default_factory=list)
    action_topics: List[str] = field(default_factory=list)

    # Trim settings
    apply_trim: bool = True
    apply_exclude_regions: bool = True

    # Staleness thresholds
    quality_warning_multiplier: float = 2.0
    quality_error_multiplier: float = 4.0

    # ---- Conversion selection knobs (StartConversion.srv) ----
    # Empty / None = use defaults from robot_config (legacy behaviour).
    selected_cameras: List[str] = field(default_factory=list)
    camera_rotations: Dict[str, int] = field(default_factory=dict)
    image_resize: Optional[Tuple[int, int]] = None  # (height, width)
    selected_state_topics: List[str] = field(default_factory=list)
    selected_action_topics: List[str] = field(default_factory=list)
    selected_joints: List[str] = field(default_factory=list)
    # Audit metadata for the root info.json conversion_config snapshot.
    source_rosbags: List[str] = field(default_factory=list)


@dataclass
class EpisodeData:
    """Data container for a single episode."""

    episode_index: int
    timestamps: List[float] = field(default_factory=list)
    observation_state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    video_files: Dict[str, Path] = field(default_factory=dict)
    tasks: List[str] = field(default_factory=list)
    length: int = 0
    # Absolute MCAP log_time (seconds since epoch) for each row of
    # ``timestamps``. Populated by ``_resample_to_fps`` so the video
    # sync step can map per-camera MP4 frames onto the same grid.
    grid_log_times_sec: List[float] = field(default_factory=list)


class RosbagToLerobotConverterBase:
    """
    Base class for ROSbag-to-LeRobot conversion.

    Owns all logic that is independent of the LeRobot dataset format
    version: bag reading, joint extraction, causal-sync resampling,
    video discovery, feature building, and per-episode statistics.

    Format-specific subclasses (``RosbagToLerobotConverter`` for v2.1,
    ``RosbagToLerobotV30Converter`` for v3.0) supply the writers.
    """

    def __init__(self, config: ConversionConfig, logger=None):
        self.config = config
        self.logger = logger
        self._metadata_manager = MetadataManager(logger)
        self._video_extractor = VideoMetadataExtractor(logger)

        self._features: Dict[str, Dict] = {}
        self._tasks: Dict[int, str] = {}
        self._task_to_index: Dict[str, int] = {}
        self._episodes: Dict[int, Dict] = {}
        self._episodes_stats: Dict[int, Dict] = {}
        self._total_frames = 0
        self._total_episodes = 0
        self._staleness_reports: Dict[int, Dict[str, StalenessMetrics]] = {}
        # Init for v3.0's enable_quality_report path; populated by future
        # quality-report machinery, empty dict short-circuits the writer.
        self._quality_reports: Dict[int, Dict[str, Any]] = {}

        self._state_joint_names: List[str] = []
        self._action_joint_names: List[str] = []
        self._camera_mapping: Dict[str, str] = {}  # topic -> camera_name
        self._camera_rotations: Dict[str, int] = {}  # camera_name -> rotation_deg
        self._joint_order: List[str] = []  # Ordered list of joints to include
        self._joint_order_by_group: Dict[str, List[str]] = {}  # group_key -> joint names
        self._state_topic_key_map: Dict[str, str] = {}  # topic -> group key
        self._action_topic_key_map: Dict[str, str] = {}  # topic -> group key

        # Per-episode bisect-keys cache for ``_find_previous_value(_in_list)``.
        # Previously this lived in a mutable-default-arg dict on the method
        # (the classic Python footgun) — it survived across episodes within
        # a worker process and could in principle return stale keys if a
        # new ``state_messages`` list reused a GC'd address. Instance scope
        # plus an explicit ``clear()`` in ``_extract_joint_data`` keeps the
        # per-call lookup speedup with zero risk of cross-episode leakage.
        self._bisect_keys_cache: Dict[int, List[float]] = {}

        # Load robot config — caller's explicit path wins, otherwise
        # auto-resolve via the schema (which searches the standard
        # ORCHESTRATOR_CONFIG_PATH / /orchestrator_config / source-tree
        # locations). The ROS service callers don't pass
        # ``robot_config_path`` on every request, so without this
        # fallback ``_joint_order_by_group`` stays empty and the
        # downstream ``_build_features`` ``names`` field collapses to
        # the placeholder ``joint_N`` list — the symptom the user
        # spotted in v2.1 ``info.json``.
        robot_config_path = config.robot_config_path
        if not robot_config_path and config.robot_type and config.robot_type != "unknown":
            try:
                resolved = robot_schema.find_robot_config_path(config.robot_type)
                robot_config_path = str(resolved)
                self._log_info(
                    f"robot_config_path auto-resolved for "
                    f"robot_type={config.robot_type!r}: {robot_config_path}"
                )
            except Exception as exc:
                self._log_warning(
                    f"robot_config_path auto-resolve failed for "
                    f"robot_type={config.robot_type!r}: {exc!r}"
                )
        if robot_config_path:
            self._load_robot_config_file(robot_config_path)

        # Apply selection knobs after robot_config has populated the
        # discovered defaults — empty selection lists mean "use all
        # discovered" so the legacy behaviour is preserved.
        self._apply_selection_knobs()

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"[ERROR] {msg}")

    def _log_warning(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    def _load_robot_config_file(self, config_path: str):
        """Load robot config from YAML file (e.g., ffw_sg2_rev1_config.yaml).

        Phase 4: yaml is VLA-semantic. State / action topic→key maps still
        carry the legacy ``follower_<group>`` / ``leader_<modality>`` keys
        because _resolve_filter_target_names + _merge_*_messages downstream
        key off that prefix to align state and action slices.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            self._log_error(f"Robot config not found: {config_path}")
            return

        # Recover robot_type from the yaml's top-level key first.
        # ConversionConfig.robot_type defaults to "unknown" — without this
        # step we'd pass an empty/wrong key to schema.load_robot_section
        # and bail out, leaving every topic→key map blank and forcing the
        # downstream heuristics (which can't recover yaml-driven group
        # ordering) to take over.
        try:
            import yaml as _yaml
            with open(config_path, 'r') as f:
                raw = _yaml.safe_load(f) or {}
            params = raw.get("orchestrator", {}).get("ros__parameters", {})
        except Exception as e:
            self._log_error(f"Failed to read robot config: {e}")
            return
        if not params:
            self._log_error(
                f"orchestrator.ros__parameters missing in {config_path}"
            )
            return

        if self.config.robot_type == "unknown" or self.config.robot_type not in params:
            self.config.robot_type = next(iter(params.keys()))

        try:
            section = robot_schema.load_robot_section(
                self.config.robot_type,
                explicit_path=str(config_path),
            )
        except Exception as e:
            self._log_error(f"Failed to load robot config: {e}")
            return

        state_groups = robot_schema.get_state_groups(section)
        action_groups = robot_schema.get_action_groups(section)
        image_groups = robot_schema.get_image_topics(section)

        state_topics: Dict[str, str] = {}
        for name, cfg in state_groups.items():
            key = f"follower_{name}"
            state_topics[key] = cfg["topic"]
            self._state_topic_key_map[cfg["topic"]] = key

        action_topics: Dict[str, str] = {}
        for modality, cfg in action_groups.items():
            key = f"leader_{modality}"
            action_topics[key] = cfg["topic"]
            self._action_topic_key_map[cfg["topic"]] = key

        self.config.state_topics = list(state_topics.values())
        self.config.action_topics = list(action_topics.values())
        self._log_info(
            f"Loaded topics — state: {list(state_topics.keys())}, "
            f"action: {list(action_topics.keys())}"
        )

        for cam_name, cfg in image_groups.items():
            self._camera_mapping[cfg["topic"]] = cam_name
            # ``rotation_deg`` is consumed at recording time by the
            # transcoder, not here. Kept on ``self._camera_rotations``
            # only as informational metadata.
            self._camera_rotations[cam_name] = int(
                cfg.get("rotation_deg") or 0
            )

        # _joint_order_by_group keyed by ``leader_<modality>`` — preserved
        # for _resolve_filter_target_names and the per-group merge logic.
        # Flat _joint_order is the concatenation in yaml insertion order.
        flattened: List[str] = []
        self._joint_order_by_group = {}
        for modality, cfg in action_groups.items():
            joints = list(cfg["joint_names"])
            self._joint_order_by_group[f"leader_{modality}"] = joints
            flattened.extend(joints)
        self._joint_order = flattened
        self._log_info(
            f"Loaded joint_order: {list(self._joint_order_by_group.keys())} "
            f"(total {len(self._joint_order)} joints)"
        )

    def _joint_names_from_config(self, group_prefix: str) -> List[str]:
        """Concatenate joint_order entries whose group key starts with ``prefix``.

        Mirrors the order used by _merge_state_messages /
        _merge_action_messages: groups sorted alphabetically, joints
        within each group taken verbatim from joint_order_by_group.
        Used as a fallback in _build_features when the per-episode
        accumulators (_state_joint_names / _action_joint_names) are
        empty — that happens in the parallel parsing path where worker
        children's attributes don't propagate back to the main process.
        """
        keys = sorted(
            k for k in self._joint_order_by_group if k.startswith(group_prefix)
        )
        names: List[str] = []
        for k in keys:
            names.extend(self._joint_order_by_group[k])
        return names

    def _resolve_filter_target_names(self, group_key: str) -> List[str]:
        """Resolve the ordered joint_names a given state/action group should
        be sliced down to.

        State and action are symmetric for VLA: every joint we command
        (action's leader_<X>) we also observe (state's follower_<X>).
        Predecessor configs only listed leader_* in joint_order; deriving
        the state-side filter from that single source keeps the yaml
        non-redundant and matches that expectation.

        Resolution order for a state group_key:
          1. Direct hit in joint_order_by_group (caller's explicit override).
          2. follower_<modality> → leader_<modality> joint_names.
          3. follower_upper_body (collapsed multi-arm follower) → union of
             every leader_* group except leader_mobile, in joint_order
             insertion order. Matches the per-arm 8/8/2/1 layout the
             leaders advertise.
        Returns an empty list when nothing maps — callers treat that as
        "no filter, take the message verbatim".
        """
        if group_key in self._joint_order_by_group:
            return list(self._joint_order_by_group[group_key])

        if group_key.startswith("follower_"):
            modality = group_key[len("follower_"):]
            leader_key = f"leader_{modality}"
            if leader_key in self._joint_order_by_group:
                return list(self._joint_order_by_group[leader_key])

            if group_key == "follower_upper_body":
                names: List[str] = []
                for k, joints in self._joint_order_by_group.items():
                    if not k.startswith("leader_"):
                        continue
                    if "mobile" in k.lower():
                        continue
                    names.extend(joints)
                return names

        return []

    def _apply_selection_knobs(self) -> None:
        """Apply ConversionConfig selection lists to the discovered defaults.

        Called from ``__init__`` after the robot_config has populated
        state_topics / action_topics / _joint_order / _camera_mapping.
        Empty selection lists are no-ops.
        """
        # State topic subset.
        if self.config.selected_state_topics:
            wanted = set(self.config.selected_state_topics)
            kept = [t for t in self.config.state_topics if t in wanted]
            if kept:
                self._log_info(
                    f"selected_state_topics filter: "
                    f"{len(self.config.state_topics)} → {len(kept)}"
                )
                self.config.state_topics = kept
            else:
                self._log_warning(
                    f"selected_state_topics {self.config.selected_state_topics} "
                    f"didn't match any of {self.config.state_topics}; "
                    f"keeping all"
                )

        # Action topic subset.
        if self.config.selected_action_topics:
            wanted = set(self.config.selected_action_topics)
            kept = [t for t in self.config.action_topics if t in wanted]
            if kept:
                self._log_info(
                    f"selected_action_topics filter: "
                    f"{len(self.config.action_topics)} → {len(kept)}"
                )
                self.config.action_topics = kept
            else:
                self._log_warning(
                    f"selected_action_topics {self.config.selected_action_topics} "
                    f"didn't match any of {self.config.action_topics}; "
                    f"keeping all"
                )

        # Joint subset / reorder. Preserve the order from
        # selected_joints (caller's intent) rather than _joint_order's
        # original order.
        if self.config.selected_joints:
            available = set(self._joint_order)
            kept = [j for j in self.config.selected_joints if j in available]
            if kept:
                self._log_info(
                    f"selected_joints filter: "
                    f"{len(self._joint_order)} → {len(kept)}"
                )
                self._joint_order = kept
                # Also subset each per-group list to the survivors.
                kept_set = set(kept)
                self._joint_order_by_group = {
                    g: [j for j in joints if j in kept_set]
                    for g, joints in self._joint_order_by_group.items()
                }
                # Drop empty groups.
                self._joint_order_by_group = {
                    g: joints for g, joints in self._joint_order_by_group.items()
                    if joints
                }
            else:
                self._log_warning(
                    f"selected_joints {self.config.selected_joints} "
                    f"didn't match any of {self._joint_order}; keeping all"
                )

    def _log_staleness_summary(self, staleness_metrics: Dict[str, StalenessMetrics]):
        for topic, metrics in staleness_metrics.items():
            if metrics.status == "GOOD":
                continue
            self._log_warning(
                f"Staleness {metrics.status} for {topic}: "
                f"warnings={metrics.stale_warning_count}, errors={metrics.stale_error_count}, "
                f"max={metrics.max_staleness_ms:.1f}ms, mean={metrics.mean_staleness_ms:.1f}ms"
            )

    def convert_single_rosbag(
        self,
        bag_path: Path,
        episode_index: int,
    ) -> Optional[EpisodeData]:
        bag_path = Path(bag_path)
        if not bag_path.exists():
            self._log_error(f"Bag path does not exist: {bag_path}")
            return None

        # Refuse to convert episodes whose background H.264 transcode
        # isn't finished — converting against the raw MJPEG source would
        # silently drop the yaml-driven rotation (and any other future
        # transcode-time treatment), producing a misoriented LeRobot
        # dataset. Better to fail loud and let the caller wait/retry.
        if not self._can_convert_transcode_state(bag_path):
            return None

        self._log_info(f"Converting rosbag: {bag_path} (episode {episode_index})")

        # Load per-episode robot_config.yaml if exists and no global config was loaded
        if not self.config.robot_config_path:
            robot_config = self._metadata_manager.load_robot_config(bag_path)
            if robot_config:
                self._update_config_from_robot_config(robot_config)

        trim_points = None
        exclude_regions = []
        if self.config.apply_trim:
            trim_points = self._metadata_manager.get_trim_points(bag_path)
        if self.config.apply_exclude_regions:
            exclude_regions = self._metadata_manager.get_exclude_regions(bag_path)

        episode_data = self._extract_joint_data(
            bag_path, episode_index, trim_points, exclude_regions
        )
        if episode_data is None:
            return None

        video_files = self._find_video_files(bag_path)
        episode_data.video_files = video_files

        # Recording format v2 path: every camera has a sidecar parquet
        # under ``videos/<cam>_timestamps.parquet``. Build a synced MP4
        # per camera so that frame N == grid step N (LeRobot's video
        # reader assumes 1:1 with the parquet rows). For each grid
        # step we pick the most recent MP4 frame with ``recv_ns`` <=
        # ``grid_log_time``; both are subscriber-side clocks so they
        # share an origin.
        episode_data = self._sync_videos_to_grid(bag_path, episode_data)

        # Legacy fallback: when no sidecars exist (recordings made by
        # the v1 pipeline that called rosbag2mp4 before this rewrite)
        # the synced MP4 step is a no-op and the parquet rows still
        # need a 1:1 trim against the raw MP4 frame count.
        if video_files and not self._episode_has_sidecars(bag_path):
            video_frame_counts = {}
            for cam_name, vpath in video_files.items():
                fc = self._get_video_frame_count(vpath)
                if fc is not None:
                    video_frame_counts[cam_name] = fc

            if video_frame_counts:
                target_frames = min(video_frame_counts.values())
                if episode_data.length > target_frames:
                    excess = episode_data.length - target_frames
                    self._log_info(
                        f"Trimming parquet from {episode_data.length} to "
                        f"{target_frames} rows to match video frames "
                        f"(removing {excess} from end)"
                    )
                    episode_data.timestamps = episode_data.timestamps[:target_frames]
                    episode_data.observation_state = episode_data.observation_state[:target_frames]
                    episode_data.action = episode_data.action[:target_frames]
                    episode_data.grid_log_times_sec = (
                        episode_data.grid_log_times_sec[:target_frames]
                    )
                    episode_data.length = target_frames

        task_markers = self._metadata_manager.get_task_markers(bag_path)
        if task_markers:
            episode_data.tasks = list(
                set(m.get("instruction", "default_task") for m in task_markers)
            )
        else:
            # Fall back to episode_info.json which records the
            # task_instruction from the recording session.
            episode_info_path = bag_path / "episode_info.json"
            instruction = ""
            if episode_info_path.exists():
                try:
                    import json as _json
                    with open(episode_info_path) as f:
                        info = _json.load(f)
                    instruction = str(info.get("task_instruction", "") or "")
                except Exception as e:
                    self._log_warning(f"Failed to read {episode_info_path}: {e}")
            episode_data.tasks = [instruction or "default_task"]

        return episode_data

    def _update_config_from_robot_config(self, robot_config: Dict):
        """Update conversion config from robot_config.yaml."""
        if "robot_type" in robot_config:
            self.config.robot_type = robot_config["robot_type"]

        if "state_topics" in robot_config:
            topics = robot_config["state_topics"]
            if isinstance(topics, dict):
                self.config.state_topics = list(topics.values())
                # Build topic -> group key mapping
                for key, topic_path in topics.items():
                    self._state_topic_key_map[topic_path] = key
            elif isinstance(topics, list):
                self.config.state_topics = topics

        if "action_topics" in robot_config:
            topics = robot_config["action_topics"]
            if isinstance(topics, dict):
                self.config.action_topics = list(topics.values())
                for key, topic_path in topics.items():
                    self._action_topic_key_map[topic_path] = key
            elif isinstance(topics, list):
                self.config.action_topics = topics

        if "fps" in robot_config:
            self.config.fps = robot_config["fps"]

        if "camera_mapping" in robot_config:
            self._camera_mapping = robot_config["camera_mapping"]
            self._log_info(f"Loaded camera mapping: {self._camera_mapping}")

        # Load joint_order (nested dict) for per-group ordering
        if "joint_order" in robot_config:
            joint_order = robot_config["joint_order"]
            if isinstance(joint_order, dict):
                self._joint_order_by_group = {}
                flattened = []
                for key, joints in joint_order.items():
                    if isinstance(joints, list):
                        self._joint_order_by_group[key] = joints
                        flattened.extend(joints)
                    else:
                        self._joint_order_by_group[key] = [joints]
                        flattened.append(joints)
                self._joint_order = flattened
                self._log_info(
                    f"Loaded joint_order by group: {list(self._joint_order_by_group.keys())} "
                    f"(total {len(self._joint_order)} joints)"
                )
            else:
                self._joint_order = joint_order
                self._log_info(
                    f"Loaded joint_order with {len(self._joint_order)} joints"
                )

        # Prefer total_joint_order (flat list) if explicitly provided
        if "total_joint_order" in robot_config:
            self._joint_order = robot_config["total_joint_order"]
            self._log_info(
                f"Overriding with total_joint_order: {len(self._joint_order)} joints"
            )

    def _extract_velocity_from_odometry(self, msg) -> Optional[np.ndarray]:
        """Extract velocity values from Odometry message."""
        if hasattr(msg, "twist") and hasattr(msg.twist, "twist"):
            twist = msg.twist.twist
            return np.array([
                twist.linear.x,
                twist.linear.y,
                twist.angular.z,
            ], dtype=np.float32)
        return None

    def _extract_velocity_from_twist(self, msg) -> Optional[np.ndarray]:
        """Extract velocity values from Twist message."""
        if hasattr(msg, "linear") and hasattr(msg, "angular"):
            return np.array([
                msg.linear.x,
                msg.linear.y,
                msg.angular.z,
            ], dtype=np.float32)
        return None

    def _get_topic_group_key(self, topic: str, role: str) -> str:
        """Get the group key for a topic, using config mapping or deriving from path."""
        if role == "state" and topic in self._state_topic_key_map:
            return self._state_topic_key_map[topic]
        if role == "action" and topic in self._action_topic_key_map:
            return self._action_topic_key_map[topic]
        # Derive from topic path
        parts = topic.strip("/").split("/")
        for part in parts:
            if "follower" in part or "leader" in part:
                role_word = "follower" if "follower" in part else "leader"
                body_part = part.replace(f"_{role_word}", "").replace(f"{role_word}_", "")
                return f"{role_word}_{body_part}"
        if "odom" in topic.lower():
            return "follower_mobile"
        if "cmd_vel" in topic.lower():
            return "leader_mobile"
        return topic

    def _extract_joint_data(
        self,
        bag_path: Path,
        episode_index: int,
        trim_points: Optional[Dict],
        exclude_regions: List[Dict],
    ) -> Optional[EpisodeData]:
        """Extract joint state and action data from ROSbag."""
        # Clear the per-instance bisect-keys cache. Each episode builds a
        # fresh state_messages / action_messages list, but Python may reuse
        # the previous list's memory address for the new one — without this
        # reset, ``_find_previous_value(_in_list)`` would silently return
        # keys belonging to the prior episode and corrupt the resample.
        self._bisect_keys_cache.clear()

        reader = BagReader(bag_path, self.logger)
        if not reader.open():
            self._log_error(f"Failed to open rosbag: {bag_path}")
            return None

        episode = EpisodeData(episode_index=episode_index)

        # Determine time bounds from trim points
        trim_start = (
            trim_points.get("start", {}).get("time", 0.0) if trim_points else 0.0
        )
        trim_end = (
            trim_points.get("end", {}).get("time", float("inf"))
            if trim_points
            else float("inf")
        )

        # Group both state and action messages by topic
        state_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        state_joint_names_by_topic: Dict[str, List[str]] = {}
        action_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        action_joint_names_by_topic: Dict[str, List[str]] = {}

        topic_types = reader.get_topic_types()

        # Build topic filter to avoid decoding unnecessary messages (TF, CameraInfo, etc.)
        topics_to_read = None
        if self.config.state_topics or self.config.action_topics:
            topics_to_read = list(
                set(self.config.state_topics + self.config.action_topics)
            )
            self._log_info(f"Reading {len(topics_to_read)} topics (filtered)")

        for topic, msg, timestamp in reader.read_messages(topic_filter=topics_to_read):
            # Skip if outside trim bounds
            if timestamp < trim_start or timestamp > trim_end:
                continue

            # Skip if in exclude region
            if self._is_in_exclude_region(timestamp, exclude_regions):
                continue

            topic_type = topic_types.get(topic, "")

            # Process state topics
            if self._is_state_topic(topic, topic_types):
                positions = None
                joint_names = []

                if "Odometry" in topic_type:
                    positions = self._extract_velocity_from_odometry(msg)
                    group_key = self._get_topic_group_key(topic, "state")
                    if group_key in self._joint_order_by_group:
                        joint_names = self._joint_order_by_group[group_key]
                    else:
                        joint_names = ["linear_x", "linear_y", "angular_z"]
                elif hasattr(msg, "position") and msg.position:
                    positions = np.array(msg.position, dtype=np.float32)
                    joint_names = list(msg.name) if hasattr(msg, "name") and msg.name else []

                if positions is not None:
                    if topic not in state_messages_by_topic:
                        state_messages_by_topic[topic] = []
                    state_messages_by_topic[topic].append((timestamp, positions))
                    if topic not in state_joint_names_by_topic and joint_names:
                        state_joint_names_by_topic[topic] = joint_names

            # Process action topics
            elif self._is_action_topic(topic, topic_types):
                positions = None
                joint_names = []

                if "Twist" in topic_type:
                    positions = self._extract_velocity_from_twist(msg)
                    group_key = self._get_topic_group_key(topic, "action")
                    if group_key in self._joint_order_by_group:
                        joint_names = self._joint_order_by_group[group_key]
                    else:
                        joint_names = ["linear_x", "linear_y", "angular_z"]
                else:
                    positions = self._extract_action_positions(msg)
                    joint_names = self._extract_joint_names(msg)

                if positions is not None:
                    if topic not in action_messages_by_topic:
                        action_messages_by_topic[topic] = []
                    action_messages_by_topic[topic].append((timestamp, positions))
                    if topic not in action_joint_names_by_topic and joint_names:
                        action_joint_names_by_topic[topic] = joint_names

        if not state_messages_by_topic:
            self._log_warning(f"No state messages found in {bag_path}")
            return None

        state_messages = self._merge_state_messages(
            state_messages_by_topic, state_joint_names_by_topic
        )
        action_messages = self._merge_action_messages(
            action_messages_by_topic, action_joint_names_by_topic
        )

        if not state_messages:
            self._log_warning(f"No valid merged state messages in {bag_path}")
            return None

        episode, staleness_metrics = self._resample_to_fps(
            episode, state_messages, action_messages, trim_start
        )

        self._staleness_reports[episode_index] = staleness_metrics
        self._log_staleness_summary(staleness_metrics)

        return episode

    def _is_state_topic(self, topic: str, topic_types: Dict[str, str]) -> bool:
        """Check if topic is a state topic."""
        if self.config.state_topics:
            return topic in self.config.state_topics

        # Default heuristics — check action indicators first to avoid false positives
        topic_type = topic_types.get(topic, "")
        topic_lower = topic.lower()
        is_action_indicator = (
            "leader" in topic_lower
            or "action" in topic_lower
            or "command" in topic_lower
            or "cmd_vel" in topic_lower
        )
        if "JointState" in topic_type or "JointTrajectory" in topic_type:
            if not is_action_indicator and "follower" in topic_lower:
                return True
        if "Odometry" in topic_type:
            if not is_action_indicator:
                return True
        return False

    def _is_action_topic(self, topic: str, topic_types: Dict[str, str]) -> bool:
        """Check if topic is an action topic."""
        if self.config.action_topics:
            return topic in self.config.action_topics

        # Default heuristics
        topic_type = topic_types.get(topic, "")
        topic_lower = topic.lower()
        if "JointTrajectory" in topic_type or "JointState" in topic_type:
            if (
                "leader" in topic_lower
                or "action" in topic_lower
                or "command" in topic_lower
            ):
                return True
        if "Twist" in topic_type:
            if "leader" in topic_lower or "cmd_vel" in topic_lower:
                return True
        return False

    def _extract_action_positions(self, msg) -> Optional[np.ndarray]:
        """Extract position values from action message."""
        # JointTrajectory message
        if hasattr(msg, "points") and msg.points:
            point = msg.points[0]
            if hasattr(point, "positions") and point.positions:
                return np.array(point.positions, dtype=np.float32)

        # JointState message
        if hasattr(msg, "position") and msg.position:
            return np.array(msg.position, dtype=np.float32)

        return None

    def _extract_joint_names(self, msg) -> List[str]:
        """Extract joint names from message."""
        if hasattr(msg, "joint_names") and msg.joint_names:
            return list(msg.joint_names)
        if hasattr(msg, "name") and msg.name:
            return list(msg.name)
        return []

    def _filter_positions_by_joint_order(
        self,
        positions: np.ndarray,
        msg_names: List[str],
        joint_order: List[str],
    ) -> Optional[np.ndarray]:
        """
        Filter positions array to only include joints in joint_order.

        Args:
            positions: Array of joint positions from message
            msg_names: Joint names from message (same order as positions)
            joint_order: Ordered list of joints to include in output

        Returns:
            Filtered positions array with only joints in joint_order,
            or None if any joint in joint_order is missing from msg_names.
        """
        if len(positions) != len(msg_names):
            self._log_warning(
                f"Position/name length mismatch: {len(positions)} vs {len(msg_names)}"
            )
            return None

        # Build name-to-index mapping
        name_to_idx = {name: idx for idx, name in enumerate(msg_names)}

        # Extract positions in joint_order
        filtered = []
        for joint_name in joint_order:
            if joint_name not in name_to_idx:
                self._log_warning(
                    f"Joint '{joint_name}' from joint_order not found in message"
                )
                return None
            filtered.append(positions[name_to_idx[joint_name]])

        return np.array(filtered, dtype=np.float32)

    def _is_in_exclude_region(
        self, timestamp: float, exclude_regions: List[Dict]
    ) -> bool:
        """Check if timestamp falls within any exclude region."""
        for region in exclude_regions:
            start = region.get("start", {}).get("time", 0)
            end = region.get("end", {}).get("time", 0)
            if start <= timestamp <= end:
                return True
        return False

    def _merge_action_messages(
        self,
        action_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]],
        action_joint_names_by_topic: Dict[str, List[str]],
    ) -> List[Tuple[float, np.ndarray]]:
        """Merge action messages from multiple topics into a single action vector."""
        if not action_messages_by_topic:
            return []

        # Determine topic ordering using group keys
        topic_to_group: Dict[str, str] = {}
        for topic in action_messages_by_topic.keys():
            group_key = self._get_topic_group_key(topic, "action")
            topic_to_group[topic] = group_key

        # Sort topics by their group key for consistent ordering
        sorted_topics = sorted(
            action_messages_by_topic.keys(),
            key=lambda t: topic_to_group.get(t, t)
        )

        # Build combined joint names, applying per-group joint_order if available
        combined_names = []
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            if group_key in self._joint_order_by_group:
                combined_names.extend(self._joint_order_by_group[group_key])
            else:
                names = action_joint_names_by_topic.get(topic, [])
                combined_names.extend(names)
        self._action_joint_names = combined_names

        # Use timestamps from the first topic as reference
        reference_topic = sorted_topics[0]
        reference_timestamps = sorted(
            t for t, _ in action_messages_by_topic[reference_topic]
        )

        # For each reference timestamp, concatenate actions from all topics
        # Only include timestamps where ALL topics have valid previous values
        merged_messages: List[Tuple[float, np.ndarray]] = []

        for timestamp in reference_timestamps:
            combined_action = []
            all_topics_have_data = True

            for topic in sorted_topics:
                msgs = action_messages_by_topic[topic]
                prev_value, _ = self._find_previous_value_in_list(
                    msgs, timestamp, tolerance=0.05
                )
                if prev_value is not None:
                    # Apply per-group joint_order filtering if available
                    group_key = topic_to_group[topic]
                    if group_key in self._joint_order_by_group:
                        group_names = action_joint_names_by_topic.get(topic, [])
                        target_names = self._joint_order_by_group[group_key]
                        if group_names:
                            filtered = self._filter_positions_by_joint_order(
                                prev_value, group_names, target_names
                            )
                            if filtered is not None:
                                combined_action.extend(filtered.tolist())
                            else:
                                all_topics_have_data = False
                                break
                        else:
                            combined_action.extend(prev_value.tolist())
                    else:
                        combined_action.extend(prev_value.tolist())
                else:
                    all_topics_have_data = False
                    break

            if all_topics_have_data and combined_action:
                merged_messages.append(
                    (timestamp, np.array(combined_action, dtype=np.float32))
                )

        return merged_messages

    def _merge_state_messages(
        self,
        state_messages_by_topic: Dict[str, List[Tuple[float, np.ndarray]]],
        state_joint_names_by_topic: Dict[str, List[str]],
    ) -> List[Tuple[float, np.ndarray]]:
        """Merge state messages from multiple topics into a single state vector.

        Uses joint_order_by_group to filter/reorder each topic's joints,
        then concatenates them in sorted group key order.
        """
        if not state_messages_by_topic:
            return []

        # If only one topic and no grouping needed, use simple path
        if len(state_messages_by_topic) == 1 and not self._joint_order_by_group:
            topic = list(state_messages_by_topic.keys())[0]
            names = state_joint_names_by_topic.get(topic, [])
            if names:
                self._state_joint_names = names
            return state_messages_by_topic[topic]

        # Determine topic ordering using group keys
        topic_to_group: Dict[str, str] = {}
        for topic in state_messages_by_topic.keys():
            group_key = self._get_topic_group_key(topic, "state")
            topic_to_group[topic] = group_key

        # Sort topics by canonical action-side ordering (leader_* keys in
        # joint_order, yaml insertion order). follower_<X> sorts to the
        # position of leader_<X>, follower_upper_body sorts to the first
        # non-mobile leader. Result: state and action emit dimensions in
        # the same per-modality order (mobile last for ffw_sg2, matching
        # the predecessor cyclo_intelligence layout).
        canonical_keys = [
            k for k in self._joint_order_by_group if k.startswith("leader_")
        ]

        def _state_sort_key(topic: str) -> Tuple[int, int, str]:
            gk = topic_to_group[topic]
            if gk.startswith("follower_"):
                modality = gk[len("follower_"):]
                leader_key = f"leader_{modality}"
                if leader_key in canonical_keys:
                    return (0, canonical_keys.index(leader_key), gk)
                if gk == "follower_upper_body":
                    for i, k in enumerate(canonical_keys):
                        if "mobile" not in k.lower():
                            return (0, i, gk)
            if gk in canonical_keys:
                return (0, canonical_keys.index(gk), gk)
            return (1, len(canonical_keys), gk)

        sorted_topics = sorted(
            state_messages_by_topic.keys(),
            key=_state_sort_key,
        )

        # Build combined joint names, applying joint_order if a state-side
        # filter resolves (directly or via follower_<X>→leader_<X>
        # symmetry — see _resolve_filter_target_names).
        combined_names = []
        for topic in sorted_topics:
            group_key = topic_to_group[topic]
            target = self._resolve_filter_target_names(group_key)
            if target:
                combined_names.extend(target)
            else:
                names = state_joint_names_by_topic.get(topic, [])
                combined_names.extend(names)
        self._state_joint_names = combined_names

        # Use timestamps from the first topic as reference
        # (all joint topics publish at ~100Hz, so any topic works)
        # This avoids creating the full cross-product of timestamps
        reference_topic = sorted_topics[0]
        reference_timestamps = sorted(
            t for t, _ in state_messages_by_topic[reference_topic]
        )

        # For each reference timestamp, merge state from all topics using causal sync
        merged_messages: List[Tuple[float, np.ndarray]] = []

        for timestamp in reference_timestamps:
            combined_state = []
            all_topics_have_data = True

            for topic in sorted_topics:
                msgs = state_messages_by_topic[topic]
                prev_value, _ = self._find_previous_value_in_list(
                    msgs, timestamp, tolerance=0.05
                )
                if prev_value is not None:
                    # Apply joint_order filtering when a target resolves;
                    # otherwise pass the message's positions through.
                    group_key = topic_to_group[topic]
                    target_names = self._resolve_filter_target_names(group_key)
                    if target_names:
                        group_names = state_joint_names_by_topic.get(topic, [])
                        if group_names:
                            filtered = self._filter_positions_by_joint_order(
                                prev_value, group_names, target_names
                            )
                            if filtered is not None:
                                combined_state.extend(filtered.tolist())
                            else:
                                all_topics_have_data = False
                                break
                        else:
                            combined_state.extend(prev_value.tolist())
                    else:
                        combined_state.extend(prev_value.tolist())
                else:
                    all_topics_have_data = False
                    break

            if all_topics_have_data and combined_state:
                merged_messages.append(
                    (timestamp, np.array(combined_state, dtype=np.float32))
                )

        self._log_info(
            f"Merged state from {len(sorted_topics)} topics: "
            f"{len(merged_messages)} merged samples, {len(combined_names)} dimensions"
        )

        return merged_messages

    def _find_previous_value_in_list(
        self,
        messages: List[Tuple[float, np.ndarray]],
        target_time: float,
        tolerance: float = float("inf"),
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find the most recent message value at or before target time (causal sync).

        Uses binary search (bisect) for O(log n) performance.
        Messages must be sorted by timestamp.

        Returns:
            Tuple of (value, staleness_ms) where staleness_ms is how old the value is.
            Returns (None, 0.0) if no valid previous value exists.

        Implementation note: keys cache lives on the instance (cleared by
        ``_extract_joint_data`` per episode). The previous default-argument
        dict was shared across all calls in the same worker process and
        could in principle return stale keys when a new ``state_messages``
        list reused a GC'd address — instance scope makes that impossible.
        """
        if not messages:
            return None, 0.0

        # Cache timestamp keys for repeated lookups on the same list
        list_id = id(messages)
        if (list_id not in self._bisect_keys_cache
                or len(self._bisect_keys_cache[list_id]) != len(messages)):
            self._bisect_keys_cache[list_id] = [t for t, _ in messages]
        keys = self._bisect_keys_cache[list_id]

        # Binary search: find rightmost index where time <= target_time
        idx = bisect.bisect_right(keys, target_time) - 1
        if idx < 0:
            return None, 0.0

        best_time = keys[idx]
        best_value = messages[idx][1]

        staleness_ms = (target_time - best_time) * 1000.0
        if staleness_ms > tolerance * 1000.0:
            return None, staleness_ms

        return best_value, staleness_ms

    def _resample_to_fps(
        self,
        episode: EpisodeData,
        state_messages: List[Tuple[float, np.ndarray]],
        action_messages: List[Tuple[float, np.ndarray]],
        start_time: float,
    ) -> Tuple[EpisodeData, Dict[str, StalenessMetrics]]:
        """Resample messages to target FPS using causal sync (previous value only)."""
        staleness_metrics: Dict[str, StalenessMetrics] = {
            "observation.state": StalenessMetrics(topic="observation.state"),
            "action": StalenessMetrics(topic="action"),
        }

        if not state_messages:
            return episode, staleness_metrics

        state_times = [t for t, _ in state_messages]
        min_time = min(state_times)
        max_time = max(state_times)

        # Find the first valid start time where both state AND action have data
        # This avoids zero-filled frames at the beginning
        effective_min_time = min_time
        if action_messages:
            action_times = [t for t, _ in action_messages]
            first_action_time = min(action_times)
            # Start from the later of first state or first action
            effective_min_time = max(min_time, first_action_time)
            if effective_min_time > min_time:
                self._log_info(
                    f"Adjusted start time: state_start={min_time:.3f}, "
                    f"action_start={first_action_time:.3f}, "
                    f"effective_start={effective_min_time:.3f}"
                )

        frame_duration = 1.0 / self.config.fps
        num_frames = int((max_time - effective_min_time) * self.config.fps) + 1

        state_staleness_values: List[float] = []
        action_staleness_values: List[float] = []

        action_dim = 0
        if action_messages:
            # Find first valid action message to get dimension
            for _, action_arr in action_messages:
                if len(action_arr) > 0:
                    action_dim = len(action_arr)
                    break

        warning_threshold_ms = (
            1000.0 / self.config.fps
        ) * self.config.quality_warning_multiplier
        error_threshold_ms = (
            1000.0 / self.config.fps
        ) * self.config.quality_error_multiplier

        for frame_idx in range(num_frames):
            target_time = effective_min_time + frame_idx * frame_duration
            # Relative time is from effective start
            relative_time = target_time - effective_min_time

            state, state_staleness_ms = self._find_previous_value(
                state_messages, target_time, frame_duration
            )
            if state is None:
                continue

            staleness_metrics["observation.state"].total_samples += 1
            state_staleness_values.append(state_staleness_ms)
            self._track_staleness(
                staleness_metrics["observation.state"],
                frame_idx,
                state_staleness_ms,
                warning_threshold_ms,
                error_threshold_ms,
            )

            if action_messages and action_dim > 0:
                action, action_staleness_ms = self._find_previous_value(
                    action_messages, target_time, frame_duration
                )
                # Skip this frame if no valid action data
                if action is None:
                    self._log_warning(
                        f"Frame {frame_idx}: No action data at t={target_time:.3f}"
                    )
                    continue

                staleness_metrics["action"].total_samples += 1
                action_staleness_values.append(action_staleness_ms)
                self._track_staleness(
                    staleness_metrics["action"],
                    frame_idx,
                    action_staleness_ms,
                    warning_threshold_ms,
                    error_threshold_ms,
                )
            else:
                action = np.zeros(len(state), dtype=np.float32)

            episode.timestamps.append(relative_time)
            episode.grid_log_times_sec.append(target_time)
            episode.observation_state.append(state)
            episode.action.append(action)

        episode.length = len(episode.timestamps)

        if state_staleness_values:
            staleness_metrics["observation.state"].mean_staleness_ms = float(
                np.mean(state_staleness_values)
            )
            staleness_metrics["observation.state"].max_staleness_ms = float(
                np.max(state_staleness_values)
            )

        if action_staleness_values:
            staleness_metrics["action"].mean_staleness_ms = float(
                np.mean(action_staleness_values)
            )
            staleness_metrics["action"].max_staleness_ms = float(
                np.max(action_staleness_values)
            )

        return episode, staleness_metrics

    def _track_staleness(
        self,
        metrics: StalenessMetrics,
        frame_idx: int,
        staleness_ms: float,
        warning_threshold_ms: float,
        error_threshold_ms: float,
    ):
        if staleness_ms > error_threshold_ms:
            metrics.stale_error_count += 1
            metrics.stale_samples.append(
                {
                    "frame_index": frame_idx,
                    "staleness_ms": round(staleness_ms, 2),
                    "severity": "error",
                }
            )
        elif staleness_ms > warning_threshold_ms:
            metrics.stale_warning_count += 1
            metrics.stale_samples.append(
                {
                    "frame_index": frame_idx,
                    "staleness_ms": round(staleness_ms, 2),
                    "severity": "warning",
                }
            )

    def _find_previous_value(
        self,
        messages: List[Tuple[float, np.ndarray]],
        target_time: float,
        expected_interval_sec: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find the most recent message value at or before target time (causal sync).

        Uses binary search (bisect) for O(log n) performance.
        Messages must be sorted by timestamp.

        Args:
            messages: List of (timestamp, value) tuples
            target_time: Target time to find previous value for
            expected_interval_sec: Expected interval between messages (for staleness calc)

        Returns:
            Tuple of (value, staleness_ms). Returns (None, 0.0) if no previous value.

        See ``_find_previous_value_in_list`` docstring for the cache rationale
        — same shared-state bug, same fix (per-instance cache cleared per
        ``_extract_joint_data`` call).
        """
        if not messages:
            return None, 0.0

        # Cache timestamp keys for repeated lookups on the same list
        list_id = id(messages)
        if (list_id not in self._bisect_keys_cache
                or len(self._bisect_keys_cache[list_id]) != len(messages)):
            self._bisect_keys_cache[list_id] = [t for t, _ in messages]
        keys = self._bisect_keys_cache[list_id]

        # Binary search: find rightmost index where time <= target_time
        idx = bisect.bisect_right(keys, target_time) - 1
        if idx < 0:
            return None, 0.0

        best_time = keys[idx]
        best_value = messages[idx][1]

        staleness_ms = (target_time - best_time) * 1000.0
        return best_value, staleness_ms

    def _ensure_video_stats_cached(
        self, video_path: Path, camera_name: str
    ) -> None:
        """Precompute + persist video stats to ``video_stats.json``.

        ``_compute_video_stats`` already looks for a ``video_stats.json``
        sidecar next to the video file and skips its cv2 decode loop on
        cache hit. The recording-v2 pipeline doesn't write that file
        anywhere, so Phase 2 (``_compute_episode_stats``) would otherwise
        re-decode 100 frames per camera per episode — that's the serial
        bottleneck that pegged Phase 2 at multi-minute runs.

        By precomputing during ``_sync_videos_to_grid`` (Phase 1) the
        decode work moves into the parallel pool *and* gets cached
        forever for subsequent runs.

        Idempotent: bails immediately if the sidecar already has this
        camera's entry. Best-effort: any failure (cv2 missing,
        unreadable mp4, json write fails) is logged at warning and the
        call is a no-op — Phase 2 falls back to its own decode path.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return
        stats_path = video_path.parent / "video_stats.json"
        existing: Dict[str, Any] = {}
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
            except (OSError, ValueError) as exc:
                self._log_warning(
                    f"{camera_name}: video_stats.json unreadable "
                    f"({exc!r}); regenerating"
                )
                loaded = None
            # Defensive: a stale sidecar that's not a dict (e.g. legacy
            # list shape, ``null``, hand-edited) would let the downstream
            # ``existing[camera_name] = stats`` raise TypeError and
            # propagate to ``_sync_videos_to_grid``'s outer except — which
            # then falls back to the raw MP4 and corrupts parquet/mp4
            # length parity. Coerce non-dicts back to {} here.
            if isinstance(loaded, dict):
                existing = loaded
            else:
                if loaded is not None:
                    self._log_warning(
                        f"{camera_name}: video_stats.json was "
                        f"{type(loaded).__name__}, expected dict; "
                        "discarding and regenerating"
                    )
                existing = {}
            if camera_name in existing:
                return
        # Compute fresh. ``_compute_video_stats`` would itself try the
        # sidecar first; since we just verified the camera isn't in it,
        # the call will fall through to decoding.
        try:
            stats = self._compute_video_stats(video_path, camera_name)
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{camera_name}: video_stats precompute raised "
                f"({exc!r}); Phase 2 will compute lazily"
            )
            return
        if not stats:
            return
        existing[camera_name] = stats
        try:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh)
        except OSError as exc:
            self._log_warning(
                f"{camera_name}: failed to write video_stats.json "
                f"({exc!r}); Phase 2 will recompute"
            )

    def _can_convert_transcode_state(self, bag_path: Path) -> bool:
        """Return True iff the episode is safe to feed to LeRobot conversion.

        For recording format v2 episodes the background transcoder bakes
        the yaml ``rotation_deg`` (and future transcode-time treatments)
        into the H.264 source MP4. Converting before the transcode has
        finished would silently drop those — so we refuse and log a
        clear actionable error per status.

        Pre-v2 episodes (no ``transcoding_status`` field, no
        ``recorder_format_version`` field) are accepted unconditionally
        for backward compatibility.
        """
        info = bag_path / "episode_info.json"
        if not info.exists():
            return True  # not a v2 episode; nothing to gate on
        try:
            import json as _json
            with open(info) as f:
                meta = _json.load(f) or {}
        except Exception:
            return True
        # No status field at all → legacy episode, allow.
        if "transcoding_status" not in meta:
            return True

        status = meta.get("transcoding_status")
        if status in ("done", "not_required"):
            return True
        if status in ("pending", "running"):
            self._log_error(
                f"{bag_path.name}: SKIPPED — transcode is still {status!r}. "
                "Wait for the background transcoder to finish (check "
                "episode_info.json) and re-run conversion."
            )
            return False
        if status == "failed":
            cams_failed = meta.get("transcoding_cameras_failed", {})
            self._log_error(
                f"{bag_path.name}: SKIPPED — previous transcode failed for "
                f"cameras {list(cams_failed.keys()) or 'unknown'}. Inspect "
                f"the recording, fix the cause, then re-trigger the "
                f"transcoder (or delete episode_info.json's transcoding_status "
                f"to retry from raw MJPEG)."
            )
            return False
        # Unknown status string — be conservative and refuse.
        self._log_error(
            f"{bag_path.name}: SKIPPED — unrecognised transcoding_status={status!r}"
        )
        return False

    def _episode_has_sidecars(self, bag_path: Path) -> bool:
        """True if ``videos/<cam>_timestamps.parquet`` exists for any cam."""
        videos = bag_path / "videos"
        if not videos.exists():
            return False
        return any(videos.glob("*_timestamps.parquet"))

    def _sync_videos_to_grid(
        self, bag_path: Path, episode: EpisodeData,
    ) -> EpisodeData:
        """Re-pack each camera's source MP4 to one frame per grid step.

        Reads ``videos/<cam>_timestamps.parquet`` written by the
        recorder, maps the EpisodeData log-time grid (sub-second
        UNIX seconds) onto frame indices via causal lookup, then
        runs ``video_sync.remux_selected_frames`` to produce a
        ``videos/<cam>_synced.mp4``. The synced MP4 replaces the raw
        recording in ``episode.video_files`` so downstream copy is
        identical to the old rosbag2mp4 flow.
        """
        if not episode.video_files or not episode.grid_log_times_sec:
            return episode
        if not self._episode_has_sidecars(bag_path):
            return episode
        from cyclo_data.converter.video_sync import remux_selected_frames
        from cyclo_data.reader.frame_timestamps import (
            FrameTimestamps,
            load_frame_timestamps,
        )

        grid_ns = np.asarray(
            [int(t * 1_000_000_000) for t in episode.grid_log_times_sec],
            dtype=np.int64,
        )
        videos_dir = bag_path / "videos"
        synced: Dict[str, Path] = {}
        # UI-supplied rotation is treated as an *additional* override on
        # top of whatever the recorder/transcoder already baked into the
        # source MP4. The yaml's ``rotation_deg`` is applied at H.264
        # transcode time in ``cyclo_data/recorder/transcoder.py``, so by
        # the time we get here the source is already correctly oriented
        # and UI default 0 means "leave as-is" — exactly what we want.
        ui_rotations = self.config.camera_rotations or {}
        for cam_name, src_path in episode.video_files.items():
            sidecar = videos_dir / f"{cam_name}_timestamps.parquet"
            if not sidecar.exists():
                self._log_warning(
                    f"{cam_name}: no sidecar {sidecar.name}; leaving raw MP4"
                )
                synced[cam_name] = src_path
                continue
            try:
                ft = load_frame_timestamps(sidecar, cam_name)
            except Exception as exc:
                self._log_error(
                    f"{cam_name}: failed to read {sidecar.name}: {exc!r}"
                )
                synced[cam_name] = src_path
                continue
            if ft.num_frames == 0:
                self._log_warning(f"{cam_name}: empty sidecar; skipping sync")
                synced[cam_name] = src_path
                continue
            indices = ft.map_to_grid(grid_ns)
            out_path = videos_dir / f"{cam_name}_synced.mp4"
            rotation_extra = int(ui_rotations.get(cam_name, 0) or 0)
            target_fps = int(self.config.fps)
            image_resize = self.config.image_resize  # (height, width) or None
            resize_key = (
                list(image_resize) if image_resize else None
            )

            # Cache reuse: when v2.1 and v3.0 run on the same dataset
            # back-to-back, v3.0's Phase 1 would otherwise redo the
            # entire sync remux that v2.1 already produced. The sidecar
            # records the params that *would* have produced this MP4 —
            # mismatch on fps / rotation / resize / frame_count
            # invalidates the cache so changing knobs via the UI always
            # regenerates.
            cache_sidecar = videos_dir / f"{cam_name}_synced.cache.json"
            desired_key = {
                "target_fps": target_fps,
                "rotation_deg": rotation_extra,
                "image_resize": resize_key,
                "frame_count": int(indices.size),
            }
            if (
                out_path.exists()
                and cache_sidecar.exists()
                and out_path.stat().st_size > 0
            ):
                try:
                    cached_key = json.loads(cache_sidecar.read_text())
                    if cached_key == desired_key:
                        cached_frames = self._get_video_frame_count(out_path)
                        if cached_frames != int(indices.size):
                            self._log_warning(
                                f"{cam_name}: cached synced MP4 frame count "
                                f"mismatch (cache={indices.size}, "
                                f"mp4={cached_frames}); regenerating"
                            )
                        else:
                            self._log_info(
                                f"{cam_name}: reusing cached synced MP4 "
                                f"({indices.size} frames, fps={target_fps}, "
                                f"rot={rotation_extra}°)"
                            )
                            # Even on cache hit, top up the video_stats.json
                            # sidecar so Phase 2 stays a cache hit. If a
                            # previous run wrote synced.mp4 + cache.json but
                            # not video_stats.json (e.g. ran before this
                            # precompute was added), this fills the gap.
                            self._ensure_video_stats_cached(out_path, cam_name)
                            synced[cam_name] = out_path
                            continue
                except (OSError, ValueError) as exc:
                    self._log_warning(
                        f"{cam_name}: cache sidecar unreadable "
                        f"({exc!r}); regenerating"
                    )
                except Exception as exc:
                    self._log_warning(
                        f"{cam_name}: cached synced MP4 validation failed "
                        f"({exc!r}); regenerating"
                    )
            try:
                remux_selected_frames(
                    src_path, indices, out_path,
                    target_fps=target_fps,
                    rotation_deg=rotation_extra,
                    image_resize=image_resize,
                )
                produced_frames = self._get_video_frame_count(out_path)
                cache_is_valid = produced_frames == int(indices.size)
                # Best-effort sidecar write — failure here just means
                # the next run won't get a cache hit. Don't fail the
                # whole conversion over a metadata write.
                try:
                    if cache_is_valid:
                        cache_sidecar.write_text(json.dumps(desired_key))
                    else:
                        cache_sidecar.unlink(missing_ok=True)
                        self._log_info(
                            f"{cam_name}: not caching synced MP4 because "
                            f"frame count is {produced_frames}, expected "
                            f"{indices.size}"
                        )
                except OSError as exc:
                    self._log_warning(
                        f"{cam_name}: failed to write cache sidecar "
                        f"({exc!r}); cache disabled for this run"
                    )
                self._log_info(
                    f"{cam_name}: synced MP4 {indices.size} frames "
                    f"-> {out_path.name} (UI extra rotation={rotation_extra}°)"
                )
                # Precompute Phase 2's per-camera video stats while we
                # have a fresh synced.mp4 and a worker process to spare.
                # Phase 2 (``_compute_episode_stats``) is single-threaded
                # — caching here moves the decode cost into Phase 1's
                # parallel pool, plus the result persists across runs.
                self._ensure_video_stats_cached(out_path, cam_name)
                synced[cam_name] = out_path
            except Exception as exc:
                self._log_error(
                    f"{cam_name}: remux failed ({exc!r}); leaving raw MP4"
                )
                synced[cam_name] = src_path

        episode.video_files = synced

        # Safety: when remux falls back to the raw MP4 for one or more
        # cameras (synced.mp4 generation raised or returned the source
        # path), the per-camera frame counts can disagree:
        #   - successful synced.mp4 → indices.size frames (= episode.length)
        #   - fallback raw mp4 → recorder frame count (often < indices.size)
        # Downstream lerobot loaders demand parquet rows == every camera's
        # mp4 frames; an off-by-N mismatch raises ``Invalid frame index``
        # at training time. We align everything to ``min(frame_count)``:
        #   1. truncate the in-memory episode (timestamps / state / action
        #      / grid) so parquet rows == min,
        #   2. trim any mp4 that's still longer (the synced.mp4 outputs
        #      we generated at the original episode.length) down to min
        #      via a fast ``-c copy`` stream copy.
        if episode.length > 0 and synced:
            counts: Dict[str, int] = {}
            for cam_name, video_path in synced.items():
                fc = self._get_video_frame_count(video_path)
                if fc is not None:
                    counts[cam_name] = fc
            if counts:
                min_frames = min(counts.values())
                if min_frames < episode.length or any(
                    c > min_frames for c in counts.values()
                ):
                    mismatched = {
                        cam: c for cam, c in counts.items()
                        if c != episode.length
                    }
                    if mismatched:
                        self._log_warning(
                            f"Episode {episode.episode_index}: "
                            f"camera frame counts disagree with grid "
                            f"length={episode.length}; aligning all "
                            f"to min={min_frames}. Disagreement: "
                            f"{mismatched}"
                        )
                    if min_frames < episode.length:
                        episode.timestamps = episode.timestamps[:min_frames]
                        if episode.observation_state:
                            episode.observation_state = (
                                episode.observation_state[:min_frames]
                            )
                        if episode.action:
                            episode.action = episode.action[:min_frames]
                        if episode.grid_log_times_sec:
                            episode.grid_log_times_sec = (
                                episode.grid_log_times_sec[:min_frames]
                            )
                        episode.length = min_frames
                    # Trim any mp4 longer than min_frames.
                    for cam_name, video_path in synced.items():
                        if counts.get(cam_name, min_frames) <= min_frames:
                            continue
                        self._trim_video_to_n_frames(video_path, min_frames)
                    # If an episode was shortened after sync, any
                    # ``*_synced.cache.json`` keyed to the pre-trim grid
                    # is no longer trustworthy. Drop it so the next
                    # conversion validates/regenerates instead of
                    # silently reusing a stale MP4.
                    for video_path in synced.values():
                        if video_path.stem.endswith("_synced"):
                            video_path.with_name(
                                video_path.stem + ".cache.json"
                            ).unlink(missing_ok=True)

        return episode

    def _trim_video_to_n_frames(self, video_path: Path, n: int) -> None:
        """Truncate ``video_path`` to its first ``n`` frames in place.

        Used as a safety step in ``_sync_videos_to_grid`` to bring
        per-camera mp4 frame counts into agreement after a fallback
        pulls episode.length down to a smaller value than the synced
        mp4 we already generated. ``-c copy`` keeps it a fast remux:
        no decode, no re-encode, output is bit-identical for the kept
        frames. Atomic rename to avoid partial files on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists() or n <= 0:
            return
        # Keep the ``.mp4`` suffix on the tmp path so ffmpeg can
        # auto-detect the container. The previous ``.mp4.trim.tmp``
        # form left ``.tmp`` as the suffix and ffmpeg refused with
        # "use a standard extension or specify the format manually".
        tmp_path = video_path.with_name(video_path.stem + ".trim_tmp.mp4")
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(video_path),
                "-frames:v", str(n),
                "-c", "copy",
                "-movflags", "+faststart",
                "-f", "mp4",
                str(tmp_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                self._log_warning(
                    f"trim {video_path.name} to {n} frames failed: "
                    f"{result.stderr[-300:]}"
                )
                tmp_path.unlink(missing_ok=True)
                return
            tmp_path.replace(video_path)
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"trim {video_path.name} to {n} frames raised: {exc!r}"
            )
            tmp_path.unlink(missing_ok=True)

    def _find_video_files(self, bag_path: Path) -> Dict[str, Path]:
        """Find MP4 video files in the rosbag directory.

        Supports MP4 converter output (cam_*.mp4 in root dir)
        and legacy format (videos/ subdirectory).
        """
        video_files = {}

        search_paths = [bag_path, bag_path / "videos"]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for mp4_file in sorted(search_path.glob("*.mp4")):
                # ``<cam>_synced.mp4`` is a derivative produced by
                # ``_sync_videos_to_grid`` on the previous conversion run;
                # skip it during initial discovery so we always start from
                # the recorder's raw MP4.
                if mp4_file.stem.endswith("_synced"):
                    continue
                camera_name = self._get_camera_name_for_video(mp4_file.stem)
                if camera_name not in video_files:
                    video_files[camera_name] = mp4_file

        if video_files:
            self._log_info(f"Found video files: {list(video_files.keys())}")

        return video_files

    def _get_camera_name_for_video(self, filename: str) -> str:
        """Get camera name from video filename.

        MP4 converter outputs files like 'cam_head_left.mp4',
        so the stem is already the camera name.
        """
        name = filename.replace("_compressed", "")

        # Direct match: MP4 converter uses cam_name as filename
        if self._camera_mapping:
            # Check if filename matches any known camera name
            for topic, camera_name in self._camera_mapping.items():
                if name == camera_name:
                    return camera_name
                # Legacy: sanitized topic match
                sanitized_topic = topic.replace("/", "_").lstrip("_")
                if sanitized_topic in name or name in sanitized_topic:
                    return camera_name

        # Filename is already the camera name (e.g., cam_head_left)
        if name.startswith("cam_"):
            return name

        return name

    def _get_video_dimensions(self, video_path: Path) -> Tuple[int, int]:
        """Get video height and width using OpenCV."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if width > 0 and height > 0:
                    return height, width
        except Exception as e:
            self._log_warning(f"Failed to get video dimensions: {e}")
        return 480, 640

    def _get_video_frame_count(self, video_path: Path) -> Optional[int]:
        """Get the number of frames in a video file using OpenCV."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count > 0:
                    return frame_count
        except Exception as e:
            self._log_warning(f"Failed to get video frame count: {e}")
        return None

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Probe a video file via ffprobe and return the LeRobot v2.1 ``info`` block.

        LeRobot v2.1 features expect every ``observation.images.*`` to carry
        an ``info`` dict with codec / pix_fmt / fps / dimensions / channels /
        is_depth_map / has_audio. Falls back to the converter's known
        encode params (h264 / yuv420p / fps from config) if ffprobe is
        unavailable or fails.
        """
        import subprocess

        height, width = self._get_video_dimensions(video_path)
        info = {
            'video.fps': float(self.config.fps),
            'video.height': int(height),
            'video.width': int(width),
            'video.channels': 3,
            'video.codec': 'h264',
            'video.pix_fmt': 'yuv420p',
            'video.is_depth_map': False,
            'has_audio': False,
        }
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_streams', '-print_format', 'json',
                    str(video_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return info
            probe = json.loads(result.stdout or '{}')
            streams = probe.get('streams') or []
            video_stream = next(
                (s for s in streams if s.get('codec_type') == 'video'), None,
            )
            if video_stream:
                if video_stream.get('codec_name'):
                    info['video.codec'] = video_stream['codec_name']
                if video_stream.get('pix_fmt'):
                    info['video.pix_fmt'] = video_stream['pix_fmt']
                # avg_frame_rate is "num/den"
                fr = video_stream.get('avg_frame_rate') or video_stream.get('r_frame_rate')
                if isinstance(fr, str) and '/' in fr:
                    num, _, den = fr.partition('/')
                    try:
                        d = float(den)
                        if d > 0:
                            info['video.fps'] = float(num) / d
                    except ValueError:
                        pass
            info['has_audio'] = any(
                s.get('codec_type') == 'audio' for s in streams
            )
        except (FileNotFoundError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            self._log_warning(f'ffprobe failed for {video_path}: {e}')
        return info

    def _build_features(self, episodes_data: List[EpisodeData]):
        """Build feature definitions from episode data."""
        # Get dimensions from first episode
        first_ep = episodes_data[0]

        state_dim = (
            len(first_ep.observation_state[0]) if first_ep.observation_state else 0
        )
        action_dim = len(first_ep.action[0]) if first_ep.action else 0

        # Default features (required by LeRobot)
        self._features = {
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        }

        # State / action joint names. Prefer per-episode names accumulated
        # by _merge_state_messages / _merge_action_messages, then fall back
        # to robot_config's joint_order grouped by side (follower_* for
        # state, leader_* for action) — that survives the
        # ProcessPoolExecutor parsing path, where worker children
        # populate the per-side _*_joint_names attributes but never
        # propagate them to the main process.
        #
        # Symmetry fallback: when joint_order only carries leader_*
        # (predecessor schema), the follower-prefix lookup returns []
        # and state would land on placeholder ``joint_N`` names. Reuse
        # the leader-prefix list for state if its dimension matches —
        # state and action describe the same joint set in our schema,
        # just observed vs commanded, so the names are identical.
        # Last resort: generic "joint_N" so the feature still has
        # names of the right length.
        state_names_from_config = self._joint_names_from_config('follower_')
        action_names_from_config = self._joint_names_from_config('leader_')

        if state_dim > 0:
            self._features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": (
                    self._state_joint_names
                    or (state_names_from_config if len(state_names_from_config) == state_dim else None)
                    or (action_names_from_config if len(action_names_from_config) == state_dim else None)
                    or [f"joint_{i}" for i in range(state_dim)]
                ),
            }

        if action_dim > 0:
            self._features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": (
                    self._action_joint_names
                    or (action_names_from_config if len(action_names_from_config) == action_dim else None)
                    or [f"joint_{i}" for i in range(action_dim)]
                ),
            }

        # Add video features. LeRobot v2.1 spec: shape is CHW (3, H, W),
        # names track shape order, and an ``info`` block carries codec /
        # pix_fmt / fps / dimensions / has_audio for downstream loaders.
        for ep in episodes_data:
            for camera_name, video_path in ep.video_files.items():
                feature_key = f"observation.images.{camera_name}"
                if feature_key not in self._features:
                    info = self._get_video_info(video_path)
                    self._features[feature_key] = {
                        "dtype": "video",
                        "shape": (3, int(info['video.height']), int(info['video.width'])),
                        "names": ["channels", "height", "width"],
                        "info": info,
                    }

    def _collect_tasks(self, episodes_data: List[EpisodeData]):
        """Populate ``self._tasks`` / ``self._task_to_index`` in episode-first-appearance order.

        Sorting alphabetically would make tasks.jsonl disagree with the
        order tasks show up in episodes.jsonl when the episode ordering
        itself isn't alphabetical (e.g. after a merge that placed source
        folders in a non-alphabetical order).
        """
        seen: set = set()
        for ep in episodes_data:
            for task in ep.tasks:
                if task in seen:
                    continue
                seen.add(task)
                idx = len(self._tasks)
                self._tasks[idx] = task
                self._task_to_index[task] = idx

    def _write_root_info_json(self) -> None:
        """Write the root-level info.json (conversion config snapshot).

        This sits at the dataset root (alongside README.md / data/ /
        meta/ / videos/) and records the choices the user made when
        running the conversion: which sources, which cameras / topics /
        joints were selected, what rotations / resize were applied,
        which episodes were skipped per source. It's a write-only audit
        artifact — downstream LeRobot loaders read meta/info.json, not
        this one — but external tools and the next reader benefit from
        knowing what knob was set when the dataset was produced.
        """
        output_dir = Path(self.config.output_dir)
        # task_name = output_dir.name with "_lerobot_v21" / "_v30" suffix removed.
        suffix = '_lerobot_v21'
        if not output_dir.name.endswith(suffix):
            suffix = '_lerobot_v30' if output_dir.name.endswith('_lerobot_v30') else ''
        task_name = (
            output_dir.name[: -len(suffix)] if suffix else output_dir.name
        )
        # Audit snapshot — fill empty selection fields from the discovered
        # robot_config defaults so the recorded config reflects what was
        # actually used in the conversion (not just what the caller
        # explicitly set).
        cameras = list(self.config.selected_cameras) or list(
            self._camera_mapping.values()
        )
        state_topics = (
            list(self.config.selected_state_topics) or list(self.config.state_topics)
        )
        action_topics = (
            list(self.config.selected_action_topics) or list(self.config.action_topics)
        )
        # selected_joints should reflect the joint names that ended up in
        # observation.state — i.e. just the state side (follower_*),
        # matching the reference layout. _joint_order is the flat
        # follower+leader concatenation (typically 2x the state dim).
        joints = list(self.config.selected_joints) or self._joint_names_from_config(
            'follower_'
        )
        # Camera rotations — include every known camera with an explicit
        # 0 for unrotated, mirroring the reference dataset's snapshot.
        rotations: Dict[str, int] = {cam: 0 for cam in cameras}
        for cam, deg in self.config.camera_rotations.items():
            rotations[cam] = int(deg)
        snapshot = {
            'source_rosbags': list(self.config.source_rosbags),
            'conversion_config': {
                'robot_type': self.config.robot_type,
                'fps': int(self.config.fps),
                'task_name': task_name,
                'selected_cameras': cameras,
                'camera_rotations': rotations,
                'image_resize': (
                    list(self.config.image_resize)
                    if self.config.image_resize else None
                ),
                'selected_joint_state_topics': state_topics,
                'selected_action_topics': action_topics,
                'selected_joints': joints,
            },
        }
        info_path = output_dir / 'info.json'
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            self._log_info(f'Wrote root info.json: {info_path}')
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f'Failed to write root info.json: {exc}')

    def _compute_episode_stats(
        self, episode: EpisodeData, global_start_index: int = 0,
    ) -> Dict[str, Dict]:
        """Compute statistics for an episode (LeRobot v2.1 format)."""
        stats = {}
        num_frames = episode.length

        if episode.observation_state:
            states = np.array(episode.observation_state)
            stats["observation.state"] = {
                "mean": np.mean(states, axis=0).tolist(),
                "std": np.maximum(np.std(states, axis=0), STATS_STD_FLOOR).tolist(),
                "min": np.min(states, axis=0).tolist(),
                "max": np.max(states, axis=0).tolist(),
                "count": [num_frames],
            }

        if episode.action:
            actions = np.array(episode.action)
            stats["action"] = {
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.maximum(np.std(actions, axis=0), STATS_STD_FLOOR).tolist(),
                "min": np.min(actions, axis=0).tolist(),
                "max": np.max(actions, axis=0).tolist(),
                "count": [num_frames],
            }

        for camera_name, video_path in episode.video_files.items():
            feature_key = f"observation.images.{camera_name}"
            video_stats = self._compute_video_stats(video_path, camera_name)
            if video_stats:
                stats[feature_key] = video_stats

        # Per-frame index / timestamp stats — LeRobot v2.1 reference
        # carries these alongside the data feature stats so downstream
        # tooling can reason about ranges without re-reading the parquet.
        if num_frames > 0:
            timestamps = np.array(episode.timestamps, dtype=np.float64) \
                if episode.timestamps else np.arange(num_frames, dtype=np.float64) / float(
                    self.config.fps if self.config.fps else 1
                )
            stats["timestamp"] = self._scalar_stats(timestamps, num_frames)

            frame_idx = np.arange(num_frames, dtype=np.int64)
            stats["frame_index"] = self._scalar_stats(frame_idx, num_frames)

            global_idx = np.arange(num_frames, dtype=np.int64) + int(global_start_index)
            stats["index"] = self._scalar_stats(global_idx, num_frames)

            ep_idx_arr = np.full(num_frames, episode.episode_index, dtype=np.int64)
            stats["episode_index"] = self._scalar_stats(ep_idx_arr, num_frames)

            task = episode.tasks[0] if episode.tasks else "default_task"
            ti = self._task_to_index.get(task, 0)
            ti_arr = np.full(num_frames, ti, dtype=np.int64)
            stats["task_index"] = self._scalar_stats(ti_arr, num_frames)

        return stats

    @staticmethod
    def _scalar_stats(arr: np.ndarray, num_frames: int) -> Dict[str, Any]:
        """min/max/mean/std/count for a 1-D scalar array, wrapped in lists."""
        return {
            "min": [arr.min().item()],
            "max": [arr.max().item()],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [num_frames],
        }

    def _load_precomputed_video_stats(
        self, video_path: Path, camera_name: str
    ) -> Optional[Dict]:
        """Try to load pre-computed video stats from video_stats.json (Stage 1)."""
        stats_path = video_path.parent / "video_stats.json"
        if not stats_path.exists():
            return None
        try:
            with open(stats_path, "r") as f:
                all_stats = json.load(f)
            if camera_name in all_stats:
                self._log_info(
                    f"Using pre-computed video stats for {camera_name}"
                )
                return all_stats[camera_name]
        except Exception as e:
            self._log_warning(
                f"Failed to load pre-computed stats from {stats_path}: {e}"
            )
        return None

    def _compute_video_stats(
        self, video_path: Path, camera_name: str = "", max_samples: int = 100
    ) -> Optional[Dict]:
        """Compute video statistics (per-channel RGB, normalized to [0,1]).

        First checks for pre-computed stats from Stage 1 (video_stats.json).
        Falls back to decoding MP4 if not available.
        """
        # Try pre-computed stats first
        precomputed = self._load_precomputed_video_stats(video_path, camera_name)
        if precomputed is not None:
            return precomputed

        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(
                0, total_frames - 1, min(max_samples, total_frames), dtype=int
            )

            samples = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    samples.append(frame_rgb)

            cap.release()

            if not samples:
                return None

            frames = np.array(samples, dtype=np.float32) / 255.0
            r_channel = frames[:, :, :, 0]
            g_channel = frames[:, :, :, 1]
            b_channel = frames[:, :, :, 2]

            def channel_stats(channel):
                return {
                    "mean": float(np.mean(channel)),
                    "std": float(np.std(channel)),
                    "min": float(np.min(channel)),
                    "max": float(np.max(channel)),
                }

            r_stats = channel_stats(r_channel)
            g_stats = channel_stats(g_channel)
            b_stats = channel_stats(b_channel)

            return {
                "min": [[[r_stats["min"]]], [[g_stats["min"]]], [[b_stats["min"]]]],
                "max": [[[r_stats["max"]]], [[g_stats["max"]]], [[b_stats["max"]]]],
                "mean": [[[r_stats["mean"]]], [[g_stats["mean"]]], [[b_stats["mean"]]]],
                "std": [[[r_stats["std"]]], [[g_stats["std"]]], [[b_stats["std"]]]],
                "count": [len(samples)],
            }
        except Exception as e:
            self._log_warning(f"Failed to compute video stats for {video_path}: {e}")
            return None

    def _serialize_stats(self, stats: Dict) -> Dict:
        """Serialize stats dictionary for JSON."""
        serialized = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_stats(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, int, float)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def _write_quality_reports(self, output_dir: Path) -> None:
        """Dump per-episode quality reports as JSON when ``enable_quality_report``.

        Currently nothing populates ``_quality_reports`` — kept as a
        forward-compatible stub so the v3.0 writer's
        ``if self.config.enable_quality_report and self._quality_reports:``
        branch doesn't AttributeError if a future producer fills the dict.
        """
        if not self._quality_reports:
            return
        report_path = Path(output_dir) / "meta" / "quality_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self._quality_reports, f, indent=2, ensure_ascii=False)
            self._log_info(f"Wrote quality report: {report_path}")
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"Failed to write quality report: {exc}")

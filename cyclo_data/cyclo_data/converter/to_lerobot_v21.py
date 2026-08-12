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
ROSbag + MP4 to LeRobot v2.1 Dataset Converter.

Converts recorded robot data (ROSbag with joint states + MP4 videos) to
LeRobot v2.1 dataset format. Shared rosbag extraction and stats logic
lives in ``base_converter.py``; this module owns the v2.1-format
writers (per-episode parquet + JSONL meta).

LeRobot v2.1 Dataset Structure:
    dataset_name/
    ├── data/
    │   └── chunk-{chunk:03d}/
    │       └── episode_{episode:06d}.parquet
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── episodes_stats.jsonl
    │   └── tasks.jsonl
    └── videos/
        └── chunk-{chunk:03d}/
            └── observation.images.rgb.{camera}/
                └── episode_{episode:06d}.mp4
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Re-export the shared dataclasses / worker so existing callers
# (pipeline_worker, scripts, tests) keep `from .to_lerobot_v21 import …`
# working unchanged.
from .base_converter import (  # noqa: F401
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FPS,
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
    StalenessMetrics,
    _active_conversion_workers,
    _clone_or_copy_file,
    _conversion_worker_init,
    _convert_rosbag_worker,
    _fast_absolute_path,
    _resolve_conversion_worker_count,
)
from .video_sync import (
    _StreamingRgbStats,
    _add_frame_yuv420p_for_stats,
    _contiguous_forward_run_length,
    _drain_exact,
    _ffmpeg,
    _ffmpeg_h264_decoder_args,
    _ffmpeg_pipe_size,
    _ffmpeg_threads_arg,
    _h264_encoder,
    _mp4_faststart_args,
    _quick_video_dimensions,
    _read_exact_into,
    _set_pipe_size,
    _splice_exact,
    _terminate_process,
    _validated_video_count,
    _video_stats_sample_positions,
    _write_repeated_frame_bytes,
    remux_selected_frames,
)


CODEBASE_VERSION = "v2.1"
V21_CHUNK_SIZE = 1000
V21_CHUNK_DIGITS = 3
V21_EPISODE_DIGITS = 6
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mp4"
)
V21_ANNOTATION_PATH = (
    "annotations/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.json"
)
_V21_DIRECT_VIDEO_DISABLE_ENV = "CYCLO_V21_DISABLE_DIRECT_VIDEO"
_V21_DIRECT_VIDEO_ENABLE_ENV = "CYCLO_V21_ENABLE_DIRECT_VIDEO"
_V21_DIRECT_VIDEO_WORKERS_ENV = "CYCLO_V21_DIRECT_VIDEO_WORKERS"
_V21_DIRECT_VIDEO_CACHE_WORKERS_ENV = "CYCLO_V21_DIRECT_VIDEO_CACHE_WORKERS"
_V21_DIRECT_VIDEO_BATCH_EPISODES_ENV = "CYCLO_V21_DIRECT_VIDEO_BATCH_EPISODES"
_V21_DIRECT_VIDEO_CACHE_DISABLE_ENV = "CYCLO_V21_DIRECT_VIDEO_CACHE_DISABLE"
_V21_DIRECT_VIDEO_VALIDATE_ENV = "CYCLO_V21_VALIDATE_DIRECT_VIDEO"
_V21_CONCAT_DECODER_DISABLE_ENV = "CYCLO_V21_CONCAT_DECODER_DISABLE"
_V21_EPISODE_PARQUET_CACHE_DISABLE_ENV = "CYCLO_V21_EPISODE_PARQUET_CACHE_DISABLE"
_V21_EPISODE_PARQUET_CACHE_VERSION = 2
_V21_SUBTASKS_PARQUET_CACHE_DISABLE_ENV = "CYCLO_V21_SUBTASKS_PARQUET_CACHE_DISABLE"
_V21_SUBTASKS_PARQUET_CACHE_VERSION = 1
# Keep v2.1 segment batches large enough to amortize ffmpeg startup while
# still leaving multiple camera jobs for portable workstation/Jetson overlap.
_DEFAULT_V21_DIRECT_VIDEO_BATCH_EPISODES = 32
_X264_GOP_ENV = "CYCLO_X264_GOP"


class _DirectConcatDecoderError(RuntimeError):
    """Raised when the v2.1 batch concat decoder path cannot feed frames."""


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy bytes through a temporary sibling and publish atomically."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _v21_chunk_dir_name(chunk_idx: int) -> str:
    return f"chunk-{chunk_idx:0{V21_CHUNK_DIGITS}d}"


def _v21_episode_stem(episode_idx: int) -> str:
    return f"episode_{episode_idx:0{V21_EPISODE_DIGITS}d}"


def _concat_file_line(video_path: Path) -> str:
    path = _fast_absolute_path(Path(video_path)).replace("'", "'\\''")
    return f"file '{path}'\n"


class RosbagToLerobotConverter(RosbagToLerobotConverterBase):
    """LeRobot v2.1 converter: per-episode parquet + JSONL meta."""

    def __init__(self, config: ConversionConfig, logger=None):
        super().__init__(config, logger=logger)
        self._quick_video_dimensions_cache: Dict[Path, Tuple[int, int]] = {}
        self._quick_video_dimensions_lock = threading.Lock()
        self._direct_v21_video_info_cache: Dict[Path, Dict[str, Any]] = {}
        self._direct_v21_video_info_lock = threading.Lock()

    def _video_dimensions_cache_key(self, video_path: Path) -> Path:
        return Path(_fast_absolute_path(Path(video_path)))

    def _remember_quick_video_dimensions(
        self,
        video_path: Path,
        width: int,
        height: int,
    ) -> None:
        width_i = int(width)
        height_i = int(height)
        if width_i <= 0 or height_i <= 0:
            return
        key = self._video_dimensions_cache_key(video_path)
        with self._quick_video_dimensions_lock:
            self._quick_video_dimensions_cache[key] = (width_i, height_i)

    def _quick_video_dimensions_cached(self, video_path: Path) -> Tuple[int, int]:
        path = self._video_dimensions_cache_key(video_path)
        with self._quick_video_dimensions_lock:
            cached = self._quick_video_dimensions_cache.get(path)
            if cached is not None:
                return cached
        dims = _quick_video_dimensions(path)
        with self._quick_video_dimensions_lock:
            self._quick_video_dimensions_cache[path] = dims
        return dims

    def _remember_direct_v21_video_info(
        self,
        video_path: Path,
        *,
        width: int,
        height: int,
        cache_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        width_i = int(width)
        height_i = int(height)
        if width_i <= 0 or height_i <= 0:
            return
        info = {
            "video.fps": float(self.config.fps),
            "video.height": height_i,
            "video.width": width_i,
            "video.channels": 3,
            "video.codec": str((cache_payload or {}).get("output_codec") or "h264"),
            "video.pix_fmt": str(
                (cache_payload or {}).get("output_pix_fmt") or "yuv420p"
            ),
            "video.is_depth_map": False,
            "has_audio": bool((cache_payload or {}).get("has_audio", False)),
        }
        key = self._video_dimensions_cache_key(video_path)
        with self._direct_v21_video_info_lock:
            self._direct_v21_video_info_cache[key] = info

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        key = self._video_dimensions_cache_key(video_path)
        with self._direct_v21_video_info_lock:
            cached = self._direct_v21_video_info_cache.get(key)
            if cached is not None:
                return dict(cached)
        return super()._get_video_info(video_path)

    def _direct_video_sources_for_bag(
        self,
        bag_path: Path,
    ) -> Optional[Dict[str, Path]]:
        """Return raw per-camera MP4s eligible for direct v2.1 video output."""
        bag_path = Path(bag_path)
        episode_info = self._metadata_manager.load_episode_info(bag_path)
        if str(episode_info.get("recording_mode", "") or "") == "subtask":
            return None
        if self._is_archived_segment_episode(bag_path, episode_info):
            segments = episode_info.get("segments") or []
            if len(segments) != 1:
                return None
            mcap_paths = self._segment_mcap_paths(bag_path, len(segments))
            if len(mcap_paths) != 1:
                return None
            videos = self._find_segment_video_files(bag_path, mcap_paths[0].stem)
        else:
            videos = self._find_video_files(bag_path)

        if not videos:
            return None
        for camera_name, video_path in videos.items():
            sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
            if not sidecar.exists():
                return None
        return {name: Path(path) for name, path in videos.items()}

    def _can_use_direct_video_output(self, bag_paths: List[Path]) -> bool:
        """True when v2.1 can skip per-episode synced MP4 intermediates."""
        self._direct_video_sources_by_episode: Dict[int, Dict[str, Path]] = {}
        self._direct_video_bag_paths_by_episode: Dict[int, Path] = {}
        if os.environ.get(_V21_DIRECT_VIDEO_DISABLE_ENV):
            return False
        profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
        direct_requested = (
            os.environ.get(_V21_DIRECT_VIDEO_ENABLE_ENV, "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if (
            not direct_requested
            and profile not in {"max", "maximum", "max_speed", "fastest"}
        ):
            return False
        if not self.config.use_videos:
            return False
        if len(bag_paths) <= 1:
            return False
        if self.config.image_resize is not None:
            return False
        if any(int(value or 0) for value in self.config.camera_rotations.values()):
            return False

        for idx, bag_path in enumerate(bag_paths):
            sources = self._direct_video_sources_for_bag(Path(bag_path))
            if not sources:
                self._direct_video_sources_by_episode = {}
                self._direct_video_bag_paths_by_episode = {}
                return False
            self._direct_video_sources_by_episode[int(idx)] = sources
            self._direct_video_bag_paths_by_episode[int(idx)] = Path(bag_path)
        return True

    def _attach_direct_video_sources(self, episodes_data: List[EpisodeData]) -> None:
        """Attach raw source videos after no-video episode extraction."""
        for episode in episodes_data:
            sources = self._direct_video_sources_by_episode.get(
                int(episode.episode_index)
            )
            if not sources:
                raise RuntimeError(
                    "direct v2.1 video source missing for "
                    f"episode {episode.episode_index}"
                )
            episode.video_files = dict(sources)

    def _v21_video_output_path(
        self,
        output_dir: Path,
        episode: EpisodeData,
        camera_name: str,
    ) -> Path:
        ep_idx = int(episode.episode_index)
        chunk_idx = ep_idx // V21_CHUNK_SIZE
        return (
            Path(output_dir)
            / "videos"
            / _v21_chunk_dir_name(chunk_idx)
            / self._video_feature_key(camera_name)
            / f"{_v21_episode_stem(ep_idx)}.mp4"
        )

    @staticmethod
    def _direct_v21_video_cache_disabled() -> bool:
        return os.environ.get(_V21_DIRECT_VIDEO_CACHE_DISABLE_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _validate_direct_v21_video() -> bool:
        raw = os.environ.get(_V21_DIRECT_VIDEO_VALIDATE_ENV)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        profile = os.environ.get("CYCLO_X264_SPEED_PROFILE", "").strip().lower()
        return profile not in {"max", "maximum", "max_speed", "fastest"}

    def _direct_v21_synced_cache_paths(
        self,
        video_path: Path,
        camera_name: str,
    ) -> Optional[Tuple[Path, Path]]:
        """Return the source-side synced-video cache paths for direct v2.1."""
        if self._direct_v21_video_cache_disabled():
            return None
        video_path = Path(video_path)
        sidecar = video_path.parent / f"{camera_name}_timestamps.parquet"
        if not sidecar.exists():
            return None
        cache_dir = self._video_sync_output_dir(video_path.parent)
        return (
            cache_dir / f"{camera_name}_synced.mp4",
            cache_dir / f"{camera_name}_synced.cache.json",
        )

    def _direct_v21_synced_cache_key(
        self,
        video_path: Path,
        indices: Sequence[int],
    ) -> Dict[str, object]:
        video_path = Path(video_path)
        src_stat = video_path.stat()
        indices_i64 = np.asarray(indices, dtype=np.int64)
        return {
            "target_fps": int(self.config.fps),
            "rotation_deg": 0,
            "image_resize": None,
            "frame_count": int(indices_i64.size),
            "frame_indices_sha256": hashlib.sha256(
                indices_i64.tobytes()
            ).hexdigest(),
            "source_size": int(src_stat.st_size),
            "source_mtime_ns": int(src_stat.st_mtime_ns),
        }

    def _direct_v21_synced_metadata_cache_key(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> Optional[Dict[str, object]]:
        grid = np.asarray(episode.grid_log_times_sec, dtype=np.float64)
        if grid.size <= 0 or grid.size != int(episode.length):
            return None
        sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
        if not sidecar.exists():
            return None
        return {
            "cache_key_version": 2,
            "target_fps": int(self.config.fps),
            "rotation_deg": 0,
            "image_resize": None,
            "frame_count": int(grid.size),
            "source": self._file_signature(Path(video_path)),
            "sidecar": self._file_signature(sidecar),
            "grid_sha256": hashlib.sha256(
                np.ascontiguousarray(grid).tobytes()
            ).hexdigest(),
        }

    def _remember_direct_v21_video_stats(
        self,
        episode: EpisodeData,
        camera_name: str,
        stats: Optional[Dict],
    ) -> None:
        if not stats:
            return
        stats_lock = getattr(self, "_direct_v21_stats_lock", None)
        cache_key = (int(episode.episode_index), camera_name)
        if stats_lock is not None:
            with stats_lock:
                self._direct_v21_video_stats_cache[cache_key] = stats
        else:
            self._direct_v21_video_stats_cache[cache_key] = stats

    def _try_reuse_direct_v21_synced_cache(
        self,
        output_dir: Path,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> Optional[Path]:
        cache_paths = self._direct_v21_synced_cache_paths(video_path, camera_name)
        if cache_paths is None:
            return None
        cache_video, cache_sidecar = cache_paths
        try:
            if (
                not cache_video.exists()
                or cache_video.stat().st_size <= 0
                or not cache_sidecar.exists()
            ):
                return None
            cached_key = json.loads(cache_sidecar.read_text(encoding="utf-8"))
            expected_frames: Optional[int] = None
            metadata_key = self._direct_v21_synced_metadata_cache_key(
                episode,
                camera_name,
                video_path,
            )
            if metadata_key is not None and self._synced_cache_identity_matches(
                cached_key,
                metadata_key,
            ):
                expected_frames = int(metadata_key["frame_count"])
            else:
                indices = self._grid_indices_for_raw_video(
                    episode,
                    camera_name,
                    video_path,
                )
                desired_key = self._direct_v21_synced_cache_key(video_path, indices)
                if not self._synced_cache_identity_matches(cached_key, desired_key):
                    return None
                expected_frames = int(indices.size)
            if self._validate_direct_v21_video():
                _validated_video_count(
                    output_mp4=cache_video,
                    expected_frames=int(expected_frames),
                    target_fps=int(self.config.fps),
                    ffmpeg=_ffmpeg(),
                    label="direct-v21-cache",
                )
            dst = self._v21_video_output_path(output_dir, episode, camera_name)
            copy_mode = _clone_or_copy_file(cache_video, dst)
            try:
                self._remember_direct_v21_video_info(
                    dst,
                    width=int(cached_key["output_width"]),
                    height=int(cached_key["output_height"]),
                    cache_payload=cached_key,
                )
            except (KeyError, TypeError, ValueError):
                pass
            stats = self._load_precomputed_video_stats(cache_video, camera_name)
            self._remember_direct_v21_video_stats(episode, camera_name, stats)
            self._record_frame_reuse_for_video(episode, camera_name, video_path)
            self._log_info(
                f"{camera_name}: reused source synced cache for episode "
                f"{int(episode.episode_index)} ({copy_mode})"
            )
            return dst
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{camera_name}: source synced cache reuse failed for episode "
                f"{int(episode.episode_index)} ({exc!r}); regenerating"
            )
            return None

    def _direct_v21_synced_cache_candidate_exists(
        self,
        video_path: Path,
        camera_name: str,
    ) -> bool:
        cache_paths = self._direct_v21_synced_cache_paths(video_path, camera_name)
        if cache_paths is None:
            return False
        cache_video, cache_sidecar = cache_paths
        try:
            return (
                cache_video.exists()
                and cache_video.stat().st_size > 0
                and cache_sidecar.exists()
            )
        except OSError:
            return False

    def _store_direct_v21_synced_cache(
        self,
        *,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
        indices: Sequence[int],
        output_path: Path,
        stats: Optional[Dict],
        width: int,
        height: int,
    ) -> None:
        cache_paths = self._direct_v21_synced_cache_paths(video_path, camera_name)
        if cache_paths is None:
            return
        cache_video, cache_sidecar = cache_paths
        tmp_sidecar: Optional[Path] = None
        try:
            cache_payload = self._direct_v21_synced_cache_key(video_path, indices)
            metadata_key = self._direct_v21_synced_metadata_cache_key(
                episode,
                camera_name,
                video_path,
            )
            if metadata_key is not None:
                cache_payload.update(metadata_key)
            cache_payload["output_height"] = int(height)
            cache_payload["output_width"] = int(width)
            cache_payload["output_codec"] = "h264"
            cache_payload["output_pix_fmt"] = "yuv420p"
            cache_payload["has_audio"] = False
            copy_mode = _clone_or_copy_file(output_path, cache_video)
            cache_sidecar.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=cache_sidecar.stem + ".",
                suffix=".tmp",
                dir=str(cache_sidecar.parent),
                delete=False,
            ) as fh:
                tmp_sidecar = Path(fh.name)
                json.dump(cache_payload, fh)
            os.replace(tmp_sidecar, cache_sidecar)
            self._store_video_stats_cached(cache_video, camera_name, stats)
            self._log_info(
                f"{camera_name}: stored v2.1 source synced cache for episode "
                f"{int(episode.episode_index)} ({copy_mode})"
            )
        except Exception as exc:  # noqa: BLE001
            if tmp_sidecar is not None:
                tmp_sidecar.unlink(missing_ok=True)
            self._log_warning(
                f"{camera_name}: failed to store v2.1 source synced cache for "
                f"episode {int(episode.episode_index)} ({exc!r})"
            )

    def _grid_indices_for_raw_video(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> np.ndarray:
        indices, _ = self._grid_indices_and_source_count_for_raw_video(
            episode,
            camera_name,
            video_path,
        )
        return indices

    def _grid_indices_and_source_count_for_raw_video(
        self,
        episode: EpisodeData,
        camera_name: str,
        video_path: Path,
    ) -> Tuple[np.ndarray, int]:
        sidecar = Path(video_path).parent / f"{camera_name}_timestamps.parquet"
        from cyclo_data.reader.frame_timestamps import load_frame_timestamps

        ft = load_frame_timestamps(sidecar, camera_name)
        grid_ns = (
            np.asarray(episode.grid_log_times_sec, dtype=np.float64)
            * 1_000_000_000
        ).astype(np.int64)
        indices = ft.map_to_grid(grid_ns, time_source="header")
        self._record_frame_reuse_report(
            episode=episode,
            camera_name=camera_name,
            indices=indices,
            grid_ns=grid_ns,
            frame_timestamps=ft,
        )
        return indices, int(ft.num_frames)

    def _pipe_selected_yuv420_frames(
        self,
        ffmpeg: str,
        video_path: Path,
        indices: Sequence[int],
        frame_size: int,
        output,
        *,
        width: int,
        height: int,
        stats: Optional[_StreamingRgbStats] = None,
    ) -> int:
        decoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            *_ffmpeg_h264_decoder_args(ffmpeg),
            "-i", str(video_path),
            "-map", "0:v:0",
            "-an",
            "-fps_mode", "passthrough",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "pipe:1",
        ]
        decoder: Optional[subprocess.Popen] = None
        discard_fd: Optional[int] = None
        try:
            decoder = subprocess.Popen(
                decoder_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert decoder.stdout is not None
            current_idx = -1
            _set_pipe_size(decoder.stdout, _ffmpeg_pipe_size(frame_size))
            last_frame: Optional[bytearray] = None
            read_buffer = bytearray(frame_size)
            drain_buffer = bytearray(min(frame_size, 1024 * 1024))
            written = 0
            sample_positions = (
                _video_stats_sample_positions(len(indices))
                if stats is not None else set()
            )
            if hasattr(os, "splice"):
                try:
                    discard_fd = os.open(os.devnull, os.O_WRONLY)
                except OSError:
                    discard_fd = None
            out_idx = 0
            indices_len = len(indices)
            while out_idx < indices_len:
                requested_idx = int(indices[out_idx])
                run_end = out_idx + 1
                while (
                    run_end < indices_len
                    and int(indices[run_end]) == requested_idx
                ):
                    run_end += 1
                run_length = run_end - out_idx
                needs_stats_sample = (
                    stats is not None
                    and any(
                        out_idx <= sample_idx < run_end
                        for sample_idx in sample_positions
                    )
                )
                while current_idx < requested_idx:
                    skipped_frames = requested_idx - current_idx - 1
                    if skipped_frames > 0:
                        skipped_bytes = skipped_frames * frame_size
                        drained = _drain_exact(
                            decoder.stdout,
                            skipped_bytes,
                            discard_fd=discard_fd,
                            buffer=drain_buffer,
                        )
                        if drained != skipped_bytes:
                            raise RuntimeError(
                                f"short drain from {Path(video_path).name}: "
                                f"{drained}/{skipped_bytes}"
                            )
                        current_idx += skipped_frames
                        last_frame = None
                        continue
                    if (
                        not needs_stats_sample
                        and run_length == 1
                        and current_idx + 1 == requested_idx
                    ):
                        forward_run = (
                            _contiguous_forward_run_length(indices, out_idx)
                            if stats is None else 1
                        )
                        splice_end = out_idx + forward_run
                        splice_bytes = frame_size * forward_run
                        spliced = _splice_exact(decoder.stdout, output, splice_bytes)
                        if spliced == splice_bytes:
                            current_idx += forward_run
                            last_frame = None
                            written += forward_run
                            out_idx = splice_end
                            run_end = splice_end
                            break
                        if spliced:
                            raise RuntimeError(
                                f"short splice from {Path(video_path).name}: "
                                f"{spliced}/{splice_bytes}"
                            )
                    n = _read_exact_into(decoder.stdout, read_buffer, frame_size)
                    if n != frame_size:
                        if last_frame is None:
                            raise RuntimeError(
                                f"no frames decoded from {Path(video_path).name}"
                            )
                        break
                    current_idx += 1
                    if last_frame is None:
                        last_frame = read_buffer
                        read_buffer = bytearray(frame_size)
                    else:
                        last_frame, read_buffer = read_buffer, last_frame
                if out_idx == run_end:
                    continue
                if last_frame is None:
                    raise RuntimeError(
                        f"no frame available for {Path(video_path).name}"
                    )
                _write_repeated_frame_bytes(output, last_frame, run_length)
                if stats is not None:
                    for sample_idx in sample_positions:
                        if out_idx <= sample_idx < run_end:
                            _add_frame_yuv420p_for_stats(
                                stats,
                                last_frame,
                                width=width,
                                height=height,
                            )
                written += run_length
                out_idx = run_end
            return written
        finally:
            try:
                if discard_fd is not None:
                    os.close(discard_fd)
            except OSError:
                pass
            _terminate_process(decoder)

    def _pipe_selected_yuv420_frames_concat_decoder(
        self,
        ffmpeg: str,
        frame_requests: Sequence[
            Tuple[EpisodeData, Path, np.ndarray, int, Optional[_StreamingRgbStats]]
        ],
        frame_size: int,
        output,
        *,
        width: int,
        height: int,
    ) -> int:
        """Decode a v2.1 camera batch through one ffmpeg concat pipe."""
        if not frame_requests:
            return 0
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as concat_file:
            for _, video_path, _, _, _ in frame_requests:
                concat_file.write(_concat_file_line(Path(video_path)))
            concat_list_path = concat_file.name

        decoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "fatal",
            *_ffmpeg_h264_decoder_args(ffmpeg),
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-map", "0:v:0",
            "-an",
            "-fps_mode", "passthrough",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "pipe:1",
        ]
        decoder: Optional[subprocess.Popen] = None
        discard_fd: Optional[int] = None
        try:
            decoder = subprocess.Popen(
                decoder_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert decoder.stdout is not None
            _set_pipe_size(decoder.stdout, _ffmpeg_pipe_size(frame_size))
            current_idx = -1
            source_offset = 0
            last_frame: Optional[bytearray] = None
            read_buffer = bytearray(frame_size)
            drain_buffer = bytearray(min(frame_size, 1024 * 1024))
            if hasattr(os, "splice"):
                try:
                    discard_fd = os.open(os.devnull, os.O_WRONLY)
                except OSError:
                    discard_fd = None
            written = 0
            for _, video_path, indices, source_count, stats in frame_requests:
                sample_positions = (
                    _video_stats_sample_positions(len(indices))
                    if stats is not None else set()
                )
                out_idx = 0
                indices_len = len(indices)
                while out_idx < indices_len:
                    requested_idx_raw = int(indices[out_idx])
                    run_end = out_idx + 1
                    while (
                        run_end < indices_len
                        and int(indices[run_end]) == requested_idx_raw
                    ):
                        run_end += 1
                    run_length = run_end - out_idx
                    requested_idx = source_offset + requested_idx_raw
                    needs_stats_sample = (
                        stats is not None
                        and any(
                            out_idx <= sample_idx < run_end
                            for sample_idx in sample_positions
                        )
                    )
                    while current_idx < requested_idx:
                        skipped_frames = requested_idx - current_idx - 1
                        if skipped_frames > 0:
                            skipped_bytes = skipped_frames * frame_size
                            drained = _drain_exact(
                                decoder.stdout,
                                skipped_bytes,
                                discard_fd=discard_fd,
                                buffer=drain_buffer,
                            )
                            if drained != skipped_bytes:
                                raise RuntimeError(
                                    "short drain from concat video batch: "
                                    f"{drained}/{skipped_bytes}"
                                )
                            current_idx += skipped_frames
                            last_frame = None
                            continue
                        if (
                            not needs_stats_sample
                            and run_length == 1
                            and current_idx + 1 == requested_idx
                        ):
                            forward_run = (
                                _contiguous_forward_run_length(indices, out_idx)
                                if stats is None else 1
                            )
                            splice_end = out_idx + forward_run
                            splice_bytes = frame_size * forward_run
                            spliced = _splice_exact(
                                decoder.stdout,
                                output,
                                splice_bytes,
                            )
                            if spliced == splice_bytes:
                                current_idx += forward_run
                                last_frame = None
                                written += forward_run
                                out_idx = splice_end
                                run_end = splice_end
                                break
                            if spliced:
                                raise RuntimeError(
                                    "short splice from concat video batch: "
                                    f"{spliced}/{splice_bytes}"
                                )
                        n = _read_exact_into(decoder.stdout, read_buffer, frame_size)
                        if n != frame_size:
                            if last_frame is None:
                                raise RuntimeError(
                                    "no frames decoded from concat video batch"
                                )
                            break
                        current_idx += 1
                        if last_frame is None:
                            last_frame = read_buffer
                            read_buffer = bytearray(frame_size)
                        else:
                            last_frame, read_buffer = read_buffer, last_frame
                    if out_idx == run_end:
                        continue
                    if last_frame is None:
                        raise RuntimeError(
                            f"no frame available for {Path(video_path).name}"
                        )
                    _write_repeated_frame_bytes(output, last_frame, run_length)
                    if stats is not None:
                        for sample_idx in sample_positions:
                            if out_idx <= sample_idx < run_end:
                                _add_frame_yuv420p_for_stats(
                                    stats,
                                    last_frame,
                                    width=width,
                                    height=height,
                                )
                    written += run_length
                    out_idx = run_end
                source_offset += int(source_count)
            return written
        finally:
            try:
                if discard_fd is not None:
                    os.close(discard_fd)
            except OSError:
                pass
            _terminate_process(decoder)
            Path(concat_list_path).unlink(missing_ok=True)

    @staticmethod
    def _segment_format_options_args() -> List[str]:
        if not _mp4_faststart_args():
            return []
        return ["-segment_format_options", "movflags=+faststart"]

    @staticmethod
    def _segment_encoder_opts(encoder: str, encoder_opts: List[str]) -> List[str]:
        # The generic max-speed libx264 fallback defaults to all-intra (`-g 1`).
        # That helps standalone clips on some hosts, but segmented output only
        # needs keyframes at episode boundaries. Keep explicit user GOP choices.
        if (
            str(encoder).strip().lower() != "libx264"
            or _X264_GOP_ENV in os.environ
        ):
            return list(encoder_opts)
        opts = list(encoder_opts)
        cleaned: List[str] = []
        idx = 0
        while idx < len(opts):
            if opts[idx] == "-g" and idx + 1 < len(opts) and opts[idx + 1] == "1":
                idx += 2
                continue
            cleaned.append(opts[idx])
            idx += 1
        return cleaned

    @staticmethod
    def _direct_video_batch_episodes() -> int:
        raw = os.environ.get(_V21_DIRECT_VIDEO_BATCH_EPISODES_ENV, "")
        if raw:
            try:
                return max(2, int(raw))
            except ValueError:
                pass
        return _DEFAULT_V21_DIRECT_VIDEO_BATCH_EPISODES

    @staticmethod
    def _resolve_direct_video_workers(job_count: int, camera_count: int) -> int:
        if job_count <= 1:
            return 1
        raw = os.environ.get(_V21_DIRECT_VIDEO_WORKERS_ENV, "")
        if raw:
            try:
                return max(1, min(int(raw), job_count))
            except ValueError:
                pass
        return max(1, min(job_count, max(1, camera_count + 1)))

    @staticmethod
    def _resolve_direct_video_cache_workers(job_count: int) -> int:
        if job_count <= 1:
            return 1
        raw = os.environ.get(_V21_DIRECT_VIDEO_CACHE_WORKERS_ENV, "")
        if not raw:
            raw = os.environ.get(_V21_DIRECT_VIDEO_WORKERS_ENV, "")
        if raw:
            try:
                return max(1, min(int(raw), job_count))
            except ValueError:
                pass
        return 1

    def _write_direct_segmented_videos(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
    ) -> None:
        camera_names = sorted({
            camera_name
            for episode in episodes_data
            for camera_name in episode.video_files
        })
        if not camera_names:
            return

        episode_by_index = {int(ep.episode_index): ep for ep in episodes_data}
        updates: Dict[Tuple[int, str], Path] = {}
        self._direct_v21_stats_lock = threading.Lock()
        self._direct_v21_video_stats_cache: Dict[
            Tuple[int, str], Dict
        ] = {}
        batch_size = self._direct_video_batch_episodes()
        misses_by_camera: Dict[str, List[Tuple[EpisodeData, Path]]] = {
            camera_name: [] for camera_name in camera_names
        }
        reuse_candidates: List[Tuple[str, EpisodeData, Path]] = []
        for camera_name in camera_names:
            for episode in episodes_data:
                if camera_name not in episode.video_files:
                    continue
                video_path = Path(episode.video_files[camera_name])
                if self._direct_v21_synced_cache_candidate_exists(
                    video_path,
                    camera_name,
                ):
                    reuse_candidates.append((camera_name, episode, video_path))
                else:
                    misses_by_camera[camera_name].append((episode, video_path))

        if reuse_candidates:
            cache_workers = self._resolve_direct_video_cache_workers(
                len(reuse_candidates)
            )

            def reuse_cache(
                camera_name: str,
                episode: EpisodeData,
                video_path: Path,
            ) -> Tuple[str, EpisodeData, Path, Optional[Path]]:
                reused_path = self._try_reuse_direct_v21_synced_cache(
                    output_dir,
                    episode,
                    camera_name,
                    video_path,
                )
                return camera_name, episode, video_path, reused_path

            if cache_workers > 1:
                with ThreadPoolExecutor(max_workers=cache_workers) as executor:
                    futures = [
                        executor.submit(
                            reuse_cache,
                            camera_name,
                            episode,
                            video_path,
                        )
                        for camera_name, episode, video_path in reuse_candidates
                    ]
                    for future in as_completed(futures):
                        camera_name, episode, video_path, reused_path = future.result()
                        if reused_path is None:
                            misses_by_camera[camera_name].append((episode, video_path))
                        else:
                            updates[
                                (int(episode.episode_index), camera_name)
                            ] = reused_path
            else:
                for camera_name, episode, video_path in reuse_candidates:
                    _, _, _, reused_path = reuse_cache(
                        camera_name,
                        episode,
                        video_path,
                    )
                    if reused_path is None:
                        misses_by_camera[camera_name].append((episode, video_path))
                    else:
                        updates[(int(episode.episode_index), camera_name)] = reused_path

        jobs: List[Tuple[str, List[Tuple[EpisodeData, Path]]]] = []
        for camera_name in camera_names:
            pairs = misses_by_camera[camera_name]
            for start in range(0, len(pairs), batch_size):
                jobs.append((camera_name, pairs[start:start + batch_size]))

        workers = self._resolve_direct_video_workers(len(jobs), len(camera_names))

        def run_job(
            camera_name: str,
            pairs: List[Tuple[EpisodeData, Path]],
        ) -> Tuple[str, Dict[int, Path]]:
            try:
                return camera_name, self._write_direct_camera_segments_with_retry(
                    output_dir,
                    camera_name,
                    pairs,
                )
            except Exception as exc:  # noqa: BLE001
                self._log_warning(
                    f"{camera_name}: direct segmented video failed "
                    f"({exc!r}); falling back to per-episode sync"
                )
                return camera_name, self._write_direct_camera_segments_fallback(
                    output_dir,
                    camera_name,
                    pairs,
                )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_camera = {
                    executor.submit(run_job, camera_name, pairs): camera_name
                    for camera_name, pairs in jobs
                }
                for future in as_completed(future_to_camera):
                    camera_name, camera_updates = future.result()
                    for episode_idx, path in camera_updates.items():
                        updates[(int(episode_idx), camera_name)] = path
        else:
            for camera_name, pairs in jobs:
                _, camera_updates = run_job(camera_name, pairs)
                for episode_idx, path in camera_updates.items():
                    updates[(int(episode_idx), camera_name)] = path

        for (episode_idx, camera_name), path in updates.items():
            episode_by_index[int(episode_idx)].video_files[camera_name] = path

    def _write_direct_camera_segments_with_retry(
        self,
        output_dir: Path,
        camera_name: str,
        pairs: List[Tuple[EpisodeData, Path]],
    ) -> Dict[int, Path]:
        use_concat_decoder = (
            len(pairs) > 1 and not os.environ.get(_V21_CONCAT_DECODER_DISABLE_ENV)
        )
        try:
            return self._write_direct_camera_segments(
                output_dir,
                camera_name,
                pairs,
                use_concat_decoder=use_concat_decoder,
            )
        except _DirectConcatDecoderError as exc:
            self._log_warning(
                f"{camera_name}: concat decoder failed ({exc!r}); "
                "retrying direct segmented video with per-file decoders"
            )
            return self._write_direct_camera_segments(
                output_dir,
                camera_name,
                pairs,
                use_concat_decoder=False,
            )

    def _write_direct_camera_segments(
        self,
        output_dir: Path,
        camera_name: str,
        pairs: List[Tuple[EpisodeData, Path]],
        *,
        use_concat_decoder: bool = True,
    ) -> Dict[int, Path]:
        if len(pairs) <= 1:
            return self._write_direct_camera_segments_fallback(
                output_dir,
                camera_name,
                pairs,
            )

        ffmpeg = _ffmpeg()
        width, height = self._quick_video_dimensions_cached(Path(pairs[0][1]))
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise RuntimeError(f"invalid direct v2.1 dimensions {width}x{height}")
        for _, video_path in pairs[1:]:
            other_width, other_height = self._quick_video_dimensions_cached(
                Path(video_path)
            )
            if (other_width, other_height) != (width, height):
                raise RuntimeError(
                    f"{camera_name}: mixed source dimensions "
                    f"{width}x{height} and {other_width}x{other_height}"
                )

        frame_size = width * height * 3 // 2
        fps = int(self.config.fps)
        fps_str = f"{float(fps):g}"
        encoder, encoder_opts = _h264_encoder(ffmpeg, width=width, height=height)
        requests: List[
            Tuple[EpisodeData, Path, np.ndarray, int, Optional[_StreamingRgbStats]]
        ] = []
        segment_frames: List[int] = []
        expected_total = 0
        sample_budget = self._video_stats_sample_budget()
        for idx, (episode, video_path) in enumerate(pairs):
            indices, source_count = self._grid_indices_and_source_count_for_raw_video(
                episode,
                camera_name,
                video_path,
            )
            stats = _StreamingRgbStats() if sample_budget > 0 else None
            requests.append((episode, video_path, indices, source_count, stats))
            expected_total += int(indices.size)
            if idx < len(pairs) - 1:
                segment_frames.append(expected_total)

        force_key_times = ",".join(
            f"{frame / float(fps):.9f}" for frame in segment_frames
        )
        with tempfile.TemporaryDirectory(
            prefix=f"v21_direct_{camera_name}_",
            dir=str(Path(output_dir)),
        ) as tmpdir:
            tmp = Path(tmpdir)
            pattern = tmp / "segment_%06d.mp4"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
                *_ffmpeg_threads_arg(),
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p",
                "-s", f"{width}x{height}",
                "-r", fps_str,
                "-i", "pipe:0",
            ]
            if force_key_times:
                cmd.extend(["-force_key_frames", force_key_times])
            cmd.extend([
                "-c:v", encoder,
                *self._segment_encoder_opts(encoder, encoder_opts),
                "-pix_fmt", "yuv420p",
                "-r", fps_str,
                "-an",
                "-f", "segment",
                "-segment_format", "mp4",
                "-segment_frames", ",".join(str(v) for v in segment_frames),
                "-reset_timestamps", "1",
                *self._segment_format_options_args(),
                str(pattern),
            ])

            process: Optional[subprocess.Popen] = None
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                assert process.stdin is not None
                if use_concat_decoder and len(requests) > 1:
                    try:
                        written_total = (
                            self._pipe_selected_yuv420_frames_concat_decoder(
                                ffmpeg,
                                requests,
                                frame_size,
                                process.stdin,
                                width=width,
                                height=height,
                            )
                        )
                    except Exception as exc:
                        raise _DirectConcatDecoderError(str(exc)) from exc
                else:
                    written_total = 0
                    for _, video_path, indices, _, stats in requests:
                        written_total += self._pipe_selected_yuv420_frames(
                            ffmpeg,
                            video_path,
                            indices,
                            frame_size,
                            process.stdin,
                            width=width,
                            height=height,
                            stats=stats,
                        )
                process.stdin.close()
                stderr = (
                    process.stderr.read().decode(errors="replace")
                    if process.stderr is not None else ""
                )
                rc = process.wait(timeout=300)
                if rc != 0:
                    raise RuntimeError(f"ffmpeg segment rc={rc}: {stderr[-500:]}")
                if written_total != expected_total:
                    raise RuntimeError(
                        f"direct segmented wrote {written_total} frames; "
                        f"expected {expected_total}"
                    )
            finally:
                _terminate_process(process, close_stdin=True)

            segment_paths = sorted(tmp.glob("segment_*.mp4"))
            if len(segment_paths) != len(requests):
                raise RuntimeError(
                    f"direct segmented produced {len(segment_paths)} files; "
                    f"expected {len(requests)}"
                )

            updates: Dict[int, Path] = {}
            for segment_path, (episode, video_path, indices, _, stats) in zip(
                segment_paths,
                requests,
            ):
                dst = self._v21_video_output_path(output_dir, episode, camera_name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(segment_path, dst)
                self._remember_direct_v21_video_info(
                    dst,
                    width=width,
                    height=height,
                )
                if self._validate_direct_v21_video():
                    _validated_video_count(
                        output_mp4=dst,
                        expected_frames=int(indices.size),
                        target_fps=fps,
                        ffmpeg=ffmpeg,
                        label="direct-v21",
                    )
                stats_payload = (
                    stats.to_stats()
                    if stats is not None and stats.frame_count > 0
                    else None
                )
                if stats is not None and stats.frame_count > 0:
                    self._remember_direct_v21_video_stats(
                        episode,
                        camera_name,
                        stats_payload,
                    )
                self._store_direct_v21_synced_cache(
                    episode=episode,
                    camera_name=camera_name,
                    video_path=video_path,
                    indices=indices,
                    output_path=dst,
                    stats=stats_payload,
                    width=width,
                    height=height,
                )
                updates[int(episode.episode_index)] = dst
            self._log_info(
                f"{camera_name}: direct-segmented {len(updates)} v2.1 videos "
                f"({expected_total} frames @ {fps_str} fps)"
            )
            return updates

    def _write_direct_camera_segments_fallback(
        self,
        output_dir: Path,
        camera_name: str,
        pairs: List[Tuple[EpisodeData, Path]],
    ) -> Dict[int, Path]:
        updates: Dict[int, Path] = {}
        for episode, video_path in pairs:
            indices = self._grid_indices_for_raw_video(
                episode,
                camera_name,
                video_path,
            )
            dst = self._v21_video_output_path(output_dir, episode, camera_name)
            result = remux_selected_frames(
                video_path,
                indices,
                dst,
                target_fps=int(self.config.fps),
            )
            stats_payload = result.stats
            if result.stats:
                self._remember_direct_v21_video_stats(
                    episode,
                    camera_name,
                    stats_payload,
                )
            if result.output_width is not None and result.output_height is not None:
                width = int(result.output_width)
                height = int(result.output_height)
            else:
                width, height = self._quick_video_dimensions_cached(dst)
            self._remember_quick_video_dimensions(dst, width, height)
            self._remember_direct_v21_video_info(
                dst,
                width=width,
                height=height,
            )
            self._store_direct_v21_synced_cache(
                episode=episode,
                camera_name=camera_name,
                video_path=video_path,
                indices=indices,
                output_path=dst,
                stats=stats_payload,
                width=width,
                height=height,
            )
            updates[int(episode.episode_index)] = dst
        return updates

    def _compute_episode_stats(
        self,
        episode: EpisodeData,
        global_start_index: int = 0,
    ) -> Dict[str, Dict]:
        """Compute v2.1 stats, using direct-path per-episode video stats."""
        if not getattr(self, "_direct_v21_video_output", False):
            return super()._compute_episode_stats(episode, global_start_index)

        video_files = dict(episode.video_files)
        episode.video_files = {}
        try:
            stats = super()._compute_episode_stats(episode, global_start_index)
        finally:
            episode.video_files = video_files

        direct_stats = getattr(self, "_direct_v21_video_stats_cache", {})
        for camera_name, video_path in video_files.items():
            feature_key = self._video_feature_key(camera_name)
            video_stats = direct_stats.get((int(episode.episode_index), camera_name))
            if video_stats is None:
                video_stats = self._compute_video_stats(
                    video_path,
                    f"{camera_name}:episode_{int(episode.episode_index)}",
                )
            if video_stats:
                stats[feature_key] = video_stats
        return stats

    def convert_multiple_rosbags(
        self,
        bag_paths: List[Path],
    ) -> bool:
        """
        Convert multiple ROSbag recordings to a single LeRobot dataset.

        Uses ProcessPoolExecutor for parallel episode parsing when multiple
        bag_paths are provided. Each worker creates its own converter instance.

        Args:
            bag_paths: List of paths to ROSbag directories

        Returns:
            True if successful, False otherwise
        """
        self._log_info(f"Converting {len(bag_paths)} rosbags to LeRobot dataset")
        self._reset_frame_reuse_reports()

        # Initialize output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        direct_video_output = self._can_use_direct_video_output(
            [Path(path) for path in bag_paths]
        )
        original_use_videos = bool(self.config.use_videos)
        if direct_video_output:
            self._log_info(
                "Using v2.1 direct segmented video fast path "
                "(raw MP4 + sidecar -> per-episode MP4)"
            )
            self.config.use_videos = False

        episodes_data: List[EpisodeData] = []
        try:
            cached_episode_indices: set[int] = set()
            if len(bag_paths) > 1:
                for idx, bag_path in enumerate(bag_paths):
                    cached_episode = self._try_load_prepared_episode_for_bag(
                        Path(bag_path),
                        idx,
                    )
                    if cached_episode is not None:
                        episodes_data.append(cached_episode)
                        cached_episode_indices.add(idx)
                if cached_episode_indices:
                    self._log_info(
                        f"Parent reused {len(cached_episode_indices)}/"
                        f"{len(bag_paths)} prepared episode cache entries"
                    )

            # Build a picklable config dict for worker processes. CRITICAL:
            # must include every selection knob the converter consults
            # in ``_sync_videos_to_grid`` / ``_compute_video_stats`` etc.;
            # missing camera_rotations / image_resize / selected_* meant the
            # worker silently defaulted to no rotation, no resize, etc. even
            # when the UI had set values.
            config_dict = {
                'repo_id': self.config.repo_id,
                'output_dir': self.config.output_dir,
                'fps': self.config.fps,
                'robot_type': self.config.robot_type,
                'use_videos': self.config.use_videos,
                'chunks_size': self.config.chunks_size,
                'robot_config_path': self.config.robot_config_path,
                'state_topics': self.config.state_topics,
                'action_topics': self.config.action_topics,
                'apply_trim': self.config.apply_trim,
                'apply_exclude_regions': self.config.apply_exclude_regions,
                'quality_warning_multiplier': self.config.quality_warning_multiplier,
                'quality_error_multiplier': self.config.quality_error_multiplier,
                # Selection knobs (camera/topic/joint filters + per-knob
                # transforms). Empty/None means "use defaults from the
                # robot_config", same as legacy behaviour.
                'selected_cameras': list(self.config.selected_cameras),
                'camera_rotations': dict(self.config.camera_rotations),
                'image_resize': (
                    tuple(self.config.image_resize)
                    if self.config.image_resize else None
                ),
                'selected_state_topics': list(self.config.selected_state_topics),
                'selected_action_topics': list(self.config.selected_action_topics),
                'selected_joints': list(self.config.selected_joints),
                'source_rosbags': list(self.config.source_rosbags),
            }

            missing_bag_paths = [
                (idx, Path(bag_path))
                for idx, bag_path in enumerate(bag_paths)
                if idx not in cached_episode_indices
            ]

            if len(bag_paths) <= 1:
                # Single episode: no parallelization overhead
                for idx, bag_path in enumerate(bag_paths):
                    episode_data = self.convert_single_rosbag(Path(bag_path), idx)
                    if episode_data is not None:
                        episodes_data.append(episode_data)
            elif not missing_bag_paths:
                # All episodes came from parent-side prepared caches; skip the
                # process pool entirely.
                episodes_data.sort(key=lambda ep: ep.episode_index)
            else:
                # Parallel episode parsing using ProcessPoolExecutor. Worker
                # count is capped at half of the host CPUs (override via
                # CYCLO_CONVERSION_MAX_WORKERS) so the ROS control loop and
                # camera drivers keep their cores. Worker initializer can lower
                # worker priority (override via CYCLO_CONVERSION_WORKER_NICE) so
                # saturated CPUs still favour higher-priority work.
                from concurrent.futures import ProcessPoolExecutor, as_completed

                max_workers = _resolve_conversion_worker_count(
                    len(missing_bag_paths)
                )
                self._log_info(
                    f"Starting parallel rosbag parsing with {max_workers} workers "
                    f"for {len(missing_bag_paths)} cache miss(es)"
                )

                with _active_conversion_workers(max_workers):
                    with ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_conversion_worker_init,
                    ) as executor:
                        futures = {}
                        for idx, bag_path in missing_bag_paths:
                            future = executor.submit(
                                _convert_rosbag_worker,
                                str(bag_path), idx, config_dict,
                            )
                            futures[future] = idx

                        for future in as_completed(futures):
                            idx = futures[future]
                            try:
                                episode_index, episode_data = future.result()
                                if episode_data is not None:
                                    episodes_data.append(episode_data)
                                    self._log_info(
                                        f"Episode {episode_index} parsed successfully"
                                    )
                                else:
                                    self._log_warning(
                                        f"Episode {idx} returned no data"
                                    )
                            except Exception as e:
                                self._log_error(
                                    f"Error parsing episode {idx}: {e}"
                                )

                # Sort by episode_index to maintain deterministic order
                episodes_data.sort(key=lambda ep: ep.episode_index)
        finally:
            self.config.use_videos = original_use_videos
        if not episodes_data:
            self._log_error("No episodes were successfully converted")
            return False

        self._direct_v21_video_output = bool(direct_video_output)
        if self._direct_v21_video_output:
            self._attach_direct_video_sources(episodes_data)

        success = self.write_from_episodes(episodes_data)
        if success:
            self._cleanup_output_temp_dirs()
            self._cleanup_source_synced_cache([Path(path) for path in bag_paths])
        return success

    def write_from_episodes(self, episodes_data: List[EpisodeData]) -> bool:
        """Write a v2.1 dataset from already parsed episodes."""
        if not episodes_data:
            self._log_error("No episodes were provided for LeRobot v2.1 writing")
            return False
        self._reset_frame_reuse_reports()
        episodes_data = self.prepare_episodes_for_writing(episodes_data)
        if not episodes_data:
            self._log_error("No complete episodes remained after subtask stitching")
            return False
        self._collect_episode_frame_reuse_reports(episodes_data)

        self._total_episodes = 0
        self._total_frames = 0
        self._features = {}
        self._tasks = {}
        self._task_to_index = {}
        self._episodes = {}
        self._episodes_stats = {}

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if getattr(self, "_direct_v21_video_output", False):
            self._write_direct_segmented_videos(output_dir, episodes_data)

        self._build_features(episodes_data)
        self._write_dataset(episodes_data)

        self._log_info(f"Successfully converted {len(episodes_data)} episodes")
        return True

    @staticmethod
    def _array_cache_signature(values: Sequence[Any]) -> Dict[str, Any]:
        if not values:
            return {"dtype": "", "shape": [0], "sha256": ""}
        array = np.ascontiguousarray(np.asarray(values))
        return {
            "dtype": str(array.dtype),
            "shape": [int(value) for value in array.shape],
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    def _v21_episode_data_cache_signature(
        self,
        episode: EpisodeData,
    ) -> Dict[str, Any]:
        prepared_cache_signature = getattr(
            episode,
            "_cyclo_prepared_cache_signature",
            None,
        )
        common = {
            "episode_index": int(episode.episode_index),
            "length": int(episode.length),
            "tasks": list(episode.tasks),
            "recording_mode": episode.recording_mode,
            "full_episode_index": episode.full_episode_index,
            "subtask_instructions": list(episode.subtask_instructions),
            "episode_success": episode.episode_success,
        }
        if isinstance(prepared_cache_signature, dict):
            common["prepared_cache"] = prepared_cache_signature
            return common
        common.update(
            {
                "timestamps": self._array_cache_signature(episode.timestamps),
                "observation_state": self._array_cache_signature(
                    episode.observation_state
                ),
                "action": self._array_cache_signature(episode.action),
                "subtask_indices": self._array_cache_signature(
                    episode.subtask_indices
                ),
            }
        )
        return common

    def _v21_episode_parquet_cache_key(
        self,
        episode: EpisodeData,
        *,
        global_start_index: int,
        has_subtask_feature: bool,
    ) -> Dict[str, Any]:
        default_task = episode.tasks[0] if episode.tasks else "default_task"
        return {
            "version": _V21_EPISODE_PARQUET_CACHE_VERSION,
            "codebase_version": CODEBASE_VERSION,
            "pyarrow_version": getattr(pa, "__version__", ""),
            "fps": int(self.config.fps),
            "global_start_index": int(global_start_index),
            "task_name": default_task,
            "task_index": int(self._task_to_index.get(default_task, 0)),
            "task_to_index": dict(sorted(self._task_to_index.items())),
            "has_subtask_feature": bool(has_subtask_feature),
            "episode": self._v21_episode_data_cache_signature(episode),
        }

    @staticmethod
    def _v21_episode_parquet_cache_digest(cache_key: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _v21_episode_parquet_cache_root(
        self,
        episode: EpisodeData,
    ) -> Optional[Path]:
        if os.environ.get(_V21_EPISODE_PARQUET_CACHE_DISABLE_ENV):
            return None
        if episode.source_path is None:
            return None
        source = Path(_fast_absolute_path(Path(episode.source_path)))
        source_root = source if source.is_dir() else source.parent
        return source_root / ".cyclo_cache" / "episode_parquet_v21"

    def _v21_episode_parquet_cache_path(
        self,
        episode: EpisodeData,
        cache_key: Dict[str, Any],
    ) -> Optional[Path]:
        cache_root = self._v21_episode_parquet_cache_root(episode)
        if cache_root is None:
            return None
        return cache_root / self._v21_episode_parquet_cache_digest(cache_key)

    def _try_reuse_v21_episode_parquet_cache(
        self,
        episode: EpisodeData,
        parquet_path: Path,
        cache_key: Dict[str, Any],
    ) -> bool:
        cache_path = self._v21_episode_parquet_cache_path(episode, cache_key)
        if cache_path is None:
            return False
        manifest_path = cache_path / "manifest.json"
        cached_parquet = cache_path / "episode.parquet"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if manifest.get("cache_key") != cache_key:
            return False
        try:
            if not cached_parquet.exists() or cached_parquet.stat().st_size <= 0:
                return False
            _atomic_copy_file(cached_parquet, parquet_path)
            self._log_info(f"Reused v2.1 episode parquet cache: {parquet_path}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"v2.1 episode parquet cache reuse failed ({exc!r}); regenerating"
            )
            return False

    def _store_v21_episode_parquet_cache(
        self,
        episode: EpisodeData,
        parquet_path: Path,
        cache_key: Dict[str, Any],
    ) -> None:
        cache_path = self._v21_episode_parquet_cache_path(episode, cache_key)
        if cache_path is None or not parquet_path.exists():
            return
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
            _atomic_copy_file(parquet_path, tmp_dir / "episode.parquet")
            (tmp_dir / "manifest.json").write_text(
                json.dumps({"cache_key": cache_key}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_dir, cache_path)
            self._log_info(f"Stored v2.1 episode parquet cache: {cache_path}")
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._log_warning(
                f"failed to store v2.1 episode parquet cache ({exc!r})"
            )

    def _v21_dataset_cache_root(
        self,
        episodes_data: List[EpisodeData],
        *,
        cache_name: str,
        disabled_env: str,
    ) -> Optional[Path]:
        if os.environ.get(disabled_env):
            return None
        roots: List[str] = []
        try:
            for episode in episodes_data:
                if episode.source_path is None:
                    return None
                source = Path(_fast_absolute_path(Path(episode.source_path)))
                roots.append(str(source if source.is_dir() else source.parent))
            if not roots:
                return None
            return Path(os.path.commonpath(roots)) / ".cyclo_cache" / cache_name
        except Exception as exc:  # noqa: BLE001
            self._log_warning(f"{cache_name} disabled ({exc!r})")
            return None

    @staticmethod
    def _v21_small_parquet_cache_digest(cache_key: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _v21_small_parquet_cache_path(
        self,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
        *,
        cache_name: str,
        disabled_env: str,
    ) -> Optional[Path]:
        cache_root = self._v21_dataset_cache_root(
            episodes_data,
            cache_name=cache_name,
            disabled_env=disabled_env,
        )
        if cache_root is None:
            return None
        return cache_root / self._v21_small_parquet_cache_digest(cache_key)

    def _try_reuse_v21_small_parquet_cache(
        self,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
        file_path: Path,
        *,
        cache_name: str,
        artifact_name: str,
        disabled_env: str,
    ) -> bool:
        cache_path = self._v21_small_parquet_cache_path(
            episodes_data,
            cache_key,
            cache_name=cache_name,
            disabled_env=disabled_env,
        )
        if cache_path is None:
            return False
        manifest_path = cache_path / "manifest.json"
        parquet_path = cache_path / artifact_name
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if manifest.get("cache_key") != cache_key:
            return False
        try:
            if not parquet_path.exists() or parquet_path.stat().st_size <= 0:
                return False
            _atomic_copy_file(parquet_path, file_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_warning(
                f"{cache_name} reuse failed ({exc!r}); regenerating"
            )
            return False

    def _store_v21_small_parquet_cache(
        self,
        episodes_data: List[EpisodeData],
        cache_key: Dict[str, Any],
        file_path: Path,
        *,
        cache_name: str,
        artifact_name: str,
        disabled_env: str,
    ) -> None:
        cache_path = self._v21_small_parquet_cache_path(
            episodes_data,
            cache_key,
            cache_name=cache_name,
            disabled_env=disabled_env,
        )
        if cache_path is None or not file_path.exists():
            return
        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
            _atomic_copy_file(file_path, tmp_dir / artifact_name)
            (tmp_dir / "manifest.json").write_text(
                json.dumps({"cache_key": cache_key}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_dir, cache_path)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._log_warning(f"failed to store {cache_name} ({exc!r})")

    def _v21_subtasks_parquet_cache_key(
        self,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "version": _V21_SUBTASKS_PARQUET_CACHE_VERSION,
            "codebase_version": CODEBASE_VERSION,
            "pyarrow_version": getattr(pa, "__version__", ""),
            "rows": rows,
        }

    def _write_subtasks_parquet(
        self,
        output_dir: Path,
        episodes_data: List[EpisodeData],
    ) -> None:
        """Write optional LeRobot subtask lookup metadata."""
        rows = self._subtask_rows_for_dataset(episodes_data)
        if not rows:
            return
        path = Path(output_dir) / "meta" / "subtasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        cache_key = self._v21_subtasks_parquet_cache_key(rows)
        if self._try_reuse_v21_small_parquet_cache(
            episodes_data,
            cache_key,
            path,
            cache_name="subtasks_parquet_v21",
            artifact_name="subtasks.parquet",
            disabled_env=_V21_SUBTASKS_PARQUET_CACHE_DISABLE_ENV,
        ):
            self._log_info("Reused v2.1 subtasks parquet cache")
            return

        table = pa.table({
            "subtask_index": pa.array(
                [int(row["subtask_index"]) for row in rows],
                type=pa.int64(),
            ),
            "subtask": pa.array(
                [str(row["subtask"]) for row in rows],
                type=pa.string(),
            ),
        })
        pq.write_table(table, path)
        self._store_v21_small_parquet_cache(
            episodes_data,
            cache_key,
            path,
            cache_name="subtasks_parquet_v21",
            artifact_name="subtasks.parquet",
            disabled_env=_V21_SUBTASKS_PARQUET_CACHE_DISABLE_ENV,
        )
        self._log_info(f"Wrote subtasks metadata: {path}")

    def _write_dataset(self, episodes_data: List[EpisodeData]):
        """Write all dataset files to output directory."""
        output_dir = Path(self.config.output_dir)

        # Create directory structure
        (output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

        # Collect tasks in episode-first-appearance order (shared with v3.0).
        self._collect_tasks(episodes_data)
        self._collect_task_names(episodes_data)

        # Write episodes
        for episode_data in episodes_data:
            self._write_episode(episode_data)

        # Optional subtask metadata mirrors the per-frame subtask_index.
        self._write_subtasks_parquet(output_dir, episodes_data)
        self._write_subtask_annotations(output_dir, episodes_data)

        # Write metadata files
        self._write_info_json()
        self._write_tasks_jsonl()
        self._write_root_info_json()
        self._write_frame_reuse_metadata(output_dir)

    def _write_episode(self, episode: EpisodeData):
        """Write a single episode's data files."""
        output_dir = Path(self.config.output_dir)
        ep_idx = episode.episode_index
        chunk_idx = ep_idx // V21_CHUNK_SIZE

        # Create chunk directories
        data_chunk_dir = output_dir / "data" / _v21_chunk_dir_name(chunk_idx)
        data_chunk_dir.mkdir(parents=True, exist_ok=True)

        video_chunk_dir = output_dir / "videos" / _v21_chunk_dir_name(chunk_idx)
        video_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet file
        parquet_path = data_chunk_dir / f"{_v21_episode_stem(ep_idx)}.parquet"
        has_subtask_feature = "subtask_index" in self._features
        parquet_cache_key = self._v21_episode_parquet_cache_key(
            episode,
            global_start_index=self._total_frames,
            has_subtask_feature=has_subtask_feature,
        )
        if not self._try_reuse_v21_episode_parquet_cache(
            episode,
            parquet_path,
            parquet_cache_key,
        ):
            self._write_parquet(episode, parquet_path)
            self._store_v21_episode_parquet_cache(
                episode,
                parquet_path,
                parquet_cache_key,
            )

        # Copy video files. ``_sync_videos_to_grid`` produces
        # ``<cam>_synced.mp4`` files in the source rosbag videos/ dir;
        # we used to delete them after the copy ("not littering the
        # tree"). Keeping them is the right call now — they're cache:
        #
        # * v3.0's Phase 1 needs the same files. When the request runs
        #   v2.1 + v3.0 back-to-back, deleting after v2.1 forces v3.0
        #   to do the same expensive remux all over again.
        # * Subsequent re-conversions (any combination) hit the
        #   ``<cam>_synced.cache.json`` gate in ``_sync_videos_to_grid``
        #   and skip remux entirely.
        #
        # Disk cost is modest (~2-3 MB per camera per episode at typical
        # resolutions). Operators who want a clean tree can wipe
        # ``<episode>/videos/*_synced.*`` after the dataset is final.
        for camera_name, src_video in episode.video_files.items():
            self._record_frame_reuse_for_video(episode, camera_name, src_video)
            video_dir = video_chunk_dir / self._video_feature_key(camera_name)
            video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = video_dir / f"{_v21_episode_stem(ep_idx)}.mp4"
            mode = _clone_or_copy_file(src_video, dst_video)
            self._log_info(
                f"Copied video ({mode}): {src_video.name} -> {dst_video}"
            )

        # Write episode metadata
        episode_dict = {
            "episode_index": ep_idx,
            "tasks": episode.tasks,
            "length": episode.length,
        }
        if episode.recording_mode == "stitched_subtask":
            episode_dict["recording_mode"] = episode.recording_mode
            episode_dict["full_episode_index"] = episode.full_episode_index
            episode_dict["subtask_instructions"] = episode.subtask_instructions
        self._episodes[ep_idx] = episode_dict
        self._append_jsonl(episode_dict, output_dir / "meta" / "episodes.jsonl")

        # Compute and write episode stats. We pass the global starting
        # index (== total frames across previously written episodes) so
        # the stats for the synthetic ``index`` column are accurate.
        ep_stats = self._compute_episode_stats(
            episode, global_start_index=self._total_frames,
        )
        self._episodes_stats[ep_idx] = ep_stats
        stats_entry = {
            "episode_index": ep_idx,
            "stats": self._serialize_stats(ep_stats),
        }
        self._append_jsonl(stats_entry, output_dir / "meta" / "episodes_stats.jsonl")

        # Update totals
        self._total_frames += episode.length
        self._total_episodes += 1

    def _write_parquet(self, episode: EpisodeData, parquet_path: Path):
        """Write episode data to parquet file with HuggingFace-compatible schema."""
        num_frames = episode.length

        # Determine dimensions
        state_dim = (
            len(episode.observation_state[0]) if episode.observation_state else 0
        )
        action_dim = len(episode.action[0]) if episode.action else 0

        # Build schema with fixed_size_list for HuggingFace compatibility
        schema_fields = [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
        ]
        has_episode_success_feature = "episode_success" in self._features
        if has_episode_success_feature:
            schema_fields.append(pa.field("episode_success", pa.bool_()))
        if action_dim > 0:
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), action_dim)))
        if state_dim > 0:
            schema_fields.append(
                pa.field("observation.state", pa.list_(pa.float32(), state_dim))
            )
        has_subtask_feature = "subtask_index" in self._features
        if has_subtask_feature:
            schema_fields.append(pa.field("subtask_index", pa.int64()))

        schema = pa.schema(schema_fields)

        # Build data arrays with explicit types
        arrays = [
            pa.array(
                list(range(self._total_frames, self._total_frames + num_frames)),
                type=pa.int64(),
            ),
            pa.array([episode.episode_index] * num_frames, type=pa.int64()),
        ]

        # Task index
        default_task = episode.tasks[0] if episode.tasks else "default_task"
        task_idx = self._task_to_index.get(default_task, 0)
        arrays.append(pa.array([task_idx] * num_frames, type=pa.int64()))

        arrays.append(
            pa.array(
                [float(episode.timestamps[i]) for i in range(num_frames)],
                type=pa.float64(),
            )
        )
        if has_episode_success_feature:
            arrays.append(
                pa.array(
                    [bool(episode.episode_success)] * num_frames,
                    type=pa.bool_(),
                )
            )

        # Add action as fixed_size_list
        if episode.action:
            action_values = [[float(v) for v in action] for action in episode.action]
            arrays.append(
                pa.array(action_values, type=pa.list_(pa.float32(), action_dim))
            )

        # Add observation.state as fixed_size_list
        if episode.observation_state:
            state_values = [
                [float(v) for v in state] for state in episode.observation_state
            ]
            arrays.append(
                pa.array(state_values, type=pa.list_(pa.float32(), state_dim))
            )

        if has_subtask_feature:
            if len(episode.subtask_indices) == num_frames:
                subtask_values = [int(idx) for idx in episode.subtask_indices]
            else:
                subtask_values = [0] * num_frames
            arrays.append(pa.array(subtask_values, type=pa.int64()))

        # Build HuggingFace metadata
        hf_features = {
            "index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
            "timestamp": {"dtype": "float64", "_type": "Value"},
        }
        if has_episode_success_feature:
            hf_features["episode_success"] = {
                "dtype": "bool",
                "_type": "Value",
            }

        if action_dim > 0:
            hf_features["action"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": action_dim,
                "_type": "Sequence",
            }
        if state_dim > 0:
            hf_features["observation.state"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": state_dim,
                "_type": "Sequence",
            }
        if has_subtask_feature:
            hf_features["subtask_index"] = {"dtype": "int64", "_type": "Value"}

        hf_metadata = json.dumps({"info": {"features": hf_features}})

        # Add metadata to schema
        schema = schema.with_metadata({"huggingface": hf_metadata})

        # Create table with schema
        table = pa.table(
            dict(zip([f.name for f in schema_fields], arrays)), schema=schema
        )
        pq.write_table(table, parquet_path)
        self._log_info(f"Wrote parquet: {parquet_path}")

    def _v21_features_for_info(self) -> dict:
        """Return v2.1 feature metadata, excluding parquet-only leftovers."""
        ordered = {}
        for key in ("observation.state", "action"):
            if key in self._features:
                ordered[key] = self._features[key]
        for key in self._features:
            if key.startswith("observation.images."):
                ordered[key] = self._features[key]
        for key in (
            "timestamp",
            "episode_index",
            "index",
            "task_index",
            "episode_success",
            "subtask_index",
        ):
            if key in self._features:
                value = dict(self._features[key])
                if key == "timestamp":
                    value["dtype"] = "float64"
                ordered[key] = value
        return ordered

    def _write_info_json(self):
        """Write info.json metadata file."""
        output_dir = Path(self.config.output_dir)

        features = self._v21_features_for_info()
        num_video_keys = sum(
            1 for k in features if k.startswith("observation.images.")
        )
        total_chunks = (
            (self._total_episodes + V21_CHUNK_SIZE - 1) // V21_CHUNK_SIZE
            if self._total_episodes > 0
            else 0
        )

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.config.robot_type,
            "total_episodes": self._total_episodes,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": self._total_episodes * num_video_keys,
            "total_chunks": total_chunks,
            "chunks_size": V21_CHUNK_SIZE,
            "fps": self.config.fps,
            "splits": {"train": f"0:{self._total_episodes}"},
            "data_path": V21_DATA_PATH,
            "video_path": V21_VIDEO_PATH if self.config.use_videos else None,
            "features": features,
        }
        if "subtask_index" in features:
            info["annotation_path"] = V21_ANNOTATION_PATH

        info_path = output_dir / "meta" / "info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        self._log_info(f"Wrote info.json: {info_path}")

    def _annotation_chunk_dir_name(self, chunk_idx: int) -> str:
        return _v21_chunk_dir_name(chunk_idx)

    def _annotation_episode_filename(self, episode_idx: int) -> str:
        return f"{_v21_episode_stem(episode_idx)}.json"

    def _episode_chunk_index(self, episode_idx: int) -> int:
        return episode_idx // V21_CHUNK_SIZE

    def _video_feature_key(self, camera_name: str) -> str:
        return f"observation.images.rgb.{camera_name}"

    def _write_tasks_jsonl(self):
        """Write tasks.jsonl metadata file."""
        output_dir = Path(self.config.output_dir)
        tasks_path = output_dir / "meta" / "tasks.jsonl"

        with open(tasks_path, "w", encoding="utf-8") as f:
            for task_idx, task in self._tasks.items():
                task_names = getattr(self, "_task_names_by_task", {})
                entry = {
                    "task_index": task_idx,
                    "task": task,
                    "task_name": task_names.get(task, task),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._log_info(f"Wrote tasks.jsonl: {tasks_path}")

    def _append_jsonl(self, data: dict, filepath: Path):
        """Append a single entry to a JSONL file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def convert_rosbags_to_lerobot(
    bag_paths: List[str],
    output_dir: str,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    robot_type: str = "unknown",
    logger=None,
) -> bool:
    """
    Convenience function to convert multiple ROSbags to LeRobot dataset.

    Args:
        bag_paths: List of paths to ROSbag directories
        output_dir: Output directory for the dataset
        repo_id: Repository ID for the dataset (e.g., "user/dataset_name")
        fps: Target frames per second
        robot_type: Robot type identifier
        logger: Optional logger instance

    Returns:
        True if successful, False otherwise

    Example:
        >>> convert_rosbags_to_lerobot(
        ...     bag_paths=["/data/rosbag_001", "/data/rosbag_002"],
        ...     output_dir="/datasets/my_robot_dataset",
        ...     repo_id="robotis/ai_worker_pick_place",
        ...     fps=30,
        ...     robot_type="ai_worker",
        ... )
    """
    config = ConversionConfig(
        repo_id=repo_id,
        output_dir=Path(output_dir),
        fps=fps,
        robot_type=robot_type,
    )

    converter = RosbagToLerobotConverter(config, logger)
    return converter.convert_multiple_rosbags([Path(p) for p in bag_paths])

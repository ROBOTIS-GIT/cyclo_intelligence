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

"""Unit tests for RosbagToLerobotConverter."""

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

# Mock ROS2 modules that are not available outside Docker
for mod_name in [
    "rosbag2_py", "rclpy", "rclpy.serialization",
    "sensor_msgs", "sensor_msgs.msg",
    "trajectory_msgs", "trajectory_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg",
    "rosbag_recorder", "rosbag_recorder.msg",
    "mcap", "mcap.reader", "mcap_ros2", "mcap_ros2.decoder",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Add mock classes to sensor_msgs.msg etc.
sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["trajectory_msgs.msg"].JointTrajectory = MagicMock
sys.modules["nav_msgs.msg"].Odometry = MagicMock
sys.modules["geometry_msgs.msg"].Twist = MagicMock
sys.modules["rosbag2_py"].SequentialReader = MagicMock
sys.modules["rosbag2_py"].StorageOptions = MagicMock
sys.modules["rosbag2_py"].ConverterOptions = MagicMock
sys.modules["rclpy.serialization"].deserialize_message = MagicMock
sys.modules["mcap.reader"].make_reader = MagicMock
sys.modules["mcap_ros2.decoder"].DecoderFactory = MagicMock

from cyclo_data.converter.to_lerobot_v21 import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverter,
)
from cyclo_data.converter import to_lerobot_v21 as v21
from cyclo_data.converter import base_converter
from cyclo_data.converter.base_converter import RosbagToLerobotConverterBase

# The host test environment may carry a pandas/numpy ABI mismatch. The v3.0
# converter only needs pandas for parquet aggregation, not for the video concat
# tests below, so provide a tiny import stub before loading it.
if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = MagicMock
    pandas_stub.__version__ = "0.0.0"
    sys.modules["pandas"] = pandas_stub

from cyclo_data.converter import to_lerobot_v30 as v30
from cyclo_data.converter.to_lerobot_v30 import (
    DEFAULT_VIDEO_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    EpisodeMetadata,
    RosbagToLerobotV30Converter,
    V30ConversionConfig,
)
from cyclo_data.converter import video_sync
from cyclo_data.converter.scripts import convert_rosbag_to_lerobot as convert_cli


class TestConversionConfig(unittest.TestCase):
    """Tests for ConversionConfig dataclass."""

    def test_default_values(self):
        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
        )
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.robot_type, "unknown")
        self.assertTrue(config.use_videos)
        self.assertEqual(config.chunks_size, 1000)
        self.assertTrue(config.apply_trim)
        self.assertTrue(config.apply_exclude_regions)

    def test_custom_values(self):
        config = ConversionConfig(
            repo_id="user/my_dataset",
            output_dir=Path("/datasets/output"),
            fps=60,
            robot_type="ai_worker",
            chunks_size=500,
            apply_trim=False,
        )
        self.assertEqual(config.fps, 60)
        self.assertEqual(config.robot_type, "ai_worker")
        self.assertEqual(config.chunks_size, 500)
        self.assertFalse(config.apply_trim)


class TestConvertCliDiscovery(unittest.TestCase):
    """Tests for input directory rosbag discovery."""

    def test_find_rosbags_prunes_video_cache_and_converted_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bag = root / "2"
            bag.mkdir()
            (bag / "metadata.yaml").write_text("rosbag", encoding="utf-8")
            (bag / "2_0.mcap").write_bytes(b"mcap")
            (bag / "episode_info.json").write_text(
                json.dumps({"episode_index": 2}),
                encoding="utf-8",
            )

            nested_video_bag = bag / "videos" / "2_0" / "not_a_bag"
            nested_video_bag.mkdir(parents=True)
            (nested_video_bag / "metadata.yaml").write_text(
                "ignore",
                encoding="utf-8",
            )
            (nested_video_bag / "fake.mcap").write_bytes(b"ignore")

            cache_bag = root / ".cyclo_cache" / "cached_bag"
            cache_bag.mkdir(parents=True)
            (cache_bag / "metadata.yaml").write_text("ignore", encoding="utf-8")
            (cache_bag / "fake.mcap").write_bytes(b"ignore")

            converted_bag = root / "old_converted" / "3"
            converted_bag.mkdir(parents=True)
            (converted_bag / "metadata.yaml").write_text("ignore", encoding="utf-8")
            (converted_bag / "fake.mcap").write_bytes(b"ignore")

            found = convert_cli.find_rosbags_in_directory(root)

        self.assertEqual(found, [bag])

    def test_find_rosbags_still_finds_nested_group_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bag = root / "group" / "7"
            bag.mkdir(parents=True)
            (bag / "metadata.yaml").write_text("rosbag", encoding="utf-8")
            (bag / "7_0.mcap").write_bytes(b"mcap")

            found = convert_cli.find_rosbags_in_directory(root)

        self.assertEqual(found, [bag])


class TestVideoProbeCaching(unittest.TestCase):
    """Tests for repeated video metadata probes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = RosbagToLerobotConverterBase(
            ConversionConfig(
                repo_id="test/probe-cache",
                output_dir=Path(self.temp_dir),
            )
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_video_frame_count_caches_by_file_signature(self):
        video_path = Path(self.temp_dir) / "video.mp4"
        video_path.write_bytes(b"video")
        calls = []

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return 7

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=1,
            VideoCapture=lambda path: calls.append(path) or FakeCapture(),
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            first = self.converter._get_video_frame_count(video_path)
            second = self.converter._get_video_frame_count(video_path)

        self.assertEqual(first, 7)
        self.assertEqual(second, 7)
        self.assertEqual(calls, [str(video_path)])

    def test_probe_video_streams_caches_by_file_signature(self):
        video_path = Path(self.temp_dir) / "video.mp4"
        video_path.write_bytes(b"video")
        payload = {"streams": [{"codec_type": "video", "codec_name": "h264"}]}

        with patch.object(
            base_converter.subprocess,
            "run",
            return_value=types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
            ),
        ) as run:
            first = self.converter._probe_video_streams(video_path)
            second = self.converter._probe_video_streams(video_path)

        self.assertEqual(first, payload)
        self.assertEqual(second, payload)
        run.assert_called_once()


class TestFileSignature(unittest.TestCase):
    """Tests for file signature generation used by caches."""

    def test_file_signature_uses_absolute_path_without_resolve_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.bin"
            path.write_bytes(b"data")

            with patch.object(
                Path,
                "resolve",
                side_effect=AssertionError("resolve should not be needed"),
            ):
                signature = RosbagToLerobotConverterBase._file_signature(path)

        self.assertEqual(signature["path"], os.path.abspath(os.fspath(path)))
        self.assertEqual(signature["size"], 4)
        self.assertIsInstance(signature["mtime_ns"], int)


class TestRawCdrExtraction(unittest.TestCase):
    """Tests for the direct CDR state/action fast parser."""

    @staticmethod
    def _align(buf: bytearray, alignment: int, origin: int = 4) -> None:
        while (len(buf) - origin) % alignment:
            buf.append(0)

    @classmethod
    def _write_i32(cls, buf: bytearray, value: int) -> None:
        cls._align(buf, 4)
        buf.extend(struct.pack("<i", value))

    @classmethod
    def _write_u32(cls, buf: bytearray, value: int) -> None:
        cls._align(buf, 4)
        buf.extend(struct.pack("<I", value))

    @classmethod
    def _write_string(cls, buf: bytearray, value: str) -> None:
        raw = value.encode("utf-8") + b"\x00"
        cls._write_u32(buf, len(raw))
        buf.extend(raw)

    @classmethod
    def _write_header(cls, buf: bytearray) -> None:
        cls._write_i32(buf, 1)
        cls._write_u32(buf, 2)
        cls._write_string(buf, "")

    @classmethod
    def _write_string_sequence(cls, buf: bytearray, values) -> None:
        cls._write_u32(buf, len(values))
        for value in values:
            cls._write_string(buf, value)

    @classmethod
    def _write_float64_sequence(cls, buf: bytearray, values) -> None:
        cls._write_u32(buf, len(values))
        cls._align(buf, 8)
        for value in values:
            buf.extend(struct.pack("<d", float(value)))

    def test_parse_raw_cdr_twist(self):
        buf = bytearray(b"\x00\x01\x00\x00")
        for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            buf.extend(struct.pack("<d", value))

        positions, names = RosbagToLerobotConverterBase._extract_raw_cdr_positions(
            "geometry_msgs/msg/Twist",
            bytes(buf),
        )

        np.testing.assert_allclose(
            positions,
            np.array([1.0, 2.0, 6.0], dtype=np.float32),
        )
        self.assertEqual(names, [])

    def test_parse_raw_cdr_joint_state(self):
        buf = bytearray(b"\x00\x01\x00\x00")
        self._write_header(buf)
        self._write_string_sequence(buf, ["joint_a", "joint_b"])
        self._write_float64_sequence(buf, [0.25, -0.5])

        positions, names = RosbagToLerobotConverterBase._extract_raw_cdr_positions(
            "sensor_msgs/msg/JointState",
            bytes(buf),
        )

        self.assertEqual(names, ["joint_a", "joint_b"])
        np.testing.assert_allclose(
            positions,
            np.array([0.25, -0.5], dtype=np.float32),
        )

    def test_parse_raw_cdr_joint_trajectory_first_point(self):
        buf = bytearray(b"\x00\x01\x00\x00")
        self._write_header(buf)
        self._write_string_sequence(buf, ["joint_a", "joint_b"])
        self._write_u32(buf, 1)
        self._write_float64_sequence(buf, [1.25, 2.5])

        positions, names = RosbagToLerobotConverterBase._extract_raw_cdr_positions(
            "trajectory_msgs/msg/JointTrajectory",
            bytes(buf),
        )

        self.assertEqual(names, ["joint_a", "joint_b"])
        np.testing.assert_allclose(
            positions,
            np.array([1.25, 2.5], dtype=np.float32),
        )


class TestEpisodeData(unittest.TestCase):
    """Tests for EpisodeData dataclass."""

    def test_default_initialization(self):
        episode = EpisodeData(episode_index=0)
        self.assertEqual(episode.episode_index, 0)
        self.assertEqual(episode.timestamps, [])
        self.assertEqual(episode.observation_state, [])
        self.assertEqual(episode.action, [])
        self.assertEqual(episode.video_files, {})
        self.assertEqual(episode.tasks, [])
        self.assertEqual(episode.length, 0)
        self.assertEqual(episode.recording_mode, "single")
        self.assertEqual(episode.subtask_instructions, [])

    def test_with_data(self):
        episode = EpisodeData(
            episode_index=5,
            timestamps=[0.0, 0.033, 0.066],
            observation_state=[
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                np.array([1.2, 2.2]),
            ],
            action=[
                np.array([0.1, 0.2]),
                np.array([0.11, 0.21]),
                np.array([0.12, 0.22]),
            ],
            tasks=["pick object"],
            length=3,
        )
        self.assertEqual(episode.episode_index, 5)
        self.assertEqual(len(episode.timestamps), 3)
        self.assertEqual(episode.length, 3)


class TestVideoSyncWorkerPolicy(unittest.TestCase):
    """Tests for adaptive camera-sync worker defaults."""

    def test_single_episode_can_use_all_camera_workers(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                RosbagToLerobotConverterBase._resolve_video_sync_camera_workers(4),
                4,
            )

    def test_multi_episode_divides_global_sync_budget(self):
        with patch.dict(
            os.environ,
            {"CYCLO_CONVERSION_ACTIVE_WORKERS": "8"},
            clear=True,
        ):
            self.assertEqual(
                RosbagToLerobotConverterBase._resolve_video_sync_camera_workers(4),
                1,
            )

        with patch.dict(
            os.environ,
            {"CYCLO_CONVERSION_ACTIVE_WORKERS": "4"},
            clear=True,
        ):
            self.assertEqual(
                RosbagToLerobotConverterBase._resolve_video_sync_camera_workers(4),
                1,
            )

    def test_explicit_camera_worker_override_still_wins(self):
        with patch.dict(
            os.environ,
            {
                "CYCLO_CONVERSION_ACTIVE_WORKERS": "8",
                "CYCLO_VIDEO_SYNC_CAMERA_WORKERS": "3",
            },
            clear=True,
        ):
            self.assertEqual(
                RosbagToLerobotConverterBase._resolve_video_sync_camera_workers(4),
                3,
            )

    def test_conversion_worker_default_caps_at_four(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            base_converter.os,
            "cpu_count",
            return_value=24,
        ):
            self.assertEqual(
                base_converter._resolve_conversion_worker_count(8),
                4,
            )

    def test_v21_direct_video_requires_max_profile_or_opt_in(self):
        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
            fps=15,
            use_videos=True,
        )
        converter = RosbagToLerobotConverter(config)
        sources = {"cam": Path("/tmp/cam.mp4")}
        with patch.object(
            converter, "_direct_video_sources_for_bag", return_value=sources
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(
                    converter._can_use_direct_video_output(
                        [Path("/bags/0"), Path("/bags/1")]
                    )
                )
            with patch.dict(
                os.environ,
                {"CYCLO_X264_SPEED_PROFILE": "max"},
                clear=True,
            ):
                self.assertTrue(
                    converter._can_use_direct_video_output(
                        [Path("/bags/0"), Path("/bags/1")]
                    )
                )
            with patch.dict(
                os.environ,
                {"CYCLO_V21_ENABLE_DIRECT_VIDEO": "1"},
                clear=True,
            ):
                self.assertTrue(
                    converter._can_use_direct_video_output(
                        [Path("/bags/0"), Path("/bags/1")]
                    )
                )

    def test_v21_direct_video_batch_defaults_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                RosbagToLerobotConverter._direct_video_batch_episodes(),
                32,
            )
            self.assertEqual(
                RosbagToLerobotConverter._resolve_direct_video_workers(
                    job_count=20,
                    camera_count=4,
                ),
                5,
            )
            self.assertEqual(
                RosbagToLerobotConverter._resolve_direct_video_cache_workers(
                    job_count=20,
                ),
                1,
            )
        with patch.dict(
            os.environ,
            {
                "CYCLO_V21_DIRECT_VIDEO_BATCH_EPISODES": "7",
                "CYCLO_V21_DIRECT_VIDEO_WORKERS": "9",
            },
            clear=True,
        ):
            self.assertEqual(
                RosbagToLerobotConverter._direct_video_batch_episodes(),
                7,
            )
            self.assertEqual(
                RosbagToLerobotConverter._resolve_direct_video_workers(
                    job_count=20,
                    camera_count=4,
                ),
                9,
            )
            self.assertEqual(
                RosbagToLerobotConverter._resolve_direct_video_cache_workers(
                    job_count=20,
                ),
                9,
            )

        with patch.dict(
            os.environ,
            {
                "CYCLO_V21_DIRECT_VIDEO_WORKERS": "9",
                "CYCLO_V21_DIRECT_VIDEO_CACHE_WORKERS": "2",
            },
            clear=True,
        ):
            self.assertEqual(
                RosbagToLerobotConverter._resolve_direct_video_cache_workers(
                    job_count=20,
                ),
                2,
            )

    def test_v21_direct_video_validation_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(RosbagToLerobotConverter._validate_direct_v21_video())
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertFalse(RosbagToLerobotConverter._validate_direct_v21_video())
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V21_VALIDATE_DIRECT_VIDEO": "1",
            },
            clear=True,
        ):
            self.assertTrue(RosbagToLerobotConverter._validate_direct_v21_video())
        with patch.dict(
            os.environ,
            {"CYCLO_V21_VALIDATE_DIRECT_VIDEO": "0"},
            clear=True,
        ):
            self.assertFalse(RosbagToLerobotConverter._validate_direct_v21_video())
        with patch.dict(
            os.environ,
            {"CYCLO_V21_VALIDATE_DIRECT_VIDEO": "yes"},
            clear=True,
        ):
            self.assertTrue(RosbagToLerobotConverter._validate_direct_v21_video())

    def test_v21_direct_video_removes_default_x264_all_intra_gop(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                RosbagToLerobotConverter._segment_encoder_opts(
                    "libx264",
                    ["-preset", "ultrafast", "-g", "1", "-qp", "51"],
                ),
                ["-preset", "ultrafast", "-qp", "51"],
            )
        with patch.dict(os.environ, {"CYCLO_X264_GOP": "1"}, clear=True):
            self.assertEqual(
                RosbagToLerobotConverter._segment_encoder_opts(
                    "libx264",
                    ["-preset", "ultrafast", "-g", "1"],
                ),
                ["-preset", "ultrafast", "-g", "1"],
            )

    def test_v30_direct_aggregate_removes_default_x264_all_intra_gop(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                RosbagToLerobotV30Converter._direct_aggregate_encoder_opts(
                    "libx264",
                    ["-preset", "ultrafast", "-g", "1", "-qp", "51"],
                ),
                ["-preset", "ultrafast", "-qp", "51"],
            )
        with patch.dict(os.environ, {"CYCLO_X264_GOP": "1"}, clear=True):
            self.assertEqual(
                RosbagToLerobotV30Converter._direct_aggregate_encoder_opts(
                    "libx264",
                    ["-preset", "ultrafast", "-g", "1"],
                ),
                ["-preset", "ultrafast", "-g", "1"],
            )

    def test_v21_direct_video_restores_use_videos_after_parse_error(self):
        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
            fps=15,
            use_videos=True,
        )
        converter = RosbagToLerobotConverter(config)
        with patch.object(
            converter,
            "_can_use_direct_video_output",
            return_value=True,
        ), patch.object(
            converter,
            "_try_load_prepared_episode_for_bag",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                converter.convert_multiple_rosbags([Path("/bags/0"), Path("/bags/1")])

        self.assertTrue(converter.config.use_videos)

    def test_v21_direct_video_stores_and_reuses_source_synced_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            source_dir.mkdir()
            source_video = source_dir / "cam.mp4"
            source_video.write_bytes(b"raw-video")
            (source_dir / "cam_timestamps.parquet").write_bytes(b"sidecar")

            config = ConversionConfig(
                repo_id="test/dataset",
                output_dir=tmp / "out",
                fps=15,
                use_videos=True,
            )
            converter = RosbagToLerobotConverter(config)
            converter._direct_v21_video_stats_cache = {}
            episode = EpisodeData(
                episode_index=0,
                length=2,
                grid_log_times_sec=[0.0, 1.0 / 15.0],
            )
            indices = np.asarray([0, 1], dtype=np.int64)
            encoded = tmp / "encoded.mp4"
            encoded.write_bytes(b"encoded-video")

            converter._store_direct_v21_synced_cache(
                episode=episode,
                camera_name="cam",
                video_path=source_video,
                indices=indices,
                output_path=encoded,
                stats={"mean": [0.5], "count": [2]},
                width=640,
                height=480,
            )

            cache_video = source_dir / "cam_synced.mp4"
            cache_sidecar = source_dir / "cam_synced.cache.json"
            self.assertTrue(cache_video.exists())
            self.assertTrue(cache_sidecar.exists())
            self.assertEqual(cache_video.read_bytes(), b"encoded-video")
            self.assertIn("cam", json.loads((source_dir / "video_stats.json").read_text()))

            with patch.object(
                converter,
                "_grid_indices_for_raw_video",
                return_value=indices,
            ), patch(
                "cyclo_data.converter.to_lerobot_v21._validated_video_count"
            ) as validate:
                reused = converter._try_reuse_direct_v21_synced_cache(
                    tmp / "reuse",
                    episode,
                    "cam",
                    source_video,
                )

            validate.assert_called_once()
            self.assertIsNotNone(reused)
            self.assertEqual(Path(reused).read_bytes(), b"encoded-video")
            self.assertIn((0, "cam"), converter._direct_v21_video_stats_cache)

    def test_v21_direct_video_cache_metadata_hit_skips_grid_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            source_dir.mkdir()
            source_video = source_dir / "cam.mp4"
            source_video.write_bytes(b"raw-video")
            (source_dir / "cam_timestamps.parquet").write_bytes(b"sidecar")

            config = ConversionConfig(
                repo_id="test/dataset",
                output_dir=tmp / "out",
                fps=15,
                use_videos=True,
            )
            converter = RosbagToLerobotConverter(config)
            converter._direct_v21_video_stats_cache = {}
            episode = EpisodeData(
                episode_index=0,
                length=2,
                grid_log_times_sec=[0.0, 1.0 / 15.0],
            )
            indices = np.asarray([0, 1], dtype=np.int64)
            encoded = tmp / "encoded.mp4"
            encoded.write_bytes(b"encoded-video")

            converter._store_direct_v21_synced_cache(
                episode=episode,
                camera_name="cam",
                video_path=source_video,
                indices=indices,
                output_path=encoded,
                stats=None,
                width=640,
                height=480,
            )

            with patch.object(
                converter,
                "_grid_indices_for_raw_video",
                side_effect=AssertionError("metadata hit should not scan sidecar"),
            ), patch(
                "cyclo_data.converter.to_lerobot_v21._validated_video_count"
            ) as validate:
                reused = converter._try_reuse_direct_v21_synced_cache(
                    tmp / "reuse",
                    episode,
                    "cam",
                    source_video,
                )

            self.assertIsNotNone(reused)
            self.assertEqual(Path(reused).read_bytes(), b"encoded-video")
            self.assertEqual(validate.call_args.kwargs["expected_frames"], 2)
            with patch.object(
                RosbagToLerobotConverterBase,
                "_get_video_info",
                side_effect=AssertionError("cached direct info expected"),
            ):
                info = converter._get_video_info(Path(reused))
            self.assertEqual(info["video.width"], 640)
            self.assertEqual(info["video.height"], 480)

    def test_v21_direct_video_cache_validation_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            source_dir.mkdir()
            source_video = source_dir / "cam.mp4"
            source_video.write_bytes(b"raw-video")
            (source_dir / "cam_timestamps.parquet").write_bytes(b"sidecar")

            config = ConversionConfig(
                repo_id="test/dataset",
                output_dir=tmp / "out",
                fps=15,
                use_videos=True,
            )
            converter = RosbagToLerobotConverter(config)
            converter._direct_v21_video_stats_cache = {}
            episode = EpisodeData(episode_index=0, length=1)
            indices = np.asarray([0], dtype=np.int64)
            encoded = tmp / "encoded.mp4"
            encoded.write_bytes(b"encoded-video")
            converter._store_direct_v21_synced_cache(
                episode=episode,
                camera_name="cam",
                video_path=source_video,
                indices=indices,
                output_path=encoded,
                stats=None,
                width=640,
                height=480,
            )

            with patch.dict(
                os.environ,
                {"CYCLO_V21_VALIDATE_DIRECT_VIDEO": "0"},
                clear=True,
            ), patch.object(
                converter,
                "_grid_indices_for_raw_video",
                return_value=indices,
            ), patch(
                "cyclo_data.converter.to_lerobot_v21._validated_video_count"
            ) as validate:
                reused = converter._try_reuse_direct_v21_synced_cache(
                    tmp / "reuse",
                    episode,
                    "cam",
                    source_video,
                )

            self.assertIsNotNone(reused)
            validate.assert_not_called()

    def test_v21_direct_video_cache_rejects_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            source_dir.mkdir()
            source_video = source_dir / "cam.mp4"
            source_video.write_bytes(b"raw-video")
            (source_dir / "cam_timestamps.parquet").write_bytes(b"sidecar")

            config = ConversionConfig(
                repo_id="test/dataset",
                output_dir=tmp / "out",
                fps=15,
                use_videos=True,
            )
            converter = RosbagToLerobotConverter(config)
            converter._direct_v21_video_stats_cache = {}
            episode = EpisodeData(episode_index=0, length=1)
            indices = np.asarray([0], dtype=np.int64)
            encoded = tmp / "encoded.mp4"
            encoded.write_bytes(b"encoded-video")
            converter._store_direct_v21_synced_cache(
                episode=episode,
                camera_name="cam",
                video_path=source_video,
                indices=indices,
                output_path=encoded,
                stats=None,
                width=640,
                height=480,
            )
            source_video.write_bytes(b"changed-video")

            with patch.object(
                converter,
                "_grid_indices_for_raw_video",
                return_value=indices,
            ), patch(
                "cyclo_data.converter.to_lerobot_v21._validated_video_count"
            ) as validate:
                reused = converter._try_reuse_direct_v21_synced_cache(
                    tmp / "reuse",
                    episode,
                    "cam",
                    source_video,
                )

            self.assertIsNone(reused)
            validate.assert_not_called()

    def test_v21_direct_video_retries_per_file_decoder_after_concat_error(self):
        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
            fps=15,
            use_videos=True,
        )
        converter = RosbagToLerobotConverter(config)
        pairs = [
            (EpisodeData(episode_index=0), Path("/tmp/cam0.mp4")),
            (EpisodeData(episode_index=1), Path("/tmp/cam1.mp4")),
        ]
        expected = {0: Path("/tmp/out0.mp4"), 1: Path("/tmp/out1.mp4")}

        with patch.object(
            converter,
            "_write_direct_camera_segments",
            side_effect=[
                v21._DirectConcatDecoderError("concat rejected input"),
                expected,
            ],
        ) as write_batch, patch.object(converter, "_log_warning") as log_warning:
            result = converter._write_direct_camera_segments_with_retry(
                Path("/tmp/out"),
                "cam",
                pairs,
            )

        self.assertEqual(result, expected)
        self.assertEqual(write_batch.call_count, 2)
        self.assertTrue(write_batch.call_args_list[0].kwargs["use_concat_decoder"])
        self.assertFalse(write_batch.call_args_list[1].kwargs["use_concat_decoder"])
        log_warning.assert_called_once()

    def test_v21_concat_decoder_coalesces_contiguous_splice_runs(self):
        class FakeDecoder:
            def __init__(self, payload: bytes):
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()

            def poll(self):
                return 0

            def kill(self):
                pass

        config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path("/tmp/test"),
            fps=15,
            use_videos=True,
        )
        converter = RosbagToLerobotConverter(config)
        frame_size = 6
        frames = [bytes([idx]) * frame_size for idx in range(5)]
        output = io.BytesIO()
        splice_sizes = []

        def fake_splice(src, dst, size):
            data = src.read(size)
            dst.write(data)
            splice_sizes.append(size)
            return len(data)

        with patch(
            "cyclo_data.converter.to_lerobot_v21.subprocess.Popen",
            return_value=FakeDecoder(b"".join(frames)),
        ), patch(
            "cyclo_data.converter.to_lerobot_v21._splice_exact",
            side_effect=fake_splice,
        ):
            written = converter._pipe_selected_yuv420_frames_concat_decoder(
                "ffmpeg",
                [
                    (
                        EpisodeData(episode_index=0),
                        Path("a.mp4"),
                        np.array([0, 1, 2, 4]),
                        5,
                        None,
                    ),
                ],
                frame_size,
                output,
                width=2,
                height=2,
            )

        self.assertEqual(written, 4)
        self.assertEqual(
            output.getvalue(),
            frames[0] + frames[1] + frames[2] + frames[4],
        )
        self.assertEqual(splice_sizes, [frame_size * 3, frame_size])

    def test_v21_raw_video_grid_mapping_uses_header_timestamp_order(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = tmp / "cam.mp4"
            video_path.write_bytes(b"video")
            pq.write_table(
                pa.table({
                    "frame_index": pa.array([0, 1, 2], type=pa.int32()),
                    "header_stamp_ns": pa.array([0, 100, 200], type=pa.int64()),
                    "recv_ns": pa.array([200, 0, 100], type=pa.int64()),
                }),
                tmp / "cam_timestamps.parquet",
            )
            converter = RosbagToLerobotConverter(
                ConversionConfig(
                    repo_id="test/dataset",
                    output_dir=tmp / "out",
                    fps=15,
                    use_videos=True,
                )
            )
            episode = EpisodeData(
                episode_index=0,
                length=3,
                grid_log_times_sec=[0.0, 0.0000001, 0.0000002],
            )

            indices, source_count = (
                converter._grid_indices_and_source_count_for_raw_video(
                    episode,
                    "cam",
                    video_path,
                )
            )

        self.assertEqual(source_count, 3)
        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_frame_reuse_report_compresses_duplicate_runs(self):
        from cyclo_data.reader.frame_timestamps import (
            FrameTimestamps,
            build_frame_reuse_report,
        )

        ft = FrameTimestamps(
            camera="cam",
            frame_index=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            header_stamp_ns=np.array([1, 10, 20, 30, 40, 50], dtype=np.int64),
            recv_ns=np.array([101, 110, 120, 130, 140, 150], dtype=np.int64),
        )

        report = build_frame_reuse_report(
            np.array([0, 0, 1, 2, 2, 2, 5], dtype=np.int64),
            np.array([1, 5, 10, 20, 25, 26, 50], dtype=np.int64),
            ft,
            episode_index=7,
            camera="cam",
            fps=15,
        )

        self.assertEqual(report["episode_index"], 7)
        self.assertEqual(report["camera"], "cam")
        self.assertEqual(report["time_source"], "header")
        self.assertEqual(report["reused_target_frames"], 3)
        self.assertEqual(report["clamped_before_first_count"], 0)
        self.assertEqual(
            report["runs"],
            [
                {
                    "target_start_frame": 1,
                    "target_end_frame": 1,
                    "count": 1,
                    "source_frame_index": 0,
                    "source_header_stamp_ns": 1,
                    "source_recv_ns": 101,
                },
                {
                    "target_start_frame": 4,
                    "target_end_frame": 5,
                    "count": 2,
                    "source_frame_index": 2,
                    "source_header_stamp_ns": 20,
                    "source_recv_ns": 120,
                },
            ],
        )

    def test_frame_reuse_report_marks_recv_fallback_and_initial_clamp(self):
        from cyclo_data.reader.frame_timestamps import (
            FrameTimestamps,
            build_frame_reuse_report,
        )

        ft = FrameTimestamps(
            camera="cam",
            frame_index=np.array([0, 1], dtype=np.int32),
            header_stamp_ns=np.array([0, 0], dtype=np.int64),
            recv_ns=np.array([100, 200], dtype=np.int64),
        )
        grid_ns = np.array([50, 100, 200], dtype=np.int64)
        indices = ft.map_to_grid(grid_ns, time_source="header")

        report = build_frame_reuse_report(
            indices,
            grid_ns,
            ft,
            episode_index=0,
            camera="cam",
            fps=15,
        )

        self.assertEqual(indices.tolist(), [0, 0, 1])
        self.assertEqual(report["time_source"], "recv")
        self.assertEqual(report["clamped_before_first_count"], 1)
        self.assertEqual(report["runs"][0]["source_header_stamp_ns"], None)

    def test_frame_reuse_metadata_writer_overwrites_sorted_reports(self):
        from cyclo_data.reader.frame_timestamps import FrameTimestamps

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            converter = RosbagToLerobotConverter(
                ConversionConfig(
                    repo_id="test/dataset",
                    output_dir=tmp,
                    fps=15,
                    use_videos=True,
                )
            )
            ft = FrameTimestamps(
                camera="cam",
                frame_index=np.array([0, 1], dtype=np.int32),
                header_stamp_ns=np.array([1, 2], dtype=np.int64),
                recv_ns=np.array([1, 2], dtype=np.int64),
            )
            episode = EpisodeData(
                episode_index=2,
                length=3,
                grid_log_times_sec=[0.000000001, 0.000000001, 0.000000002],
            )
            converter._record_frame_reuse_report(
                episode=episode,
                camera_name="cam",
                indices=np.array([0, 0, 1], dtype=np.int64),
                grid_ns=np.array([1, 1, 2], dtype=np.int64),
                frame_timestamps=ft,
            )
            import pyarrow.parquet as pq

            reuse_path = tmp / "meta" / "frame_reuse.parquet"
            legacy_jsonl_path = tmp / "meta" / "frame_reuse.jsonl"
            legacy_gzip_path = tmp / "meta" / "frame_reuse.json.gz"
            reuse_path.parent.mkdir(parents=True)
            legacy_jsonl_path.write_text("old\n", encoding="utf-8")
            legacy_gzip_path.write_bytes(b"old\n")

            converter._write_frame_reuse_metadata(tmp)

            rows = pq.read_table(reuse_path).to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["episode_index"], 2)
            self.assertEqual(rows[0]["camera"], "cam")
            self.assertEqual(rows[0]["target_frame_index"], 1)
            self.assertNotIn("source_frame_index", rows[0])
            self.assertFalse(legacy_jsonl_path.exists())
            self.assertFalse(legacy_gzip_path.exists())

    def test_frame_reuse_metadata_writer_uses_episode_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            converter = RosbagToLerobotConverter(
                ConversionConfig(repo_id="test/dataset", output_dir=tmp, fps=15)
            )
            episode = EpisodeData(
                episode_index=4,
                length=3,
                frame_reuse_reports=[{
                    "episode_index": 4,
                    "camera": "cam_left_head",
                    "target_fps": 15,
                    "time_source": "header",
                    "total_target_frames": 3,
                    "total_source_frames": 2,
                    "reused_target_frames": 1,
                    "reuse_ratio": 1 / 3,
                    "clamped_before_first_count": 0,
                    "runs": [{
                        "target_start_frame": 2,
                        "target_end_frame": 2,
                        "count": 1,
                        "source_frame_index": 1,
                        "source_header_stamp_ns": 100,
                        "source_recv_ns": 200,
                    }],
                }],
            )

            converter._collect_episode_frame_reuse_reports([episode])
            converter._write_frame_reuse_metadata(tmp)

            import pyarrow.parquet as pq

            rows = pq.read_table(tmp / "meta" / "frame_reuse.parquet").to_pylist()
            self.assertEqual(
                rows,
                [{
                    "episode_index": 4,
                    "camera": "cam_left_head",
                    "target_frame_index": 2,
                }],
            )

    def test_frame_reuse_metadata_accepts_legacy_cached_episode_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            converter = RosbagToLerobotConverter(
                ConversionConfig(repo_id="test/dataset", output_dir=tmp, fps=15)
            )
            episode = EpisodeData(episode_index=1, length=3)
            delattr(episode, "frame_reuse_reports")

            converter._remember_episode_frame_reuse_report(
                episode,
                {
                    "episode_index": 1,
                    "camera": "cam_left_head",
                    "target_fps": 15,
                    "time_source": "header",
                    "total_target_frames": 3,
                    "total_source_frames": 2,
                    "reused_target_frames": 1,
                    "reuse_ratio": 1 / 3,
                    "clamped_before_first_count": 0,
                    "runs": [{
                        "target_start_frame": 1,
                        "target_end_frame": 1,
                        "count": 1,
                        "source_frame_index": 0,
                        "source_header_stamp_ns": 100,
                        "source_recv_ns": 200,
                    }],
                },
            )
            converter._collect_episode_frame_reuse_reports([episode])
            converter._write_frame_reuse_metadata(tmp)

            import pyarrow.parquet as pq

            rows = pq.read_table(tmp / "meta" / "frame_reuse.parquet").to_pylist()
            self.assertEqual(
                rows,
                [{
                    "episode_index": 1,
                    "camera": "cam_left_head",
                    "target_frame_index": 1,
                }],
            )

    def test_conversion_worker_max_profile_caps_at_sixteen(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ), patch.object(
            base_converter.os,
            "cpu_count",
            return_value=32,
        ):
            self.assertEqual(
                base_converter._resolve_conversion_worker_count(53),
                16,
            )

    def test_max_profile_suppresses_internal_info_logs_by_default(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertFalse(base_converter._converter_info_logs_enabled())
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_CONVERTER_INFO_LOGS": "1",
            },
            clear=True,
        ):
            self.assertTrue(base_converter._converter_info_logs_enabled())

    def test_max_profile_quiets_cli_input_listing_by_default(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertTrue(convert_cli.max_profile_quiet_info())
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_CONVERTER_INFO_LOGS": "1",
            },
            clear=True,
        ):
            self.assertFalse(convert_cli.max_profile_quiet_info())

    def test_cli_speed_profile_sets_encoder_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            convert_cli.apply_speed_profile("max")
            self.assertEqual(os.environ["CYCLO_X264_SPEED_PROFILE"], "max")

        with patch.dict(os.environ, {"CYCLO_X264_SPEED_PROFILE": "quality"}, clear=True):
            convert_cli.apply_speed_profile(None)
            self.assertEqual(os.environ["CYCLO_X264_SPEED_PROFILE"], "quality")

    def test_cli_h264_encoder_sets_encoder_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            convert_cli.apply_h264_encoder("software")
            self.assertEqual(os.environ["CYCLO_H264_ENCODER"], "software")

        with patch.dict(os.environ, {"CYCLO_H264_ENCODER": "libx264"}, clear=True):
            convert_cli.apply_h264_encoder(None)
            self.assertEqual(os.environ["CYCLO_H264_ENCODER"], "libx264")

    def test_cli_h264_encoder_label_mentions_max_libx264(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertEqual(
                convert_cli.active_h264_encoder_label(),
                "libx264 (max profile)",
            )

        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_H264_ENCODER": "software",
            },
            clear=True,
        ):
            self.assertEqual(
                convert_cli.active_h264_encoder_label(),
                "software",
            )

    def test_conversion_worker_env_override_can_exceed_default_cap(self):
        with patch.dict(
            os.environ,
            {"CYCLO_CONVERSION_MAX_WORKERS": "8"},
            clear=True,
        ):
            self.assertEqual(
                base_converter._resolve_conversion_worker_count(8),
                8,
            )

    def test_video_sync_staging_dir_hashes_source_video_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "bag" / "videos"
            source.mkdir(parents=True)
            staging = Path(tmpdir) / "stage"
            with patch.dict(
                os.environ,
                {"CYCLO_VIDEO_SYNC_STAGING_DIR": str(staging)},
                clear=True,
            ):
                out_dir = RosbagToLerobotConverterBase._video_sync_output_dir(
                    source
                )

            self.assertTrue(str(out_dir).startswith(str(staging)))
            self.assertTrue(out_dir.exists())
            self.assertNotEqual(out_dir, source)


class TestCloneOrCopyFile(unittest.TestCase):
    """Tests for reflink/copy helper used by video writers."""

    def test_same_path_returns_without_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "video.mp4"
            src.write_bytes(b"video")

            with patch.object(
                Path,
                "resolve",
                side_effect=AssertionError("resolve should not be needed"),
            ), patch.object(
                base_converter.shutil,
                "copyfile",
                side_effect=AssertionError("copy should not run"),
            ):
                mode = base_converter._clone_or_copy_file(src, src)

            self.assertEqual(mode, "same_path")

    def test_reflink_failure_falls_back_to_copyfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src.mp4"
            dst = tmp / "dst.mp4"
            src.write_bytes(b"video")

            fake_fcntl = types.SimpleNamespace(
                ioctl=MagicMock(side_effect=OSError("no reflink"))
            )
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), patch.object(
                base_converter.shutil,
                "copyfile",
                wraps=base_converter.shutil.copyfile,
            ) as mock_copy:
                mode = base_converter._clone_or_copy_file(src, dst)

            self.assertEqual(mode, "copy")
            mock_copy.assert_called_once()
            self.assertEqual(dst.read_bytes(), b"video")

    def test_reflink_failure_is_memoized_by_device_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src.mp4"
            dst_a = tmp / "dst-a.mp4"
            dst_b = tmp / "dst-b.mp4"
            src.write_bytes(b"video")

            unsupported_dev_pairs = set(
                base_converter._REFLINK_UNSUPPORTED_DEV_PAIRS
            )
            base_converter._REFLINK_UNSUPPORTED_DEV_PAIRS.clear()
            fake_fcntl = types.SimpleNamespace(
                ioctl=MagicMock(side_effect=OSError("no reflink"))
            )
            try:
                with patch.dict(sys.modules, {"fcntl": fake_fcntl}), patch.object(
                    base_converter.shutil,
                    "copyfile",
                    wraps=base_converter.shutil.copyfile,
                ) as mock_copy:
                    first = base_converter._clone_or_copy_file(src, dst_a)
                    second = base_converter._clone_or_copy_file(src, dst_b)
            finally:
                base_converter._REFLINK_UNSUPPORTED_DEV_PAIRS.clear()
                base_converter._REFLINK_UNSUPPORTED_DEV_PAIRS.update(
                    unsupported_dev_pairs
                )

            self.assertEqual(first, "copy")
            self.assertEqual(second, "copy")
            fake_fcntl.ioctl.assert_called_once()
            self.assertEqual(mock_copy.call_count, 2)

    def test_hardlink_mode_uses_os_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src.mp4"
            dst = tmp / "dst.mp4"
            src.write_bytes(b"video")

            with patch.dict(os.environ, {"CYCLO_VIDEO_COPY_MODE": "hardlink"}):
                mode = base_converter._clone_or_copy_file(src, dst)

            self.assertEqual(mode, "hardlink")
            self.assertEqual(dst.read_bytes(), b"video")
            self.assertEqual(src.stat().st_ino, dst.stat().st_ino)


class TestEpisodeExtractCache(unittest.TestCase):
    """Tests for persistent state/action extraction cache."""

    def test_cache_path_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            mcap = bag / "episode.mcap"
            mcap.write_bytes(b"one")
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            )

            first = converter._episode_extract_cache_path(bag, None, [])
            mcap.write_bytes(b"two")
            second = converter._episode_extract_cache_path(bag, None, [])

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)

    def test_extract_joint_data_uses_cache_before_opening_bag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            config = ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            writer = RosbagToLerobotConverterBase(config)
            writer._state_joint_names = ["joint_a"]
            writer._action_joint_names = ["joint_b"]
            cached = EpisodeData(
                episode_index=1,
                timestamps=[0.0],
                observation_state=[np.array([1.0], dtype=np.float32)],
                action=[np.array([2.0], dtype=np.float32)],
                video_files={"cam": Path("raw.mp4")},
                source_path=Path("source"),
                length=1,
            )
            cache_path = writer._episode_extract_cache_path(bag, None, [])
            self.assertIsNotNone(cache_path)
            writer._store_episode_extract_cache(
                cache_path,
                episode=cached,
                staleness_metrics={},
            )

            reader = RosbagToLerobotConverterBase(config)
            with patch(
                "cyclo_data.converter.base_converter.BagReader.open",
                side_effect=AssertionError("cache miss"),
            ):
                episode = reader._extract_joint_data(bag, 7, None, [])

            self.assertIsNotNone(episode)
            self.assertEqual(episode.episode_index, 7)
            self.assertEqual(episode.length, 1)
            self.assertEqual(episode.video_files, {})
            self.assertIsNone(episode.source_path)
            self.assertEqual(reader._state_joint_names, ["joint_a"])
            self.assertEqual(reader._action_joint_names, ["joint_b"])

    def test_extract_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            )

            with patch.dict(os.environ, {"CYCLO_EXTRACT_CACHE_DISABLE": "1"}):
                self.assertIsNone(
                    converter._episode_extract_cache_path(bag, None, [])
                )


class TestPreparedEpisodeCache(unittest.TestCase):
    """Tests for fully prepared episode cache."""

    def _make_cached_episode(self, bag: Path) -> EpisodeData:
        video_dir = bag / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        synced = video_dir / "cam_left_head_synced.mp4"
        synced.write_bytes(b"synced")
        return EpisodeData(
            episode_index=1,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            video_files={"cam_left_head": synced},
            tasks=["cached task"],
            length=1,
        )

    def test_convert_single_uses_prepared_cache_before_extract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            config = ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            writer = RosbagToLerobotConverterBase(config)
            episode = self._make_cached_episode(bag)
            cache_path = writer._prepared_episode_cache_path(bag, {}, None, [])
            self.assertIsNotNone(cache_path)
            writer._store_prepared_episode_cache(cache_path, episode)
            self.assertEqual(
                episode._cyclo_prepared_cache_signature["path"],
                str(cache_path),
            )

            reader = RosbagToLerobotConverterBase(config)
            with patch.object(
                reader,
                "_extract_joint_data",
                side_effect=AssertionError("prepared cache miss"),
            ):
                loaded = reader.convert_single_rosbag(bag, 7)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.episode_index, 7)
            self.assertIsNone(loaded.full_episode_index)
            self.assertEqual(loaded.length, 1)
            self.assertEqual(loaded.tasks, ["cached task"])
            self.assertEqual(
                loaded._cyclo_prepared_cache_signature["path"],
                str(cache_path),
            )
            self.assertEqual(
                loaded.video_files["cam_left_head"].name,
                "cam_left_head_synced.mp4",
            )

    def test_prepared_cache_rejects_changed_synced_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            config = ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            writer = RosbagToLerobotConverterBase(config)
            episode = self._make_cached_episode(bag)
            cache_path = writer._prepared_episode_cache_path(bag, {}, None, [])
            self.assertIsNotNone(cache_path)
            writer._store_prepared_episode_cache(cache_path, episode)

            episode.video_files["cam_left_head"].write_bytes(b"changed")
            reader = RosbagToLerobotConverterBase(config)
            self.assertIsNone(
                reader._load_prepared_episode_cache(
                    cache_path,
                    episode_index=1,
                    bag_path=bag,
                )
            )

    def test_parent_can_load_prepared_cache_without_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            config = ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            writer = RosbagToLerobotConverterBase(config)
            writer._state_joint_names = ["joint_a"]
            writer._action_joint_names = ["joint_b"]
            episode = self._make_cached_episode(bag)
            cache_path = writer._prepared_episode_cache_path(bag, {}, None, [])
            self.assertIsNotNone(cache_path)
            writer._store_prepared_episode_cache(cache_path, episode)

            reader = RosbagToLerobotConverterBase(config)
            loaded = reader._try_load_prepared_episode_for_bag(bag, 3)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.episode_index, 3)
            self.assertEqual(loaded.length, 1)
            self.assertEqual(reader._state_joint_names, ["joint_a"])
            self.assertEqual(reader._action_joint_names, ["joint_b"])

    def test_synced_cache_identity_ignores_metadata(self):
        cached = {
            "target_fps": 15,
            "frame_count": 2,
            "frame_indices_sha256": "abc",
            "output_height": 480,
            "output_width": 640,
        }
        desired = {
            "target_fps": 15,
            "frame_count": 2,
            "frame_indices_sha256": "abc",
        }

        self.assertTrue(
            RosbagToLerobotConverterBase._synced_cache_identity_matches(
                cached, desired
            )
        )

    def test_prepared_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(repo_id="test", output_dir=bag / "out", fps=15)
            )

            with patch.dict(
                os.environ,
                {"CYCLO_PREPARED_EPISODE_CACHE_DISABLE": "1"},
            ):
                self.assertIsNone(
                    converter._prepared_episode_cache_path(bag, {}, None, [])
                )


class TestVideoDisabledFastPath(unittest.TestCase):
    """Tests for skipping video work when videos are disabled."""

    def test_convert_single_skips_video_discovery_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "episode.mcap").write_bytes(b"fake")
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(
                    repo_id="test",
                    output_dir=bag / "out",
                    fps=15,
                    use_videos=False,
                )
            )
            extracted = EpisodeData(
                episode_index=0,
                timestamps=[0.0],
                observation_state=[np.array([1.0], dtype=np.float32)],
                action=[np.array([2.0], dtype=np.float32)],
                length=1,
            )

            with patch.object(
                converter, "_extract_joint_data", return_value=extracted
            ), patch.object(
                converter,
                "_find_video_files",
                side_effect=AssertionError("video discovery should be skipped"),
            ):
                episode = converter.convert_single_rosbag(bag, 0)

            self.assertIsNotNone(episode)
            self.assertEqual(episode.video_files, {})

    def test_archived_segments_skip_video_work_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "0_0.mcap").write_bytes(b"fake")
            (bag / "episode_info.json").write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "sub_task_instruction": "segment",
                                "frame_duration": [0.0, 0.1],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(
                    repo_id="test",
                    output_dir=bag / "out",
                    fps=10,
                    use_videos=False,
                )
            )
            segment_episode = EpisodeData(
                episode_index=0,
                timestamps=[0.0],
                observation_state=[np.array([1.0], dtype=np.float32)],
                action=[np.array([2.0], dtype=np.float32)],
                length=1,
            )

            with patch.object(
                converter, "_extract_joint_data", return_value=segment_episode
            ), patch.object(
                converter,
                "_find_segment_video_files",
                side_effect=AssertionError("segment video discovery should be skipped"),
            ), patch.object(
                converter,
                "_stitch_subtask_videos",
                side_effect=AssertionError("video stitching should be skipped"),
            ):
                episode = converter.convert_single_rosbag(bag, 0)

            self.assertIsNotNone(episode)
            self.assertEqual(episode.video_files, {})
            self.assertEqual(episode.length, 1)

    def test_single_archived_segment_reuses_episode_without_stitching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            (bag / "0_0.mcap").write_bytes(b"fake")
            video_path = bag / "videos" / "0_0" / "cam_left_head_synced.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"mp4")
            (bag / "episode_info.json").write_text(
                json.dumps(
                    {
                        "task_instruction": "main task",
                        "task_name": "task",
                        "segments": [
                            {
                                "sub_task_instruction": "single segment",
                                "frame_duration": [0.0, 0.2],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            converter = RosbagToLerobotConverterBase(
                ConversionConfig(
                    repo_id="test",
                    output_dir=bag / "out",
                    fps=10,
                    use_videos=True,
                )
            )
            segment_episode = EpisodeData(
                episode_index=99,
                timestamps=[0.0, 0.1],
                observation_state=[
                    np.array([1.0], dtype=np.float32),
                    np.array([2.0], dtype=np.float32),
                ],
                action=[
                    np.array([3.0], dtype=np.float32),
                    np.array([4.0], dtype=np.float32),
                ],
                length=2,
                video_files={"cam_left_head": video_path},
            )

            def sync_segment(_mcap_path, episode):
                episode.video_files = {"cam_left_head": video_path}
                return episode

            with patch.object(
                converter, "_extract_joint_data", return_value=segment_episode
            ), patch.object(
                converter,
                "_find_segment_video_files",
                return_value={"cam_left_head": video_path},
            ), patch.object(
                converter, "_sync_videos_to_grid", side_effect=sync_segment
            ), patch.object(
                converter,
                "_stitch_subtask_videos",
                side_effect=AssertionError("single segment should not stitch"),
            ):
                episode = converter.convert_single_rosbag(bag, 7)

            self.assertIs(episode, segment_episode)
            self.assertEqual(episode.episode_index, 7)
            self.assertEqual(episode.length, 2)
            self.assertEqual(episode.video_files, {"cam_left_head": video_path})
            self.assertEqual(episode.subtask_indices, [0, 0])
            self.assertEqual(
                episode.subtask_segments[0]["sub_task_instruction"],
                "single segment",
            )


class TestSubtaskStitching(unittest.TestCase):
    """Tests for recording-time subtask episodes stitched at conversion time."""

    def _make_subtask_episode(self, raw_idx, full_idx, sub_idx, total, values):
        return EpisodeData(
            episode_index=raw_idx,
            timestamps=[0.0, 0.1],
            observation_state=[
                np.array([values[0]], dtype=np.float32),
                np.array([values[1]], dtype=np.float32),
            ],
            action=[
                np.array([values[0] + 10], dtype=np.float32),
                np.array([values[1] + 10], dtype=np.float32),
            ],
            tasks=["main task"],
            length=2,
            recording_mode="subtask",
            full_episode_index=full_idx,
            subtask_index=sub_idx,
            subtask_total=total,
            subtask_instruction=f"subtask {sub_idx}",
        )

    def test_complete_subtasks_stitch_into_one_episode(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )
        episodes = [
            self._make_subtask_episode(0, 0, 0, 3, [1, 2]),
            self._make_subtask_episode(1, 0, 1, 3, [3, 4]),
            self._make_subtask_episode(2, 0, 2, 3, [5, 6]),
        ]

        stitched = converter.prepare_episodes_for_writing(episodes)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0].episode_index, 0)
        self.assertEqual(stitched[0].recording_mode, "stitched_subtask")
        self.assertEqual(stitched[0].tasks, ["main task"])
        self.assertEqual(stitched[0].subtask_instructions, [
            "subtask 0", "subtask 1", "subtask 2",
        ])
        self.assertEqual(stitched[0].length, 6)
        self.assertEqual([round(t, 3) for t in stitched[0].timestamps], [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        ])
        self.assertEqual(
            [float(state[0]) for state in stitched[0].observation_state],
            [1, 2, 3, 4, 5, 6],
        )

    def test_complete_subtasks_shift_frame_reuse_reports(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )
        episodes = [
            self._make_subtask_episode(0, 0, 0, 2, [1, 2]),
            self._make_subtask_episode(1, 0, 1, 2, [3, 4]),
        ]
        for ep in episodes:
            ep.frame_reuse_reports = [{
                "episode_index": int(ep.episode_index),
                "camera": "cam_left_head",
                "target_fps": 10,
                "time_source": "header",
                "total_target_frames": 2,
                "total_source_frames": 1,
                "reused_target_frames": 1,
                "reuse_ratio": 0.5,
                "clamped_before_first_count": 0,
                "runs": [{
                    "target_start_frame": 1,
                    "target_end_frame": 1,
                    "count": 1,
                    "source_frame_index": 0,
                    "source_header_stamp_ns": 100,
                    "source_recv_ns": 200,
                }],
            }]

        stitched = converter.prepare_episodes_for_writing(episodes)

        self.assertEqual(len(stitched), 1)
        reports = stitched[0].frame_reuse_reports
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["episode_index"], 0)
        self.assertEqual(reports[0]["camera"], "cam_left_head")
        self.assertEqual(reports[0]["total_target_frames"], 4)
        self.assertEqual(reports[0]["reused_target_frames"], 2)
        self.assertEqual(
            [
                (
                    run["target_start_frame"],
                    run["target_end_frame"],
                )
                for run in reports[0]["runs"]
            ],
            [(1, 1), (3, 3)],
        )

    def test_complete_subtasks_shift_frame_reuse_reports_without_off_by_one(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )

        def make_episode(raw_idx, sub_idx, length):
            return EpisodeData(
                episode_index=raw_idx,
                timestamps=[idx / 10.0 for idx in range(length)],
                observation_state=[
                    np.array([idx], dtype=np.float32) for idx in range(length)
                ],
                action=[
                    np.array([idx + 10], dtype=np.float32) for idx in range(length)
                ],
                tasks=["main task"],
                length=length,
                recording_mode="subtask",
                full_episode_index=0,
                subtask_index=sub_idx,
                subtask_total=2,
                subtask_instruction=f"subtask {sub_idx}",
            )

        episodes = [
            make_episode(0, 0, 100),
            make_episode(1, 1, 140),
        ]
        episodes[0].frame_reuse_reports = [{
            "episode_index": 0,
            "camera": "cam_left_head",
            "target_fps": 10,
            "time_source": "header",
            "total_target_frames": 100,
            "total_source_frames": 99,
            "reused_target_frames": 1,
            "reuse_ratio": 0.01,
            "clamped_before_first_count": 0,
            "runs": [{
                "target_start_frame": 99,
                "target_end_frame": 99,
                "count": 1,
                "source_frame_index": 98,
                "source_header_stamp_ns": 100,
                "source_recv_ns": 200,
            }],
        }]
        episodes[1].frame_reuse_reports = [{
            "episode_index": 1,
            "camera": "cam_left_head",
            "target_fps": 10,
            "time_source": "header",
            "total_target_frames": 140,
            "total_source_frames": 137,
            "reused_target_frames": 4,
            "reuse_ratio": 4 / 140,
            "clamped_before_first_count": 0,
            "runs": [
                {
                    "target_start_frame": 70,
                    "target_end_frame": 72,
                    "count": 3,
                    "source_frame_index": 69,
                    "source_header_stamp_ns": 300,
                    "source_recv_ns": 400,
                },
                {
                    "target_start_frame": 139,
                    "target_end_frame": 139,
                    "count": 1,
                    "source_frame_index": 138,
                    "source_header_stamp_ns": 500,
                    "source_recv_ns": 600,
                },
            ],
        }]

        stitched = converter.prepare_episodes_for_writing(episodes)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0].length, 240)
        report = stitched[0].frame_reuse_reports[0]
        self.assertEqual(report["total_target_frames"], 240)
        self.assertEqual(report["reused_target_frames"], 5)
        self.assertEqual(
            [
                (
                    run["target_start_frame"],
                    run["target_end_frame"],
                    run["count"],
                )
                for run in report["runs"]
            ],
            [
                (99, 99, 1),
                (170, 172, 3),
                (239, 239, 1),
            ],
        )

    def test_incomplete_subtask_group_is_skipped(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )
        episodes = [
            self._make_subtask_episode(0, 0, 0, 3, [1, 2]),
            self._make_subtask_episode(1, 0, 2, 3, [5, 6]),
        ]

        stitched = converter.prepare_episodes_for_writing(episodes)

        self.assertEqual(stitched, [])

    def test_mixed_single_and_subtask_counts_are_rejected(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )
        single = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["main task"],
            length=1,
        )
        subtask = EpisodeData(
            episode_index=1,
            timestamps=[0.0],
            observation_state=[np.array([3.0], dtype=np.float32)],
            action=[np.array([4.0], dtype=np.float32)],
            tasks=["main task"],
            length=1,
            subtask_segments=[
                {
                    "subtask_index": 0,
                    "sub_task_instruction": "subtask",
                    "frame_duration": [0.0, 0.1],
                }
            ],
        )

        prepared = converter.prepare_episodes_for_writing([single, subtask])

        self.assertEqual(prepared, [])

    def test_matching_single_segment_counts_are_allowed(self):
        converter = RosbagToLerobotConverter(
            ConversionConfig(repo_id="test", output_dir=Path("/tmp/out"), fps=10)
        )
        episodes = [
            EpisodeData(
                episode_index=idx,
                timestamps=[0.0],
                observation_state=[np.array([float(idx)], dtype=np.float32)],
                action=[np.array([float(idx)], dtype=np.float32)],
                tasks=["main task"],
                length=1,
                subtask_segments=[
                    {
                        "subtask_index": 0,
                        "sub_task_instruction": "main task",
                        "frame_duration": [0.0, 0.1],
                    }
                ],
                subtask_indices=[0],
            )
            for idx in range(2)
        ]

        prepared = converter.prepare_episodes_for_writing(episodes)

        self.assertEqual(len(prepared), 2)
        self.assertEqual([ep.episode_index for ep in prepared], [0, 1])


class TestRosbagToLerobotConverter(unittest.TestCase):
    """Tests for RosbagToLerobotConverter class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
        )
        self.converter = RosbagToLerobotConverter(self.config)

    def test_initialization(self):
        self.assertEqual(self.converter.config.repo_id, "test/dataset")
        self.assertEqual(self.converter._total_episodes, 0)
        self.assertEqual(self.converter._total_frames, 0)

    def test_single_segment_archive_is_detected(self):
        episode_dir = Path(self.temp_dir) / "0"
        episode_dir.mkdir()
        (episode_dir / "0_0.mcap").touch()
        episode_info = {
            "segments": [
                {
                    "sub_task_instruction": "main task",
                    "frame_duration": [0.0, 1.0],
                }
            ]
        }

        self.assertTrue(
            self.converter._is_archived_segment_episode(
                episode_dir,
                episode_info,
            )
        )

    def test_empty_segments_remain_legacy(self):
        episode_dir = Path(self.temp_dir) / "0"
        episode_dir.mkdir(exist_ok=True)
        (episode_dir / "episode.mcap").touch()

        self.assertFalse(
            self.converter._is_archived_segment_episode(
                episode_dir,
                {"segments": []},
            )
        )

    def test_get_topic_group_key_basic(self):
        result = self.converter._get_topic_group_key(
            "/robot/arm_left_follower/joint_states", "state")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("follower", result)

    def test_is_in_exclude_region(self):
        exclude_regions = [
            {"start": {"time": 1.0}, "end": {"time": 2.0}},
            {"start": {"time": 5.0}, "end": {"time": 6.0}},
        ]

        self.assertFalse(self.converter._is_in_exclude_region(0.5, exclude_regions))
        self.assertTrue(self.converter._is_in_exclude_region(1.5, exclude_regions))
        self.assertFalse(self.converter._is_in_exclude_region(3.0, exclude_regions))
        self.assertTrue(self.converter._is_in_exclude_region(5.5, exclude_regions))
        self.assertFalse(self.converter._is_in_exclude_region(7.0, exclude_regions))

    def test_find_previous_value_in_list(self):
        messages = [
            (0.0, np.array([1.0])),
            (1.0, np.array([2.0])),
            (2.0, np.array([3.0])),
        ]

        result, staleness = self.converter._find_previous_value_in_list(messages, 0.4)
        np.testing.assert_array_equal(result, np.array([1.0]))

        result, staleness = self.converter._find_previous_value_in_list(messages, 1.5)
        np.testing.assert_array_equal(result, np.array([2.0]))

        result, staleness = self.converter._find_previous_value_in_list(messages, 2.0)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_find_previous_value_in_list_empty(self):
        result, staleness = self.converter._find_previous_value_in_list([], 1.0)
        self.assertIsNone(result)

    def test_build_features(self):
        episodes = [
            EpisodeData(
                episode_index=0,
                observation_state=[np.array([1.0, 2.0, 3.0])],
                action=[np.array([0.1, 0.2, 0.3])],
                length=1,
            )
        ]
        self.converter._state_joint_names = ["j1", "j2", "j3"]
        self.converter._action_joint_names = ["a1", "a2", "a3"]

        self.converter._build_features(episodes)

        self.assertIn("observation.state", self.converter._features)
        self.assertIn("action", self.converter._features)
        self.assertEqual(self.converter._features["observation.state"]["shape"], (3,))
        self.assertEqual(self.converter._features["action"]["shape"], (3,))

    def test_get_video_info_uses_synced_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "cam_left_head_synced.mp4"
            video_path.write_bytes(b"video")
            video_path.with_name(video_path.stem + ".cache.json").write_text(
                json.dumps({
                    "target_fps": self.converter.config.fps,
                    "output_height": 480,
                    "output_width": 640,
                    "output_codec": "h264",
                    "output_pix_fmt": "yuv420p",
                    "has_audio": False,
                }),
                encoding="utf-8",
            )

            with patch.object(
                self.converter,
                "_get_video_dimensions",
                side_effect=AssertionError("dimensions should come from cache"),
            ), patch(
                "cyclo_data.converter.base_converter.subprocess.run",
                side_effect=AssertionError("ffprobe should not run"),
            ):
                info = self.converter._get_video_info(video_path)

        self.assertEqual(info["video.height"], 480)
        self.assertEqual(info["video.width"], 640)
        self.assertEqual(info["video.codec"], "h264")
        self.assertEqual(info["video.pix_fmt"], "yuv420p")

    def test_v21_video_feature_key_uses_rgb_prefix(self):
        self.assertEqual(
            self.converter._video_feature_key("cam_left_head"),
            "observation.images.rgb.cam_left_head",
        )

    @patch("cyclo_data.converter.base_converter.subprocess.run")
    def test_segment_video_prepare_uses_stream_copy_when_compatible(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            list_path = tmp / "concat.txt"
            list_path.write_text("", encoding="utf-8")
            srcs = [tmp / "a.mp4", tmp / "b.mp4"]
            for src in srcs:
                src.write_bytes(b"mp4")
            out_path = tmp / "out.mp4"

            def fake_run(cmd, *args, **kwargs):
                out_path.write_bytes(b"copy")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            mock_run.side_effect = fake_run
            with patch.object(
                self.converter,
                "_segment_copy_compatibility",
                return_value=(5, 300_000_000),
            ), patch.object(
                self.converter, "_get_video_frame_count", return_value=5
            ), patch.object(
                self.converter, "_video_decodes_successfully", return_value=True
            ):
                result = self.converter._try_prepare_segment_video_copy(
                    "ffmpeg", list_path, srcs, out_path
                )

            cmd = mock_run.call_args.args[0]
            self.assertTrue(result)
            self.assertIn("-c:v", cmd)
            self.assertIn("copy", cmd)

    @patch("cyclo_data.converter.base_converter.subprocess.run")
    def test_segment_video_copy_skips_tiny_clips(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            list_path = tmp / "concat.txt"
            srcs = [tmp / "a.mp4"]
            srcs[0].write_bytes(b"mp4")
            out_path = tmp / "out.mp4"

            with patch.object(
                self.converter,
                "_segment_copy_compatibility",
                return_value=(5, 1),
            ):
                result = self.converter._try_prepare_segment_video_copy(
                    "ffmpeg", list_path, srcs, out_path
                )

            self.assertFalse(result)
            mock_run.assert_not_called()

    @patch("cyclo_data.converter.base_converter.subprocess.run")
    def test_single_subtask_stitch_reuses_synced_video(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "cam_left_head_synced.mp4"
            video_path.write_bytes(b"mp4")
            episode = EpisodeData(
                episode_index=0,
                video_files={"cam_left_head": video_path},
            )

            stitched = self.converter._stitch_subtask_videos(0, [episode])

            self.assertEqual(stitched, {"cam_left_head": video_path})
            mock_run.assert_not_called()

    @patch("cyclo_data.converter.base_converter.subprocess.run")
    def test_multi_subtask_stitch_tries_stream_copy(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_a = tmp / "a.mp4"
            video_b = tmp / "b.mp4"
            video_a.write_bytes(b"a")
            video_b.write_bytes(b"b")
            episodes = [
                EpisodeData(episode_index=0, video_files={"cam_left_head": video_a}),
                EpisodeData(episode_index=0, video_files={"cam_left_head": video_b}),
            ]

            with patch.object(
                self.converter,
                "_try_prepare_segment_video_copy",
                return_value=True,
            ) as mock_copy:
                stitched = self.converter._stitch_subtask_videos(0, episodes)

            self.assertIn("cam_left_head", stitched)
            self.assertTrue(stitched["cam_left_head"].name.endswith(".mp4"))
            mock_copy.assert_called_once()
            mock_run.assert_not_called()

    def test_synced_video_cache_is_kept_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            synced = tmp / "cam_left_head_synced.mp4"
            cache = tmp / "cam_left_head_synced.cache.json"
            synced.write_bytes(b"mp4")
            cache.write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                removed = self.converter._cleanup_source_synced_cache([tmp])

            self.assertEqual(removed, 0)
            self.assertTrue(synced.exists())
            self.assertTrue(cache.exists())

    def test_synced_video_cache_cleanup_can_be_forced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            synced = tmp / "cam_left_head_synced.mp4"
            cache = tmp / "cam_left_head_synced.cache.json"
            synced.write_bytes(b"mp4")
            cache.write_text("{}", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"CYCLO_VIDEO_SYNC_CLEAN_CACHE": "1"},
                clear=True,
            ):
                removed = self.converter._cleanup_source_synced_cache([tmp])

            self.assertEqual(removed, 2)
            self.assertFalse(synced.exists())
            self.assertFalse(cache.exists())

    def test_stitched_video_stats_merge_without_decode(self):
        left = {
            "min": [[[0.0]], [[0.1]], [[0.2]]],
            "max": [[[0.4]], [[0.5]], [[0.6]]],
            "mean": [[[0.2]], [[0.3]], [[0.4]]],
            "std": [[[0.1]], [[0.1]], [[0.1]]],
            "count": [10],
        }
        right = {
            "min": [[[0.1]], [[0.0]], [[0.3]]],
            "max": [[[0.8]], [[0.7]], [[0.9]]],
            "mean": [[[0.6]], [[0.5]], [[0.7]]],
            "std": [[[0.2]], [[0.2]], [[0.2]]],
            "count": [10],
        }

        merged = self.converter._merge_video_stats([left, right])

        self.assertEqual(merged["count"], [20])
        self.assertEqual(merged["min"], [[[0.0]], [[0.0]], [[0.2]]])
        self.assertEqual(merged["max"], [[[0.8]], [[0.7]], [[0.9]]])
        np.testing.assert_allclose(
            np.asarray(merged["mean"]).reshape(3),
            np.array([0.4, 0.4, 0.55]),
        )

    def test_stitched_video_stats_are_cached_from_source_stats(self):
        source_stats = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.1]], [[0.1]], [[0.1]]],
            "count": [5],
        }
        out_path = Path(self.temp_dir) / "stitched.mp4"

        with patch.object(
            self.converter,
            "_load_precomputed_video_stats",
            return_value=source_stats,
        ) as mock_load, patch.object(
            self.converter,
            "_store_video_stats_cached",
        ) as mock_store:
            self.converter._store_stitched_video_stats_from_sources(
                out_path,
                "cam_left_head",
                [Path("a.mp4"), Path("b.mp4")],
            )

        self.assertEqual(mock_load.call_count, 2)
        mock_store.assert_called_once()

    def test_video_stats_samples_zero_skips_lazy_decode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "cam_left_head_synced.mp4"
            video_path.write_bytes(b"mp4")

            with patch.dict(
                os.environ,
                {"CYCLO_VIDEO_STATS_SAMPLES": "0"},
            ), patch.object(
                self.converter,
                "_load_precomputed_video_stats",
                return_value=None,
            ), patch.dict(
                sys.modules,
                {"cv2": None},
            ):
                self.assertIsNone(
                    self.converter._compute_video_stats(
                        video_path,
                        "cam_left_head",
                    )
                )

    def test_segment_copy_eligibility_rejects_non_h264_and_audio(self):
        h264_stream = {
            "codec_type": "video",
            "codec_name": "h264",
            "has_b_frames": 0,
            "width": 64,
            "height": 48,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": "15/1",
            "time_base": "1/15360",
        }

        with patch.object(
            self.converter,
            "_probe_video_streams",
            return_value={"streams": [h264_stream]},
        ), patch.object(
            self.converter, "_get_video_frame_count", return_value=2
        ):
            self.assertTrue(
                self.converter._segment_videos_support_copy_concat(
                    [Path("a.mp4"), Path("b.mp4")]
                )
            )

        with patch.object(
            self.converter,
            "_probe_video_streams",
            return_value={
                "streams": [
                    {**h264_stream, "codec_name": "mjpeg"},
                ]
            },
        ), patch.object(
            self.converter, "_get_video_frame_count", return_value=2
        ):
            self.assertFalse(
                self.converter._segment_videos_support_copy_concat(
                    [Path("a.mp4")]
                )
            )

        with patch.object(
            self.converter,
            "_probe_video_streams",
            return_value={
                "streams": [
                    h264_stream,
                    {"codec_type": "audio", "codec_name": "aac"},
                ]
            },
        ), patch.object(
            self.converter, "_get_video_frame_count", return_value=2
        ):
            self.assertFalse(
                self.converter._segment_videos_support_copy_concat(
                    [Path("a.mp4")]
                )
            )

    def test_v21_parquet_omits_frame_index(self):
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0 / 15.0],
            observation_state=[
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
            action=[
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.3, 0.4], dtype=np.float32),
            ],
            tasks=["task"],
            length=2,
            subtask_indices=[0, 0],
        )
        self.converter._features = {
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
        self.converter._task_to_index = {"task": 0}
        parquet_path = Path(self.temp_dir) / "episode.parquet"

        self.converter._write_parquet(episode, parquet_path)

        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        self.assertEqual(
            table.column_names,
            [
                "index",
                "episode_index",
                "task_index",
                "timestamp",
                "action",
                "observation.state",
                "subtask_index",
            ],
        )
        self.assertNotIn("frame_index", table.column_names)
        self.assertEqual(str(table.schema.field("timestamp").type), "double")

    def test_v21_episode_parquet_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0 / 15.0],
            observation_state=[
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ],
            action=[
                np.array([3.0], dtype=np.float32),
                np.array([4.0], dtype=np.float32),
            ],
            tasks=["task"],
            length=2,
            source_path=source_dir,
        )
        self.converter._features = {}
        self.converter._task_to_index = {"task": 0}
        self.converter._write_episode(episode)

        first_path = (
            Path(self.temp_dir)
            / "data/chunk-000/episode_000000.parquet"
        )
        original_bytes = first_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "episode_parquet_v21"
                ).glob("*/manifest.json")
            )
        )

        second_output = Path(self.temp_dir) / "second_output"
        self.converter.config.output_dir = second_output
        self.converter._total_frames = 0
        self.converter._total_episodes = 0
        self.converter._episodes = {}
        self.converter._episodes_stats = {}

        with patch.object(
            self.converter,
            "_write_parquet",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_episode(episode)

        second_path = (
            second_output
            / "data/chunk-000/episode_000000.parquet"
        )
        self.assertEqual(second_path.read_bytes(), original_bytes)

    def test_v21_parquet_cache_key_uses_prepared_cache_signature(self):
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
        )
        episode._cyclo_prepared_cache_signature = {
            "path": "/tmp/prepared.pickle",
            "size": 10,
            "mtime_ns": 20,
        }
        self.converter._task_to_index = {"task": 0}

        with patch.object(
            self.converter,
            "_array_cache_signature",
            side_effect=AssertionError("prepared cache should skip array hashing"),
        ):
            cache_key = self.converter._v21_episode_parquet_cache_key(
                episode,
                global_start_index=0,
                has_subtask_feature=False,
            )

        self.assertEqual(
            cache_key["episode"]["prepared_cache"]["path"],
            "/tmp/prepared.pickle",
        )
        self.assertNotIn("observation_state", cache_key["episode"])

    def test_v21_subtasks_parquet_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
            source_path=source_dir,
            subtask_instructions=["Pick"],
        )
        self.converter._write_subtasks_parquet(Path(self.temp_dir), [episode])

        subtasks_path = Path(self.temp_dir) / "meta/subtasks.parquet"
        original_bytes = subtasks_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "subtasks_parquet_v21"
                ).glob("*/manifest.json")
            )
        )

        subtasks_path.unlink()
        with patch(
            "cyclo_data.converter.to_lerobot_v21.pq.write_table",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_subtasks_parquet(Path(self.temp_dir), [episode])

        self.assertEqual(subtasks_path.read_bytes(), original_bytes)

    def test_v21_subtask_annotations_are_frame_based(self):
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0 / 15.0, 2.0 / 15.0],
            tasks=["main task"],
            length=3,
            subtask_segments=[
                {
                    "subtask_index": 0,
                    "sub_task_instruction": "main task",
                    "frame_duration": [0.0, 3.0 / 15.0],
                }
            ],
            subtask_indices=[0, 0, 0],
        )

        self.converter._write_subtask_annotations(Path(self.temp_dir), [episode])

        annotation_path = (
            Path(self.temp_dir)
            / "annotations/chunk-000/episode_000000.json"
        )
        with open(annotation_path, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["meta_data"]["task_duration"], 3)
        self.assertEqual(payload["meta_data"]["valid_duration"], [0, 3])
        self.assertNotIn("failure_recovery_annotation", payload)
        self.assertEqual(
            payload["sub_task_annotation"][0]["frame_duration"],
            [0, 3],
        )

    def test_failure_hint_uses_nearest_retained_grid_and_clamps(self):
        episode = EpisodeData(
            episode_index=0,
            length=3,
            grid_log_times_sec=[10.2, 10.3, 10.4],
        )
        info = {
            "failure_recovery_annotation": {
                "events": [
                    {
                        "event_id": "fr_000",
                        "sub_task_idx": 0,
                        "source_mark": {"source": "ui_failed", "time_sec": 0.31},
                    },
                    {
                        "event_id": "fr_001",
                        "sub_task_idx": 0,
                        "source_mark": {"source": "left_button", "time_sec": 99.0},
                    },
                ]
            }
        }

        annotation = self.converter._resolve_failure_annotation(
            info,
            episode,
            recording_start_sec=10.0,
        )

        self.assertEqual(
            [event["source_mark"]["hint_frame"] for event in annotation["events"]],
            [1, 2],
        )
        self.assertEqual(annotation["events"][0]["event_id"], "fr_000")
        self.assertIsNone(annotation["events"][0]["failure"]["frame_duration"])

    def test_failure_hint_is_resolved_after_video_alignment_shortens_grid(self):
        bag_path = Path(self.temp_dir) / "legacy_episode"
        bag_path.mkdir()
        (bag_path / "metadata.yaml").write_text(
            base_converter.yaml.safe_dump({
                "rosbag2_bagfile_information": {
                    "starting_time": {"nanoseconds_since_epoch": 10_000_000_000},
                }
            }),
            encoding="utf-8",
        )
        info = {
            "task_instruction": "task",
            "failure_recovery_annotation": {
                "events": [{
                    "event_id": "fr_000",
                    "sub_task_idx": 0,
                    "source_mark": {"source": "ui_failed", "time_sec": 0.2},
                }]
            },
        }
        extracted = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 0.1, 0.2],
            observation_state=[np.zeros(1)] * 3,
            action=[np.zeros(1)] * 3,
            grid_log_times_sec=[10.0, 10.1, 10.2],
            length=3,
        )

        def shorten_grid(_bag_path, episode):
            episode.timestamps = episode.timestamps[:2]
            episode.observation_state = episode.observation_state[:2]
            episode.action = episode.action[:2]
            episode.grid_log_times_sec = episode.grid_log_times_sec[:2]
            episode.length = 2
            return episode

        with patch.object(
            self.converter, "_can_convert_transcode_state", return_value=True
        ), patch.object(
            self.converter._metadata_manager, "load_robot_config", return_value=None
        ), patch.object(
            self.converter._metadata_manager, "get_trim_points", return_value=None
        ), patch.object(
            self.converter._metadata_manager, "get_exclude_regions", return_value=[]
        ), patch.object(
            self.converter._metadata_manager, "load_episode_info", return_value=info
        ), patch.object(
            self.converter._metadata_manager, "get_task_markers", return_value=[]
        ), patch.object(
            self.converter, "_prepared_episode_cache_path", return_value=None
        ), patch.object(
            self.converter, "_is_archived_segment_episode", return_value=False
        ), patch.object(
            self.converter, "_extract_joint_data", return_value=extracted
        ), patch.object(
            self.converter, "_find_video_files", return_value={"cam": bag_path / "cam.mp4"}
        ), patch.object(
            self.converter, "_sync_videos_to_grid", side_effect=shorten_grid
        ), patch.object(
            self.converter, "_episode_has_sidecars", return_value=True
        ):
            episode = self.converter.convert_single_rosbag(bag_path, 0)

        self.assertEqual(episode.length, 2)
        self.assertEqual(
            episode.failure_recovery_annotation["events"][0]["source_mark"]["hint_frame"],
            1,
        )

    def test_single_retained_archived_segment_keeps_original_subtask_index(self):
        bag_path = Path(self.temp_dir) / "0"
        bag_path.mkdir()
        mcaps = [bag_path / "0_0.mcap", bag_path / "0_1.mcap"]
        for mcap in mcaps:
            mcap.touch()
        (bag_path / "metadata.yaml").write_text(
            base_converter.yaml.safe_dump({
                "rosbag2_bagfile_information": {
                    "files": [
                        {
                            "path": "0_0.mcap",
                            "starting_time": {
                                "nanoseconds_since_epoch": 10_000_000_000,
                            },
                        },
                        {
                            "path": "0_1.mcap",
                            "starting_time": {
                                "nanoseconds_since_epoch": 20_000_000_000,
                            },
                        },
                    ],
                }
            }),
            encoding="utf-8",
        )
        info = {
            "segments": [
                {"sub_task_instruction": "first", "frame_duration": [0.0, 1.0]},
                {"sub_task_instruction": "second", "frame_duration": [1.0, 2.0]},
            ],
            "failure_recovery_annotation": {
                "events": [{
                    "event_id": "fr_000",
                    "sub_task_idx": 1,
                    "source_mark": {"source": "left_button", "time_sec": 1.5},
                }]
            },
        }
        converter = RosbagToLerobotConverter(
            ConversionConfig(
                repo_id="test/partial",
                output_dir=Path(self.temp_dir) / "partial",
                use_videos=False,
            )
        )
        retained = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 0.5],
            observation_state=[np.zeros(1)] * 2,
            action=[np.zeros(1)] * 2,
            grid_log_times_sec=[20.0, 20.5],
            length=2,
        )
        with patch.object(
            converter,
            "_extract_joint_data",
            side_effect=[None, retained],
        ):
            episode = converter._convert_archived_segment_episode(bag_path, 0, info)

        self.assertEqual(episode.subtask_index, 1)
        self.assertEqual(episode.subtask_indices, [1, 1])
        self.assertEqual(episode.subtask_segments[0]["subtask_index"], 1)
        self.assertEqual(
            episode.failure_recovery_annotation["events"][0]["source_mark"]["hint_frame"],
            1,
        )

    def test_failure_annotation_writer_is_optional(self):
        empty = EpisodeData(episode_index=0, tasks=["task"], length=2)
        failed = EpisodeData(
            episode_index=1,
            tasks=["task"],
            length=2,
            failure_recovery_annotation={
                "schema_version": 1,
                "review_status": "pending",
                "events": [{
                    "event_id": "fr_000",
                    "sub_task_idx": 0,
                    "annotation_status": "pending",
                    "source_mark": {
                        "source": "left_button",
                        "time_sec": 0.0,
                        "hint_frame": 0,
                    },
                    "failure": {"frame_duration": None},
                    "recovery": {"frame_duration": None},
                }],
            },
        )

        self.converter._write_subtask_annotations(
            Path(self.temp_dir),
            [empty, failed],
        )

        empty_path = Path(self.temp_dir) / "annotations/chunk-000/episode_000000.json"
        failed_path = Path(self.temp_dir) / "annotations/chunk-000/episode_000001.json"
        self.assertFalse(empty_path.exists())
        payload = json.loads(failed_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["sub_task_annotation"], [])
        self.assertEqual(
            payload["failure_recovery_annotation"]["events"][0]["event_id"],
            "fr_000",
        )

    def test_archived_failure_hint_adds_segment_frame_offset(self):
        bag_path = Path(self.temp_dir) / "0"
        bag_path.mkdir()
        metadata = {
            "rosbag2_bagfile_information": {
                "starting_time": {"nanoseconds_since_epoch": 10_000_000_000},
                "files": [
                    {
                        "path": "0_0.mcap",
                        "starting_time": {"nanoseconds_since_epoch": 10_000_000_000},
                    },
                    {
                        "path": "0_1.mcap",
                        "starting_time": {"nanoseconds_since_epoch": 20_000_000_000},
                    },
                ],
            }
        }
        (bag_path / "metadata.yaml").write_text(
            base_converter.yaml.safe_dump(metadata),
            encoding="utf-8",
        )
        mcaps = [bag_path / "0_0.mcap", bag_path / "0_1.mcap"]
        for mcap in mcaps:
            mcap.touch()
        episode_info = {
            "segments": [
                {"frame_duration": [0.0, 1.0]},
                {"frame_duration": [1.0, 2.0]},
            ],
            "failure_recovery_annotation": {
                "events": [{
                    "event_id": "fr_000",
                    "sub_task_idx": 1,
                    "source_mark": {"source": "left_button", "time_sec": 1.19},
                }]
            },
        }
        segment_episodes = {
            0: EpisodeData(
                episode_index=0,
                length=3,
                grid_log_times_sec=[10.0, 10.1, 10.2],
            ),
            1: EpisodeData(
                episode_index=0,
                length=3,
                grid_log_times_sec=[20.0, 20.2, 20.4],
            ),
        }

        annotation = self.converter._resolve_archived_failure_annotation(
            episode_info,
            bag_path,
            segment_episodes,
            {0: 0, 1: 3},
            mcaps,
            6,
        )

        self.assertEqual(
            annotation["events"][0]["source_mark"]["hint_frame"],
            4,
        )

    def test_archived_fixture_writes_failure_annotations_for_v21_and_v30(self):
        source = Path(self.temp_dir) / "source" / "0"
        source.mkdir(parents=True)
        segments = [
            {
                "sub_task_instruction": f"subtask {idx}",
                "frame_duration": [float(idx), float(idx + 1)],
            }
            for idx in range(3)
        ]
        metadata_files = []
        for idx in range(3):
            filename = f"0_{idx}.mcap"
            (source / filename).touch()
            metadata_files.append({
                "path": filename,
                "starting_time": {
                    "nanoseconds_since_epoch": (idx + 1) * 10_000_000_000,
                },
            })
        (source / "metadata.yaml").write_text(
            base_converter.yaml.safe_dump({
                "rosbag2_bagfile_information": {
                    "files": metadata_files,
                }
            }),
            encoding="utf-8",
        )
        info_path = source / "episode_info.json"
        info = {
            "task_name": "failure fixture",
            "segments": segments,
            "failure_recovery_annotation": {
                "schema_version": 1,
                "review_status": "pending",
                "events": [
                    {
                        "event_id": f"fr_{idx:03d}",
                        "sub_task_idx": idx,
                        "annotation_status": "pending",
                        "source_mark": {
                            "source": (
                                "left_button" if idx != 1 else "ui_failed"
                            ),
                            "time_sec": float(idx) + 0.5,
                        },
                        "failure": {"frame_duration": None},
                        "recovery": {"frame_duration": None},
                    }
                    for idx in range(3)
                ],
            },
        }
        info_path.write_text(json.dumps(info), encoding="utf-8")

        converters = [
            RosbagToLerobotConverter(
                ConversionConfig(
                    repo_id="test/v21",
                    output_dir=Path(self.temp_dir) / "v21",
                    fps=15,
                    use_videos=False,
                )
            ),
            RosbagToLerobotV30Converter(
                V30ConversionConfig(
                    repo_id="test/v30",
                    output_dir=Path(self.temp_dir) / "v30",
                    fps=15,
                    use_videos=False,
                )
            ),
        ]

        for converter in converters:
            def fake_extract(mcap_path, episode_index, trim_points, exclude_regions):
                start = converter._recording_start_sec(source, Path(mcap_path).name)
                return EpisodeData(
                    episode_index=episode_index,
                    timestamps=[0.0, 0.3, 0.6],
                    observation_state=[np.zeros(1)] * 3,
                    action=[np.zeros(1)] * 3,
                    length=3,
                    grid_log_times_sec=[start + 0.2, start + 0.5, start + 0.8],
                )

            with patch.dict(
                os.environ,
                {"CYCLO_PREPARED_EPISODE_CACHE_DISABLE": "1"},
            ), patch.object(
                converter,
                "_extract_joint_data",
                side_effect=fake_extract,
            ):
                episode = converter.convert_single_rosbag(source, 0)
            self.assertIsNotNone(episode)
            converter._write_subtask_annotations(converter.config.output_dir, [episode])
            annotation_path = (
                converter.config.output_dir
                / "annotations/chunk-000/episode_000000.json"
            )
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            events = payload["failure_recovery_annotation"]["events"]
            self.assertEqual(
                [event["source_mark"]["hint_frame"] for event in events],
                [1, 4, 7],
            )
            self.assertEqual(
                [event["sub_task_idx"] for event in events],
                [0, 1, 2],
            )
            self.assertTrue(all(
                0 <= event["source_mark"]["hint_frame"] < 9
                for event in events
            ))

    def test_compute_episode_stats(self):
        episode = EpisodeData(
            episode_index=0,
            observation_state=[
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                np.array([5.0, 6.0]),
            ],
            action=[
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
                np.array([0.5, 0.6]),
            ],
            length=3,
        )

        stats = self.converter._compute_episode_stats(episode)

        self.assertIn("observation.state", stats)
        self.assertIn("action", stats)
        self.assertIn("mean", stats["observation.state"])
        self.assertIn("std", stats["observation.state"])
        self.assertIn("min", stats["observation.state"])
        self.assertIn("max", stats["observation.state"])

        np.testing.assert_array_almost_equal(
            stats["observation.state"]["mean"], [3.0, 4.0]
        )

    def test_direct_v21_episode_stats_use_per_episode_video_cache(self):
        video_dir = Path(self.temp_dir) / "videos/chunk-000/cam_left_head"
        video_dir.mkdir(parents=True)
        ep0_video = video_dir / "episode_000000.mp4"
        ep1_video = video_dir / "episode_000001.mp4"
        ep0_video.write_bytes(b"mp4")
        ep1_video.write_bytes(b"mp4")
        ep0_stats = {"mean": [[[0.1]], [[0.2]], [[0.3]]], "count": [2]}
        ep1_stats = {"mean": [[[0.7]], [[0.8]], [[0.9]]], "count": [3]}

        self.converter._direct_v21_video_output = True
        self.converter._direct_v21_video_stats_cache = {
            (0, "cam_left_head"): ep0_stats,
            (1, "cam_left_head"): ep1_stats,
        }
        episode0 = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0],
            length=2,
            video_files={"cam_left_head": ep0_video},
        )
        episode1 = EpisodeData(
            episode_index=1,
            timestamps=[0.0, 1.0, 2.0],
            length=3,
            video_files={"cam_left_head": ep1_video},
        )

        with patch.object(
            self.converter,
            "_compute_video_stats",
            side_effect=AssertionError("direct cache should be used"),
        ):
            stats0 = self.converter._compute_episode_stats(episode0)
            stats1 = self.converter._compute_episode_stats(episode1)

        feature_key = "observation.images.rgb.cam_left_head"
        self.assertIs(stats0[feature_key], ep0_stats)
        self.assertIs(stats1[feature_key], ep1_stats)

    def test_serialize_stats(self):
        stats = {
            "observation.state": {
                "mean": np.array([1.0, 2.0]),
                "std": np.array([0.1, 0.2]),
            }
        }

        serialized = self.converter._serialize_stats(stats)

        self.assertIsInstance(serialized["observation.state"]["mean"], list)
        self.assertEqual(serialized["observation.state"]["mean"], [1.0, 2.0])

    def test_resample_to_fps(self):
        episode = EpisodeData(episode_index=0)
        state_messages = [
            (0.0, np.array([1.0])),
            (0.5, np.array([2.0])),
            (1.0, np.array([3.0])),
        ]
        action_messages = [
            (0.0, np.array([0.1])),
            (0.5, np.array([0.2])),
            (1.0, np.array([0.3])),
        ]

        self.converter.config.fps = 10
        result, staleness = self.converter._resample_to_fps(
            episode, state_messages, action_messages, 0.0
        )

        self.assertGreater(result.length, 0)
        self.assertEqual(len(result.timestamps), result.length)
        self.assertEqual(len(result.observation_state), result.length)
        self.assertEqual(len(result.action), result.length)

    def test_is_state_topic_new_naming(self):
        topic_types = {
            "/robot/arm_left_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/arm_right_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/head_follower/joint_states": "sensor_msgs/msg/JointState",
            "/odom": "nav_msgs/msg/Odometry",
            "/robot/arm_left_leader/joint_states": "sensor_msgs/msg/JointState",
            "/cmd_vel": "geometry_msgs/msg/Twist",
        }
        self.assertTrue(self.converter._is_state_topic(
            "/robot/arm_left_follower/joint_states", topic_types))
        self.assertTrue(self.converter._is_state_topic("/odom", topic_types))
        self.assertFalse(self.converter._is_state_topic(
            "/robot/arm_left_leader/joint_states", topic_types))
        self.assertFalse(self.converter._is_state_topic("/cmd_vel", topic_types))

    def test_is_action_topic_new_naming(self):
        topic_types = {
            "/robot/arm_left_follower/joint_states": "sensor_msgs/msg/JointState",
            "/robot/arm_left_leader/joint_states": "sensor_msgs/msg/JointState",
            "/cmd_vel": "geometry_msgs/msg/Twist",
            "/odom": "nav_msgs/msg/Odometry",
        }
        self.assertTrue(self.converter._is_action_topic(
            "/robot/arm_left_leader/joint_states", topic_types))
        self.assertTrue(self.converter._is_action_topic("/cmd_vel", topic_types))
        self.assertFalse(self.converter._is_action_topic(
            "/robot/arm_left_follower/joint_states", topic_types))
        self.assertFalse(self.converter._is_action_topic("/odom", topic_types))

    def test_extract_joint_data_uses_header_time_for_state_only(self):
        state_topic = "/robot/arm_left_follower/joint_states"
        action_topic = "/robot/arm_left_leader/joint_states"
        self.converter.config.state_topics = [state_topic]
        self.converter.config.action_topics = [action_topic]

        def stamp(sec: int, nanosec: int = 0):
            return types.SimpleNamespace(sec=sec, nanosec=nanosec)

        def joint_msg(value: float, header_stamp):
            return types.SimpleNamespace(
                header=types.SimpleNamespace(stamp=header_stamp),
                name=["joint_a"],
                position=[value],
            )

        messages = [
            (state_topic, joint_msg(1.0, stamp(1, 100_000_000)), 100.0),
            (action_topic, joint_msg(2.0, stamp(9, 900_000_000)), 100.1),
            (state_topic, joint_msg(3.0, stamp(1, 200_000_000)), 100.2),
            (action_topic, joint_msg(4.0, stamp(9, 800_000_000)), 100.3),
        ]

        fake_reader = MagicMock()
        fake_reader.open.return_value = True
        fake_reader.get_topic_types.return_value = {
            state_topic: "sensor_msgs/msg/JointState",
            action_topic: "sensor_msgs/msg/JointState",
        }
        fake_reader.read_messages.return_value = iter(messages)

        captured = {}

        def fake_resample(episode, state_messages, action_messages, trim_start):
            captured["state_times"] = [round(t, 3) for t, _ in state_messages]
            captured["action_times"] = [round(t, 3) for t, _ in action_messages]
            episode.length = 1
            return episode, {}

        with tempfile.TemporaryDirectory() as tmpdir:
            bag = Path(tmpdir)
            with patch.dict(os.environ, {"CYCLO_EXTRACT_CACHE_DISABLE": "1"}):
                with patch(
                    "cyclo_data.converter.base_converter.BagReader",
                    return_value=fake_reader,
                ):
                    with patch.object(
                        self.converter,
                        "_resample_to_fps",
                        side_effect=fake_resample,
                    ):
                        episode = self.converter._extract_joint_data(
                            bag,
                            episode_index=0,
                            trim_points=None,
                            exclude_regions=[],
                        )

        self.assertIsNotNone(episode)
        self.assertEqual(captured["state_times"], [1.1, 1.2])
        self.assertEqual(captured["action_times"], [100.1, 100.3])

    def test_merge_state_messages_multi_topic(self):
        state_msgs = {
            "/robot/arm_left_follower/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
                (0.01, np.array([1.1, 2.1], dtype=np.float32)),
            ],
            "/robot/arm_right_follower/joint_states": [
                (0.0, np.array([3.0, 4.0], dtype=np.float32)),
                (0.01, np.array([3.1, 4.1], dtype=np.float32)),
            ],
        }
        state_names = {
            "/robot/arm_left_follower/joint_states": ["j1", "j2"],
            "/robot/arm_right_follower/joint_states": ["j3", "j4"],
        }

        merged = self.converter._merge_state_messages(state_msgs, state_names)

        self.assertGreater(len(merged), 0)
        # Merged vector should be 4 dimensions (2 + 2)
        self.assertEqual(len(merged[0][1]), 4)
        self.assertEqual(self.converter._state_joint_names, ["j1", "j2", "j3", "j4"])

    def test_merge_state_messages_with_joint_order(self):
        self.converter._joint_order_by_group = {
            "follower_arm_left": ["j1", "j2"],
            "follower_arm_right": ["j3", "j4"],
        }
        self.converter._state_topic_key_map = {
            "/robot/arm_left_follower/joint_states": "follower_arm_left",
            "/robot/arm_right_follower/joint_states": "follower_arm_right",
        }

        state_msgs = {
            "/robot/arm_left_follower/joint_states": [
                (0.0, np.array([1.0, 2.0], dtype=np.float32)),
            ],
            "/robot/arm_right_follower/joint_states": [
                (0.0, np.array([3.0, 4.0], dtype=np.float32)),
            ],
        }
        state_names = {
            "/robot/arm_left_follower/joint_states": ["j1", "j2"],
            "/robot/arm_right_follower/joint_states": ["j3", "j4"],
        }

        merged = self.converter._merge_state_messages(state_msgs, state_names)

        self.assertEqual(len(merged), 1)
        np.testing.assert_array_almost_equal(
            merged[0][1], [1.0, 2.0, 3.0, 4.0]
        )

    def test_get_topic_group_key(self):
        self.assertEqual(
            self.converter._get_topic_group_key(
                "/robot/arm_left_follower/joint_states", "state"),
            "follower_arm_left"
        )
        self.assertEqual(
            self.converter._get_topic_group_key(
                "/robot/head_leader/joint_states", "action"),
            "leader_head"
        )
        self.assertEqual(
            self.converter._get_topic_group_key("/odom", "state"),
            "follower_mobile"
        )
        self.assertEqual(
            self.converter._get_topic_group_key("/cmd_vel", "action"),
            "leader_mobile"
        )

    def test_update_config_from_robot_config_with_grouped_joint_order(self):
        robot_config = {
            "robot_type": "ffw_sg2_rev1",
            "state_topics": {
                "follower_arm_left": "/robot/arm_left_follower/joint_states",
                "follower_arm_right": "/robot/arm_right_follower/joint_states",
            },
            "action_topics": {
                "leader_arm_left": "/robot/arm_left_leader/joint_states",
                "leader_arm_right": "/robot/arm_right_leader/joint_states",
            },
            "joint_order": {
                "follower_arm_left": ["j1", "j2"],
                "follower_arm_right": ["j3", "j4"],
                "leader_arm_left": ["j1", "j2"],
                "leader_arm_right": ["j3", "j4"],
            },
        }

        self.converter._update_config_from_robot_config(robot_config)

        self.assertEqual(self.converter.config.robot_type, "ffw_sg2_rev1")
        self.assertIn("follower_arm_left", self.converter._joint_order_by_group)
        self.assertEqual(
            self.converter._joint_order_by_group["follower_arm_left"], ["j1", "j2"])
        self.assertEqual(
            self.converter._state_topic_key_map["/robot/arm_left_follower/joint_states"],
            "follower_arm_left"
        )
        self.assertEqual(self.converter._joint_order, ["j1", "j2", "j3", "j4", "j1", "j2", "j3", "j4"])


class TestInfoJsonGeneration(unittest.TestCase):
    """Tests for info.json generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
            fps=30,
            robot_type="test_robot",
        )
        self.converter = RosbagToLerobotConverter(self.config)

    def test_write_info_json(self):
        self.converter._features = {
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
            "action": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.images.rgb.cam_left_head": {
                "dtype": "video",
                "shape": (3, 720, 1280),
                "names": ["channels", "height", "width"],
                "info": {"video.fps": 30.0},
            },
        }
        self.converter._tasks = {0: "test_task"}
        self.converter._total_episodes = 5
        self.converter._total_frames = 500

        (Path(self.temp_dir) / "meta").mkdir(parents=True, exist_ok=True)

        self.converter._write_info_json()

        info_path = Path(self.temp_dir) / "meta" / "info.json"
        self.assertTrue(info_path.exists())

        with open(info_path) as f:
            info = json.load(f)

        self.assertEqual(info["codebase_version"], "v2.1")
        self.assertEqual(info["robot_type"], "test_robot")
        self.assertEqual(info["fps"], 30)
        self.assertEqual(info["total_episodes"], 5)
        self.assertEqual(info["total_frames"], 500)
        self.assertIn("features", info)
        self.assertEqual(info["chunks_size"], 1000)
        self.assertEqual(
            info["data_path"],
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        self.assertEqual(
            info["video_path"],
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        self.assertNotIn("frame_index", info["features"])
        self.assertEqual(info["features"]["timestamp"]["dtype"], "float64")

    def test_write_root_info_json_uses_robot_config_rotations(self):
        self.converter.config.selected_cameras = [
            "cam_left_head",
            "cam_left_wrist",
        ]
        self.converter.config.camera_rotations = {
            "cam_left_head": 0,
            "cam_left_wrist": 0,
        }
        self.converter._camera_rotations = {
            "cam_left_head": 0,
            "cam_left_wrist": 270,
        }
        self.converter.config.source_rosbags = ["Task_X_MCAP"]
        self.converter.config.state_topics = ["/joint_states", "/odom"]
        self.converter.config.action_topics = [
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            "/cmd_vel",
        ]
        self.converter._state_topic_key_map = {
            "/joint_states": "follower_upper_body",
            "/odom": "follower_mobile",
        }
        self.converter._action_topic_key_map = {
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory":
                "leader_arm_left",
            "/cmd_vel": "leader_mobile",
        }
        self.converter._tasks = {0: "wipe table"}

        self.converter._write_root_info_json()

        info_path = Path(self.temp_dir) / "info.json"
        self.assertTrue(info_path.exists())
        with open(info_path) as f:
            info = json.load(f)

        self.assertEqual(info["source_rosbags"], ["Task_X_MCAP"])
        config = info["conversion_config"]
        self.assertEqual(
            list(config.keys()),
            [
                "robot_type",
                "task_name",
                "fps",
                "camera_rotations",
                "selected_end_effector_topics",
                "selected_cameras",
                "output_dataset_name",
                "image_resize",
                "selected_joint_state_topics",
                "primitive_instructions",
                "selected_action_topics",
            ],
        )
        self.assertEqual(
            config["camera_rotations"],
            {"cam_left_wrist": 270},
        )
        self.assertEqual(config["task_name"], "wipe table")
        self.assertEqual(
            config["selected_joint_state_topics"],
            ["upper_body", "mobile"],
        )
        self.assertEqual(
            config["selected_action_topics"],
            ["arm_left", "mobile"],
        )
        self.assertEqual(config["selected_end_effector_topics"], [])
        self.assertEqual(config["primitive_instructions"], [])
        self.assertNotIn("selected_joints", config)


class TestRosbagToLerobotV30VideoConcat(unittest.TestCase):
    """Tests for v3.0 aggregate video writing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = V30ConversionConfig(
            repo_id="test/dataset",
            output_dir=Path(self.temp_dir),
            fps=15,
        )
        self.converter = RosbagToLerobotV30Converter(self.config)
        self.converter._episode_metadata_list = [
            EpisodeMetadata(episode_index=0, length=2, tasks=["task"]),
            EpisodeMetadata(episode_index=1, length=3, tasks=["task"]),
        ]
        self.input_a = Path(self.temp_dir) / "episode_000000.mp4"
        self.input_b = Path(self.temp_dir) / "episode_000001.mp4"
        self.input_a.touch()
        self.input_b.touch()

    def test_v30_video_feature_key_uses_rgb_prefix(self):
        self.assertEqual(
            self.converter._video_feature_key("cam_left_head"),
            "observation.images.rgb.cam_left_head",
        )

    def test_v30_video_info_describes_generated_aggregate(self):
        with patch.object(
            self.converter,
            "_get_video_dimensions",
            return_value=(480, 640),
        ):
            info = self.converter._get_video_info(self.input_a)

        self.assertEqual(info["video.fps"], 15.0)
        self.assertEqual(info["video.codec"], "h264")
        self.assertEqual(info["video.pix_fmt"], "yuv420p")
        self.assertFalse(info["has_audio"])

    def test_v30_video_info_uses_aggregate_cache_dimensions_without_probe(self):
        self.converter._aggregated_video_cache_path(self.input_a).write_text(
            json.dumps({"output_width": 640, "output_height": 480}),
            encoding="utf-8",
        )

        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            side_effect=AssertionError("video probe should not run"),
        ), patch.object(
            self.converter,
            "_get_video_dimensions",
            side_effect=AssertionError("OpenCV fallback should not run"),
        ):
            info = self.converter._get_video_info(self.input_a)

        self.assertEqual(info["video.height"], 480)
        self.assertEqual(info["video.width"], 640)

    def test_quick_video_dimensions_are_cached_per_converter(self):
        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(640, 480),
        ) as quick_dims, patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("resolve should not be needed"),
        ):
            first = self.converter._quick_video_dimensions_cached(self.input_a)
            second = self.converter._quick_video_dimensions_cached(self.input_a)

        self.assertEqual(first, (640, 480))
        self.assertEqual(second, (640, 480))
        quick_dims.assert_called_once()

    def test_v30_default_video_file_size_keeps_portable_default(self):
        self.assertEqual(DEFAULT_VIDEO_FILE_SIZE_IN_MB, 200)
        self.assertEqual(
            V30ConversionConfig(
                repo_id="test/default",
                output_dir=Path(self.temp_dir) / "default",
            ).video_file_size_in_mb,
            200,
        )

    def test_v30_data_flush_accepts_numpy_state_action_columns(self):
        frames = [
            {
                "timestamp": 0.0,
                "frame_index": 0,
                "episode_index": 0,
                "index": 0,
                "task_index": 0,
                "observation.state": np.array([1.0, 2.0], dtype=np.float32),
                "action": np.array([0.1, 0.2], dtype=np.float32),
            },
            {
                "timestamp": 1.0 / 15.0,
                "frame_index": 1,
                "episode_index": 0,
                "index": 1,
                "task_index": 0,
                "observation.state": np.array([3.0, 4.0], dtype=np.float32),
                "action": np.array([0.3, 0.4], dtype=np.float32),
            },
        ]

        self.converter._flush_data_file(Path(self.temp_dir), frames)

        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(self.temp_dir) / "data/chunk-000/file-000.parquet"
        )
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(
            table.column("observation.state").to_pylist(),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_v30_aggregated_data_subtask_column_does_not_need_features(self):
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0 / 15.0],
            observation_state=[
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ],
            action=[
                np.array([3.0], dtype=np.float32),
                np.array([4.0], dtype=np.float32),
            ],
            tasks=["task"],
            length=2,
            subtask_indices=[7, 8],
        )
        self.converter._features = {}
        self.converter._task_to_index = {"task": 0}
        self.converter._write_aggregated_data([episode])

        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(self.temp_dir) / "data/chunk-000/file-000.parquet"
        )
        self.assertIn("subtask_index", table.column_names)
        self.assertEqual(table.column("subtask_index").to_pylist(), [7, 8])

    def test_v30_aggregated_data_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0, 1.0 / 15.0],
            observation_state=[
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ],
            action=[
                np.array([3.0], dtype=np.float32),
                np.array([4.0], dtype=np.float32),
            ],
            tasks=["task"],
            length=2,
            source_path=source_dir,
        )
        self.converter._tasks = {0: "task"}
        self.converter._task_to_index = {"task": 0}
        self.converter._episode_metadata_list = []
        self.converter._write_aggregated_data([episode])

        data_path = Path(self.temp_dir) / "data/chunk-000/file-000.parquet"
        original_bytes = data_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "data_aggregate_v30"
                ).glob("*/manifest.json")
            )
        )

        data_path.unlink()
        self.converter._episode_metadata_list = []
        self.converter._episode_metadata_by_index = {}
        self.converter._total_episodes = 0
        self.converter._total_frames = 0

        with patch.object(
            self.converter,
            "_flush_episode_data_file",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_aggregated_data([episode])

        self.assertEqual(data_path.read_bytes(), original_bytes)
        self.assertEqual(self.converter._total_episodes, 1)
        self.assertEqual(self.converter._total_frames, 2)
        self.assertEqual(len(self.converter._episode_metadata_list), 1)
        self.assertEqual(
            self.converter._episode_metadata_list[0].stats["observation.state/min"],
            [1.0],
        )

    def test_v30_data_cache_key_uses_prepared_cache_signature(self):
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
        )
        episode._cyclo_prepared_cache_signature = {
            "path": "/tmp/prepared.pickle",
            "size": 10,
            "mtime_ns": 20,
        }
        self.converter._tasks = {0: "task"}
        self.converter._task_to_index = {"task": 0}

        with patch.object(
            self.converter,
            "_array_cache_signature",
            side_effect=AssertionError("prepared cache should skip array hashing"),
        ):
            cache_key = self.converter._data_aggregate_cache_key(
                [episode],
                has_subtask_feature=False,
            )

        self.assertEqual(
            cache_key["episodes"][0]["prepared_cache"]["path"],
            "/tmp/prepared.pickle",
        )
        self.assertNotIn("observation_state", cache_key["episodes"][0])

    def test_v30_episodes_parquet_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
            source_path=source_dir,
        )
        self.converter._tasks = {0: "task"}
        self.converter._task_to_index = {"task": 0}
        self.converter._episode_metadata_list = []
        self.converter._write_aggregated_data([episode])
        self.converter._write_episodes_parquet([episode])

        episodes_path = (
            Path(self.temp_dir)
            / "meta/episodes/chunk-000/file-000.parquet"
        )
        original_bytes = episodes_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "episodes_parquet_v30"
                ).glob("*/manifest.json")
            )
        )

        episodes_path.unlink()
        with patch(
            "cyclo_data.converter.to_lerobot_v30.pq.write_table",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_episodes_parquet([episode])

        self.assertEqual(episodes_path.read_bytes(), original_bytes)

    def test_v30_tasks_parquet_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
            source_path=source_dir,
        )
        self.converter._tasks = {0: "task"}
        self.converter._task_to_index = {"task": 0}
        self.converter._task_names_by_task = {"task": "Task"}
        self.converter._episode_metadata_list = []
        self.converter._write_aggregated_data([episode])
        self.converter._write_tasks_parquet([episode])

        tasks_path = Path(self.temp_dir) / "meta/tasks.parquet"
        original_bytes = tasks_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "tasks_parquet_v30"
                ).glob("*/manifest.json")
            )
        )

        tasks_path.unlink()
        with patch(
            "cyclo_data.converter.to_lerobot_v30.pq.write_table",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_tasks_parquet([episode])

        self.assertEqual(tasks_path.read_bytes(), original_bytes)

    def test_v30_subtasks_parquet_reuses_source_cache_on_repeat(self):
        source_dir = Path(self.temp_dir) / "source_episode"
        source_dir.mkdir()
        (source_dir / "episode_info.json").write_text("{}", encoding="utf-8")
        episode = EpisodeData(
            episode_index=0,
            timestamps=[0.0],
            observation_state=[np.array([1.0], dtype=np.float32)],
            action=[np.array([2.0], dtype=np.float32)],
            tasks=["task"],
            length=1,
            source_path=source_dir,
            subtask_instructions=["Pick"],
        )
        self.converter._tasks = {0: "task"}
        self.converter._task_to_index = {"task": 0}
        self.converter._episode_metadata_list = []
        self.converter._write_aggregated_data([episode])
        self.converter._write_subtasks_parquet(Path(self.temp_dir), [episode])

        subtasks_path = Path(self.temp_dir) / "meta/subtasks.parquet"
        original_bytes = subtasks_path.read_bytes()
        self.assertTrue(
            list(
                (
                    source_dir
                    / ".cyclo_cache"
                    / "subtasks_parquet_v30"
                ).glob("*/manifest.json")
            )
        )

        subtasks_path.unlink()
        with patch(
            "cyclo_data.converter.to_lerobot_v30.pq.write_table",
            side_effect=AssertionError("cache miss"),
        ):
            self.converter._write_subtasks_parquet(Path(self.temp_dir), [episode])

        self.assertEqual(subtasks_path.read_bytes(), original_bytes)

    def test_direct_aggregate_cache_key_reuses_supplied_video_probe(self):
        sidecar = Path(self.temp_dir) / "cam_left_head_timestamps.parquet"
        sidecar.touch()
        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }

        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            side_effect=AssertionError("dimensions should be supplied"),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            side_effect=AssertionError("encoder should be supplied"),
        ), patch.dict(os.environ, {}, clear=True):
            cache_key = self.converter._direct_aggregated_video_cache_key(
                [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
                expected_frames=5,
                episode_by_index=episode_by_index,
                camera_name="cam_left_head",
                ffmpeg="ffmpeg",
                width=64,
                height=48,
                encoder="libx264",
                encoder_opts=["-preset", "ultrafast"],
            )

        self.assertEqual(cache_key["encoder"], "libx264")
        self.assertEqual(cache_key["encoder_opts"], ["-preset", "ultrafast"])
        self.assertEqual(cache_key["output_width"], 64)
        self.assertEqual(cache_key["output_height"], 48)
        self.assertNotIn("source_frame_count_validation", cache_key)

    def test_direct_aggregate_cache_key_records_output_dimensions(self):
        cache_key = self.converter._direct_aggregated_video_cache_key(
            [],
            expected_frames=0,
            episode_by_index={},
            camera_name="cam_left_head",
            width=640,
            height=480,
            encoder="libx264",
            encoder_opts=["-preset", "ultrafast"],
            content_key={
                "version": 1,
                "mode": "direct_v3_raw_sidecar",
                "target_fps": 15.0,
                "expected_frames": 0,
                "inputs": [],
            },
        )

        self.assertEqual(cache_key["output_width"], 640)
        self.assertEqual(cache_key["output_height"], 480)

    def test_max_direct_output_cache_hit_can_skip_encoder_probe(self):
        sidecar = Path(self.temp_dir) / "cam_left_head_timestamps.parquet"
        sidecar.touch()
        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]
        camera_key = "observation.images.rgb.cam_left_head"
        output_path = Path(self.temp_dir) / DEFAULT_VIDEO_PATH.format(
            video_key=camera_key,
            chunk_index=0,
            file_index=0,
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"cached-video")
        with patch.dict(os.environ, {"CYCLO_X264_SPEED_PROFILE": "max"}, clear=True):
            content_key = self.converter._direct_aggregated_video_content_cache_key(
                videos,
                5,
                episode_by_index,
                "cam_left_head",
            )
        cached_key = {
            **content_key,
            "encoder": "libx264",
            "encoder_opts": ["-preset", "ultrafast"],
        }
        self.converter._aggregated_video_cache_path(output_path).write_text(
            json.dumps(cached_key),
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(640, 480),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            side_effect=AssertionError("encoder probe should be skipped"),
        ):
            self.converter._write_direct_aggregated_video(
                Path(self.temp_dir),
                camera_key,
                0,
                0,
                videos,
                episode_by_index,
            )

    def test_max_direct_source_cache_hit_can_skip_encoder_probe(self):
        sidecar = Path(self.temp_dir) / "cam_left_head_timestamps.parquet"
        sidecar.touch()
        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]
        with patch.dict(os.environ, {"CYCLO_X264_SPEED_PROFILE": "max"}, clear=True):
            content_key = self.converter._direct_aggregated_video_content_cache_key(
                videos,
                5,
                episode_by_index,
                "cam_left_head",
            )
        cached_key = {
            **content_key,
            "encoder": "libx264",
            "encoder_opts": ["-preset", "ultrafast"],
        }
        source_paths = self.converter._direct_source_aggregate_cache_paths(
            videos,
            cached_key,
        )
        self.assertIsNotNone(source_paths)
        source_video, source_meta = source_paths
        source_video.parent.mkdir(parents=True)
        source_video.write_bytes(b"cached-video")
        source_meta.write_text(json.dumps(cached_key), encoding="utf-8")
        camera_key = "observation.images.rgb.cam_left_head"
        output_path = Path(self.temp_dir) / DEFAULT_VIDEO_PATH.format(
            video_key=camera_key,
            chunk_index=0,
            file_index=0,
        )

        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(640, 480),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            side_effect=AssertionError("encoder probe should be skipped"),
        ):
            self.converter._write_direct_aggregated_video(
                Path(self.temp_dir),
                camera_key,
                0,
                0,
                videos,
                episode_by_index,
            )

        self.assertEqual(output_path.read_bytes(), b"cached-video")
        self.assertFalse(
            self.converter._aggregated_video_cache_path(output_path).exists()
        )

    def test_explicit_h264_encoder_disables_probe_free_direct_cache_reuse(self):
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_H264_ENCODER": "software",
            },
            clear=True,
        ):
            self.assertFalse(
                self.converter._can_reuse_direct_cache_without_encoder_probe()
            )

    def test_direct_source_aggregate_cache_can_be_disabled(self):
        with patch.dict(
            os.environ,
            {"CYCLO_V30_DISABLE_SOURCE_AGGREGATE_CACHE": "1"},
        ):
            self.assertIsNone(
                self.converter._direct_source_aggregate_cache_paths(
                    [(0, self.input_a, 0.0)],
                    {"key": "value"},
                )
            )

    def test_direct_source_aggregate_cache_paths_use_common_source_root(self):
        with patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("resolve should not be needed"),
        ):
            cache_paths = self.converter._direct_source_aggregate_cache_paths(
                [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
                {"key": "value"},
            )

        self.assertIsNotNone(cache_paths)
        video_cache, meta_cache = cache_paths
        self.assertEqual(
            video_cache.parent,
            Path(self.temp_dir) / ".cyclo_cache" / "direct_aggregate_v30",
        )
        self.assertEqual(video_cache.suffix, ".mp4")
        self.assertTrue(meta_cache.name.endswith(".cache.json"))

    def test_direct_source_aggregate_cache_dir_is_cached_per_video_batch(self):
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]

        with patch.object(
            self.converter,
            "_direct_source_video_cache_root",
            wraps=self.converter._direct_source_video_cache_root,
        ) as root:
            first = self.converter._direct_source_aggregate_cache_dir(videos)
            second = self.converter._direct_source_aggregate_cache_dir(videos)

        self.assertEqual(first, second)
        self.assertEqual(root.call_count, len(videos))

    def test_direct_source_aggregate_cache_uses_dataset_root_for_task_folder(self):
        dataset = Path(self.temp_dir) / "task"
        ep_a = dataset / "0"
        ep_b = dataset / "1"
        video_a = ep_a / "videos" / "0_0" / "cam_left_head.mp4"
        video_b = ep_b / "videos" / "1_0" / "cam_left_head.mp4"
        video_a.parent.mkdir(parents=True)
        video_b.parent.mkdir(parents=True)
        video_a.touch()
        video_b.touch()
        (ep_a / "episode_info.json").write_text("{}", encoding="utf-8")
        (ep_b / "episode_info.json").write_text("{}", encoding="utf-8")

        cache_paths = self.converter._direct_source_aggregate_cache_paths(
            [(0, video_a, 0.0)],
            {"key": "value"},
        )

        self.assertIsNotNone(cache_paths)
        video_cache, _ = cache_paths
        self.assertEqual(
            video_cache.parent,
            dataset / ".cyclo_cache" / "direct_aggregate_v30",
        )

    def test_direct_source_video_cache_root_is_cached_per_source_video(self):
        dataset = Path(self.temp_dir) / "task"
        ep_a = dataset / "0"
        ep_b = dataset / "1"
        video_a = ep_a / "videos" / "0_0" / "cam_left_head.mp4"
        video_a.parent.mkdir(parents=True)
        video_a.touch()
        (ep_a / "episode_info.json").write_text("{}", encoding="utf-8")
        ep_b.mkdir(parents=True)
        sibling_info = ep_b / "episode_info.json"
        sibling_info.write_text("{}", encoding="utf-8")

        with patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("resolve should not be needed"),
        ):
            first = self.converter._direct_source_video_cache_root(video_a)
            sibling_info.unlink()
            second = self.converter._direct_source_video_cache_root(video_a)

        self.assertEqual(first, dataset)
        self.assertEqual(second, dataset)

    def test_direct_source_video_cache_root_scans_dataset_once(self):
        dataset = Path(self.temp_dir) / "task"
        ep_a = dataset / "0"
        ep_b = dataset / "1"
        video_a = ep_a / "videos" / "0_0" / "cam_left_head.mp4"
        video_b = ep_b / "videos" / "1_0" / "cam_left_head.mp4"
        video_a.parent.mkdir(parents=True)
        video_b.parent.mkdir(parents=True)
        video_a.touch()
        video_b.touch()
        (ep_a / "episode_info.json").write_text("{}", encoding="utf-8")
        (ep_b / "episode_info.json").write_text("{}", encoding="utf-8")
        original_iterdir = Path.iterdir
        calls = 0

        def counted_iterdir(path):
            nonlocal calls
            if path == dataset:
                calls += 1
            return original_iterdir(path)

        with patch.object(Path, "iterdir", counted_iterdir):
            first = self.converter._direct_source_video_cache_root(video_a)
            second = self.converter._direct_source_video_cache_root(video_b)

        self.assertEqual(first, dataset)
        self.assertEqual(second, dataset)
        self.assertEqual(calls, 1)

    def test_direct_source_aggregate_cache_keeps_standalone_bag_root(self):
        bag = Path(self.temp_dir) / "single"
        video = bag / "videos" / "0_0" / "cam_left_head.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        (bag / "episode_info.json").write_text("{}", encoding="utf-8")

        cache_paths = self.converter._direct_source_aggregate_cache_paths(
            [(0, video, 0.0)],
            {"key": "value"},
        )

        self.assertIsNotNone(cache_paths)
        video_cache, _ = cache_paths
        self.assertEqual(
            video_cache.parent,
            bag / ".cyclo_cache" / "direct_aggregate_v30",
        )

    def test_direct_source_aggregate_cache_population_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                self.converter._populate_direct_source_aggregate_cache()
            )

        for value in ["1", "true", "yes", "on", "write"]:
            with patch.dict(
                os.environ,
                {"CYCLO_V30_POPULATE_SOURCE_AGGREGATE_CACHE": value},
                clear=True,
            ):
                self.assertTrue(
                    self.converter._populate_direct_source_aggregate_cache()
                )

    def test_direct_source_cache_copy_requires_explicit_populate_env(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertFalse(
                self.converter._allow_direct_source_aggregate_cache_copy(
                    "libx264",
                )
            )

        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V30_POPULATE_SOURCE_AGGREGATE_CACHE": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                self.converter._allow_direct_source_aggregate_cache_copy(
                    "libx264",
                )
            )

        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V30_POPULATE_SOURCE_AGGREGATE_CACHE": "0",
            },
            clear=True,
        ):
            self.assertFalse(
                self.converter._allow_direct_source_aggregate_cache_copy(
                    "libx264",
                )
            )

    def test_direct_source_aggregate_cache_reuse_copies_to_output(self):
        cache_dir = Path(self.temp_dir) / ".cyclo_cache" / "direct_aggregate_v30"
        source_video = cache_dir / "cached.mp4"
        source_meta = cache_dir / "cached.cache.json"
        source_video.parent.mkdir(parents=True)
        source_video.write_bytes(b"aggregate")
        cache_key = {
            "key": "value",
            "output_width": 640,
            "output_height": 480,
            "inputs": [{"video": {"path": str(self.input_a)}}],
        }
        source_meta.write_text(json.dumps(cache_key), encoding="utf-8")
        output_path = Path(self.temp_dir) / "out" / "file.mp4"
        output_cache = output_path.with_name(output_path.name + ".cache.json")
        self.converter._validate_aggregated_video = MagicMock()

        reused = self.converter._try_reuse_direct_source_aggregate_cache(
            source_video,
            source_meta,
            output_path,
            output_cache,
            cache_key,
            expected_frames=5,
            require_decode=False,
        )

        self.assertTrue(reused)
        self.assertEqual(output_path.read_bytes(), b"aggregate")
        self.assertEqual(
            json.loads(output_cache.read_text(encoding="utf-8")),
            cache_key,
        )
        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            side_effect=AssertionError("video probe should not run"),
        ):
            info = self.converter._get_video_info(output_path)
        self.assertEqual(info["video.width"], 640)
        self.assertEqual(info["video.height"], 480)
        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            side_effect=AssertionError("source video probe should not run"),
        ):
            source_info = self.converter._get_video_info(self.input_a)
        self.assertEqual(source_info["video.width"], 640)
        self.assertEqual(source_info["video.height"], 480)

    def test_max_source_aggregate_reuse_skips_output_cache_by_default(self):
        cache_dir = Path(self.temp_dir) / ".cyclo_cache" / "direct_aggregate_v30"
        source_video = cache_dir / "cached.mp4"
        source_meta = cache_dir / "cached.cache.json"
        source_video.parent.mkdir(parents=True)
        source_video.write_bytes(b"aggregate")
        cache_key = {
            "key": "value",
            "output_width": 640,
            "output_height": 480,
        }
        source_meta.write_text(json.dumps(cache_key), encoding="utf-8")
        output_path = Path(self.temp_dir) / "out" / "file.mp4"
        output_cache = output_path.with_name(output_path.name + ".cache.json")
        self.converter._validate_aggregated_video = MagicMock()

        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            reused = self.converter._try_reuse_direct_source_aggregate_cache(
                source_video,
                source_meta,
                output_path,
                output_cache,
                cache_key,
                expected_frames=5,
                require_decode=False,
            )

        self.assertTrue(reused)
        self.assertEqual(output_path.read_bytes(), b"aggregate")
        self.assertFalse(output_cache.exists())
        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            side_effect=AssertionError("video probe should not run"),
        ):
            info = self.converter._get_video_info(output_path)
        self.assertEqual(info["video.width"], 640)
        self.assertEqual(info["video.height"], 480)

    def test_max_source_aggregate_reuse_output_cache_can_be_forced(self):
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V30_WRITE_SOURCE_REUSE_OUTPUT_CACHE": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                self.converter._write_output_cache_for_source_reuse()
            )

    def test_direct_source_aggregate_cache_can_skip_validation(self):
        cache_dir = Path(self.temp_dir) / ".cyclo_cache" / "direct_aggregate_v30"
        source_video = cache_dir / "cached.mp4"
        source_meta = cache_dir / "cached.cache.json"
        source_video.parent.mkdir(parents=True)
        source_video.write_bytes(b"aggregate")
        cache_key = {"key": "value"}
        source_meta.write_text(json.dumps(cache_key), encoding="utf-8")
        output_path = Path(self.temp_dir) / "out" / "file.mp4"
        output_cache = output_path.with_name(output_path.name + ".cache.json")
        self.converter._validate_aggregated_video = MagicMock(
            side_effect=AssertionError("validation should be skipped")
        )

        reused = self.converter._try_reuse_direct_source_aggregate_cache(
            source_video,
            source_meta,
            output_path,
            output_cache,
            cache_key,
            expected_frames=5,
            require_decode=False,
            validate=False,
        )

        self.assertTrue(reused)
        self.assertEqual(output_path.read_bytes(), b"aggregate")

    def test_aggregate_output_cache_can_skip_validation(self):
        output_path = Path(self.temp_dir) / "out" / "file.mp4"
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"aggregate")
        cache_path = output_path.with_name(output_path.name + ".cache.json")
        cache_key = {"key": "value"}
        cache_path.write_text(json.dumps(cache_key), encoding="utf-8")
        self.converter._validate_aggregated_video = MagicMock(
            side_effect=AssertionError("validation should be skipped")
        )

        reused = self.converter._try_reuse_aggregated_video_cache(
            output_path,
            cache_path,
            cache_key,
            expected_frames=5,
            require_decode=False,
            validate=False,
        )

        self.assertTrue(reused)

    def test_direct_source_aggregate_cache_store_writes_video_and_metadata(self):
        output_path = Path(self.temp_dir) / "out.mp4"
        output_path.write_bytes(b"aggregate")
        source_video = (
            Path(self.temp_dir)
            / ".cyclo_cache"
            / "direct_aggregate_v30"
            / "cached.mp4"
        )
        source_meta = source_video.with_name("cached.cache.json")
        cache_key = {"key": "value"}

        self.converter._store_direct_source_aggregate_cache(
            output_path,
            source_video,
            source_meta,
            cache_key,
        )

        self.assertEqual(source_video.read_bytes(), b"aggregate")
        self.assertEqual(
            json.loads(source_meta.read_text(encoding="utf-8")),
            cache_key,
        )

    def test_direct_source_aggregate_cache_store_writes_content_index(self):
        output_path = Path(self.temp_dir) / "out.mp4"
        output_path.write_bytes(b"aggregate")
        source_video = (
            Path(self.temp_dir)
            / ".cyclo_cache"
            / "direct_aggregate_v30"
            / "cached.mp4"
        )
        source_meta = source_video.with_name("cached.cache.json")
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]
        content_key = {"mode": "direct_v3_raw_sidecar", "inputs": [1, 2]}
        cache_key = {**content_key, "encoder": "libx264"}

        self.converter._store_direct_source_aggregate_cache(
            output_path,
            source_video,
            source_meta,
            cache_key,
            videos=videos,
            content_key=content_key,
        )

        index_path = self.converter._direct_source_aggregate_content_index_path(
            videos,
            content_key,
        )
        self.assertIsNotNone(index_path)
        self.assertTrue(index_path.exists())
        with patch.object(
            Path,
            "glob",
            side_effect=AssertionError("content index should avoid cache scan"),
        ):
            self.assertEqual(
                self.converter._find_direct_source_aggregate_cache_by_content(
                    videos,
                    content_key,
                ),
                (source_video, source_meta, cache_key),
            )

    def test_direct_source_aggregate_cache_store_can_skip_copy_fallback(self):
        output_path = Path(self.temp_dir) / "out.mp4"
        output_path.write_bytes(b"aggregate")
        source_video = (
            Path(self.temp_dir)
            / ".cyclo_cache"
            / "direct_aggregate_v30"
            / "cached.mp4"
        )
        source_meta = source_video.with_name("cached.cache.json")

        with patch.object(
            self.converter,
            "_clone_or_copy_no_hardlink",
            return_value=None,
        ):
            self.converter._store_direct_source_aggregate_cache(
                output_path,
                source_video,
                source_meta,
                {"key": "value"},
                allow_copy=False,
            )

        self.assertFalse(source_video.exists())
        self.assertFalse(source_meta.exists())

    def test_direct_source_cache_allow_copy_false_still_uses_reflink(self):
        output_path = Path(self.temp_dir) / "out.mp4"
        output_path.write_bytes(b"aggregate")
        source_video = Path(self.temp_dir) / "cached.mp4"

        fake_fcntl = types.SimpleNamespace(ioctl=MagicMock(return_value=None))
        unsupported_dev_pairs = set(v30._REFLINK_UNSUPPORTED_DEV_PAIRS)
        v30._REFLINK_UNSUPPORTED_DEV_PAIRS.clear()
        try:
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), patch.object(
                v30.shutil,
                "copyfile",
                side_effect=AssertionError("byte copy fallback should not run"),
            ):
                mode = self.converter._clone_or_copy_no_hardlink(
                    output_path,
                    source_video,
                    allow_copy=False,
                )
        finally:
            v30._REFLINK_UNSUPPORTED_DEV_PAIRS.clear()
            v30._REFLINK_UNSUPPORTED_DEV_PAIRS.update(unsupported_dev_pairs)

        self.assertEqual(mode, "reflink")
        fake_fcntl.ioctl.assert_called_once()

    def test_video_aggregation_workers_scale_with_jobs(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CYCLO_VIDEO_AGG_CAMERA_WORKERS", None)
            os.environ.pop("CYCLO_X264_SPEED_PROFILE", None)
            with patch(
                "cyclo_data.converter.to_lerobot_v30.os.cpu_count",
                return_value=8,
            ):
                self.assertEqual(
                    self.converter._resolve_video_aggregation_workers(
                        camera_count=4,
                        job_count=4,
                    ),
                    4,
                )
                self.assertEqual(
                    self.converter._resolve_video_aggregation_workers(
                        camera_count=4,
                        job_count=18,
                    ),
                    4,
                )
            with patch(
                "cyclo_data.converter.to_lerobot_v30.os.cpu_count",
                return_value=12,
            ):
                self.assertEqual(
                    self.converter._resolve_video_aggregation_workers(
                        camera_count=4,
                        job_count=18,
                    ),
                    5,
                )
            with patch.dict(
                os.environ,
                {"CYCLO_X264_SPEED_PROFILE": "max"},
            ), patch(
                "cyclo_data.converter.to_lerobot_v30.os.cpu_count",
                return_value=12,
            ):
                self.assertEqual(
                    self.converter._resolve_video_aggregation_workers(
                        camera_count=4,
                        job_count=18,
                    ),
                    6,
                )
            with patch.dict(
                os.environ,
                {"CYCLO_X264_SPEED_PROFILE": "max"},
            ), patch(
                "cyclo_data.converter.to_lerobot_v30.os.cpu_count",
                return_value=32,
            ):
                self.assertEqual(
                    self.converter._resolve_video_aggregation_workers(
                        camera_count=4,
                        job_count=18,
                    ),
                    8,
                )

        with patch.dict(
            os.environ,
            {"CYCLO_VIDEO_AGG_CAMERA_WORKERS": "8"},
        ):
            self.assertEqual(
                self.converter._resolve_video_aggregation_workers(
                    camera_count=4,
                    job_count=18,
                ),
                8,
            )

    def test_video_aggregation_workers_use_one_for_full_source_cache_hit(self):
        jobs = [
            ("observation.images.rgb.cam_left_head", 0, 0, [(0, self.input_a, 0.0)]),
            ("observation.images.rgb.cam_right_head", 0, 0, [(1, self.input_b, 0.0)]),
        ]
        matches = {
            (camera_key, chunk_idx, file_idx): (
                Path(self.temp_dir) / f"{camera_key}.mp4",
                Path(self.temp_dir) / f"{camera_key}.cache.json",
                {"key": camera_key},
            )
            for camera_key, chunk_idx, file_idx, _ in jobs
        }
        self.converter._direct_video_aggregation = True

        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertEqual(
                self.converter._resolve_video_aggregation_workers_for_jobs(
                    camera_count=2,
                    jobs=jobs,
                    source_cache_matches=matches,
                ),
                1,
            )

        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_VIDEO_AGG_CAMERA_WORKERS": "2",
            },
            clear=True,
        ):
            self.assertEqual(
                self.converter._resolve_video_aggregation_workers_for_jobs(
                    camera_count=2,
                    jobs=jobs,
                    source_cache_matches=matches,
                ),
                2,
            )

    def test_warm_direct_video_encoder_populates_probe_cache_once(self):
        jobs = [
            (
                "observation.images.rgb.cam_left_head",
                0,
                0,
                [(0, self.input_a, 0.0)],
            )
        ]

        with patch.object(
            self.converter,
            "_quick_video_dimensions_cached",
            return_value=(640, 480),
        ) as dims, patch(
            "cyclo_data.converter.to_lerobot_v30._ffmpeg",
            return_value="ffmpeg",
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            return_value=("libx264", ["-preset", "ultrafast"]),
        ) as encoder:
            self.converter._warm_direct_video_encoder(jobs)

        dims.assert_called_once_with(self.input_a)
        encoder.assert_called_once_with("ffmpeg", width=640, height=480)

    def test_max_speed_profile_trusts_sidecar_frame_count_by_default(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertTrue(self.converter._trust_sidecar_frame_count())
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V30_TRUST_SIDECAR_FRAME_COUNT": "0",
            },
            clear=True,
        ):
            self.assertFalse(self.converter._trust_sidecar_frame_count())

    def test_max_speed_profile_skips_direct_output_validation_by_default(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertFalse(
                self.converter._direct_aggregate_requires_output_validation()
            )
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_V30_VALIDATE_DIRECT_AGGREGATE": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                self.converter._direct_aggregate_requires_output_validation()
            )

    def test_grid_indices_for_raw_video_maps_header_sidecar(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        video_dir = Path(self.temp_dir) / "videos"
        video_dir.mkdir()
        video_path = video_dir / "cam_left_head.mp4"
        video_path.touch()
        pq.write_table(
            pa.table({
                "frame_index": pa.array([0, 1, 2], type=pa.int32()),
                "header_stamp_ns": pa.array(
                    [1_000_000_000, 2_000_000_000, 4_000_000_000],
                    type=pa.int64(),
                ),
                "recv_ns": pa.array(
                    [4_000_000_000, 1_000_000_000, 2_000_000_000],
                    type=pa.int64(),
                ),
            }),
            video_dir / "cam_left_head_timestamps.parquet",
        )
        episode = EpisodeData(
            episode_index=0,
            length=7,
            grid_log_times_sec=[0.5, 1.0, 1.5, 2.0, 3.999, 4.0, 5.0],
        )

        indices = self.converter._grid_indices_for_raw_video(
            episode,
            "cam_left_head",
            video_path,
        )

        np.testing.assert_array_equal(indices, np.array([0, 0, 0, 1, 1, 2, 2]))
        report = self.converter._frame_reuse_reports[(0, "cam_left_head")]
        self.assertEqual(report["time_source"], "header")
        self.assertEqual(report["reused_target_frames"], 4)
        self.assertEqual(report["clamped_before_first_count"], 1)

    def test_direct_video_aggregation_rejects_resize(self):
        self.converter.config.image_resize = (240, 320)

        self.assertFalse(
            self.converter._can_use_direct_video_aggregation(
                [Path(self.temp_dir) / "bag"]
            )
        )

    def test_write_aggregated_videos_uses_direct_writer_when_enabled(self):
        self.converter._direct_video_aggregation = True
        episodes = [
            EpisodeData(
                episode_index=0,
                timestamps=[0.0, 1 / 15],
                observation_state=[np.array([1.0]), np.array([2.0])],
                action=[np.array([1.0]), np.array([2.0])],
                video_files={"cam_left_head": self.input_a},
                tasks=["task"],
                length=2,
            ),
            EpisodeData(
                episode_index=1,
                timestamps=[0.0, 1 / 15, 2 / 15],
                observation_state=[np.array([1.0])] * 3,
                action=[np.array([1.0])] * 3,
                video_files={"cam_left_head": self.input_b},
                tasks=["task"],
                length=3,
            ),
        ]

        with patch.object(
            self.converter,
            "_write_direct_aggregated_video",
        ) as mock_direct, patch.object(
            self.converter,
            "_concatenate_videos",
            side_effect=AssertionError("concat path should not run"),
        ):
            self.converter._write_aggregated_videos(episodes)

        mock_direct.assert_called_once()
        args = mock_direct.call_args.args
        self.assertEqual(args[1], "observation.images.rgb.cam_left_head")
        self.assertEqual([item[0] for item in args[4]], [0, 1])

    def test_write_aggregated_videos_runs_largest_jobs_first(self):
        self.converter._direct_video_aggregation = True
        episodes = [
            EpisodeData(
                episode_index=0,
                timestamps=[0.0, 1 / 15],
                observation_state=[np.array([1.0]), np.array([2.0])],
                action=[np.array([1.0]), np.array([2.0])],
                video_files={"cam_small": self.input_a},
                tasks=["task"],
                length=2,
            ),
            EpisodeData(
                episode_index=1,
                timestamps=[0.0] * 10,
                observation_state=[np.array([1.0])] * 10,
                action=[np.array([1.0])] * 10,
                video_files={"cam_large": self.input_b},
                tasks=["task"],
                length=10,
            ),
        ]

        with patch.object(
            self.converter,
            "_plan_video_batches",
            side_effect=[
                [(0, 0, [(0, self.input_a, 0.0)])],
                [(0, 0, [(1, self.input_b, 0.0)])],
            ],
        ), patch.object(
            self.converter,
            "_resolve_video_aggregation_workers",
            return_value=1,
        ), patch.object(
            self.converter,
            "_write_direct_aggregated_video",
        ) as mock_direct, patch.object(
            self.converter,
            "_update_video_metadata",
        ):
            self.converter._write_aggregated_videos(episodes)

        self.assertEqual(mock_direct.call_args_list[0].args[4][0][0], 1)
        self.assertEqual(mock_direct.call_args_list[1].args[4][0][0], 0)

    def test_write_aggregated_videos_prioritizes_larger_source_bytes(self):
        self.converter._direct_video_aggregation = True
        self.input_a.write_bytes(b"a")
        self.input_b.write_bytes(b"b" * 1024)
        episodes = [
            EpisodeData(
                episode_index=0,
                timestamps=[0.0] * 20,
                observation_state=[np.array([1.0])] * 20,
                action=[np.array([1.0])] * 20,
                video_files={"cam_many_frames": self.input_a},
                tasks=["task"],
                length=20,
            ),
            EpisodeData(
                episode_index=1,
                timestamps=[0.0],
                observation_state=[np.array([1.0])],
                action=[np.array([1.0])],
                video_files={"cam_big_source": self.input_b},
                tasks=["task"],
                length=1,
            ),
        ]

        with patch.object(
            self.converter,
            "_plan_video_batches",
            side_effect=[
                [(0, 0, [(0, self.input_a, 0.0)])],
                [(0, 0, [(1, self.input_b, 0.0)])],
            ],
        ), patch.object(
            self.converter,
            "_resolve_video_aggregation_workers",
            return_value=1,
        ), patch.object(
            self.converter,
            "_write_direct_aggregated_video",
        ) as mock_direct, patch.object(
            self.converter,
            "_update_video_metadata",
        ):
            self.converter._write_aggregated_videos(episodes)

        self.assertEqual(mock_direct.call_args_list[0].args[4][0][0], 1)
        self.assertEqual(mock_direct.call_args_list[1].args[4][0][0], 0)

    def test_direct_aggregate_uses_concat_decoder_when_counts_match(self):
        class FakeProcess:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                pass

        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [
            (0, self.input_a, 0.0),
            (1, self.input_b, 0.0),
        ]

        with patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(64, 48),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            return_value=("libx264", ["-preset", "ultrafast"]),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._validated_video_count",
            return_value=5,
        ) as mock_validate, patch.object(
            self.converter,
            "_direct_aggregated_video_cache_key",
            return_value={},
        ), patch.object(
            self.converter,
            "_try_reuse_aggregated_video_cache",
            return_value=False,
        ) as mock_reuse, patch.object(
            self.converter,
            "_write_aggregated_video_cache",
        ), patch.object(
            self.converter,
            "_store_direct_source_aggregate_cache",
        ) as mock_store_source_cache, patch.object(
            self.converter,
            "_try_reuse_direct_source_aggregate_cache",
            return_value=False,
        ) as mock_source_reuse, patch.object(
            self.converter,
            "_grid_indices_and_source_count_for_raw_video",
            side_effect=[
                (np.array([0, 1]), 2),
                (np.array([0, 1, 2]), 3),
            ],
        ), patch.object(
            self.converter,
            "_get_video_frame_count",
            side_effect=[2, 3],
        ), patch.object(
            self.converter,
            "_pipe_selected_yuv420_frames_concat_decoder",
            return_value=5,
        ) as mock_concat, patch.object(
            self.converter,
            "_pipe_selected_yuv420_frames",
            side_effect=AssertionError("per-video decoder should not run"),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            return_value=FakeProcess(),
        ):
            self.converter._write_direct_aggregated_video(
                Path(self.temp_dir),
                "observation.images.rgb.cam_left_head",
                0,
                0,
                videos,
                episode_by_index,
            )

        mock_concat.assert_called_once()
        self.assertTrue(mock_reuse.call_args.kwargs["require_decode"])
        self.assertTrue(mock_reuse.call_args.kwargs["validate"])
        self.assertTrue(mock_source_reuse.call_args.kwargs["validate"])
        self.assertTrue(mock_validate.call_args.kwargs["require_decode"])
        mock_store_source_cache.assert_called_once()
        self.assertFalse(mock_store_source_cache.call_args.kwargs["allow_copy"])

    def test_concat_decoder_drains_skipped_frames_and_preserves_duplicates(self):
        class FakeDecoder:
            def __init__(self, payload: bytes):
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()

            def poll(self):
                return 0

            def kill(self):
                pass

        frame_size = 6
        frames = [bytes([idx]) * frame_size for idx in range(8)]
        output = io.BytesIO()

        with patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            return_value=FakeDecoder(b"".join(frames)),
        ):
            written = self.converter._pipe_selected_yuv420_frames_concat_decoder(
                "ffmpeg",
                [
                    (0, Path("a.mp4"), np.array([0, 2, 2]), 5, None),
                    (1, Path("b.mp4"), np.array([1]), 3, None),
                ],
                frame_size,
                output,
                width=2,
                height=2,
            )

        self.assertEqual(written, 4)
        self.assertEqual(
            output.getvalue(),
            frames[0] + frames[2] + frames[2] + frames[6],
        )

    def test_concat_decoder_coalesces_contiguous_splice_runs(self):
        class FakeDecoder:
            def __init__(self, payload: bytes):
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()

            def poll(self):
                return 0

            def kill(self):
                pass

        frame_size = 6
        frames = [bytes([idx]) * frame_size for idx in range(5)]
        output = io.BytesIO()
        splice_sizes = []

        def fake_splice(src, dst, size):
            data = src.read(size)
            dst.write(data)
            splice_sizes.append(size)
            return len(data)

        with patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            return_value=FakeDecoder(b"".join(frames)),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._splice_exact",
            side_effect=fake_splice,
        ):
            written = self.converter._pipe_selected_yuv420_frames_concat_decoder(
                "ffmpeg",
                [(0, Path("a.mp4"), np.array([0, 1, 2, 4]), 5, None)],
                frame_size,
                output,
                width=2,
                height=2,
            )

        self.assertEqual(written, 4)
        self.assertEqual(
            output.getvalue(),
            frames[0] + frames[1] + frames[2] + frames[4],
        )
        self.assertEqual(splice_sizes, [frame_size * 3, frame_size])

    def test_direct_aggregate_synced_fallback_forces_software_encoder(self):
        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]
        encoders = []

        def fake_remux(src, indices, out_path, target_fps):
            encoder, _ = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )
            encoders.append(encoder)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"video")

        def fake_concat(*args, **kwargs):
            encoder, _ = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )
            encoders.append(encoder)

        with patch.dict(
            os.environ,
            {"CYCLO_H264_ENCODER": "auto"},
            clear=True,
        ), patch.object(
            self.converter,
            "_grid_indices_for_raw_video",
            return_value=np.array([0, 1]),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30.remux_selected_frames",
            side_effect=fake_remux,
        ), patch.object(
            self.converter,
            "_concatenate_videos",
            side_effect=fake_concat,
        ):
            self.converter._write_direct_aggregate_synced_fallback(
                Path(self.temp_dir),
                "observation.images.rgb.cam_left_head",
                0,
                0,
                videos,
                episode_by_index,
            )

        self.assertEqual(encoders, ["libx264", "libx264", "libx264"])

    def test_concat_decoder_splices_non_sampled_frames_when_stats_enabled(self):
        class FakeDecoder:
            def __init__(self, payload: bytes):
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()

            def poll(self):
                return 0

            def kill(self):
                pass

        frame_size = 6
        frames = [bytes([idx]) * frame_size for idx in range(4)]
        output = io.BytesIO()
        stats = video_sync._StreamingRgbStats()
        splice_calls = []

        def fake_splice(src, dst, size):
            data = src.read(size)
            dst.write(data)
            splice_calls.append(data)
            return len(data)

        with patch.dict(
            os.environ,
            {"CYCLO_VIDEO_STATS_SAMPLES": "2"},
        ), patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            return_value=FakeDecoder(b"".join(frames)),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._splice_exact",
            side_effect=fake_splice,
        ):
            written = self.converter._pipe_selected_yuv420_frames_concat_decoder(
                "ffmpeg",
                [(0, Path("a.mp4"), np.array([0, 1, 2, 3]), 4, stats)],
                frame_size,
                output,
                width=2,
                height=2,
            )

        self.assertEqual(written, 4)
        self.assertEqual(output.getvalue(), b"".join(frames))
        self.assertEqual(splice_calls, [frames[1], frames[2]])
        self.assertEqual(stats.frame_count, 2)

    def test_direct_aggregate_can_trust_sidecar_frame_count(self):
        class FakeProcess:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                pass

        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]

        with patch.dict(
            os.environ,
            {"CYCLO_V30_TRUST_SIDECAR_FRAME_COUNT": "1"},
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(64, 48),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            return_value=("libx264", ["-preset", "ultrafast"]),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._validated_video_count",
            return_value=5,
        ) as mock_validate, patch.object(
            self.converter,
            "_direct_aggregated_video_cache_key",
            return_value={},
        ), patch.object(
            self.converter,
            "_try_reuse_aggregated_video_cache",
            return_value=False,
        ) as mock_reuse, patch.object(
            self.converter,
            "_write_aggregated_video_cache",
        ), patch.object(
            self.converter,
            "_store_direct_source_aggregate_cache",
        ), patch.object(
            self.converter,
            "_try_reuse_direct_source_aggregate_cache",
            return_value=False,
        ) as mock_source_reuse, patch.object(
            self.converter,
            "_grid_indices_and_source_count_for_raw_video",
            side_effect=[
                (np.array([0, 1]), 2),
                (np.array([0, 1, 2]), 3),
            ],
        ), patch.object(
            self.converter,
            "_get_video_frame_count",
            side_effect=AssertionError("frame count should come from sidecar"),
        ), patch.object(
            self.converter,
            "_pipe_selected_yuv420_frames_concat_decoder",
            return_value=5,
        ) as mock_concat, patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            return_value=FakeProcess(),
        ):
            self.converter._write_direct_aggregated_video(
                Path(self.temp_dir),
                "observation.images.rgb.cam_left_head",
                0,
                0,
                videos,
                episode_by_index,
            )

        mock_concat.assert_called_once()
        self.assertFalse(mock_reuse.call_args.kwargs["require_decode"])
        self.assertFalse(mock_reuse.call_args.kwargs["validate"])
        self.assertFalse(mock_source_reuse.call_args.kwargs["validate"])
        mock_validate.assert_not_called()

    def test_direct_aggregate_retries_per_file_decoder_after_concat_error(self):
        class FakeProcess:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO()
                self.running = True

            def wait(self, timeout=None):
                self.running = False
                return 0

            def poll(self):
                return None if self.running else 0

            def kill(self):
                self.running = False

        episode_by_index = {
            0: EpisodeData(episode_index=0, length=2, grid_log_times_sec=[0, 1]),
            1: EpisodeData(episode_index=1, length=3, grid_log_times_sec=[0, 1, 2]),
        }
        videos = [(0, self.input_a, 0.0), (1, self.input_b, 0.0)]

        def fake_per_file_decode(ffmpeg, video_path, indices, *args, **kwargs):
            return int(len(indices))

        with patch.dict(
            os.environ,
            {"CYCLO_V30_TRUST_SIDECAR_FRAME_COUNT": "1"},
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._quick_video_dimensions",
            return_value=(64, 48),
        ), patch(
            "cyclo_data.converter.to_lerobot_v30._h264_encoder",
            return_value=("libx264", ["-preset", "ultrafast"]),
        ), patch.object(
            self.converter,
            "_direct_aggregated_video_cache_key",
            return_value={},
        ), patch.object(
            self.converter,
            "_try_reuse_aggregated_video_cache",
            return_value=False,
        ), patch.object(
            self.converter,
            "_write_aggregated_video_cache",
        ), patch.object(
            self.converter,
            "_direct_source_aggregate_cache_paths",
            return_value=None,
        ), patch.object(
            self.converter,
            "_grid_indices_and_source_count_for_raw_video",
            side_effect=[
                (np.array([0, 1]), 2),
                (np.array([0, 1, 2]), 3),
                (np.array([0, 1]), 2),
                (np.array([0, 1, 2]), 3),
            ],
        ), patch.object(
            self.converter,
            "_pipe_selected_yuv420_frames_concat_decoder",
            side_effect=RuntimeError("concat rejected input"),
        ) as mock_concat, patch.object(
            self.converter,
            "_pipe_selected_yuv420_frames",
            side_effect=fake_per_file_decode,
        ) as mock_per_file, patch.object(
            self.converter,
            "_write_direct_aggregate_synced_fallback",
        ) as mock_fallback, patch.object(
            self.converter,
            "_log_warning",
        ) as mock_warning, patch(
            "cyclo_data.converter.to_lerobot_v30.subprocess.Popen",
            side_effect=lambda *args, **kwargs: FakeProcess(),
        ):
            self.converter._write_direct_aggregated_video(
                Path(self.temp_dir),
                "observation.images.rgb.cam_left_head",
                0,
                0,
                videos,
                episode_by_index,
            )

        mock_concat.assert_called_once()
        self.assertEqual(mock_per_file.call_count, 2)
        mock_fallback.assert_not_called()
        mock_warning.assert_called_once()

    def test_convert_multiple_uses_parent_prepared_cache_without_pool(self):
        bag_a = Path(self.temp_dir) / "bag_a"
        bag_b = Path(self.temp_dir) / "bag_b"
        bag_a.mkdir()
        bag_b.mkdir()
        cached = [
            EpisodeData(
                episode_index=0,
                timestamps=[0.0],
                observation_state=[np.array([1.0], dtype=np.float32)],
                action=[np.array([2.0], dtype=np.float32)],
                tasks=["task"],
                length=1,
            ),
            EpisodeData(
                episode_index=1,
                timestamps=[0.0],
                observation_state=[np.array([3.0], dtype=np.float32)],
                action=[np.array([4.0], dtype=np.float32)],
                tasks=["task"],
                length=1,
            ),
        ]

        with patch.object(
            self.converter,
            "_try_load_prepared_episode_for_bag",
            side_effect=lambda _path, idx: cached[idx],
        ), patch.object(
            self.converter,
            "write_from_episodes",
            return_value=True,
        ) as mock_write, patch(
            "concurrent.futures.ProcessPoolExecutor",
            side_effect=AssertionError("process pool should be skipped"),
        ):
            success = self.converter.convert_multiple_rosbags([bag_a, bag_b])

        self.assertTrue(success)
        mock_write.assert_called_once()
        written = mock_write.call_args.args[0]
        self.assertEqual([ep.episode_index for ep in written], [0, 1])

    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_concatenate_videos_reuses_valid_aggregate_cache(self, mock_run):
        output_path = (
            Path(self.temp_dir)
            / "videos"
            / "observation.images.rgb.cam_left_head"
            / "chunk-000"
            / "file-000.mp4"
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"aggregate")
        cache_key = self.converter._aggregated_video_cache_key(
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
            expected_frames=5,
        )
        self.converter._aggregated_video_cache_path(output_path).write_text(
            json.dumps(cache_key),
            encoding="utf-8",
        )
        self.converter._validate_aggregated_video = MagicMock()

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_left_head",
            0,
            0,
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
        )

        mock_run.assert_not_called()
        self.converter._validate_aggregated_video.assert_called_once()

    @patch("cyclo_data.converter.to_lerobot_v30._h264_encoder")
    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_concatenate_videos_falls_back_to_cfr_reencode(
        self, mock_run, mock_encoder
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_encoder.return_value = ("libx264", ["-preset", "ultrafast", "-crf", "23"])
        self.converter._videos_support_copy_concat = MagicMock(return_value=False)
        self.converter._get_video_frame_count = MagicMock(return_value=5)
        self.converter._probe_video_fps = MagicMock(return_value=15.0)
        self.converter._video_decodes_successfully = MagicMock(return_value=True)

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_right_wrist",
            0,
            0,
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
        )

        cmd = mock_run.call_args.args[0]
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("-an", cmd)
        self.assertIn("-r", cmd)
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("[0:v]fps=15,setpts=PTS-STARTPTS[v0]", filter_complex)
        self.assertIn("[1:v]fps=15,setpts=PTS-STARTPTS[v1]", filter_complex)
        self.assertIn(
            "[v0][v1]concat=n=2:v=1:a=0,fps=15,setpts=N/(15*TB)[outv]",
            filter_complex,
        )
        self.assertIn("-map", cmd)
        self.assertIn("[outv]", cmd)
        self.assertNotIn("copy", cmd)

    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_concatenate_videos_uses_stream_copy_when_compatible(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self.converter._videos_support_copy_concat = MagicMock(return_value=True)
        self.converter._validate_aggregated_video = MagicMock()
        self.converter._video_decodes_successfully = MagicMock(return_value=True)

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_left_head",
            0,
            0,
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
        )

        cmd = mock_run.call_args.args[0]
        self.assertIn("-c:v", cmd)
        self.assertIn("copy", cmd)
        self.assertNotIn("libx264", cmd)

    @patch("cyclo_data.converter.to_lerobot_v30._clone_or_copy_file")
    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_single_compatible_video_uses_direct_copy(
        self, mock_run, mock_clone
    ):
        self.converter._videos_support_copy_concat = MagicMock(return_value=True)
        self.converter._validate_aggregated_video = MagicMock()
        self.converter._video_decodes_successfully = MagicMock(return_value=True)

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_left_head",
            0,
            0,
            [(0, self.input_a, 0.0)],
        )

        mock_clone.assert_called_once()
        mock_run.assert_not_called()

    def test_single_video_copy_validation_keeps_same_path_input(self):
        same_path = self.input_a
        same_path.write_bytes(b"video")
        self.converter._validate_aggregated_video = MagicMock(
            side_effect=RuntimeError("bad copy")
        )

        result = self.converter._copy_single_compatible_video(
            same_path, same_path, expected_frames=1
        )

        self.assertFalse(result)
        self.assertTrue(same_path.exists())

    def test_trusted_synced_outputs_skip_per_input_ffprobe(self):
        input_a = Path(self.temp_dir) / "cam_left_head_synced.mp4"
        input_b = Path(self.temp_dir) / "cam_left_head_2_synced.mp4"
        input_a.write_bytes(b"video-a")
        input_b.write_bytes(b"video-b")
        for path, frames in ((input_a, 2), (input_b, 3)):
            path.with_name(path.stem + ".cache.json").write_text(
                json.dumps({"target_fps": 15, "frame_count": frames}),
                encoding="utf-8",
            )

        with patch.object(
            self.converter,
            "_get_video_dimensions",
            return_value=(480, 640),
        ), patch.object(
            self.converter, "_probe_video_streams"
        ) as mock_probe, patch.object(
            self.converter, "_get_video_frame_count"
        ) as mock_count:
            self.assertTrue(
                self.converter._videos_support_copy_concat(
                    [(0, input_a, 0.0), (1, input_b, 0.0)]
                )
            )

        mock_probe.assert_not_called()
        mock_count.assert_not_called()
        self.assertEqual(
            json.loads(input_a.with_name(input_a.stem + ".cache.json").read_text())[
                "output_height"
            ],
            480,
        )

    def test_trusted_synced_outputs_use_cached_dimensions(self):
        input_a = Path(self.temp_dir) / "cam_left_head_synced.mp4"
        input_b = Path(self.temp_dir) / "cam_left_head_2_synced.mp4"
        input_a.write_bytes(b"video-a")
        input_b.write_bytes(b"video-b")
        for path, frames in ((input_a, 2), (input_b, 3)):
            path.with_name(path.stem + ".cache.json").write_text(
                json.dumps({
                    "target_fps": 15,
                    "frame_count": frames,
                    "output_height": 480,
                    "output_width": 640,
                }),
                encoding="utf-8",
            )

        with patch.object(
            self.converter,
            "_get_video_dimensions",
            side_effect=AssertionError("dimensions should come from cache"),
        ), patch.object(
            self.converter, "_probe_video_streams"
        ) as mock_probe:
            self.assertTrue(
                self.converter._videos_support_copy_concat(
                    [(0, input_a, 0.0), (1, input_b, 0.0)]
                )
            )

        mock_probe.assert_not_called()

    def test_trusted_synced_outputs_reject_dimension_mismatch(self):
        input_a = Path(self.temp_dir) / "cam_left_head_synced.mp4"
        input_b = Path(self.temp_dir) / "cam_left_head_2_synced.mp4"
        input_a.write_bytes(b"video-a")
        input_b.write_bytes(b"video-b")
        for path, frames in ((input_a, 2), (input_b, 3)):
            path.with_name(path.stem + ".cache.json").write_text(
                json.dumps({"target_fps": 15, "frame_count": frames}),
                encoding="utf-8",
            )

        with patch.object(
            self.converter,
            "_get_video_dimensions",
            side_effect=[(480, 640), (720, 1280)],
        ), patch.object(
            self.converter, "_probe_video_streams"
        ) as mock_probe:
            self.assertFalse(
                self.converter._videos_support_copy_concat(
                    [(0, input_a, 0.0), (1, input_b, 0.0)]
                )
            )

        mock_probe.assert_not_called()

    @patch("cyclo_data.converter.to_lerobot_v30._h264_encoder")
    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_stream_copy_validation_failure_reencodes(
        self, mock_run, mock_encoder
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        mock_encoder.return_value = ("libx264", ["-preset", "ultrafast", "-crf", "23"])
        self.converter._videos_support_copy_concat = MagicMock(return_value=True)
        self.converter._validate_aggregated_video = MagicMock(
            side_effect=[RuntimeError("bad copy"), None]
        )

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_left_head",
            0,
            0,
            [(0, self.input_a, 0.0), (1, self.input_b, 0.0)],
        )

        copy_cmd = mock_run.call_args_list[0].args[0]
        fallback_cmd = mock_run.call_args_list[1].args[0]
        self.assertIn("copy", copy_cmd)
        self.assertIn("libx264", fallback_cmd)

    @patch("cyclo_data.converter.to_lerobot_v30._h264_encoder")
    @patch("cyclo_data.converter.to_lerobot_v30.subprocess.run")
    def test_ffmpeg_failure_raises_with_stderr(self, mock_run, mock_encoder):
        mock_run.return_value = MagicMock(returncode=1, stderr="bad concat")
        mock_encoder.return_value = ("libx264", ["-preset", "ultrafast", "-crf", "23"])
        self.converter._videos_support_copy_concat = MagicMock(return_value=False)

        with self.assertRaisesRegex(RuntimeError, "bad concat"):
            self.converter._concatenate_videos(
                Path(self.temp_dir),
                "observation.images.rgb.cam_left_head",
                0,
                0,
                [(0, self.input_a, 0.0)],
            )

    def test_validate_aggregated_video_rejects_frame_mismatch(self):
        self.converter._get_video_frame_count = MagicMock(return_value=4)

        with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
            self.converter._validate_aggregated_video(self.input_a, 5)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are required for integration video concat test",
    )
    def test_synthetic_video_concat_frame_count_and_fps(self):
        try:
            import cv2
        except Exception:
            self.skipTest("cv2 is required for integration video concat test")

        input_dir = Path(self.temp_dir) / "inputs"
        input_dir.mkdir()
        video_a = input_dir / "a.mp4"
        video_b = input_dir / "b.mp4"
        self._write_synthetic_video(cv2, video_a, 2, (0, 0, 255))
        self._write_synthetic_video(cv2, video_b, 3, (0, 255, 0))

        self.converter._concatenate_videos(
            Path(self.temp_dir),
            "observation.images.rgb.cam_left_head",
            0,
            0,
            [(0, video_a, 2 / 15), (1, video_b, 3 / 15)],
        )

        output = (
            Path(self.temp_dir)
            / "videos/observation.images.rgb.cam_left_head/chunk-000/file-000.mp4"
        )
        self.assertEqual(self.converter._get_video_frame_count(output), 5)
        self.assertAlmostEqual(self.converter._probe_video_fps(output), 15.0, places=2)

    def _write_synthetic_video(self, cv2, path: Path, frames: int, color):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.config.fps),
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for idx in range(frames):
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                str(idx),
                (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()


class TestVideoSyncTempResources(unittest.TestCase):
    """Tests for video_sync temporary-directory resource controls."""

    def test_default_temp_parent_is_output_adjacent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            parent, cleanup = video_sync._resolve_tmp_parent(out)
            try:
                self.assertEqual(parent, out.parent / ".video_sync_tmp")
                self.assertTrue(parent.exists())
                self.assertTrue(cleanup)
            finally:
                video_sync._cleanup_tmp_parent(parent, cleanup)
            self.assertFalse(parent.exists())

    def test_temp_parent_env_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "custom_tmp"
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            with patch.dict(
                os.environ, {"CYCLO_VIDEO_SYNC_TMPDIR": str(override)}
            ):
                parent, cleanup = video_sync._resolve_tmp_parent(out)
            self.assertEqual(parent, override)
            self.assertTrue(parent.exists())
            self.assertFalse(cleanup)

    def test_min_free_space_gate_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            fake_usage = types.SimpleNamespace(free=10 * 1024 * 1024)
            with patch.dict(
                os.environ, {"CYCLO_VIDEO_SYNC_MIN_FREE_MB": "20"}
            ), patch.object(
                video_sync.shutil, "disk_usage", return_value=fake_usage
            ):
                with self.assertRaisesRegex(RuntimeError, "requires at least 20"):
                    video_sync._check_tmp_free_space(parent)

    def test_h264_encoder_defaults_to_libx264(self):
        with patch.dict(os.environ, {}, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            ["-preset", "ultrafast", "-crf", "32", "-tune", "zerolatency"],
        )

    def test_h264_encoder_thread_local_software_override_uses_libx264(self):
        with patch.dict(
            os.environ,
            {"CYCLO_H264_ENCODER": "auto"},
            clear=True,
        ):
            with video_sync._force_h264_software_encoder():
                encoder, opts = video_sync._h264_encoder(
                    "ffmpeg",
                    width=640,
                    height=480,
                )

        self.assertEqual(encoder, "libx264")
        self.assertIn("-preset", opts)

    def test_h264_encoder_quality_profile_restores_legacy_crf(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "quality"},
            clear=True,
        ):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(opts, ["-preset", "ultrafast", "-crf", "23"])

    def test_h264_encoder_max_speed_profile_defaults_to_libx264(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            [
                "-preset", "ultrafast",
                "-qp", "51",
                "-tune", "zerolatency",
                "-g", "1",
                "-threads", "1",
            ],
        )

    def test_h264_encoder_max_speed_software_uses_fastest_qp(self):
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_H264_ENCODER": "software",
            },
            clear=True,
        ):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            [
                "-preset", "ultrafast",
                "-qp", "51",
                "-tune", "zerolatency",
                "-g", "1",
                "-threads", "1",
            ],
        )

    def test_h264_encoder_honors_x264_qp_env(self):
        env = {
            "CYCLO_X264_PRESET": "superfast",
            "CYCLO_X264_QP": "45",
            "CYCLO_X264_CRF": "28",
        }
        with patch.dict(os.environ, env, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            ["-preset", "superfast", "-qp", "45", "-tune", "zerolatency"],
        )

    def test_h264_encoder_honors_x264_gop_env(self):
        env = {
            "CYCLO_X264_SPEED_PROFILE": "max",
            "CYCLO_H264_ENCODER": "software",
            "CYCLO_X264_GOP": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
        )

        self.assertEqual(encoder, "libx264")
        self.assertIn("-g", opts)
        self.assertEqual(opts[opts.index("-g") + 1], "120")

    def test_h264_encoder_honors_x264_threads_env(self):
        env = {
            "CYCLO_H264_ENCODER": "software",
            "CYCLO_X264_THREADS": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(opts[-2:], ["-threads", "2"])

    def test_h264_encoder_auto_uses_current_x264_threads_env(self):
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_X264_THREADS": "2",
            },
            clear=True,
        ):
            first_encoder, first_opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_X264_THREADS": "3",
            },
            clear=True,
        ):
            second_encoder, second_opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(first_encoder, "libx264")
        self.assertEqual(second_encoder, "libx264")
        self.assertEqual(first_opts[-2:], ["-threads", "2"])
        self.assertEqual(second_opts[-2:], ["-threads", "3"])

    def test_h264_encoder_honors_x264_tuning_env(self):
        env = {
            "CYCLO_X264_PRESET": "superfast",
            "CYCLO_X264_CRF": "28",
            "CYCLO_X264_TUNE": "zerolatency",
        }
        with patch.dict(os.environ, env, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
            )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            ["-preset", "superfast", "-crf", "28", "-tune", "zerolatency"],
        )

    def test_h264_encoder_unsupported_value_uses_x264_tuning_env(self):
        env = {
            "CYCLO_H264_ENCODER": "unsupported_encoder",
            "CYCLO_X264_PRESET": "veryfast",
            "CYCLO_X264_CRF": "30",
        }
        with patch.dict(os.environ, env, clear=True):
            encoder, opts = video_sync._h264_encoder(
                "ffmpeg",
                width=640,
                height=480,
        )

        self.assertEqual(encoder, "libx264")
        self.assertEqual(
            opts,
            ["-preset", "veryfast", "-crf", "30", "-tune", "zerolatency"],
        )

    def test_h264_encoder_auto_uses_libx264_without_probe(self):
        with patch.dict(os.environ, {"CYCLO_H264_ENCODER": "auto"}, clear=True), \
            patch.object(video_sync.subprocess, "run") as mock_run:
            first = video_sync._h264_encoder("ffmpeg", width=640, height=480)
            second = video_sync._h264_encoder("ffmpeg", width=640, height=480)

        self.assertEqual(first, second)
        self.assertEqual(first[0], "libx264")
        mock_run.assert_not_called()

    def test_h264_encoder_keeps_tiny_outputs_on_libx264(self):
        with patch.dict(os.environ, {"CYCLO_H264_ENCODER": "auto"}, clear=True):
            encoder, _ = video_sync._h264_encoder(
                "ffmpeg",
                width=64,
                height=48,
            )

        self.assertEqual(encoder, "libx264")

    def test_decode_validation_uses_opencv_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "ok.mp4"
            video_path.write_bytes(b"mp4")

            class FakeCapture:
                def __init__(self, _path):
                    pass

                def isOpened(self):
                    return True

                def read(self):
                    return True, object()

                def release(self):
                    pass

            fake_cv2 = types.SimpleNamespace(VideoCapture=FakeCapture)
            with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(
                video_sync.subprocess,
                "run",
                side_effect=AssertionError("ffmpeg should not run"),
            ):
                self.assertTrue(
                    video_sync._video_decodes_successfully(video_path, "ffmpeg")
                )

    def test_output_validation_uses_opencv_count_fps_decode_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "ok.mp4"
            video_path.write_bytes(b"mp4")

            class FakeCapture:
                def __init__(self, _path):
                    pass

                def isOpened(self):
                    return True

                def get(self, prop):
                    if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                        return 7
                    if prop == fake_cv2.CAP_PROP_FPS:
                        return 15.0
                    return 0

                def read(self):
                    return True, object()

                def release(self):
                    pass

            fake_cv2 = types.SimpleNamespace(
                VideoCapture=FakeCapture,
                CAP_PROP_FRAME_COUNT=1,
                CAP_PROP_FPS=2,
            )
            with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(
                video_sync.subprocess,
                "run",
                side_effect=AssertionError("ffprobe should not run"),
            ):
                count = video_sync._validated_video_count(
                    output_mp4=video_path,
                    expected_frames=7,
                    target_fps=15,
                    ffmpeg="ffmpeg",
                    label="streaming",
                )

            self.assertEqual(count, 7)

    def test_output_validation_can_skip_decode_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "ok.mp4"
            video_path.write_bytes(b"mp4")

            class FakeCapture:
                def __init__(self, _path):
                    pass

                def isOpened(self):
                    return True

                def get(self, prop):
                    if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                        return 7
                    if prop == fake_cv2.CAP_PROP_FPS:
                        return 15.0
                    return 0

                def read(self):
                    raise AssertionError("decode read should be skipped")

                def release(self):
                    pass

            fake_cv2 = types.SimpleNamespace(
                VideoCapture=FakeCapture,
                CAP_PROP_FRAME_COUNT=1,
                CAP_PROP_FPS=2,
            )
            with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(
                video_sync.subprocess,
                "run",
                side_effect=AssertionError("ffprobe should not run"),
            ):
                count = video_sync._validated_video_count(
                    output_mp4=video_path,
                    expected_frames=7,
                    target_fps=15,
                    ffmpeg="ffmpeg",
                    label="streaming",
                    require_decode=False,
                )

            self.assertEqual(count, 7)

    def test_decode_validation_strict_mode_uses_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "ok.mp4"
            video_path.write_bytes(b"mp4")
            result = types.SimpleNamespace(returncode=0)

            with patch.dict(
                os.environ,
                {"CYCLO_VIDEO_SYNC_STRICT_FFMPEG_DECODE": "1"},
            ), patch.object(
                video_sync.subprocess,
                "run",
                return_value=result,
            ) as mock_run:
                self.assertTrue(
                    video_sync._video_decodes_successfully(video_path, "ffmpeg")
                )

            mock_run.assert_called_once()

    def test_capture_advance_grabs_skipped_frames(self):
        class FakeCapture:
            def __init__(self):
                self.grabs = 0
                self.reads = 0

            def grab(self):
                self.grabs += 1
                return True

            def read(self):
                self.reads += 1
                return True, np.array([[[self.reads]]], dtype=np.uint8)

        cap = FakeCapture()
        current_idx, frame, clamped = video_sync._advance_capture_to_index(
            cap,
            current_idx=-1,
            requested_idx=3,
            last_frame=None,
            input_name="input.mp4",
        )

        self.assertEqual(current_idx, 3)
        self.assertFalse(clamped)
        self.assertEqual(cap.grabs, 3)
        self.assertEqual(cap.reads, 1)
        self.assertEqual(int(frame[0, 0, 0]), 1)

    def test_write_frame_bgr_writes_numpy_buffer_without_tobytes(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        out = io.BytesIO()

        video_sync._write_frame_bgr(out, frame)

        self.assertEqual(out.getvalue(), frame.tobytes())

    def test_write_frame_bytes_uses_fileno_pipe(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(write_fd, "wb", buffering=0) as out:
                video_sync._write_frame_bytes(out, bytearray(b"abcde"))
            with os.fdopen(read_fd, "rb", buffering=0) as inp:
                self.assertEqual(inp.read(), b"abcde")
        finally:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_write_frame_bytes_falls_back_without_fileno(self):
        out = io.BytesIO()

        video_sync._write_frame_bytes(out, bytearray(b"abcde"))

        self.assertEqual(out.getvalue(), b"abcde")

    def test_write_repeated_frame_bytes_uses_writev_pipe(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(write_fd, "wb", buffering=0) as out:
                video_sync._write_repeated_frame_bytes(out, bytearray(b"ab"), 3)
            with os.fdopen(read_fd, "rb", buffering=0) as inp:
                self.assertEqual(inp.read(), b"ababab")
        finally:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_write_repeated_frame_bytes_caches_iov_limit(self):
        video_sync._writev_iov_batch_limit.cache_clear()
        with patch.object(video_sync.os, "sysconf", return_value=16) as sysconf:
            first = video_sync._writev_iov_batch_limit()
            second = video_sync._writev_iov_batch_limit()

        sysconf.assert_called_once_with("SC_IOV_MAX")
        self.assertEqual(first, 16)
        self.assertEqual(second, 16)

    def test_write_repeated_frame_bytes_falls_back_without_fileno(self):
        out = io.BytesIO()

        video_sync._write_repeated_frame_bytes(out, bytearray(b"ab"), 3)

        self.assertEqual(out.getvalue(), b"ababab")

    def test_contiguous_forward_run_stops_before_duplicate_run(self):
        self.assertEqual(
            video_sync._contiguous_forward_run_length(np.array([0, 1, 2, 3]), 0),
            4,
        )
        self.assertEqual(
            video_sync._contiguous_forward_run_length(np.array([0, 1, 1]), 0),
            1,
        )
        self.assertEqual(
            video_sync._contiguous_forward_run_length(np.array([0, 1, 2, 2]), 0),
            2,
        )

    def test_splice_exact_moves_pipe_bytes(self):
        if not hasattr(os, "splice"):
            self.skipTest("os.splice is not available on this platform")
        src_read, src_write = os.pipe()
        dst_read, dst_write = os.pipe()
        try:
            os.write(src_write, b"abcdef")
            with os.fdopen(src_read, "rb", buffering=0) as src, os.fdopen(
                dst_write, "wb", buffering=0
            ) as dst:
                moved = video_sync._splice_exact(src, dst, 6)
            self.assertEqual(moved, 6)
            self.assertEqual(os.read(dst_read, 6), b"abcdef")
        finally:
            for fd in (src_read, src_write, dst_read, dst_write):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_drain_exact_discards_pipe_bytes(self):
        src_read, src_write = os.pipe()
        discard_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.write(src_write, b"abcdef")
            with os.fdopen(src_read, "rb", buffering=0) as src:
                drained = video_sync._drain_exact(
                    src,
                    4,
                    discard_fd=discard_fd,
                )
                self.assertEqual(drained, 4)
                self.assertEqual(src.read(2), b"ef")
        finally:
            for fd in (src_read, src_write, discard_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def test_read_exact_into_handles_partial_pipe_reads(self):
        class PartialPipe:
            def __init__(self, chunks):
                self.chunks = list(chunks)

            def readinto(self, target):
                if not self.chunks:
                    return 0
                chunk = self.chunks.pop(0)
                n = min(len(target), len(chunk))
                target[:n] = chunk[:n]
                if n < len(chunk):
                    self.chunks.insert(0, chunk[n:])
                return n

        buffer = bytearray(5)

        count = video_sync._read_exact_into(
            PartialPipe([b"ab", b"c", b"de"]),
            buffer,
            5,
        )

        self.assertEqual(count, 5)
        self.assertEqual(bytes(buffer), b"abcde")

    def test_yuv420_pipe_eligibility(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                video_sync._use_yuv420_pipe(
                    width=640,
                    height=480,
                    rotation_deg=0,
                    image_resize=None,
                )
            )
            self.assertFalse(
                video_sync._use_yuv420_pipe(
                    width=641,
                    height=480,
                    rotation_deg=0,
                    image_resize=None,
                )
            )
            self.assertFalse(
                video_sync._use_yuv420_pipe(
                    width=640,
                    height=480,
                    rotation_deg=90,
                    image_resize=None,
                )
            )
        with patch.dict(
            os.environ,
            {"CYCLO_VIDEO_SYNC_DISABLE_YUV420_PIPE": "1"},
        ):
            self.assertFalse(
                video_sync._use_yuv420_pipe(
                    width=640,
                    height=480,
                    rotation_deg=0,
                    image_resize=None,
                )
            )

    def test_streaming_prefers_yuv420_ffmpeg_pipe(self):
        expected = video_sync.VideoSyncResult(
            frame_count=1,
            mode="stream_encode",
            output_height=480,
            output_width=640,
        )
        with patch.dict(os.environ, {}, clear=True), \
            patch.object(
                video_sync,
                "_remux_selected_frames_ffmpeg_yuv420_pipe",
                return_value=expected,
            ) as yuv_pipe, \
            patch.object(
                video_sync,
                "_remux_selected_frames_opencv_streaming",
                side_effect=AssertionError("OpenCV fallback should not run"),
            ):
            result = video_sync._remux_selected_frames_streaming(
                input_mp4=Path("input.mp4"),
                frame_indices=[0],
                output_mp4=Path("output.mp4"),
                target_fps=30,
                rotation_deg=0,
                image_resize=None,
                ffmpeg="ffmpeg",
            )

        self.assertIs(result, expected)
        yuv_pipe.assert_called_once()

    def test_streaming_can_disable_yuv420_ffmpeg_pipe(self):
        expected = video_sync.VideoSyncResult(frame_count=1, mode="stream_encode")
        with patch.dict(
            os.environ,
            {"CYCLO_VIDEO_SYNC_DISABLE_YUV420_PIPE": "1"},
            clear=True,
        ), patch.object(
            video_sync,
            "_remux_selected_frames_ffmpeg_yuv420_pipe",
            side_effect=AssertionError("YUV pipe should be skipped"),
        ), patch.object(
            video_sync,
            "_remux_selected_frames_opencv_streaming",
            return_value=expected,
        ) as opencv_stream:
            result = video_sync._remux_selected_frames_streaming(
                input_mp4=Path("input.mp4"),
                frame_indices=[0],
                output_mp4=Path("output.mp4"),
                target_fps=30,
                rotation_deg=0,
                image_resize=None,
                ffmpeg="ffmpeg",
            )

        self.assertIs(result, expected)
        opencv_stream.assert_called_once()

    def test_ffmpeg_threads_default_is_portable_decoder_side_default(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            video_sync.os,
            "cpu_count",
            return_value=8,
        ):
            self.assertEqual(video_sync._ffmpeg_threads_arg(), ["-threads", "2"])
        with patch.dict(os.environ, {}, clear=True), patch.object(
            video_sync.os,
            "cpu_count",
            return_value=2,
        ):
            self.assertEqual(video_sync._ffmpeg_threads_arg(), ["-threads", "1"])
        with patch.dict(os.environ, {"CYCLO_FFMPEG_THREADS": "4"}, clear=True):
            self.assertEqual(video_sync._ffmpeg_threads_arg(), ["-threads", "4"])

    def test_h264_decoder_uses_portable_software_default(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ), patch.object(video_sync.os, "cpu_count", return_value=8):
            self.assertEqual(
                video_sync._ffmpeg_h264_decoder_args("ffmpeg"),
                ["-threads", "2"],
            )

    def test_h264_decoder_uses_portable_threads_in_max_profile(self):
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ), patch.object(video_sync.os, "cpu_count", return_value=8):
            self.assertEqual(
                video_sync._ffmpeg_h264_decoder_args("ffmpeg"),
                ["-threads", "2"],
            )

    def test_ffmpeg_pipe_size_is_explicit_host_tuning_knob(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(video_sync._ffmpeg_pipe_size(640 * 480 * 3 // 2), 0)
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertEqual(
                video_sync._ffmpeg_pipe_size(640 * 480 * 3 // 2),
                0,
            )
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_FFMPEG_PIPE_SIZE": "1048576",
            },
            clear=True,
        ):
            self.assertEqual(
                video_sync._ffmpeg_pipe_size(640 * 480 * 3 // 2),
                1_048_576,
            )

    def test_max_speed_profile_skips_faststart_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(video_sync._mp4_faststart_args(), ["-movflags", "+faststart"])
        with patch.dict(
            os.environ,
            {"CYCLO_X264_SPEED_PROFILE": "max"},
            clear=True,
        ):
            self.assertEqual(video_sync._mp4_faststart_args(), [])
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_MP4_FASTSTART": "1",
            },
            clear=True,
        ):
            self.assertEqual(video_sync._mp4_faststart_args(), ["-movflags", "+faststart"])

    def test_streaming_rgb_stats_match_numpy_stats(self):
        samples = [
            np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [[255, 128, 64], [1, 2, 3]],
                    [[4, 5, 6], [7, 8, 9]],
                ],
                dtype=np.uint8,
            ),
        ]

        stats = video_sync._stats_from_samples(samples)
        frames = np.asarray(samples, dtype=np.float32) / 255.0

        self.assertEqual(stats["count"], [2])
        for channel in range(3):
            channel_data = frames[:, :, :, channel]
            self.assertAlmostEqual(
                stats["mean"][channel][0][0],
                float(channel_data.mean()),
                places=7,
            )
            self.assertAlmostEqual(
                stats["std"][channel][0][0],
                float(channel_data.std()),
                places=7,
            )
            self.assertAlmostEqual(
                stats["min"][channel][0][0],
                float(channel_data.min()),
                places=7,
            )
            self.assertAlmostEqual(
                stats["max"][channel][0][0],
                float(channel_data.max()),
                places=7,
            )

    def test_video_stats_sample_positions_default_and_env(self):
        with patch.dict(os.environ, {}, clear=True):
            default_positions = video_sync._video_stats_sample_positions(100)
        self.assertEqual(len(default_positions), 8)
        self.assertIn(0, default_positions)
        self.assertIn(99, default_positions)

        with patch.dict(os.environ, {"CYCLO_VIDEO_STATS_SAMPLES": "0"}):
            self.assertEqual(video_sync._video_stats_sample_positions(100), set())

        with patch.dict(os.environ, {"CYCLO_VIDEO_STATS_SAMPLES": "5"}):
            self.assertEqual(
                len(video_sync._video_stats_sample_positions(100)),
                5,
            )

    def test_max_speed_profile_defaults_video_stats_to_zero(self):
        with patch.dict(os.environ, {"CYCLO_X264_SPEED_PROFILE": "max"}, clear=True):
            self.assertEqual(
                base_converter.RosbagToLerobotConverterBase._video_stats_sample_budget(),
                0,
            )
        with patch.dict(
            os.environ,
            {
                "CYCLO_X264_SPEED_PROFILE": "max",
                "CYCLO_VIDEO_STATS_SAMPLES": "5",
            },
            clear=True,
        ):
            self.assertEqual(
                base_converter.RosbagToLerobotConverterBase._video_stats_sample_budget(),
                5,
            )

    def test_remux_uses_resolved_temp_parent_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "videos" / "cam_synced.mp4"
            input_mp4 = Path(tmpdir) / "input.mp4"
            input_mp4.write_bytes(b"fake")
            seen = {}

            class FakeTemporaryDirectory:
                def __init__(self, prefix=None, dir=None):
                    seen["prefix"] = prefix
                    seen["dir"] = Path(dir)
                    self.path = Path(dir) / f"{prefix}fake"

                def __enter__(self):
                    self.path.mkdir(parents=True, exist_ok=True)
                    return str(self.path)

                def __exit__(self, exc_type, exc, tb):
                    shutil.rmtree(self.path, ignore_errors=True)

            def fake_inner(**kwargs):
                raise RuntimeError("stop before ffmpeg")

            with patch.object(
                video_sync, "_ffmpeg", return_value="ffmpeg"
            ), patch.object(
                video_sync, "_can_packet_copy_sync", return_value=False
            ), patch.object(
                video_sync,
                "_remux_selected_frames_streaming",
                side_effect=RuntimeError("stream failed"),
            ), patch.object(
                video_sync.tempfile, "TemporaryDirectory", FakeTemporaryDirectory
            ), patch.object(
                video_sync, "_remux_selected_frames_in_tmp", side_effect=fake_inner
            ):
                with self.assertRaisesRegex(RuntimeError, "stop before ffmpeg"):
                    video_sync.remux_selected_frames(
                        input_mp4, [0], out, target_fps=15
                    )

            self.assertEqual(seen["prefix"], "video_sync_")
            self.assertEqual(seen["dir"], out.parent / ".video_sync_tmp")
            self.assertFalse((out.parent / ".video_sync_tmp").exists())

    def test_packet_copy_helper_uses_setts_and_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_mp4 = tmp / "input.mp4"
            output_mp4 = tmp / "out.mp4"
            input_mp4.write_bytes(b"fake")
            commands = []

            def fake_run(cmd, *args, **kwargs):
                commands.append(list(cmd))
                output_mp4.write_bytes(b"mp4")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(
                video_sync.subprocess, "run", side_effect=fake_run
            ), patch.object(
                video_sync, "_video_frame_count_and_fps", return_value=(3, 15.0)
            ), patch.object(
                video_sync, "_video_decodes_successfully", return_value=True
            ):
                result = video_sync._remux_selected_frames_packet_copy(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1, 2],
                    output_mp4=output_mp4,
                    target_fps=15,
                    ffmpeg="ffmpeg",
                )

            cmd = commands[0]
            self.assertEqual(result.mode, "packet_copy")
            self.assertIn("-c:v", cmd)
            self.assertIn("copy", cmd)
            self.assertIn("-bsf:v", cmd)
            self.assertTrue(any("setts=pts=N*6000" in item for item in cmd))
            self.assertIn("+faststart", cmd)

    def test_yuv420_pipe_uses_portable_decoder_threads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_mp4 = tmp / "input.mp4"
            output_mp4 = tmp / "out.mp4"
            input_mp4.write_bytes(b"fake")
            frame0 = bytes([0, 1, 2, 3, 4, 5])
            frame1 = bytes([6, 7, 8, 9, 10, 11])
            commands = []
            encoder_stdin = io.BytesIO()

            class DecoderProcess:
                def __init__(self):
                    self.stdout = io.BytesIO(frame0 + frame1)
                    self.stderr = io.BytesIO()

                def poll(self):
                    return None

                def kill(self):
                    pass

            class EncoderStdin:
                closed = False

                def write(self, data):
                    encoder_stdin.write(data)

                def close(self):
                    self.closed = True

            class EncoderProcess:
                def __init__(self):
                    self.stdin = EncoderStdin()
                    self.stderr = io.BytesIO()

                def wait(self, timeout=None):
                    output_mp4.write_bytes(b"mp4")
                    return 0

                def poll(self):
                    return 0

                def kill(self):
                    pass

            def fake_popen(cmd, *args, **kwargs):
                commands.append(list(cmd))
                if "pipe:1" in cmd:
                    return DecoderProcess()
                return EncoderProcess()

            with patch.object(
                video_sync, "_quick_video_dimensions", return_value=(2, 2)
            ), patch.object(
                video_sync.subprocess, "Popen", side_effect=fake_popen
            ), patch.object(
                video_sync, "_h264_encoder", return_value=("libx264", [])
            ), patch.object(
                video_sync, "_video_frame_count_and_fps", return_value=(2, 15.0)
            ), patch.object(
                video_sync, "_video_decodes_successfully", return_value=True
            ):
                result = video_sync._remux_selected_frames_ffmpeg_yuv420_pipe(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    output_mp4=output_mp4,
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )

            self.assertEqual(result.mode, "stream_encode")
            self.assertEqual(encoder_stdin.getvalue(), frame0 + frame1)
            self.assertIn("-threads", commands[0])
            self.assertIn("-fps_mode", commands[0])
            self.assertIn("passthrough", commands[0])

    def test_streaming_falls_back_to_opencv_after_yuv420_failure(self):
        expected = video_sync.VideoSyncResult(frame_count=1, mode="stream_encode")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = Path(tmpdir) / "input.mp4"
            output_mp4 = Path(tmpdir) / "out.mp4"
            input_mp4.write_bytes(b"fake")
            with patch.dict(os.environ, {}, clear=True), patch.object(
                video_sync,
                "_remux_selected_frames_ffmpeg_yuv420_pipe",
                side_effect=RuntimeError("not eligible"),
            ) as mock_yuv420, patch.object(
                video_sync,
                "_remux_selected_frames_opencv_streaming",
                return_value=expected,
            ) as mock_opencv:
                result = video_sync._remux_selected_frames_streaming(
                    input_mp4=input_mp4,
                    frame_indices=[0],
                    output_mp4=output_mp4,
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )

        self.assertEqual(result, expected)
        mock_yuv420.assert_called_once()
        mock_opencv.assert_called_once()

    def test_opencv_streaming_uses_ffmpeg_encoder_by_default(self):
        expected = video_sync.VideoSyncResult(frame_count=1, mode="stream_encode")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = Path(tmpdir) / "input.mp4"
            output_mp4 = Path(tmpdir) / "out.mp4"
            input_mp4.write_bytes(b"fake")
            with patch.dict(os.environ, {}, clear=True), patch.object(
                video_sync,
                "_remux_selected_frames_opencv_ffmpeg_encoder",
                return_value=expected,
            ) as mock_ffmpeg:
                result = video_sync._remux_selected_frames_opencv_streaming(
                    input_mp4=input_mp4,
                    frame_indices=[0],
                    output_mp4=output_mp4,
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )

        self.assertEqual(result, expected)
        mock_ffmpeg.assert_called_once()

    def test_packet_copy_failure_falls_back_to_streaming_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = Path(tmpdir) / "input.mp4"
            output_mp4 = Path(tmpdir) / "out.mp4"
            input_mp4.write_bytes(b"fake")
            expected = video_sync.VideoSyncResult(
                frame_count=1,
                stats=None,
                used_fallback=False,
                mode="stream_encode",
            )

            with patch.object(
                video_sync, "_ffmpeg", return_value="ffmpeg"
            ), patch.object(
                video_sync, "_can_packet_copy_sync", return_value=True
            ), patch.object(
                video_sync,
                "_remux_selected_frames_packet_copy",
                side_effect=RuntimeError("copy failed"),
            ), patch.object(
                video_sync,
                "_remux_selected_frames_streaming",
                return_value=expected,
            ) as mock_stream:
                result = video_sync.remux_selected_frames(
                    input_mp4, [0], output_mp4, target_fps=15
                )

            self.assertEqual(result.mode, "stream_encode")
            mock_stream.assert_called_once()

    def test_rotation_skips_packet_copy_and_uses_streaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = Path(tmpdir) / "input.mp4"
            output_mp4 = Path(tmpdir) / "out.mp4"
            input_mp4.write_bytes(b"fake")
            expected = video_sync.VideoSyncResult(frame_count=1, mode="stream_encode")

            with patch.object(
                video_sync, "_ffmpeg", return_value="ffmpeg"
            ), patch.object(
                video_sync,
                "_remux_selected_frames_streaming",
                return_value=expected,
            ) as mock_stream:
                result = video_sync.remux_selected_frames(
                    input_mp4, [0], output_mp4, target_fps=15, rotation_deg=90
                )

            self.assertEqual(result.mode, "stream_encode")
            mock_stream.assert_called_once()

    def test_packet_copy_eligibility_rejects_non_identity_cases(self):
        input_mp4 = Path("input.mp4")

        with patch.object(
            video_sync,
            "_probe_video_stream",
            return_value={"codec_name": "h264", "has_b_frames": 0},
        ), patch.object(video_sync, "_ffmpeg_supports_setts", return_value=True):
            self.assertTrue(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1, 2],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )
            )
            self.assertFalse(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 0, 1],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )
            )
            self.assertFalse(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 2],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )
            )
            self.assertFalse(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=(32, 32),
                    ffmpeg="ffmpeg",
                )
            )

        with patch.object(
            video_sync,
            "_probe_video_stream",
            return_value={"codec_name": "mjpeg", "has_b_frames": 0},
        ), patch.object(video_sync, "_ffmpeg_supports_setts", return_value=True):
            self.assertFalse(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )
            )

        with patch.dict(
            os.environ,
            {video_sync._VIDEO_SYNC_DISABLE_COPY_FASTPATH_ENV: "1"},
        ):
            self.assertFalse(
                video_sync._can_packet_copy_sync(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                )
            )

    def test_jpeg_fallback_reports_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = Path(tmpdir) / "input.mp4"
            output_mp4 = Path(tmpdir) / "out.mp4"
            input_mp4.write_bytes(b"fake")

            def fake_inner(**kwargs):
                output_mp4.write_bytes(b"mp4")

            with patch.object(
                video_sync, "_ffmpeg", return_value="ffmpeg"
            ), patch.object(
                video_sync, "_can_packet_copy_sync", return_value=False
            ), patch.object(
                video_sync,
                "_remux_selected_frames_streaming",
                side_effect=RuntimeError("stream failed"),
            ), patch.object(
                video_sync, "_remux_selected_frames_in_tmp", side_effect=fake_inner
            ), patch.object(
                video_sync, "_video_frame_count", return_value=1
            ):
                result = video_sync.remux_selected_frames(
                    input_mp4, [0], output_mp4, target_fps=15
                )

            self.assertTrue(result.used_fallback)
            self.assertEqual(result.mode, "jpeg_fallback")

    def test_h264_extract_regenerates_monotonic_pts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_mp4 = tmp / "input.mp4"
            output_mp4 = tmp / "out.mp4"
            frames_dir = tmp / "frames"
            seq_dir = tmp / "seq"
            input_mp4.write_bytes(b"fake")
            frames_dir.mkdir()
            seq_dir.mkdir()
            commands = []

            def fake_run(cmd, *args, **kwargs):
                commands.append(list(cmd))
                if "ffprobe" in cmd[0]:
                    return types.SimpleNamespace(stdout="h264\n", returncode=0)
                if str(frames_dir / "f_%08d.jpg") in cmd:
                    for idx in range(2):
                        (frames_dir / f"f_{idx:08d}.jpg").write_bytes(
                            b"jpeg"
                        )
                    return types.SimpleNamespace(stdout="", returncode=0)
                if str(output_mp4) in cmd:
                    output_mp4.write_bytes(b"mp4")
                    return types.SimpleNamespace(stdout="", returncode=0)
                return types.SimpleNamespace(stdout="", returncode=0)

            with patch.object(
                video_sync.subprocess, "run", side_effect=fake_run
            ), patch.object(
                video_sync, "_h264_encoder", return_value=("libx264", [])
            ):
                video_sync._remux_selected_frames_in_tmp(
                    input_mp4=input_mp4,
                    frame_indices=[0, 1],
                    output_mp4=output_mp4,
                    target_fps=15,
                    rotation_deg=0,
                    image_resize=None,
                    ffmpeg="ffmpeg",
                    frames_dir=frames_dir,
                    seq_dir=seq_dir,
                )

            extract_cmd = next(
                cmd for cmd in commands
                if str(frames_dir / "f_%08d.jpg") in cmd
            )
            self.assertIn("-vf", extract_cmd)
            self.assertIn("setpts=N/TB", extract_cmd)
            self.assertIn("-fps_mode", extract_cmd)
            self.assertIn("passthrough", extract_cmd)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are required for packet-copy integration test",
    )
    def test_h264_packet_copy_prefix_preserves_decoded_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_mp4 = tmp / "input.mp4"
            output_mp4 = tmp / "out.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi",
                    "-i", "testsrc=size=64x48:rate=15:duration=0.4",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p",
                    str(input_mp4),
                ],
                check=True,
            )

            result = video_sync.remux_selected_frames(
                input_mp4, [0, 1, 2], output_mp4, target_fps=15
            )

            self.assertEqual(result.mode, "packet_copy")
            self.assertEqual(video_sync._video_frame_count(output_mp4), 3)
            self.assertAlmostEqual(video_sync._video_fps(output_mp4), 15.0, places=2)
            self.assertEqual(
                self._framehash(input_mp4, frames=3),
                self._framehash(output_mp4),
            )

    def _framehash(self, video_path: Path, frames: Optional[int] = None):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
        ]
        if frames is not None:
            cmd.extend(["-frames:v", str(frames)])
        cmd.extend(["-f", "framehash", "-"])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [
            line.split(",", maxsplit=5)[-1].strip()
            for line in result.stdout.splitlines()
            if line and not line.startswith("#")
        ]


if __name__ == "__main__":
    unittest.main()

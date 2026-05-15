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
# Author: Dongyun Kim, Seongwoo Kim

import json
import os
from pathlib import Path
import queue
import shutil
import socket
import threading
import time
from typing import Optional

from huggingface_hub import HfApi
from interfaces.msg import RecordingStatus
from cyclo_data.converter.orchestrator import DataConverter
from cyclo_data.hub.progress_tracker import (
    HuggingFaceLogCapture,
    HuggingFaceProgressTqdm,
)
# NOTE: cyclo_data importing from orchestrator.internal.* is a layering
# violation — device_manager fits better in shared/ since it's
# robot-agnostic. Tracking as a follow-up; for now mirror the actual
# install path (D4 moved these under internal/).
from orchestrator.internal.device_manager.cpu_checker import CPUChecker
from orchestrator.internal.device_manager.ram_checker import RAMChecker
from orchestrator.internal.device_manager.storage_checker import StorageChecker


def _atomic_write_text(path, content: str, encoding: str = 'utf-8') -> None:
    """Write ``content`` to ``path`` atomically (temp file + os.replace).

    Prevents partial/truncated writes from being observed by concurrent
    readers (e.g. the converter reading episode_info.json while recorder
    saves it, or an HF upload streaming README.md while a checkbox toggle
    rewrites it). A crash between the temp write and the rename leaves
    the original file unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _atomic_write_json(path, obj, indent: int = 2) -> None:
    """JSON-serialize and write atomically. See ``_atomic_write_text``."""
    _atomic_write_text(path, json.dumps(obj, indent=indent))


# README building helpers — shared between recording (DataManager._ensure_task_readme),
# upload fallback (DataManager._create_dataset_card), and merge
# (cyclo_data.editor.episode_editor.merge_rosbag_task_folders) so the
# dataset README looks identical no matter which path created it.
#
# Layout choice (2026-04-28): heading at line 1, no YAML frontmatter.
# HF Hub still picks up the dataset name as the page title, but the
# raw file is human-friendly to read in IDEs / GitHub. License (when
# included) goes in the body right after the heading. Tags / license
# metadata can be added on the HF Hub UI side per repo if the user
# wants the badge display.

_ROBOTIS_LICENSE_BLOCK = (
    'Copyright {year} ROBOTIS CO., LTD.\n'
    '\n'
    'Licensed under the Apache License, Version 2.0 (the "License");\n'
    'you may not use this file except in compliance with the License.\n'
    'You may obtain a copy of the License at\n'
    '\n'
    '    http://www.apache.org/licenses/LICENSE-2.0\n'
    '\n'
    'Unless required by applicable law or agreed to in writing, software\n'
    'distributed under the License is distributed on an "AS IS" BASIS,\n'
    'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n'
    'See the License for the specific language governing permissions and\n'
    'limitations under the License.\n'
)

_ATTRIBUTION = (
    'Created with [Cyclo Intelligence]'
    '(https://github.com/ROBOTIS-GIT/cyclo_intelligence) by ROBOTIS.'
)


def build_dataset_readme(name: str, include_license: bool, year=None) -> str:
    """Render the canonical dataset README in HF Hub dataset-card format.

    Layout:
      ---
      license: apache-2.0      # only when include_license=True
      tags:
      - robotis
      - cyclo_intelligence
      - robotics
      ---

      # <name>

      Copyright … Licensed under …  # only when include_license=True

      ---

      Created with [Cyclo Intelligence](…) by ROBOTIS.

    Frontmatter at the top is what HF Hub parses into metadata badges
    (license, tags). When the user opts into the ROBOTIS license, the
    full Apache 2.0 NOTICE goes in the body so it's also visible to
    humans reading the raw .md.
    """
    if year is None:
        year = time.strftime('%Y')

    # 2-space indented list items match HF Hub's YAML parser convention
    # (other dataset cards on the Hub use the same form). The
    # zero-indent variant `- item` parsed as `tags: null` on the Hub
    # frontend even though it's technically valid YAML.
    out = ['---']
    if include_license:
        out.append('license: apache-2.0')
    out.append('tags:')
    out.append('  - robotis')
    out.append('  - cyclo_intelligence')
    out.append('  - robotics')
    out.append('---')
    out.append('')
    out.append(f'# {name}')
    out.append('')
    if include_license:
        out.append(_ROBOTIS_LICENSE_BLOCK.format(year=year).rstrip())
        out.append('')
        out.append('---')
        out.append('')
    out.append(_ATTRIBUTION)
    out.append('')
    return '\n'.join(out)


def readme_has_license(readme_path: Path) -> bool:
    """Best-effort detection of the ROBOTIS license header in an
    existing README.md. Used by merge to decide what license-status the
    merged output should claim."""
    try:
        head = readme_path.read_text(encoding='utf-8', errors='ignore')[:2000]
    except Exception:
        return False
    return 'Apache License, Version 2.0' in head or 'Copyright' in head and 'ROBOTIS CO., LTD.' in head


class DataManager:

    # Progress queue for multiprocessing communication
    _progress_queue = None

    def __init__(
            self,
            save_root_path,
            robot_type,
            task_info):
        self._robot_type = robot_type
        # Folder naming: Task_{task_num}_{task_name}_MCAP for recordings,
        # Task_{task_num}_{task_name}_Inference_MCAP for inference-time
        # recordings so the two data sources stay visually separated.
        task_num = getattr(task_info, 'task_num', '') or ''
        task_type = getattr(task_info, 'task_type', '') or ''
        suffix = '_Inference_MCAP' if task_type == 'inference' else '_MCAP'
        self._save_repo_name = f'Task_{task_num}_{task_info.task_name}{suffix}'
        self._save_path = save_root_path / self._save_repo_name
        self._save_rosbag_path = '/workspace/rosbag2/' + self._save_repo_name
        self._single_task = len(task_info.task_instruction) == 1
        self._task_info = task_info
        # Per-recording opt-in flag from the UI checkbox; getattr guards
        # against TaskInfo messages built before the field was added.
        self._include_robotis_license = bool(
            getattr(task_info, 'include_robotis_license', False)
        )

        # Find next available episode number from existing folders
        self._record_episode_count = self._find_next_episode_number()
        self._start_time_s = 0
        self._proceed_time = 0
        self._status = 'idle'  # Start in idle state (simplified mode)
        self._cpu_checker = CPUChecker()
        self.data_converter = DataConverter()
        self.current_instruction = ''
        self._init_task_limits()
        self._current_scenario_number = 0
        # Last README content written for this task. Cached so the
        # per-episode save path doesn't re-read README.md from disk
        # on every save_robotis_metadata() call.
        self._cached_readme_content: Optional[str] = None
        # Protects the recording-state group (_status, _record_episode_count,
        # _start_time_s, _proceed_time, _current_scenario_number).
        # orchestrator runs under MultiThreadedExecutor so the timer
        # callback that publishes RecordingStatus and the service
        # callbacks that mutate state (START_RECORD / STOP_RECORD etc.)
        # can fire concurrently; the lock guarantees the snapshot read
        # in ``get_current_record_status`` is consistent.
        self._state_lock = threading.Lock()

    def _find_next_episode_number(self) -> int:
        """
        Find the next available episode number by scanning existing directories.

        Checks the rosbag save path for existing episode folders (0, 1, 2, ...)
        and returns the next available number.

        Returns:
            Next available episode number (0 if no existing episodes).
        """
        rosbag_dir = self._save_rosbag_path

        if not os.path.exists(rosbag_dir):
            print(f'[DataManager] No existing folder at {rosbag_dir}, starting from episode 0')
            return 0

        # Find all numeric folder names
        existing_episodes = []
        try:
            for item in os.listdir(rosbag_dir):
                item_path = os.path.join(rosbag_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    existing_episodes.append(int(item))
        except OSError as e:
            print(f'[DataManager] Error scanning directory: {e}, starting from episode 0')
            return 0

        if not existing_episodes:
            print(f'[DataManager] No existing episodes in {rosbag_dir}, starting from episode 0')
            return 0

        next_episode = max(existing_episodes) + 1
        print(f'[DataManager] Found existing episodes {sorted(existing_episodes)}, '
              f'starting from episode {next_episode}')
        return next_episode

    def get_status(self):
        with self._state_lock:
            return self._status

    # ========== Simplified Recording Methods (rosbag2-only mode) ==========

    def start_recording(self):
        """
        Start recording (simplified mode).

        Changes status to 'recording' for rosbag to begin writing.
        """
        with self._state_lock:
            self._status = 'recording'
            self._start_time_s = time.perf_counter()
            episode = self._record_episode_count
        self.current_instruction = self._task_info.task_instruction[0] \
            if self._task_info.task_instruction else ''
        print(f'[DataManager] Recording started - Episode {episode}')

    def stop_recording(self):
        """
        Stop recording and save (simplified mode).

        Changes status to 'idle' and increments episode count.
        """
        with self._state_lock:
            self._status = 'idle'
            self._record_episode_count += 1
            self._start_time_s = 0
            total = self._record_episode_count
        print(f'[DataManager] Recording stopped - Episode saved. '
              f'Total episodes: {total}')

    def discard_recording(self):
        """
        Stop without saving — flip to idle but leave the episode counter
        untouched so the discarded slot is reused by the next START.

        Caller is responsible for removing the episode directory on disk
        (rosbag's stop_and_delete + a defensive rmtree in
        RecordingService._do_discard).
        """
        with self._state_lock:
            self._status = 'idle'
            self._start_time_s = 0
            unchanged = self._record_episode_count
        print(f'[DataManager] Recording discarded - episode count unchanged '
              f'({unchanged})')

    def is_recording(self):
        """Check if currently recording."""
        with self._state_lock:
            return self._status == 'recording'

    # ========== End Simplified Recording Methods ==========

    def get_save_rosbag_path(self, allow_idle: bool = False):
        """Get rosbag save path for current episode."""
        # For simplified mode, return path when recording.
        # `allow_idle` is used during START pre-check before status flips to recording.
        with self._state_lock:
            status = self._status
            episode = self._record_episode_count
        if status == 'idle' and not allow_idle:
            return None  # Not recording
        if status == 'warmup':
            return None  # Legacy: Not ready yet
        return self._save_rosbag_path + f'/{episode}'

    def update_task_info(self, task_info):
        """Refresh per-session config from a new task_info.

        Called by RecordingService when a second START_RECORD arrives
        for the same task — the user may have toggled the
        ``include_robotis_license`` checkbox between episodes. Without
        this refresh the existing manager would keep its first-START
        snapshot and the README would never reflect later choices.
        """
        self._task_info = task_info
        self._include_robotis_license = bool(
            getattr(task_info, 'include_robotis_license', False)
        )

    def _ensure_task_readme(self):
        """Write or refresh ``<task_folder>/README.md`` based on the
        current ``_include_robotis_license`` flag.

        Two variants:

        * Default (false): minimal — tool attribution + license-is-yours
          reminder. Recording outputs are the user's intellectual
          property, not ROBOTIS' to license. Same pattern as LeRobot,
          Audacity, OBS — tool license != output license.

        * Opt-in (true): tool attribution + ROBOTIS Apache 2.0 license
          header. For ROBOTIS-internal captures where ROBOTIS is the
          actual data owner; the license rides through conversion +
          HF upload.

        Called on every ``save_robotis_metadata`` so the README always
        reflects the user's latest choice, even if the checkbox flipped
        between episodes (episode 0 unchecked → episode 1 checked etc.).
        Skips the write when on-disk content already matches, so we
        don't churn mtimes when the flag is stable.
        """
        task_dir = Path(self._save_rosbag_path)
        readme_path = task_dir / 'README.md'

        desired = build_dataset_readme(
            name=self._save_repo_name,
            include_license=self._include_robotis_license,
        )
        if self._cached_readme_content == desired and readme_path.exists():
            return
        if readme_path.exists():
            try:
                if readme_path.read_text(encoding='utf-8') == desired:
                    self._cached_readme_content = desired
                    return
            except Exception:
                pass  # fall through and rewrite

        try:
            _atomic_write_text(readme_path, desired)
            self._cached_readme_content = desired
            variant = 'with ROBOTIS license' if self._include_robotis_license else 'minimal'
            print(f'[ROBOTIS] README.md written at: {readme_path} ({variant})')
        except Exception as e:
            print(f'[ROBOTIS] Failed to write README.md at {task_dir}: {e}')

    def save_robotis_metadata(
        self,
        urdf_path: str = None,
        video_stats: dict | None = None,
        camera_info_files: dict | None = None,
        camera_rotations: dict | None = None,
    ):
        """
        Save URDF and metadata for ROBOTIS format.

        Called after each episode rosbag is saved.
        Copies URDF file and all referenced mesh files.

        Args:
            urdf_path: Path to URDF file to copy.
            video_stats: ``{cam_name: {frames_received, frames_written, ...}}`` from VideoRecorder.
            camera_info_files: ``{cam_name: yaml_path}`` from CameraInfoSnapshot.
        """
        rosbag_path = self.get_save_rosbag_path()
        if rosbag_path is None:
            return

        # Create rosbag directory if not exists
        os.makedirs(rosbag_path, exist_ok=True)

        # Drop the ROBOTIS / Apache 2.0 README at the task-folder root the
        # first time any episode is saved — idempotent (skip if exists),
        # so user edits + later upload-time additions are preserved.
        # The downstream conversion stages copy it forward into v21 / v30
        # outputs so the same notice rides all the way to HF Hub.
        self._ensure_task_readme()

        # Copy URDF only — meshes used to be bundled into each rosbag for
        # self-contained replay, but every recording then carried tens of
        # MBs of duplicate STL data that the local replay path doesn't
        # need (shared/robot_configs/ffw_description/ is already on
        # disk). Drop the mesh copy; URDF stays since it's small and
        # describes the joint topology used to interpret the bag.
        if urdf_path and os.path.exists(urdf_path):
            urdf_dest = os.path.join(rosbag_path, 'robot.urdf')
            try:
                shutil.copy2(urdf_path, urdf_dest)
                print(f'[ROBOTIS] URDF copied to: {urdf_dest}')
            except Exception as e:
                print(f'[ROBOTIS] Failed to copy URDF: {e}')

        # Save metadata JSON.
        # fps is intentionally NOT recorded here — recording is rate-
        # agnostic (every sensor publishes at its own natural rate and
        # the rosbag captures verbatim with timestamps). fps becomes
        # relevant only at convert time, when the user picks a target
        # rate for MP4 encoding + LeRobot info.json. That value rides
        # on the StartConversion srv (request.fps), so it doesn't
        # belong in the per-episode recording metadata.
        # Recording format v2 (images-as-MP4 + camera_info-as-yaml + MCAP
        # without images). format_version: 'robotis_v2'. Older recordings
        # used 'robotis_v1' (images embedded in MCAP) — the converter
        # branches on this field. video_files / camera_info_files are
        # paths relative to the episode dir so the manifest survives a
        # move of the parent workspace tree.
        videos_dir = os.path.join(rosbag_path, 'videos')
        video_files = {}
        if os.path.isdir(videos_dir):
            for entry in sorted(os.listdir(videos_dir)):
                full = os.path.join(videos_dir, entry)
                if entry.endswith('.mp4') and os.path.isfile(full):
                    cam = entry[:-4]
                    video_files[cam] = os.path.relpath(full, rosbag_path)
        camera_info_rel = {}
        if camera_info_files:
            for cam, path in camera_info_files.items():
                try:
                    camera_info_rel[cam] = os.path.relpath(path, rosbag_path)
                except ValueError:
                    camera_info_rel[cam] = path

        # ``transcoding_status`` default depends on whether this episode
        # actually has any cameras to transcode. The TranscodeWorker
        # patches this field again once it runs.
        initial_status = 'pending' if video_files else 'not_required'

        meta_data = {
            'task_instruction': self.current_instruction,
            'robot_type': self._robot_type,
            'episode_index': self._record_episode_count,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'format_version': 'robotis_v2',
            'recorder_format_version': 2,
            'device_serial': socket.gethostname(),
            'video_files': video_files,
            'camera_info_files': camera_info_rel,
            'video_stats': video_stats or {},
            # ``camera_rotations`` is ``{cam_name: degrees}`` (0/90/180/270)
            # straight from the robot config yaml. The background
            # transcoder reads this when re-encoding to H.264 and applies
            # ``-vf transpose=N`` so the stored MP4 has the correct
            # orientation (e.g. wrist cameras mounted upside down at 270°).
            'camera_rotations': dict(camera_rotations or {}),
            'transcoding_status': initial_status,
        }

        meta_data_path = os.path.join(rosbag_path, 'episode_info.json')
        try:
            _atomic_write_json(meta_data_path, meta_data)
            print(f'[ROBOTIS] Metadata saved to: {meta_data_path}')
        except Exception as e:
            print(f'[ROBOTIS] Failed to save metadata: {e}')

    def should_record_rosbag2(self):
        """In simplified mode, always record rosbag2."""
        # Always return True in rosbag2-only mode
        # Legacy: return self._task_info.record_rosbag2
        return True

    def get_current_record_status(self):
        current_status = RecordingStatus()
        current_status.robot_type = self._robot_type
        current_status.task_info = self._task_info

        with self._state_lock:
            status = self._status
            start_time_s = self._start_time_s

        if status == 'idle':
            current_status.record_phase = RecordingStatus.READY
        elif status == 'recording':
            current_status.record_phase = RecordingStatus.RECORDING
            if start_time_s > 0:
                elapsed = time.perf_counter() - start_time_s
                with self._state_lock:
                    self._proceed_time = int(elapsed)
        elif status == 'save' or status == 'finish':
            is_saving, encoding_progress = self._get_encoding_progress()
            current_status.record_phase = RecordingStatus.SAVING
            with self._state_lock:
                self._proceed_time = int(0)
            if is_saving:
                current_status.encoding_progress = encoding_progress
            else:
                current_status.encoding_progress = 0.0

        with self._state_lock:
            proceed_time = int(getattr(self, '_proceed_time', 0))
            episode_count = int(self._record_episode_count)
            scenario_number = self._current_scenario_number
        current_status.current_task_instruction = self.current_instruction
        current_status.proceed_time = proceed_time
        current_status.current_episode_number = episode_count

        total_storage, used_storage = StorageChecker.get_storage_gb('/')
        current_status.used_storage_size = float(used_storage)
        current_status.total_storage_size = float(total_storage)

        current_status.used_cpu = float(self._cpu_checker.get_cpu_usage())

        ram_total, ram_used = RAMChecker.get_ram_gb()
        current_status.used_ram_size = float(ram_used)
        current_status.total_ram_size = float(ram_total)
        if not self._single_task:
            current_status.current_scenario_number = scenario_number

        return current_status

    def _get_encoding_progress(self):
        """Get encoding progress. Always returns not-saving for rosbag2-only mode."""
        return False, 100.0

    def _init_task_limits(self):
        if not self._single_task:
            if hasattr(self._task_info, 'num_episodes'):
                self._task_info.num_episodes = 1_000_000
            if hasattr(self._task_info, 'episode_time_s'):
                self._task_info.episode_time_s = 1_000_000

    @staticmethod
    def get_robot_type_from_info_json(info_json_path):
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get('robot_type', '')

    @staticmethod
    def whoami_huggingface(endpoint, token, timeout_s=5.0):
        """Validate ``token`` against ``endpoint`` and return the user's
        identifier list (primary user + every org they belong to).

        Returns ``None`` on timeout, raises on invalid token / network error.
        Both ``endpoint`` and ``token`` are required — there is no global
        fallback.
        """
        if not endpoint:
            raise ValueError('endpoint is required')
        if not token:
            raise ValueError('token is required')

        def api_call():
            api = HfApi(endpoint=endpoint, token=token)
            user_info = api.whoami()
            user_ids = [user_info['name']]
            for org_info in user_info.get('orgs', []) or []:
                org_name = org_info.get('name')
                if org_name:
                    user_ids.append(org_name)
            return user_ids

        result_queue = queue.Queue()

        def worker():
            try:
                result_queue.put(('success', api_call()))
            except Exception as e:
                result_queue.put(('error', e))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            status, data = result_queue.get(timeout=timeout_s)
        except queue.Empty:
            print(f'Token validation timed out after {timeout_s}s '
                  f'for endpoint {endpoint}')
            return None

        if status == 'success':
            return data
        raise data

    # Default download roots used when the caller does not pass an explicit
    # ``local_dir``. The previous LeRobot defaults are kept as a fallback for
    # back-compat, but new flows should always pass the destination from the
    # UI so the user can pick where downloads land.
    DEFAULT_DOWNLOAD_PATHS = {
        'dataset': Path('/workspace/rosbag2'),
        'model': Path('/workspace/model'),
    }

    @staticmethod
    def download_huggingface_repo(
        repo_id,
        repo_type='dataset',
        local_dir=None,
        endpoint=None,
        token=None,
    ):
        """Download a HuggingFace repo via the ``hf`` CLI in a PTY.

        We shell out to the CLI (instead of the huggingface_hub Python API)
        because the in-process tqdm wrapper fails to track byte counts when
        the hf-xet accelerator is used; running the CLI under a real PTY
        gives us native tqdm bars on the backend log.
        """
        import ctypes
        import os
        import pty
        import re
        import select
        import signal
        import subprocess

        if local_dir:
            save_dir = Path(local_dir) / repo_id
        else:
            base = DataManager.DEFAULT_DOWNLOAD_PATHS.get(repo_type)
            if base is None:
                raise ValueError(f'Invalid repo type: {repo_type}')
            save_dir = base / repo_id
        save_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if token:
            env['HF_TOKEN'] = token
        if endpoint:
            env['HF_ENDPOINT'] = endpoint
        env.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)

        cmd = [
            'hf', 'download', repo_id,
            '--repo-type', repo_type,
            '--local-dir', str(save_dir),
        ]
        print(
            f'Starting download of {repo_id} ({repo_type}) from '
            f'{endpoint or "<default endpoint>"} via hf CLI'
        )

        # Child becomes a process-group leader + dies if our worker dies.
        # If libc cannot be loaded (musl, stripped image), surface a
        # warning so an operator can see why the child outlives its
        # parent on crash — silent failure here previously left orphan
        # hf download processes running after the worker died.
        try:
            _libc = ctypes.CDLL('libc.so.6', use_errno=True)
        except OSError as exc:
            print(
                f'[DataManager] WARNING: failed to load libc for '
                f'PR_SET_PDEATHSIG ({exc}); hf download child may '
                f'outlive this worker on crash.'
            )
            _libc = None

        def _preexec():
            os.setsid()
            if _libc is not None:
                try:
                    _libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
                except Exception:
                    pass

        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                env=env,
                preexec_fn=_preexec,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        ansi_re = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
        buf = b''
        try:
            while True:
                if proc.poll() is not None:
                    try:
                        data = os.read(master_fd, 8192)
                    except OSError:
                        data = b''
                    if data:
                        buf += data
                    if not data:
                        break

                try:
                    # 1s blocks the loop just long enough that an idle
                    # multi-minute download polls proc.poll() ~ once per
                    # second instead of twice. tqdm-style progress lines
                    # arrive far more frequently than that anyway, so
                    # latency-to-log doesn't change in practice.
                    r, _, _ = select.select([master_fd], [], [], 1.0)
                except (OSError, ValueError):
                    break
                if r:
                    try:
                        data = os.read(master_fd, 8192)
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data

                # Split on CR or LF: tqdm uses CR for in-place updates.
                while True:
                    idx = -1
                    for sep in (b'\r', b'\n'):
                        i = buf.find(sep)
                        if i >= 0 and (idx == -1 or i < idx):
                            idx = i
                    if idx < 0:
                        break
                    raw, buf = buf[:idx], buf[idx + 1:]
                    line = ansi_re.sub('', raw.decode('utf-8', errors='replace')).strip()
                    if not line:
                        continue
                    print(f'[hf cli] {line}')
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

        return_code = proc.wait()
        if return_code == 0:
            print(f'Download completed: {repo_id}')
            return str(save_dir)

        print(f'Error downloading HuggingFace repo (exit={return_code}): {repo_id}')
        return False

    @classmethod
    def set_progress_queue(cls, progress_queue):
        """Set progress queue for multiprocessing communication."""
        cls._progress_queue = progress_queue

    @staticmethod
    def _collect_task_instructions(local_dir):
        """Collect distinct task_instruction strings from a dataset folder.

        Handles both layouts:
        * Raw rosbag2: each ``<episode>/episode_info.json`` has its own
          ``task_instruction`` (recorder writes it during the session).
        * LeRobot v3.0: ``meta/tasks.parquet`` has a ``task`` column
          aggregated by the converter (one row per distinct task).

        Returns a list with insertion order preserved (no dedup churn).
        """
        local_path = Path(local_dir)
        seen = set()
        out = []

        for sub in sorted(local_path.iterdir()) if local_path.is_dir() else []:
            if not sub.is_dir():
                continue
            info = sub / 'episode_info.json'
            if not info.exists():
                continue
            try:
                with open(info, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ti = (data.get('task_instruction') or '').strip()
                if ti and ti not in seen:
                    seen.add(ti)
                    out.append(ti)
            except Exception:
                continue

        if not out:
            tasks_parquet = local_path / 'meta' / 'tasks.parquet'
            if tasks_parquet.exists():
                try:
                    import pyarrow.parquet as pq
                    tbl = pq.read_table(str(tasks_parquet))
                    for t in tbl.column('task').to_pylist():
                        t = (t or '').strip()
                        if t and t not in seen:
                            seen.add(t)
                            out.append(t)
                except Exception:
                    pass

        return out

    @staticmethod
    def _create_dataset_card(local_dir, readme_path):
        """Write a minimal dataset README.

        Just task instructions + provenance (created via cyclo_intelligence
        by ROBOTIS). No HF DatasetCard template, no embedded info.json
        block — those were inherited from cyclo_intelligence and added
        clutter without giving HF Hub anything it needs beyond the
        license / tags frontmatter.
        """
        local_path = Path(local_dir)
        name = local_path.name
        task_instructions = DataManager._collect_task_instructions(local_dir)

        # Fallback for upload paths where the recorder didn't write a
        # README (legacy datasets). The license is the user's call —
        # default to no license. Same builder as the recording-time
        # README so the rendered shape is identical.
        body = build_dataset_readme(name=name, include_license=False)
        if task_instructions:
            extras = ['## Task instructions', '']
            for ti in task_instructions:
                extras.append(f'- {ti}')
            extras.append('')
            body += '\n'.join(extras) + '\n'

        Path(readme_path).write_text(body, encoding='utf-8')
        print(f'Dataset README.md created ({len(task_instructions)} tasks)')

    @staticmethod
    def _create_model_card(local_dir, readme_path):
        """Write a minimal model README.

        Provenance line + (when train_config.json exists) the source
        dataset repo. Same minimalism as _create_dataset_card.
        """
        local_path = Path(local_dir)
        name = local_path.name

        # train_config.json is optional — the upload may be a raw
        # checkpoint folder without config. Look in obvious places, fall
        # back to a recursive find as a last resort.
        train_config = None
        candidates = [
            local_path / 'train_config.json',
            local_path / 'config' / 'train_config.json',
            local_path / 'pretrained_model' / 'train_config.json',
        ]
        for cfg in candidates:
            if cfg.exists():
                try:
                    with open(cfg, 'r', encoding='utf-8') as f:
                        train_config = json.load(f)
                    break
                except Exception:
                    continue
        if train_config is None:
            for cfg in local_path.rglob('train_config.json'):
                try:
                    with open(cfg, 'r', encoding='utf-8') as f:
                        train_config = json.load(f)
                    break
                except Exception:
                    continue

        dataset_repo = ''
        if train_config:
            dataset_repo = (train_config.get('dataset') or {}).get('repo_id', '')

        # Tool attribution only — see _create_dataset_card for why we
        # don't auto-stamp a license on user-trained models.
        lines = [
            '---',
            'pipeline_tag: robotics',
            'tags:',
            '- robotis',
            '- cyclo_intelligence',
            '- robotics',
        ]
        if dataset_repo:
            lines.append('datasets:')
            lines.append(f'- {dataset_repo}')
        lines += [
            '---',
            '',
            f'# {name}',
            '',
            'Created with [Cyclo Intelligence]'
            '(https://github.com/ROBOTIS-GIT/cyclo_intelligence) by ROBOTIS.',
            '',
        ]
        if dataset_repo:
            lines.append(f'Trained on: [{dataset_repo}]'
                         f'(https://huggingface.co/datasets/{dataset_repo})')
            lines.append('')

        Path(readme_path).write_text('\n'.join(lines), encoding='utf-8')
        print(f'Model README.md created (dataset={dataset_repo or "<none>"})')

    @staticmethod
    def _create_readme_if_not_exists(local_dir, repo_type):
        """
        Create README.md file if it doesn't exist in the folder.

        Uses HuggingFace Hub's DatasetCard or ModelCard.

        """
        readme_path = Path(local_dir) / 'README.md'

        if readme_path.exists():
            print(f'README.md already exists in {local_dir}')
            return

        print(f'Creating README.md in {local_dir}')

        try:
            if repo_type == 'dataset':
                DataManager._create_dataset_card(local_dir, readme_path)
            elif repo_type == 'model':
                DataManager._create_model_card(local_dir, readme_path)
        except Exception as e:
            print(f'Warning: Failed to create README.md: {e}')
            import traceback
            print(f'Traceback: {traceback.format_exc()}')

    @staticmethod
    def upload_huggingface_repo(
        repo_id,
        repo_type,
        local_dir,
        endpoint=None,
        token=None,
    ):
        try:
            api = HfApi(endpoint=endpoint, token=token)

            # Verify authentication first
            try:
                user_info = api.whoami()
                print(
                    f'Authenticated as: {user_info["name"]} '
                    f'({endpoint or "<default endpoint>"})'
                )
            except Exception as auth_e:
                print(f'Authentication failed: {auth_e}')
                print(
                    'Please make sure a valid token is registered for this '
                    f'endpoint: {endpoint or "<default>"}'
                )
                return False

            # Create repository
            print(f'Creating HuggingFace repository: {repo_id}')
            url = api.create_repo(
                repo_id,
                repo_type=repo_type,
                private=False,
                exist_ok=True,
            )
            print(f'Repository created/verified: {url}')

            # Delete .cache folder before upload
            DataManager._delete_dot_cache_folder_before_upload(local_dir)

            # Create README.md if it doesn't exist
            DataManager._create_readme_if_not_exists(
                local_dir, repo_type
            )

            print(f'Uploading folder {local_dir} to repository {repo_id}')

            # Capture stdout for logging
            from contextlib import redirect_stdout

            # Use log capture with progress queue
            log_capture = HuggingFaceLogCapture(progress_queue=DataManager._progress_queue)

            with redirect_stdout(log_capture):
                # Upload folder contents via the HfApi instance so it picks up
                # the per-call endpoint+token without touching env vars.
                api.upload_large_folder(
                    repo_id=repo_id,
                    folder_path=local_dir,
                    repo_type=repo_type,
                    print_report=True,
                    print_report_every=1,
                )

            # Create tag
            if repo_type == 'dataset':
                try:
                    print(f'Creating tag for {repo_id} ({repo_type})')
                    api.create_tag(repo_id=repo_id, tag='v2.1', repo_type=repo_type)
                    print(f'Tag "v2.1" created successfully for {repo_id}')
                except Exception as e:
                    print(f'Warning: Failed to create tag for {repo_id} ({repo_type}): {e}')
                    # Don't fail the entire upload just because tag creation failed

            return True
        except Exception as e:
            print(f'Error Uploading HuggingFace repo: {e}')
            # Print more detailed error information
            import traceback
            print(f'Detailed error traceback:\n{traceback.format_exc()}')
            return False

    @staticmethod
    def _delete_dot_cache_folder_before_upload(local_dir):
        dot_cache_path = Path(local_dir) / '.cache'
        if dot_cache_path.exists():
            shutil.rmtree(dot_cache_path)
            print(f'Deleted {local_dir}/.cache folder before upload')

    @staticmethod
    def delete_huggingface_repo(
        repo_id,
        repo_type='dataset',
        endpoint=None,
        token=None,
    ):
        try:
            api = HfApi(endpoint=endpoint, token=token)
            return api.delete_repo(repo_id, repo_type=repo_type)
        except Exception as e:
            print(f'Error deleting HuggingFace repo: {e}')
            return False

    @staticmethod
    def get_huggingface_repo_list(
        author,
        data_type='dataset',
        endpoint=None,
        token=None,
    ):
        api = HfApi(endpoint=endpoint, token=token)
        repo_id_list = []
        if data_type == 'dataset':
            for dataset in api.list_datasets(author=author):
                repo_id_list.append(dataset.id)
        elif data_type == 'model':
            for model in api.list_models(author=author):
                repo_id_list.append(model.id)
        return repo_id_list[::-1]

    @staticmethod
    def get_collections_repo_list(
        collection_id,
        endpoint=None,
        token=None,
    ):
        api = HfApi(endpoint=endpoint, token=token)
        collection_list = api.get_collection(collection_id)
        return [item.item_id for item in collection_list.items]

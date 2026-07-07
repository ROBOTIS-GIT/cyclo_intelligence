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
import errno
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import threading
import time
from typing import Optional

from huggingface_hub import HfApi
import yaml
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
    _atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False))


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
        self._save_repo_name = self._make_save_repo_name(
            save_root_path,
            task_info,
        )
        self._save_path = save_root_path / self._save_repo_name
        self._save_rosbag_path = '/workspace/rosbag2/' + self._save_repo_name
        self._task_info = task_info
        self._main_task_instruction = self._get_main_task_instruction(task_info)
        self._subtask_instructions = self._get_subtask_instructions(task_info)
        self._subtask_mode = bool(self._subtask_instructions)
        self._subtask_total = len(self._subtask_instructions)
        self._physical_segment_total = max(1, self._subtask_total)
        self._segmented_storage_mode = True
        self._single_task = not self._subtask_mode
        self._saved_subtasks_cache: dict[int, set[int]] = {}
        self._episode_info_scan_cache = self._collect_episode_info_entries()
        try:
            self._validate_existing_segment_count()

            # Per-recording opt-in flag from the UI checkbox; getattr guards
            # against TaskInfo messages built before the field was added.
            self._include_robotis_license = bool(
                getattr(task_info, 'include_robotis_license', False)
            )

            # Find next available raw recording number from existing folders.
            # In subtask mode this is a raw subtask counter used only for
            # metadata continuity; the on-disk path is full_episode/subtasks/N.
            self._record_episode_count = self._find_next_episode_number()
            (
                self._current_full_episode_index,
                self._current_subtask_index,
            ) = self._find_next_subtask_position()
        finally:
            self._episode_info_scan_cache = None
        self._start_time_s = 0
        self._proceed_time = 0
        self._status = 'idle'  # Start in idle state (simplified mode)
        self._cpu_checker = CPUChecker()
        self.data_converter = DataConverter()
        self.current_instruction = self._main_task_instruction
        self._init_task_limits()
        self._current_scenario_number = self._current_subtask_index
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

    @classmethod
    def _make_save_repo_name(cls, save_root_path, task_info) -> str:
        """Return the task folder name and normalise inference metadata."""
        task_type = getattr(task_info, 'task_type', '') or ''
        if task_type == 'inference':
            record_id = cls._make_unique_inference_record_id(save_root_path)
            task_info.task_num = record_id
            task_info.task_name = 'inference'
            return f'Task_{record_id}_inference_MCAP'

        task_num = getattr(task_info, 'task_num', '') or ''
        task_name = getattr(task_info, 'task_name', '') or ''
        return f'Task_{task_num}_{task_name}_MCAP'

    @staticmethod
    def _make_unique_inference_record_id(save_root_path) -> str:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        root = Path(save_root_path)
        record_id = timestamp
        suffix = 1
        while (root / f'Task_{record_id}_inference_MCAP').exists():
            record_id = f'{timestamp}_{suffix:02d}'
            suffix += 1
        return record_id

    @staticmethod
    def _read_episode_info(episode_dir: Path) -> dict:
        info_path = Path(episode_dir) / 'episode_info.json'
        if not info_path.exists():
            return {}
        try:
            return json.loads(info_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    @staticmethod
    def _is_rosbag_leaf(path: Path) -> bool:
        path = Path(path)
        return (
            path.is_dir()
            and (path / 'metadata.yaml').exists()
            and any(path.glob('*.mcap'))
        )

    @staticmethod
    def _skip_scan_path(path: Path) -> bool:
        return any(
            part.endswith('_converted') or part in {
                '_stitched_subtasks',
                '.legacy_subtask_migration_tmp',
                '.subtasks_archive',
            }
            for part in path.parts
        )

    def _collect_episode_info_entries(self) -> list[tuple[Path, dict]]:
        """Return saved episode_info entries under the task root."""
        root = Path(self._save_rosbag_path)
        if not root.exists():
            return []

        entries: list[tuple[Path, dict]] = []
        for info_path in root.rglob('episode_info.json'):
            try:
                rel_parent = info_path.parent.relative_to(root)
            except ValueError:
                rel_parent = info_path.parent
            if self._skip_scan_path(rel_parent):
                continue
            entries.append((
                info_path.parent,
                self._read_episode_info(info_path.parent),
            ))
        return entries

    def _episode_info_entries(self) -> list[tuple[Path, dict]]:
        cached = getattr(self, '_episode_info_scan_cache', None)
        if cached is not None:
            return list(cached)
        return self._collect_episode_info_entries()

    def _existing_episode_segment_counts(self) -> set[int]:
        """Return semantic subtask counts already present in this task folder."""
        root = Path(self._save_rosbag_path)
        counts: set[int] = set()
        if not root.exists():
            return counts

        for episode_dir, info in self._episode_info_entries():
            mode = info.get('recording_mode')
            if mode == 'single_segment':
                counts.add(1)
                continue
            if mode == 'subtask':
                try:
                    counts.add(int(info.get('subtask_total', 1) or 1))
                except (TypeError, ValueError):
                    counts.add(1)
                continue
            segments = info.get('segments')
            if isinstance(segments, list):
                counts.add(len(segments))
                continue
            if self._is_rosbag_leaf(episode_dir):
                counts.add(0)
        return counts

    def _validate_existing_segment_count(self) -> None:
        existing_counts = self._existing_episode_segment_counts()
        if not existing_counts:
            return
        requested = self._physical_segment_total
        if existing_counts == {requested}:
            return
        raise ValueError(
            'Cannot mix subtask counts in one task folder: '
            f'existing={sorted(existing_counts)}, requested={requested}. '
            'Use a new task folder for a different Number of Subtasks.'
        )

    def _iter_subtask_episode_dirs(self) -> list[Path]:
        """Return all saved raw segment rosbag dirs, old flat + new nested."""
        root = Path(self._save_rosbag_path)
        if not root.exists():
            return []

        matches: list[Path] = []
        for episode_dir, info in self._episode_info_entries():
            if info.get('recording_mode') not in {'subtask', 'single_segment'}:
                continue
            matches.append(episode_dir)

        def sort_key(path: Path):
            info = self._read_episode_info(path)
            try:
                full_idx = int(info.get('full_episode_index', 0))
            except (TypeError, ValueError):
                full_idx = 0
            try:
                subtask_idx = int(info.get('subtask_index', 0))
            except (TypeError, ValueError):
                subtask_idx = 0
            try:
                raw_idx = int(info.get('episode_index', 0))
            except (TypeError, ValueError):
                raw_idx = 0
            return (full_idx, subtask_idx, raw_idx, str(path))

        return sorted(matches, key=sort_key)

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

        # Find all numeric folder names and any recursively stored raw
        # subtask metadata. This preserves legacy flat recordings and the
        # nested full_episode/subtasks layout used for new subtask data.
        existing_episodes = []
        try:
            for item in os.listdir(rosbag_dir):
                item_path = os.path.join(rosbag_dir, item)
                if not (os.path.isdir(item_path) and item.isdigit()):
                    continue
                if self._segmented_storage_mode and not self._episode_dir_has_data(
                    Path(item_path)
                ):
                    continue
                existing_episodes.append(int(item))
            for _episode_dir, info in self._episode_info_entries():
                try:
                    existing_episodes.append(int(info.get('episode_index')))
                except (TypeError, ValueError):
                    pass
        except OSError as e:
            print(f'[DataManager] Error scanning directory: {e}, starting from episode 0')
            return 0

        if not existing_episodes:
            print(f'[DataManager] No existing episodes in {rosbag_dir}, starting from episode 0')
            return 0

        existing_unique = sorted(set(existing_episodes))
        next_episode = existing_unique[-1] + 1
        if len(existing_unique) <= 20:
            existing_text = str(existing_unique)
        else:
            existing_text = (
                f'{len(existing_unique)} episodes '
                f'({existing_unique[0]}..{existing_unique[-1]})'
            )
        print(f'[DataManager] Found existing episodes {existing_text}, '
              f'starting from episode {next_episode}')
        return next_episode

    def _episode_dir_has_data(self, episode_dir: Path) -> bool:
        """Return True when a numeric episode folder contains real data."""
        episode_dir = Path(episode_dir)
        if self._is_rosbag_leaf(episode_dir):
            return True
        for info_path in episode_dir.rglob('episode_info.json'):
            rel_parent = info_path.parent.relative_to(episode_dir)
            if not self._skip_scan_path(rel_parent):
                return True
        return any(episode_dir.rglob('*.mcap'))

    @staticmethod
    def _get_main_task_instruction(task_info) -> str:
        instructions = getattr(task_info, 'task_instruction', []) or []
        for instruction in instructions:
            text = str(instruction or '').strip()
            if text:
                return text
        return ''

    @staticmethod
    def _get_subtask_instructions(task_info) -> list:
        raw = getattr(task_info, 'subtask_instruction', []) or []
        return [str(item or '').strip() for item in raw if str(item or '').strip()]

    def _find_next_subtask_position(self) -> tuple[int, int]:
        """Find the next full-episode/subtask cursor from saved metadata."""
        if not self._segmented_storage_mode:
            return self._record_episode_count, 0

        groups: dict[int, set[int]] = {}
        totals: dict[int, int] = {}
        for episode_dir in self._iter_subtask_episode_dirs():
            info = self._read_episode_info(episode_dir)
            try:
                full_idx = int(info.get('full_episode_index', 0))
                subtask_idx = int(info.get('subtask_index', 0))
                total = int(info.get('subtask_total', self._physical_segment_total))
            except (TypeError, ValueError):
                continue
            groups.setdefault(full_idx, set()).add(subtask_idx)
            totals[full_idx] = max(1, total)

        root = Path(self._save_rosbag_path)
        if root.exists():
            for _episode_dir, info in self._episode_info_entries():
                # New archived full-episode summaries intentionally keep
                # episode_info.json minimal: episode_index + segments, no
                # recording_mode/full_episode_index/subtask_total. Treat
                # that shape as a complete full episode so a recreated
                # DataManager continues at the next numeric episode folder.
                segments = info.get('segments')
                if isinstance(segments, list):
                    try:
                        full_idx = int(info.get('episode_index'))
                    except (TypeError, ValueError):
                        continue
                    total = max(1, len(segments))
                    groups.setdefault(full_idx, set()).update(range(total))
                    totals[full_idx] = total
                    continue

                if info.get('recording_mode') != 'subtask_episode':
                    continue
                try:
                    full_idx = int(info.get('full_episode_index', 0))
                    total = int(info.get('subtask_total', self._physical_segment_total))
                except (TypeError, ValueError):
                    continue
                total = max(1, total)
                groups.setdefault(full_idx, set()).update(range(total))
                totals[full_idx] = total

        if not groups:
            self._saved_subtasks_cache = {}
            return self._record_episode_count, 0

        self._saved_subtasks_cache = {
            int(full_idx): set(saved)
            for full_idx, saved in groups.items()
        }

        for full_idx in sorted(groups):
            total = totals.get(full_idx, self._physical_segment_total)
            missing = [idx for idx in range(total) if idx not in groups[full_idx]]
            if missing:
                return full_idx, missing[0]

        return max(groups) + 1, 0

    def _current_subtask_instruction(self) -> str:
        if not self._subtask_mode:
            return ''
        if 0 <= self._current_subtask_index < len(self._subtask_instructions):
            return self._subtask_instructions[self._current_subtask_index]
        return ''

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
            full_episode = self._current_full_episode_index
            subtask = self._current_subtask_index
        self.current_instruction = self._main_task_instruction
        subtask_label = (
            f' full_episode={full_episode} subtask={subtask}/{self._subtask_total}'
            if self._subtask_mode else ''
        )
        print(f'[DataManager] Recording started - Episode {episode}{subtask_label}')

    def stop_recording(self, finish_full_episode: bool = True):
        """
        Stop recording and save (simplified mode).

        Changes status to 'idle' and increments episode count.
        """
        with self._state_lock:
            self._status = 'idle'
            self._record_episode_count += 1
            if self._subtask_mode:
                is_last_subtask = self._current_subtask_index >= self._subtask_total - 1
                if finish_full_episode:
                    self._current_full_episode_index += 1
                    self._current_subtask_index = 0
                else:
                    self._current_subtask_index = min(
                        self._current_subtask_index + 1,
                        self._subtask_total - 1 if is_last_subtask else self._subtask_total,
                    )
                self._current_scenario_number = self._current_subtask_index
            self._start_time_s = 0
            total = self._record_episode_count
        print(f'[DataManager] Recording stopped - Episode saved. '
              f'Total episodes: {total}')

    def set_current_subtask_index(self, index: int):
        """Select the subtask slot that the next START should record."""
        if not self._segmented_storage_mode:
            return
        with self._state_lock:
            bounded = max(0, min(int(index), max(0, self._physical_segment_total - 1)))
            self._current_subtask_index = bounded
            self._current_scenario_number = bounded

    def get_current_subtask_index(self) -> int:
        """Return the active segmented subtask slot."""
        with self._state_lock:
            return int(self._current_subtask_index)

    def get_current_full_episode_index(self) -> int:
        """Return the full-episode cursor used by segmented recordings."""
        with self._state_lock:
            return int(self._current_full_episode_index)

    def saved_subtask_indices_for_full_episode(
        self,
        full_idx: int | None = None,
    ) -> set[int]:
        """Return subtask indices that have a saved episode_info.json."""
        if not self._segmented_storage_mode:
            return set()
        if full_idx is None:
            with self._state_lock:
                full_idx = self._current_full_episode_index
        full_idx = int(full_idx)

        saved: set[int] = set()
        segments_root = self._full_episode_dir(full_idx) / 'segments'
        if segments_root.exists():
            for subtask_dir in segments_root.iterdir():
                if not subtask_dir.is_dir() or not subtask_dir.name.isdigit():
                    continue
                if not (subtask_dir / 'episode_info.json').exists():
                    continue
                info = self._read_episode_info(subtask_dir)
                try:
                    subtask_idx = int(info.get('subtask_index', -1))
                except (TypeError, ValueError):
                    continue
                if 0 <= subtask_idx < self._physical_segment_total:
                    saved.add(subtask_idx)
            cache = getattr(self, '_saved_subtasks_cache', {})
            cache[full_idx] = set(saved)
            self._saved_subtasks_cache = cache
            return saved

        cache = getattr(self, '_saved_subtasks_cache', {})
        return set(cache.get(full_idx, set()))

    def missing_subtasks_for_full_episode(
        self,
        full_idx: int | None = None,
    ) -> list[int]:
        """Return planned subtasks that are not saved for a full episode."""
        if not self._segmented_storage_mode:
            return []
        saved = self.saved_subtask_indices_for_full_episode(full_idx)
        return [
            idx for idx in range(self._physical_segment_total)
            if idx not in saved
        ]

    def full_episode_archive_errors(
        self,
        full_idx: int | None = None,
    ) -> list[str]:
        """Return problems that would make FINISH_EPISODE fail."""
        if not self._segmented_storage_mode:
            return []
        if full_idx is None:
            with self._state_lock:
                full_idx = self._current_full_episode_index
        full_idx = int(full_idx)

        subtask_dirs = self._episode_dirs_for_full_subtask(full_idx)
        if not subtask_dirs:
            return [f'no saved subtasks for episode {full_idx}']

        by_subtask: dict[int, Path] = {}
        for episode_dir in subtask_dirs:
            info = self._read_episode_info(episode_dir)
            try:
                subtask_idx = int(info.get('subtask_index', -1))
            except (TypeError, ValueError):
                continue
            if 0 <= subtask_idx < self._physical_segment_total:
                by_subtask.setdefault(subtask_idx, episode_dir)

        errors: list[str] = []
        missing = [
            idx for idx in range(self._physical_segment_total)
            if idx not in by_subtask
        ]
        if missing:
            errors.append(f'missing subtask(s) {missing}')

        out_dir = self._full_episode_dir(full_idx)
        for subtask_idx, seg_dir in sorted(by_subtask.items()):
            if not (seg_dir / 'metadata.yaml').exists():
                errors.append(f'subtask {subtask_idx}: missing metadata.yaml')
            archived_mcaps = (
                sorted(out_dir.glob(f'{full_idx}_{subtask_idx}.mcap'))
                + sorted(out_dir.glob(f'{full_idx}_{subtask_idx}_*.mcap'))
            )
            if not list(seg_dir.glob('*.mcap')) and not archived_mcaps:
                errors.append(f'subtask {subtask_idx}: missing .mcap file')
        return errors

    def finish_full_episode(self) -> Optional[Path]:
        """Advance the full-episode cursor after all planned subtasks saved."""
        if not self._segmented_storage_mode:
            return None
        with self._state_lock:
            current_full = self._current_full_episode_index
        archived_dir = self._archive_full_episode(current_full)
        with self._state_lock:
            if self._current_full_episode_index != current_full:
                return archived_dir
            self._current_full_episode_index += 1
            self._current_subtask_index = 0
            self._current_scenario_number = 0
        return archived_dir

    def _episode_dirs_for_full_subtask(self, full_idx: int, subtask_idx: int | None = None):
        matches = []
        saved_indices: set[int] = set()
        segments_root = self._full_episode_dir(int(full_idx)) / 'segments'
        if segments_root.exists():
            for episode_dir in segments_root.iterdir():
                if not episode_dir.is_dir() or not episode_dir.name.isdigit():
                    continue
                if not (episode_dir / 'episode_info.json').exists():
                    continue
                info = self._read_episode_info(episode_dir)
                try:
                    candidate_full = int(info.get('full_episode_index', -1))
                    candidate_subtask = int(info.get('subtask_index', -1))
                except (TypeError, ValueError):
                    continue
                if candidate_full != int(full_idx):
                    continue
                if subtask_idx is not None and candidate_subtask != subtask_idx:
                    continue
                matches.append(episode_dir)
                if 0 <= candidate_subtask < self._physical_segment_total:
                    saved_indices.add(candidate_subtask)
            matches.sort(
                key=lambda path: int(
                    self._read_episode_info(path).get('subtask_index', 0) or 0
                )
            )
            if subtask_idx is None:
                cache = getattr(self, '_saved_subtasks_cache', {})
                cache[int(full_idx)] = saved_indices
                self._saved_subtasks_cache = cache
            return matches

        for episode_dir in self._iter_subtask_episode_dirs():
            info = self._read_episode_info(episode_dir)
            try:
                candidate_full = int(info.get('full_episode_index', -1))
                candidate_subtask = int(info.get('subtask_index', -1))
            except (TypeError, ValueError):
                continue
            if candidate_full != full_idx:
                continue
            if subtask_idx is not None and candidate_subtask != subtask_idx:
                continue
            matches.append(episode_dir)
        return matches

    def discard_saved_subtask(self, subtask_idx: int) -> int:
        """Delete saved raw episode dirs for one subtask in the active full episode."""
        if not self._segmented_storage_mode:
            return 0
        with self._state_lock:
            full_idx = self._current_full_episode_index
        deleted = 0
        for episode_dir in self._episode_dirs_for_full_subtask(full_idx, int(subtask_idx)):
            shutil.rmtree(episode_dir, ignore_errors=True)
            deleted += 1
        if not self._episode_dirs_for_full_subtask(full_idx):
            shutil.rmtree(self._full_episode_dir(full_idx), ignore_errors=True)
        cache = getattr(self, '_saved_subtasks_cache', {})
        saved = set(cache.get(full_idx, set()))
        saved.discard(int(subtask_idx))
        if saved:
            cache[full_idx] = saved
        else:
            cache.pop(full_idx, None)
        self._saved_subtasks_cache = cache
        with self._state_lock:
            self._record_episode_count = self._find_next_episode_number()
            self._current_subtask_index = int(subtask_idx)
            self._current_scenario_number = self._current_subtask_index
        return deleted

    def discard_full_episode(self, full_idx: int) -> int:
        """Delete all saved raw subtask dirs for a specific full episode."""
        if not self._segmented_storage_mode:
            return 0
        full_idx = int(full_idx)
        deleted = 0
        for episode_dir in self._episode_dirs_for_full_subtask(full_idx):
            shutil.rmtree(episode_dir, ignore_errors=True)
            deleted += 1
        shutil.rmtree(self._full_episode_dir(full_idx), ignore_errors=True)
        cache = getattr(self, '_saved_subtasks_cache', {})
        cache.pop(full_idx, None)
        self._saved_subtasks_cache = cache
        with self._state_lock:
            self._record_episode_count = self._find_next_episode_number()
            if self._current_full_episode_index == full_idx:
                self._current_subtask_index = 0
                self._current_scenario_number = 0
        return deleted

    def discard_current_full_episode(self) -> int:
        """Delete all saved raw subtask dirs for the active full episode."""
        if not self._segmented_storage_mode:
            return 0
        with self._state_lock:
            full_idx = self._current_full_episode_index
        return self.discard_full_episode(full_idx)

    def discard_recording(self, reset_subtask_index: bool = False):
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
            if reset_subtask_index and self._segmented_storage_mode:
                self._current_subtask_index = 0
                self._current_scenario_number = 0
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
            full_episode = self._current_full_episode_index
            subtask = self._current_subtask_index
        if status == 'idle' and not allow_idle:
            return None  # Not recording
        if status == 'warmup':
            return None  # Legacy: Not ready yet
        if self._segmented_storage_mode:
            return str(self._subtask_rosbag_dir(full_episode, subtask))
        return self._save_rosbag_path + f'/{episode}'

    def _full_episode_dir(self, full_idx: int) -> Path:
        """Path for one full episode in subtask mode."""
        return Path(self._save_rosbag_path) / str(full_idx)

    def _subtask_rosbag_dir(self, full_idx: int, subtask_idx: int) -> Path:
        full_dir = self._full_episode_dir(full_idx)
        full_dir.mkdir(parents=True, exist_ok=True)
        return full_dir / 'segments' / str(subtask_idx)

    @staticmethod
    def _move_file(src: Path, dst: Path) -> None:
        """Move a file, preferring atomic rename within the same filesystem."""
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.replace(dst)
        except OSError as exc:
            if getattr(exc, 'errno', None) != errno.EXDEV:
                raise
            shutil.move(str(src), str(dst))

    def _archive_full_episode(self, full_idx: int) -> Optional[Path]:
        if not self._segmented_storage_mode:
            return None

        subtask_dirs = self._episode_dirs_for_full_subtask(full_idx)
        if not subtask_dirs:
            return None
        subtask_dirs = sorted(
            subtask_dirs,
            key=lambda path: int(
                self._read_episode_info(path).get('subtask_index', 0) or 0
            ),
        )
        by_subtask = {
            int(self._read_episode_info(path).get('subtask_index', 0) or 0): path
            for path in subtask_dirs
        }
        missing = [
            idx for idx in range(self._physical_segment_total)
            if idx not in by_subtask
        ]
        if missing:
            raise RuntimeError(
                f'Cannot finish episode {full_idx}: missing subtask(s) {missing}'
            )

        out_dir = self._full_episode_dir(full_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        ordered = [by_subtask[idx] for idx in range(self._physical_segment_total)]

        for stale in ('metadata.yaml', 'episode_info.json'):
            (out_dir / stale).unlink(missing_ok=True)

        output_files = []
        file_entries = []
        all_topic_counts = {}
        global_start = None
        global_end = None
        total_messages = 0
        ros_distro = 'jazzy'
        segments_meta = []
        segment_start_s = 0.0

        for subtask_idx, seg_dir in enumerate(ordered):
            seg_info = self._read_episode_info(seg_dir)
            src_mcaps = sorted(seg_dir.glob('*.mcap'))
            dst_prefix = f'{full_idx}_{subtask_idx}'
            archived_mcaps = (
                sorted(out_dir.glob(f'{dst_prefix}.mcap'))
                + sorted(out_dir.glob(f'{dst_prefix}_*.mcap'))
            )
            mcaps = src_mcaps or archived_mcaps
            if not mcaps:
                raise FileNotFoundError(f'No .mcap file in {seg_dir}')
            meta_path = seg_dir / 'metadata.yaml'
            if not meta_path.exists():
                raise FileNotFoundError(f'No metadata.yaml in {seg_dir}')
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f) or {}
            bag_info = metadata.get('rosbag2_bagfile_information', {}) or {}
            files_info = bag_info.get('files') or []
            segment_duration_ns = 0

            for split_idx, src_mcap in enumerate(mcaps):
                dst_name = f'{full_idx}_{subtask_idx}'
                if len(mcaps) > 1:
                    dst_name += f'_{split_idx}'
                dst_name += src_mcap.suffix
                dst_path = out_dir / dst_name
                if src_mcap.resolve() != dst_path.resolve():
                    self._move_file(src_mcap, dst_path)
                output_files.append(dst_path)

                file_info = files_info[split_idx] if split_idx < len(files_info) else {}
                start_ns = int(
                    (file_info.get('starting_time') or {}).get(
                        'nanoseconds_since_epoch',
                        (bag_info.get('starting_time') or {}).get(
                            'nanoseconds_since_epoch', 0),
                    ) or 0
                )
                dur_ns = int(
                    (file_info.get('duration') or {}).get(
                        'nanoseconds',
                        (bag_info.get('duration') or {}).get('nanoseconds', 0),
                    ) or 0
                )
                msg_count = int(
                    file_info.get('message_count', bag_info.get('message_count', 0))
                    or 0
                )
                file_entries.append({
                    'path': dst_name,
                    'starting_time': {
                        'nanoseconds_since_epoch': start_ns,
                    },
                    'duration': {
                        'nanoseconds': dur_ns,
                    },
                    'message_count': msg_count,
                })
                end_ns = start_ns + dur_ns
                if global_start is None or start_ns < global_start:
                    global_start = start_ns
                if global_end is None or end_ns > global_end:
                    global_end = end_ns
                total_messages += msg_count
                segment_duration_ns += dur_ns

            ros_distro = bag_info.get('ros_distro', ros_distro)
            for entry in bag_info.get('topics_with_message_count', []):
                tm = entry.get('topic_metadata', {})
                name = tm.get('name', '')
                count = int(entry.get('message_count', 0) or 0)
                if name in all_topic_counts:
                    all_topic_counts[name]['count'] += count
                else:
                    all_topic_counts[name] = {
                        'meta': tm,
                        'count': count,
                    }
            segment_end_s = segment_start_s + (segment_duration_ns / 1_000_000_000.0)
            subtask_instruction = (
                seg_info.get('subtask_instruction', '')
                if self._subtask_mode
                else self._main_task_instruction
            )
            segments_meta.append({
                'sub_task_instruction': (
                    subtask_instruction or self._main_task_instruction
                ),
                'frame_duration': [segment_start_s, segment_end_s],
            })
            segment_start_s = segment_end_s

        topics_with_count = []
        for name in sorted(all_topic_counts):
            entry = all_topic_counts[name]
            topics_with_count.append({
                'topic_metadata': entry['meta'],
                'message_count': entry['count'],
            })

        unified = {
            'rosbag2_bagfile_information': {
                'version': 9,
                'storage_identifier': 'mcap',
                'duration': {
                    'nanoseconds': (
                        global_end - global_start
                        if global_start is not None and global_end is not None
                        else 0
                    ),
                },
                'starting_time': {
                    'nanoseconds_since_epoch': global_start or 0,
                },
                'message_count': total_messages,
                'topics_with_message_count': topics_with_count,
                'compression_format': '',
                'compression_mode': '',
                'relative_file_paths': [entry['path'] for entry in file_entries],
                'files': file_entries,
                'custom_data': None,
                'ros_distro': ros_distro,
            }
        }
        expected_mcaps = {path.resolve() for path in output_files}
        for old_mcap in out_dir.glob('*.mcap'):
            if old_mcap.resolve() not in expected_mcaps:
                old_mcap.unlink(missing_ok=True)
        with open(out_dir / 'metadata.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(unified, f, default_flow_style=False, sort_keys=False)

        self._copy_episode_sidecars(ordered, out_dir)
        video_segments, video_warnings = self._archive_episode_videos(
            ordered, out_dir, full_idx,
        )
        has_transcodable_videos = any(
            bool(segment.get('cameras')) for segment in video_segments
        )
        has_pending_remux = any(
            bool(segment.get('raw_cameras')) for segment in video_segments
        )

        summary = {
            'task_instruction': self._main_task_instruction,
            'task_num': getattr(self._task_info, 'task_num', '') or '',
            'task_name': getattr(self._task_info, 'task_name', '') or '',
            'robot_type': self._robot_type,
            'device_serial': socket.gethostname(),
            'episode_index': full_idx,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'format_version': 'robotis_v2',
            'segments': segments_meta,
            'transcoding_status': (
                'pending' if has_transcodable_videos else 'not_required'
            ),
            'video_remux_status': (
                'pending' if has_pending_remux
                else ('done' if has_transcodable_videos else 'not_required')
            ),
        }
        if video_warnings:
            summary['video_warnings'] = video_warnings
        try:
            _atomic_write_json(out_dir / 'episode_info.json', summary)
        except Exception as e:
            print(f'[ROBOTIS] Failed to save full episode summary: {e}')

        segments_root = out_dir / 'segments'
        if segments_root.exists():
            shutil.rmtree(segments_root, ignore_errors=True)
        print(
            f'[ROBOTIS] Archived full episode {full_idx}: '
            f'{len(ordered)} segment(s), {len(output_files)} mcap file(s)'
        )
        return out_dir

    @staticmethod
    def _copy_episode_sidecars(subtask_dirs: list[Path], out_dir: Path) -> None:
        for name in ('robot.urdf',):
            for seg_dir in subtask_dirs:
                src = seg_dir / name
                if src.exists():
                    shutil.copy2(src, out_dir / name)
                    break
        for seg_dir in subtask_dirs:
            src_camera_info = seg_dir / 'camera_info'
            if src_camera_info.exists():
                dst = out_dir / 'camera_info'
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src_camera_info, dst)
                break

    @staticmethod
    def _write_camera_metadata(
        episode_dir: Path,
        *,
        image_topics: dict | None = None,
        camera_info_topics: dict | None = None,
        camera_rotations: dict | None = None,
    ) -> None:
        """Persist record-time camera processing metadata beside CameraInfo.

        Per-camera ``*.yaml`` files in ``camera_info/`` mirror ROS
        CameraInfo calibration. This companion file records non-calibration
        provenance from the robot config, especially rotation that is applied
        while recording/transcoding raw video.
        """
        image_topics = dict(image_topics or {})
        camera_info_topics = dict(camera_info_topics or {})
        camera_rotations = dict(camera_rotations or {})
        camera_names = sorted(
            set(image_topics) | set(camera_info_topics) | set(camera_rotations)
        )
        if not camera_names:
            return

        cameras = {}
        for name in camera_names:
            entry = {
                'rotation_deg': int(camera_rotations.get(name, 0) or 0),
                'rotation_applied_at': 'record',
            }
            if name in image_topics:
                entry['image_topic'] = image_topics[name]
            if name in camera_info_topics:
                entry['camera_info_topic'] = camera_info_topics[name]
            cameras[name] = entry

        payload = {
            'format_version': 'robotis_camera_metadata_v1',
            'source': 'robot_config',
            'cameras': cameras,
        }
        metadata_path = episode_dir / 'camera_info' / 'camera_metadata.yaml'
        try:
            _atomic_write_text(
                metadata_path,
                yaml.safe_dump(
                    payload,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                ),
            )
        except Exception as exc:
            print(f'[ROBOTIS] Failed to save camera metadata: {exc}')

    @staticmethod
    def _read_camera_metadata_rotations(episode_dir: Path) -> dict[str, int]:
        metadata_path = episode_dir / 'camera_info' / 'camera_metadata.yaml'
        if not metadata_path.exists():
            return {}
        try:
            data = yaml.safe_load(metadata_path.read_text(encoding='utf-8')) or {}
            cameras = data.get('cameras') or {}
            if not isinstance(cameras, dict):
                return {}
            rotations: dict[str, int] = {}
            for name, entry in cameras.items():
                if not isinstance(entry, dict):
                    continue
                rotations[str(name)] = int(entry.get('rotation_deg', 0) or 0)
            return rotations
        except Exception:
            return {}

    @staticmethod
    def _video_artifact_summary(videos_dir: Path) -> tuple[bool, bool, bool]:
        """Return ``(has_any, has_raw_spool, has_mp4)`` for recorder output."""
        videos_dir = Path(videos_dir)
        if not videos_dir.exists():
            return False, False, False
        has_raw_spool = False
        has_mp4 = False
        for path in videos_dir.rglob('*'):
            size = DataManager._file_size_if_present(path)
            if size <= 0:
                continue
            if path.name.endswith('.mjpeg.tmp'):
                has_raw_spool = True
            elif (
                path.suffix == '.mp4'
                and not path.stem.endswith('_synced')
                and not path.name.endswith('.remuxing.mp4')
            ):
                has_mp4 = True
        return has_raw_spool or has_mp4, has_raw_spool, has_mp4

    @staticmethod
    def _file_size_if_present(path: Path) -> int:
        try:
            if not path.is_file():
                return 0
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _archive_episode_videos(
        subtask_dirs: list[Path],
        out_dir: Path,
        full_idx: int,
    ) -> tuple[list[dict], dict[str, dict[str, str]]]:
        dst_videos = out_dir / 'videos'
        dst_videos.mkdir(parents=True, exist_ok=True)

        video_segments = []
        warnings: dict[str, dict[str, str]] = {}
        for subtask_idx, seg_dir in enumerate(subtask_dirs):
            prefix = f'{full_idx}_{subtask_idx}'
            seg_info = DataManager._read_episode_info(seg_dir)
            seg_videos = seg_dir / 'videos'
            expected_cameras = set()
            if seg_videos.exists():
                expected_cameras.update(
                    path.stem for path in seg_videos.glob('*.mp4')
                    if not path.stem.endswith('_synced')
                )
                expected_cameras.update(
                    path.name[:-len('.mjpeg.tmp')]
                    for path in seg_videos.glob('*.mjpeg.tmp')
                    if DataManager._file_size_if_present(path) > 0
                )
            video_stats = seg_info.get('video_stats') or {}
            if isinstance(video_stats, dict):
                expected_cameras.update(str(name) for name in video_stats)

            dst_segment_dir = dst_videos / prefix
            segment_cameras = []
            segment_raw_cameras = []
            segment_warnings = {}
            for camera in sorted(expected_cameras):
                src = seg_videos / f'{camera}.mp4'
                raw_spool = seg_videos / f'{camera}.mjpeg.tmp'
                sidecar = seg_videos / f'{camera}_timestamps.parquet'
                stats = seg_videos / f'{camera}_recorder_stats.json'
                diagnostics = seg_videos / f'{camera}_diagnostics.parquet'
                dst = dst_segment_dir / f'{camera}.mp4'
                dst_raw_spool = dst_segment_dir / f'{camera}.mjpeg.tmp'
                dst_sidecar = dst_segment_dir / f'{camera}_timestamps.parquet'
                dst_stats = dst_segment_dir / f'{camera}_recorder_stats.json'
                dst_diagnostics = dst_segment_dir / f'{camera}_diagnostics.parquet'
                has_mp4 = DataManager._file_size_if_present(src) > 0
                has_raw_spool = DataManager._file_size_if_present(raw_spool) > 0
                if not has_mp4 and not has_raw_spool:
                    if (
                        (dst.exists() or dst_raw_spool.exists())
                        and dst_sidecar.exists()
                    ):
                        segment_cameras.append(camera)
                        if dst_raw_spool.exists():
                            segment_raw_cameras.append(camera)
                        continue
                    segment_warnings[camera] = 'missing video file'
                    continue
                if DataManager._file_size_if_present(sidecar) <= 0:
                    if (
                        (dst.exists() or dst_raw_spool.exists())
                        and dst_sidecar.exists()
                    ):
                        segment_cameras.append(camera)
                        if dst_raw_spool.exists():
                            segment_raw_cameras.append(camera)
                        continue
                    segment_warnings[camera] = 'missing timestamp sidecar'
                    continue
                try:
                    if has_mp4:
                        DataManager._move_file(src, dst)
                    if has_raw_spool:
                        DataManager._move_file(raw_spool, dst_raw_spool)
                        segment_raw_cameras.append(camera)
                    DataManager._move_file(sidecar, dst_sidecar)
                    if DataManager._file_size_if_present(stats) > 0:
                        DataManager._move_file(stats, dst_stats)
                    if DataManager._file_size_if_present(diagnostics) > 0:
                        DataManager._move_file(diagnostics, dst_diagnostics)
                    segment_cameras.append(camera)
                except Exception as exc:
                    segment_warnings[camera] = repr(exc)
            video_segments.append({
                'mcap': f'{prefix}.mcap',
                'video_dir': f'videos/{prefix}',
                'cameras': segment_cameras,
                'raw_cameras': segment_raw_cameras,
            })
            if segment_warnings:
                warnings[prefix] = segment_warnings

        return video_segments, warnings

    def update_task_info(self, task_info):
        """Refresh per-session config from a new task_info.

        Called by RecordingService when a second START_RECORD arrives
        for the same task — the user may have toggled the
        ``include_robotis_license`` checkbox between episodes. Without
        this refresh the existing manager would keep its first-START
        snapshot and the README would never reflect later choices.
        """
        previous_subtask_mode = getattr(self, '_subtask_mode', False)
        previous_segment_total = getattr(self, '_physical_segment_total', 1)
        self._task_info = task_info
        self._main_task_instruction = self._get_main_task_instruction(task_info)
        self._subtask_instructions = self._get_subtask_instructions(task_info)
        self._subtask_mode = bool(self._subtask_instructions)
        self._subtask_total = len(self._subtask_instructions)
        self._physical_segment_total = max(1, self._subtask_total)
        self._segmented_storage_mode = True
        self._single_task = not self._subtask_mode
        layout_changed = (
            previous_subtask_mode != self._subtask_mode
            or previous_segment_total != self._physical_segment_total
        )
        if layout_changed:
            self._episode_info_scan_cache = self._collect_episode_info_entries()
            try:
                self._validate_existing_segment_count()
            finally:
                self._episode_info_scan_cache = None
        self._include_robotis_license = bool(
            getattr(task_info, 'include_robotis_license', False)
        )
        with self._state_lock:
            if self._status == 'idle':
                if layout_changed:
                    self._episode_info_scan_cache = self._collect_episode_info_entries()
                    try:
                        (
                            self._current_full_episode_index,
                            self._current_subtask_index,
                        ) = self._find_next_subtask_position()
                    finally:
                        self._episode_info_scan_cache = None
                self._current_scenario_number = self._current_subtask_index
                self.current_instruction = self._main_task_instruction

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
        image_topics: dict | None = None,
        camera_info_topics: dict | None = None,
    ):
        """
        Save URDF and metadata for ROBOTIS format.

        Called after each episode rosbag is saved.
        Copies URDF file and all referenced mesh files.

        Args:
            urdf_path: Path to URDF file to copy.
            video_stats: ``{cam_name: {frames_received, frames_written, ...}}`` from VideoRecorder.
            camera_info_files: Deprecated; camera info files are discovered
                from the episode's camera_info/ directory when needed.
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
        # branches on this field. Per-camera files are discovered from
        # videos/ and camera_info/ directly, so episode_info stays focused
        # on semantic metadata and recording diagnostics.
        videos_dir = Path(rosbag_path) / 'videos'
        has_videos, has_raw_spool, has_mp4 = self._video_artifact_summary(videos_dir)

        # ``transcoding_status`` default depends on whether this episode
        # actually has any cameras to transcode. The TranscodeWorker
        # patches this field again once it runs.
        initial_status = 'pending' if has_videos else 'not_required'

        with self._state_lock:
            raw_episode_index = self._record_episode_count
            full_episode_index = self._current_full_episode_index
            subtask_index = self._current_subtask_index
        current_subtask_instruction = self._current_subtask_instruction()
        camera_rotations = dict(camera_rotations or {})
        image_topics = dict(image_topics or {})
        camera_info_topics = dict(camera_info_topics or {})
        self._write_camera_metadata(
            Path(rosbag_path),
            image_topics=image_topics,
            camera_info_topics=camera_info_topics,
            camera_rotations=camera_rotations,
        )

        meta_data = {
            'task_instruction': self._main_task_instruction,
            'task_num': getattr(self._task_info, 'task_num', '') or '',
            'task_name': getattr(self._task_info, 'task_name', '') or '',
            'recording_mode': 'subtask' if self._subtask_mode else 'single_segment',
            'subtask_storage_layout': 'nested',
            'subtask_instruction': current_subtask_instruction,
            'subtask_instructions': list(self._subtask_instructions),
            'full_episode_index': full_episode_index,
            'subtask_index': subtask_index,
            'subtask_total': self._physical_segment_total,
            'robot_type': self._robot_type,
            'episode_index': raw_episode_index,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'format_version': 'robotis_v2',
            'device_serial': socket.gethostname(),
            'video_stats': video_stats or {},
            'transcoding_status': initial_status,
            'video_remux_status': (
                'pending' if has_raw_spool
                else ('done' if has_mp4 else 'not_required')
            ),
        }

        meta_data_path = os.path.join(rosbag_path, 'episode_info.json')
        try:
            _atomic_write_json(meta_data_path, meta_data)
            print(f'[ROBOTIS] Metadata saved to: {meta_data_path}')
            if self._segmented_storage_mode:
                cache = getattr(self, '_saved_subtasks_cache', {})
                cache.setdefault(
                    int(full_episode_index),
                    set(),
                ).add(int(subtask_index))
                self._saved_subtasks_cache = cache
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
            full_episode_index = self._current_full_episode_index
            subtask_index = self._current_subtask_index
        current_status.current_task_instruction = self.current_instruction
        current_status.proceed_time = proceed_time
        current_status.current_episode_number = (
            full_episode_index if self._segmented_storage_mode else episode_count
        )
        current_status.current_subtask_index = subtask_index if self._subtask_mode else 0
        current_status.subtask_count = self._subtask_total if self._subtask_mode else 0
        current_status.current_subtask_instruction = self._current_subtask_instruction()
        current_status.subtask_instructions = list(self._subtask_instructions)
        if self._subtask_mode:
            current_status.saved_subtask_indices = sorted(
                self.saved_subtask_indices_for_full_episode(full_episode_index)
            )

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
        'model': Path('/workspace/model/lerobot'),
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

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
Editor for ROS2 bag based recordings.

Layout assumed on disk::

    <rosbag2_root>/
    └── <task_dir>/                ← e.g. ffw_sg2_rev1_test_0406_1
        ├── 0/
        │   ├── 0_0.mcap
        │   ├── metadata.yaml
        │   ├── episode_info.json  {"episode_index": 0, ...}
        │   ├── robot.urdf
        │   └── meshes/
        ├── 1/
        └── ...

This module provides three operations on that layout:

* ``merge_rosbag_task_folders`` — combine episodes from N source task folders
  into a single output task folder, renumbered to a contiguous 0..N-1 range.
* ``delete_rosbag_episodes`` — delete a set of episode dirs from a task folder,
  optionally compacting the surviving episodes back to 0..M-1.
* ``get_rosbag_task_info`` — summarise a task folder (episode count, fps, etc).
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
from typing import List, Optional

import yaml

from shared.io.file_io import FileIO


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RosbagMergeResult:
    output_dir: Path
    total_episodes: int
    source_episode_counts: List[int]
    moved: bool


@dataclass
class RosbagDeleteResult:
    task_dir: Path
    deleted_count: int
    remaining_count: int
    compacted: bool


@dataclass
class RosbagTaskInfo:
    task_dir: Path
    episode_count: int
    total_duration_s: float
    fps: int = 0
    robot_type: str = ''
    task_instruction: str = ''
    episode_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger('DataEditor')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.INFO)
    return logger


def _is_episode_dir(path: Path) -> bool:
    """An episode directory must contain metadata.yaml and at least one .mcap."""
    if not path.is_dir():
        return False
    if not (path / 'metadata.yaml').exists():
        return False
    return any(path.glob('*.mcap'))


def _list_episode_dirs(task_dir: Path) -> List[Path]:
    """Return episode directories under ``task_dir`` sorted by integer name.

    Folders whose name is not a plain integer are placed after the integer
    folders, sorted lexicographically — they are typically not produced by
    the recorder but we tolerate them rather than crash.
    """
    if not task_dir.is_dir():
        return []

    int_named: List[tuple] = []
    other: List[Path] = []
    for child in task_dir.iterdir():
        if not _is_episode_dir(child):
            continue
        try:
            int_named.append((int(child.name), child))
        except ValueError:
            other.append(child)
    int_named.sort(key=lambda pair: pair[0])
    other.sort(key=lambda p: p.name)
    return [p for _, p in int_named] + other


def _patch_episode_index(episode_dir: Path, new_index: int) -> None:
    """Update episode_info.json::episode_index in place."""
    info_path = episode_dir / 'episode_info.json'
    if not info_path.exists():
        return
    info = FileIO.read_json(info_path, default={}) or {}
    info['episode_index'] = new_index
    FileIO.write_json(info_path, info)


def _rename_mcap_files(episode_dir: Path, new_index: int) -> None:
    """Rename ``<old>_<n>.mcap`` files inside ``episode_dir`` to ``<new>_<n>.mcap``.

    The recorder produces a single ``<idx>_0.mcap`` file per episode, but we
    handle the general split case (``_0``, ``_1``, ...) defensively.
    """
    for mcap in list(episode_dir.glob('*.mcap')):
        stem = mcap.stem  # e.g. "3_0"
        if '_' not in stem:
            continue
        _, _, suffix = stem.partition('_')
        new_name = f'{new_index}_{suffix}{mcap.suffix}'
        if new_name != mcap.name:
            mcap.rename(episode_dir / new_name)


def _patch_metadata_paths(episode_dir: Path, new_index: int) -> None:
    """Update mcap file references inside ``metadata.yaml`` to match ``new_index``.

    rosbag2 writes ``rosbag2_bagfile_information.relative_file_paths`` and
    ``files[].path`` with the original ``<episode_index>_<split>.mcap``
    filename. After ``_rename_mcap_files`` renames the files on disk, the
    yaml's bookkeeping still points at the old name — ``ros2 bag play``
    and any other consumer that resolves files via the metadata then
    fails. Mirror the rename here so the metadata stays consistent with
    what's actually on disk.
    """
    metadata_path = episode_dir / 'metadata.yaml'
    if not metadata_path.exists():
        return
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f) or {}
    except Exception:
        return
    info = metadata.get('rosbag2_bagfile_information')
    if not isinstance(info, dict):
        return

    def _patched(name):
        if not isinstance(name, str):
            return name
        p = Path(name)
        stem = p.stem
        if '_' not in stem:
            return name
        prefix, _, suffix = stem.partition('_')
        try:
            int(prefix)
        except ValueError:
            return name
        return f'{new_index}_{suffix}{p.suffix}'

    rel = info.get('relative_file_paths')
    if isinstance(rel, list):
        info['relative_file_paths'] = [_patched(p) for p in rel]

    files = info.get('files')
    if isinstance(files, list):
        for entry in files:
            if isinstance(entry, dict) and 'path' in entry:
                entry['path'] = _patched(entry['path'])

    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)
    except Exception:
        pass


def _read_metadata_duration_ns(episode_dir: Path) -> int:
    metadata_path = episode_dir / 'metadata.yaml'
    if not metadata_path.exists():
        return 0
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f) or {}
    except Exception:
        return 0
    info = metadata.get('rosbag2_bagfile_information', {}) or {}
    duration = info.get('duration', {}) or {}
    return int(duration.get('nanoseconds', 0) or 0)


# ---------------------------------------------------------------------------
# DataEditor
# ---------------------------------------------------------------------------


class DataEditor:
    def __init__(
        self, *, verbose: bool = False, logger: Optional[logging.Logger] = None
    ):
        self.verbose = verbose
        self.logger = logger or _default_logger(verbose)

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        self.logger.log(level, msg)

    # ------------------------------------------------------------------ merge

    def merge_rosbag_task_folders(
        self,
        source_dirs: List[Path],
        output_dir: Path,
        move: bool = False,
    ) -> RosbagMergeResult:
        """Merge episodes from several task folders into ``output_dir``.

        Episodes are concatenated in the order ``source_dirs`` is given, and
        within each source the episodes are taken in numeric folder-name order.
        Output episodes are renumbered 0..N-1; ``episode_info.json::episode_index``
        and ``<idx>_*.mcap`` filenames are patched to match.

        ``output_dir`` must not already exist (or must be empty) — we never
        silently merge into a populated folder.

        With ``move=True`` the source episode dirs are moved (rename), which is
        cheap and frees disk; with ``move=False`` (default) they are copied.
        """
        if not source_dirs:
            raise ValueError('No source task folders provided')

        for src in source_dirs:
            if not src.is_dir():
                raise FileNotFoundError(f'Source task folder not found: {src}')

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f'Output task folder already exists and is not empty: {output_dir}'
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        source_episode_counts: List[int] = []
        new_index = 0

        try:
            for src in source_dirs:
                episodes = _list_episode_dirs(src)
                source_episode_counts.append(len(episodes))
                if not episodes:
                    self._log(
                        f'Source task folder has no episodes, skipping: {src}',
                        logging.WARNING,
                    )
                    continue

                for episode in episodes:
                    dest = output_dir / str(new_index)
                    if move:
                        shutil.move(str(episode), str(dest))
                    else:
                        shutil.copytree(str(episode), str(dest))
                    _rename_mcap_files(dest, new_index)
                    _patch_metadata_paths(dest, new_index)
                    _patch_episode_index(dest, new_index)
                    self._log(
                        f'Merged {episode} -> {dest} (mode={"move" if move else "copy"})'
                    )
                    new_index += 1
        except Exception:
            # Best-effort cleanup so a failed merge does not leave a half-built
            # output folder behind. Source folders are left untouched.
            self._log(
                f'Merge failed, cleaning up partial output: {output_dir}',
                logging.ERROR,
            )
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

        # README handling for the merged output. Rule:
        #   * If ANY source's README carries the ROBOTIS license, the
        #     merged output gets the license too — ROBOTIS being a
        #     partial owner is enough to claim the broader license on
        #     the combined dataset.
        #   * Otherwise (no source licensed, or no source has a README
        #     at all), generate the minimal variant.
        # Always regenerate fresh against output_dir.name rather than
        # copying a source README — the heading (`# <name>`) needs to
        # match the merged folder, not any one source's name.
        try:
            from cyclo_data.recorder.session_manager import (
                build_dataset_readme,
                readme_has_license,
            )
        except Exception as exc:
            self._log(f'README helpers unavailable, skipping merge README: {exc}',
                      logging.WARNING)
            build_dataset_readme = None
            readme_has_license = None

        src_readmes = [s / 'README.md' for s in source_dirs if (s / 'README.md').exists()]
        out_readme = output_dir / 'README.md'
        if build_dataset_readme is not None:
            try:
                any_licensed = any(readme_has_license(p) for p in src_readmes)
                out_readme.write_text(
                    build_dataset_readme(output_dir.name, include_license=any_licensed),
                    encoding='utf-8',
                )
                if any_licensed and not all(readme_has_license(p) for p in src_readmes):
                    self._log(
                        f'Merge: at least one source had the ROBOTIS license; '
                        f'merged output inherits it ({out_readme}).',
                        logging.INFO,
                    )
                else:
                    variant = 'with license' if any_licensed else 'minimal'
                    self._log(
                        f'README generated ({variant}) for merged output: {out_readme}'
                    )
            except Exception as exc:
                self._log(f'Merge README write failed: {exc}', logging.WARNING)

        # In move mode, drop the now-empty source dataset folders. shutil.move
        # only relocated the episode subdirs (0/, 1/, ...) — the parent shell
        # would otherwise stay around contradicting the "frees disk" promise.
        # rmdir() is intentional: a source that still has unrelated files
        # (custom user notes, etc.) is left alone rather than rmtree-d.
        # README is the one file we know we can safely remove (we just
        # carried its content forward to output).
        if move:
            for src in source_dirs:
                try:
                    readme = src / 'README.md'
                    if readme.exists():
                        readme.unlink()
                    src.rmdir()
                    self._log(f'Removed empty source dataset: {src}')
                except OSError as exc:
                    self._log(
                        f'Source not removed (still has contents or already gone): '
                        f'{src} ({exc})',
                        logging.DEBUG,
                    )

        return RosbagMergeResult(
            output_dir=output_dir,
            total_episodes=new_index,
            source_episode_counts=source_episode_counts,
            moved=move,
        )

    # ----------------------------------------------------------------- delete

    def delete_rosbag_episodes(
        self,
        task_dir: Path,
        indices: List[int],
        compact: bool = True,
    ) -> RosbagDeleteResult:
        """Delete the episode directories at the given indices.

        With ``compact=True`` (default) the remaining episodes are renumbered
        back to a 0..M-1 contiguous range — folder names, mcap filenames, and
        ``episode_info.json::episode_index`` are all updated.
        """
        if not task_dir.is_dir():
            raise FileNotFoundError(f'Task folder not found: {task_dir}')
        if not indices:
            raise ValueError('No episode indices provided')

        unique_indices = sorted(set(int(i) for i in indices))

        existing_dirs = {}
        for episode in _list_episode_dirs(task_dir):
            try:
                existing_dirs[int(episode.name)] = episode
            except ValueError:
                continue

        missing = [i for i in unique_indices if i not in existing_dirs]
        if missing:
            raise FileNotFoundError(
                f'Episode indices not found in {task_dir}: {missing}'
            )

        deleted = 0
        for i in unique_indices:
            shutil.rmtree(existing_dirs[i])
            self._log(f'Deleted episode {i}: {existing_dirs[i]}')
            deleted += 1

        if compact:
            self._compact_episode_indices(task_dir)

        remaining = len(_list_episode_dirs(task_dir))
        return RosbagDeleteResult(
            task_dir=task_dir,
            deleted_count=deleted,
            remaining_count=remaining,
            compacted=compact,
        )

    def _compact_episode_indices(self, task_dir: Path) -> None:
        """Renumber surviving episode dirs to 0..M-1.

        Uses a two-phase rename (final name -> ``__tmp_<i>__`` -> final name)
        to avoid collisions when an existing folder happens to occupy a target
        name. ``episode_info.json`` and mcap filenames are patched after the
        final rename so they always reflect the on-disk folder name.
        """
        episodes = _list_episode_dirs(task_dir)

        # Phase 1: move every survivor to a unique temporary name.
        temp_paths: List[Path] = []
        for i, episode in enumerate(episodes):
            tmp = task_dir / f'__tmp_{i}__'
            episode.rename(tmp)
            temp_paths.append(tmp)

        # Phase 2: rename to final 0..M-1 and patch contents.
        for new_index, tmp in enumerate(temp_paths):
            final = task_dir / str(new_index)
            tmp.rename(final)
            _rename_mcap_files(final, new_index)
            _patch_metadata_paths(final, new_index)
            _patch_episode_index(final, new_index)
            self._log(f'Compacted episode -> {final}')

    # -------------------------------------------------------------------- info

    def get_rosbag_task_info(self, task_dir: Path) -> RosbagTaskInfo:
        """Summarise a rosbag task folder.

        ``fps`` / ``robot_type`` / ``task_instruction`` are taken from the
        first episode's ``episode_info.json``. ``total_duration_s`` is the
        sum of per-episode bag durations from ``metadata.yaml``.
        """
        if not task_dir.is_dir():
            raise FileNotFoundError(f'Task folder not found: {task_dir}')

        episodes = _list_episode_dirs(task_dir)
        episode_indices: List[int] = []
        for episode in episodes:
            try:
                episode_indices.append(int(episode.name))
            except ValueError:
                continue

        total_ns = sum(_read_metadata_duration_ns(e) for e in episodes)

        fps = 0
        robot_type = ''
        task_instruction = ''
        if episodes:
            info = FileIO.read_json(
                episodes[0] / 'episode_info.json', default={}
            ) or {}
            fps = int(info.get('fps', 0) or 0)
            robot_type = str(info.get('robot_type', '') or '')
            task_instruction = str(info.get('task_instruction', '') or '')

        return RosbagTaskInfo(
            task_dir=task_dir,
            episode_count=len(episodes),
            total_duration_s=total_ns / 1e9,
            fps=fps,
            robot_type=robot_type,
            task_instruction=task_instruction,
            episode_indices=episode_indices,
        )

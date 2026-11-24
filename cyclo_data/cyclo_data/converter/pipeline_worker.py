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
# Author: Claude AI Assistant

"""
Chained Dataset Conversion Worker.

Background process that converts rosbag2 episodes through a 3-stage pipeline:
  Stage 1: rosbag → rosbag + MP4 (RosbagToMp4Converter)
  Stage 2: rosbag + MP4 → LeRobot v2.1 (RosbagToLerobotConverter)
  Stage 3: rosbag + MP4 → LeRobot v3.0 (RosbagToLerobotV30Converter)
           (Parallel from the same _converted/ input as Stage 2 — runs
            in-process via the in-tree v30 converter so we don't need
            the lerobot container available for Stage 3.)

Follows the HfApiWorker pattern using multiprocessing.Process.

Output structure:
    /workspace/rosbag2/{task}/                        # Source dataset (input)
    ├── 0/                    # Original episode
    ├── 0_converted/          # Stage 1 intermediate (MP4) — auto-cleaned
    │   ├── episode.mcap
    │   ├── cam_*.mp4
    │   ├── robot.urdf
    │   └── meshes/
    ├── 1/
    └── 1_converted/
    /workspace/lerobot/{task}_lerobot_v21/            # Stage 2 output (v2.1)
    /workspace/lerobot/{task}_lerobot_v30/            # Stage 3 output (v3.0)

The LeRobot output root (``/workspace/lerobot/``) is created on demand if
missing — keeps converted datasets out of the rosbag2 source tree.
"""

import json
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import time
from typing import Dict, List, Optional


# Where converted LeRobot datasets land. Kept separate from the rosbag2 source
# tree so the source folder stays clean (only original episodes + auto-cleaned
# *_converted/ intermediates). Created on demand inside each conversion stage
# (mkdir parents=True, exist_ok=True), so a fresh deploy doesn't need any
# manual setup.
LEROBOT_OUTPUT_ROOT = Path('/workspace/lerobot')


def lerobot_output_dirs(dataset_path, output_name, convert_v21, convert_v30):
    """Resolve only the requested writer targets, refusing to mix or overwrite data."""
    name = output_name or f'{Path(dataset_path).name}_lerobot'
    if (name != name.strip() or name in ('.', '..') or '/' in name or '\\' in name
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)):
        raise ValueError('Output dataset name must be a single folder name')
    outputs = {}
    for version, enabled in (('v21', convert_v21), ('v30', convert_v30)):
        if enabled:
            path = LEROBOT_OUTPUT_ROOT / f'{name}_{version}'
            if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
                raise ValueError(f'Output already exists: {path}. Choose another output name.')
            outputs[version] = path
    return outputs


def _copy_dataset_readme(src_dir: Path, dst_dir: Path, logger: logging.Logger) -> None:
    """Forward the recording-time README.md from the rosbag2 source folder
    to a converted LeRobot output folder.

    The recorder writes README.md (Apache 2.0 + ROBOTIS notice + HF
    frontmatter) at the task-folder root the first time any episode is
    saved. Conversion stages (v21 / v30) call this so the same legal
    notice rides forward into the converted dataset and is then ready
    for HF upload without an extra step.

    Quiet no-op for older datasets recorded before the README hook
    landed — the HF upload path's _create_dataset_card still picks up
    the slack as a fallback.
    """
    import shutil
    src = src_dir / 'README.md'
    if not src.exists():
        return
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst_dir / 'README.md'))
        logger.info(f'README forwarded: {src} -> {dst_dir / "README.md"}')
    except Exception as exc:
        logger.warning(f'README forward failed ({src} -> {dst_dir}): {exc}')


def _convert_single_episode_worker(
    episode_dir, output_dir, fps, use_hw, enable_smoothing,
    selected_cameras=None, camera_rotations=None, image_resize=None,
    camera_pairs=None,
):
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Recording format v2 fast path: the recorder already wrote per-camera
    MJPEG MP4s and Parquet sidecars at record time, so Stage 1 collapses
    into a hardlink pass that materialises ``<episode>_converted/`` for
    Stages 2/3. The synced-to-grid MP4 is produced lazily inside
    ``base_converter._sync_videos_to_grid`` at LeRobot conversion time.

    Legacy v1 episodes (images embedded in MCAP, no sidecars) still go
    through the old ``rosbag2mp4`` encoder.
    """
    import os
    import shutil
    src = Path(episode_dir)
    dst = Path(output_dir)
    videos_dir = src / 'videos'
    has_sidecars = (
        videos_dir.exists()
        and any(videos_dir.rglob('*_timestamps.parquet'))
    )
    if has_sidecars:
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        for src_file in src.rglob('*'):
            if src_file.is_dir():
                continue
            rel = src_file.relative_to(src)
            # Don't drag any stale ``*_synced.mp4`` from a previous
            # conversion attempt into the new _converted/ — they'll be
            # produced fresh by ``_sync_videos_to_grid``.
            if src_file.suffix == '.mp4' and src_file.stem.endswith('_synced'):
                continue
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists():
                dst_file.unlink()
            try:
                os.link(src_file, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)
        return str(episode_dir), True, {}

    # Recording format v1 fallback (images-in-MCAP). The rosbag2mp4 +
    # video_encoder modules will be removed once no v1 episodes need to
    # be converted; until then they remain reachable through this branch.
    from cyclo_data.converter.rosbag2mp4 import RosbagToMp4Converter
    converter = RosbagToMp4Converter(
        fps=fps,
        use_hardware_encoding=use_hw,
        camera_pairs=dict(camera_pairs or {}),
        enable_timestamp_smoothing=enable_smoothing,
        selected_cameras=list(selected_cameras or []),
        camera_rotations=dict(camera_rotations or {}),
        image_resize=tuple(image_resize) if image_resize else None,
    )
    results = converter.convert_episode(str(episode_dir), str(output_dir))
    success = any(
        result.success for result in results.values()
        if hasattr(result, 'success')
    )
    return str(episode_dir), success, results


class Mp4ConversionWorker:
    """
    Background worker for MP4 conversion.

    Uses multiprocessing.Process to run conversion in a separate process,
    following the HfApiWorker pattern.
    """

    def __init__(self):
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.progress_queue = multiprocessing.Queue()
        self.process = None
        self.logger = logging.getLogger('Mp4ConversionWorker')

        # Task state management
        self.is_processing = False
        self.current_task = None
        self.start_time = None

        # Progress tracking
        self.current_progress = {
            'current': 0,
            'total': 0,
            'percentage': 0.0,
            'current_episode': '',
            'dataset_path': ''
        }

        # Basic config for the main process logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(name)s - %(levelname)s - %(message)s'
        )

    def start(self) -> bool:
        """Start the worker process."""
        if self.process and self.process.is_alive():
            self.logger.warning('MP4 conversion worker process is already running.')
            return False

        try:
            self.logger.info('Starting MP4 conversion worker process...')

            self.process = multiprocessing.Process(
                target=self._worker_process_loop,
                args=(
                    self.input_queue,
                    self.output_queue,
                    self.progress_queue
                )
            )

            self.process.start()
            self.logger.info(
                f'MP4 conversion worker process started with PID: {self.process.pid}'
            )
            return True

        except Exception as e:
            self.logger.error(f'Failed to start MP4 conversion worker: {str(e)}')
            return False

    def stop(self, timeout: float = 3.0):
        """Stop the worker process."""
        if not self.is_alive():
            self.logger.info(
                'MP4 conversion worker process is not running or already stopped.'
            )
            return

        try:
            self.logger.info('Sending shutdown signal to MP4 conversion worker...')
            try:
                self.input_queue.put_nowait(None)
            except Exception:
                pass

            grace_timeout = max(timeout, 0.0)
            if grace_timeout > 0:
                self.process.join(grace_timeout)

            if self.process.is_alive():
                self.logger.warning(
                    'MP4 conversion worker did not terminate gracefully. '
                    'Forcing termination now.'
                )
                self.process.kill()
                self.process.join(1.0)
        except Exception as e:
            self.logger.error(f'Error stopping MP4 conversion worker process: {e}')
        finally:
            self.process = None
            self.is_processing = False
            self.current_task = None
            self.start_time = None

    def is_alive(self) -> bool:
        """Check if the worker process is alive."""
        return self.process and self.process.is_alive()

    def send_request(self, request_data: dict) -> bool:
        """
        Send a conversion request to the worker.

        Args:
            request_data: Dict containing:
                - dataset_path: Path to the dataset directory
                - robot_type: Robot type string

        Returns:
            True if request was sent successfully.
        """
        if self.is_alive():
            self.input_queue.put(request_data)
            self.is_processing = True
            self.current_task = request_data
            self.start_time = time.time()
            return True
        else:
            self.logger.error(
                'Cannot send request, MP4 conversion worker process is not running.'
            )
            return False

    def get_result(self, block: bool = False, timeout: float = 0.1) -> Optional[tuple]:
        """Get result from the output queue."""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def check_task_status(self) -> dict:
        """Check the current task status and return appropriate message."""
        result = {
            'operation': 'convert_mp4',
            'status': 'Idle',
            'dataset_path': '',
            'message': '',
            'progress': {
                'current': 0,
                'total': 0,
                'percentage': 0.0,
            }
        }

        if not self.is_alive():
            self.logger.error('MP4 conversion worker process died')
            result['status'] = 'Failed'
            result['message'] = 'MP4 conversion worker process died'
            return result

        if not self.is_processing:
            result['status'] = 'Idle'
            return result

        try:
            if self.current_task:
                result['dataset_path'] = self.current_task.get('dataset_path', '')

            # Check for progress updates from worker process
            self.current_progress = self._get_progress_from_queue()
            current = self.current_progress.get('current', 0)
            total = self.current_progress.get('total', 0)
            percentage = self.current_progress.get('percentage', 0.0)
            result['progress']['current'] = current
            result['progress']['total'] = total
            result['progress']['percentage'] = percentage

            # Check for task result
            task_result = self.get_result(block=False, timeout=0.1)
            if task_result:
                status, message = task_result
                if status == 'success':
                    log_message = f'MP4 conversion completed successfully:\n{message}'
                    self.logger.info(log_message)
                    self.is_processing = False
                    self.current_task = None

                    result['status'] = 'Success'
                    result['message'] = log_message
                    return result
                elif status == 'error':
                    log_message = f'MP4 conversion failed:\n{message}'
                    self.logger.error(log_message)
                    self.is_processing = False
                    self.current_task = None

                    result['status'] = 'Failed'
                    result['message'] = log_message
                    return result

            # Still processing
            result['status'] = 'Converting'
            current_episode = self.current_progress.get('current_episode', '')
            if current_episode:
                result['message'] = f'Converting episode {current_episode}'

            return result

        except Exception as e:
            log_message = f'Error checking MP4 conversion task status: {str(e)}'
            self.logger.error(log_message)
            result['status'] = 'Failed'
            result['message'] = log_message
            return result

    def is_busy(self) -> bool:
        """Check if the worker is currently processing a task."""
        return self.is_processing

    def _get_progress_from_queue(self) -> dict:
        """Get the latest progress information from worker process."""
        latest_progress = None
        try:
            while True:
                try:
                    latest_progress = self.progress_queue.get(block=False, timeout=0.01)
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f'Error updating progress from worker: {e}')

        return latest_progress if latest_progress else self.current_progress

    @staticmethod
    def _worker_process_loop(input_queue, output_queue, progress_queue):
        """
        Main loop for the worker process.

        Processes conversion requests from the input queue and sends
        results to the output queue.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='[MP4_CONVERSION_WORKER] %(levelname)s: %(message)s'
        )
        logger = logging.getLogger('mp4_conversion_worker')

        try:
            logger.info(f'MP4 conversion worker process started with PID: {os.getpid()}')
            logger.info('Worker is ready and waiting for requests')

            request_count = 0
            last_log_time = time.time()

            while True:
                try:
                    current_time = time.time()
                    if current_time - last_log_time > 30.0:
                        logger.info(
                            f'Worker still alive, processed {request_count} requests so far'
                        )
                        last_log_time = current_time

                    try:
                        data = input_queue.get(timeout=1.0)

                        if data is None:
                            logger.info('Received shutdown signal')
                            break

                        request_count += 1
                        logger.info(f'*** Received MP4 conversion request #{request_count} ***')

                        dataset_path = data.get('dataset_path')
                        robot_type = data.get('robot_type', '')
                        robot_config_path = data.get('robot_config_path', '')
                        source_folders = data.get('source_folders', [])

                        # fps is a conversion-time knob carried on the
                        # StartConversion srv. 0 means 'use the default'
                        # (recording is rate-agnostic; sensors stream at
                        # their natural rates and rosbag captures verbatim,
                        # so there's nothing to read off the recording).
                        DEFAULT_CONVERSION_FPS = 15
                        fps = int(data.get('fps', 0) or 0) or DEFAULT_CONVERSION_FPS
                        logger.info(f'[fps] conversion target = {fps}')

                        # Format selection. Stage 1 (MP4) is always required
                        # because Stages 2/3 read from its output. If both
                        # flags are absent/false default to running both
                        # — the StartConversion forwarder enforces the same
                        # rule, this is a second line of defence.
                        convert_v21 = bool(data.get('convert_v21', False))
                        convert_v30 = bool(data.get('convert_v30', False))
                        if not convert_v21 and not convert_v30:
                            convert_v21 = True
                            convert_v30 = True
                        outputs = lerobot_output_dirs(
                            dataset_path, data.get('conversion_output_name', ''), convert_v21, convert_v30)

                        # Selection knobs. Empty / None = use defaults
                        # from robot_config (legacy behaviour preserved).
                        selected_cameras = list(data.get('selected_cameras', []) or [])
                        camera_rotations = dict(data.get('camera_rotations', {}) or {})
                        image_resize = data.get('image_resize', None)
                        if image_resize is not None:
                            try:
                                image_resize = (
                                    int(image_resize[0]),
                                    int(image_resize[1]),
                                )
                                if image_resize[0] <= 0 or image_resize[1] <= 0:
                                    image_resize = None
                            except (TypeError, ValueError, IndexError):
                                image_resize = None
                        selected_state_topics = list(
                            data.get('selected_state_topics', []) or []
                        )
                        selected_action_topics = list(
                            data.get('selected_action_topics', []) or []
                        )
                        selected_joints = list(data.get('selected_joints', []) or [])

                        logger.info(f'Processing chained conversion for: {dataset_path}')
                        if selected_cameras or camera_rotations or image_resize:
                            logger.info(
                                f'  selected_cameras={selected_cameras or "<all>"} '
                                f'camera_rotations={camera_rotations or "<none>"} '
                                f'image_resize={image_resize or "<none>"}'
                            )
                        if selected_state_topics or selected_action_topics or selected_joints:
                            logger.info(
                                f'  selected_state_topics={selected_state_topics or "<all>"} '
                                f'selected_action_topics={selected_action_topics or "<all>"} '
                                f'selected_joints[{len(selected_joints)}]'
                            )

                        is_merge_mode = len(source_folders) > 0

                        # Compute progress bands for the enabled stages so
                        # the % bar fills smoothly regardless of which
                        # downstream formats were selected.
                        stage_names = ['mp4']
                        if convert_v21:
                            stage_names.append('v21')
                        if convert_v30:
                            stage_names.append('v30')
                        merge_end = 5.0 if is_merge_mode else 0.0
                        band_width = (100.0 - merge_end) / len(stage_names)
                        ranges = {
                            name: (merge_end + i * band_width,
                                   merge_end + (i + 1) * band_width)
                            for i, name in enumerate(stage_names)
                        }
                        n_stages = len(stage_names)

                        # Stage 0: Merge episodes (only in merge mode)
                        if is_merge_mode:
                            logger.info('=== Stage 0: Merging episodes ===')
                            success, message = Mp4ConversionWorker._merge_episodes(
                                source_folders, dataset_path,
                                progress_queue, logger,
                            )
                            if not success:
                                output_queue.put(('error', f'[Merge] {message}'))
                                continue
                            logger.info(f'Merge completed: {message}')

                        # Stage 1: MP4 conversion (always runs — Stages 2/3
                        # read its _converted/ output).
                        mp4_start, mp4_end = ranges['mp4']
                        logger.info(f'=== Stage 1/{n_stages}: Converting to MP4 ===')
                        success, message = Mp4ConversionWorker._convert_dataset(
                            dataset_path=dataset_path,
                            robot_type=robot_type,
                            robot_config_path=robot_config_path,
                            progress_queue=progress_queue,
                            logger=logger,
                            fps=fps,
                            progress_start=mp4_start,
                            progress_end=mp4_end,
                            selected_cameras=selected_cameras,
                            camera_rotations=camera_rotations,
                            image_resize=image_resize,
                        )
                        if not success:
                            logger.error(f'Stage 1 failed: {message}')
                            output_queue.put(('error', f'[Stage 1/{n_stages} MP4] {message}'))
                            continue

                        # Stage 2/3 combined fast path: parse each
                        # _converted episode once, then feed both writers.
                        if convert_v21 and convert_v30:
                            v21_start, _ = ranges['v21']
                            _, v30_end = ranges['v30']
                            logger.info(
                                '=== Stages v2.1+v3.0: shared parse + dual write ===')
                            success, message = Mp4ConversionWorker._convert_to_lerobot_both(
                                v21_output_dir=outputs['v21'],
                                v30_output_dir=outputs['v30'],
                                dataset_path=dataset_path,
                                robot_config_path=robot_config_path,
                                progress_queue=progress_queue,
                                logger=logger,
                                fps=fps,
                                progress_start=v21_start,
                                progress_end=v30_end,
                                selected_cameras=selected_cameras,
                                camera_rotations=camera_rotations,
                                image_resize=image_resize,
                                selected_state_topics=selected_state_topics,
                                selected_action_topics=selected_action_topics,
                                selected_joints=selected_joints,
                                source_rosbags=source_folders or [Path(dataset_path).name],
                            )
                            if not success:
                                logger.error(f'Shared LeRobot conversion failed: {message}')
                                output_queue.put((
                                    'error',
                                    f'[LeRobot v2.1+v3.0] {message}'))
                                continue

                        # Stage 2: LeRobot v2.1 conversion
                        elif convert_v21:
                            v21_start, v21_end = ranges['v21']
                            stage_idx = stage_names.index('v21') + 1
                            logger.info(
                                f'=== Stage {stage_idx}/{n_stages}: Converting to LeRobot v2.1 ===')
                            success, message = Mp4ConversionWorker._convert_to_lerobot_v21(
                                output_dir=outputs['v21'],
                                dataset_path=dataset_path,
                                robot_config_path=robot_config_path,
                                progress_queue=progress_queue,
                                logger=logger,
                                fps=fps,
                                progress_start=v21_start,
                                progress_end=v21_end,
                                selected_cameras=selected_cameras,
                                camera_rotations=camera_rotations,
                                image_resize=image_resize,
                                selected_state_topics=selected_state_topics,
                                selected_action_topics=selected_action_topics,
                                selected_joints=selected_joints,
                                source_rosbags=source_folders or [Path(dataset_path).name],
                            )
                            if not success:
                                logger.error(f'Stage {stage_idx} failed: {message}')
                                output_queue.put((
                                    'error',
                                    f'[Stage {stage_idx}/{n_stages} LeRobot v2.1] {message}'))
                                continue
                        else:
                            logger.info('Skipping LeRobot v2.1 (not selected)')

                        # Stage 3: LeRobot v3.0 conversion
                        if convert_v30 and not convert_v21:
                            v30_start, v30_end = ranges['v30']
                            stage_idx = stage_names.index('v30') + 1
                            logger.info(
                                f'=== Stage {stage_idx}/{n_stages}: Converting to LeRobot v3.0 ===')
                            success, message = Mp4ConversionWorker._convert_to_lerobot_v30(
                                output_dir=outputs['v30'],
                                dataset_path=dataset_path,
                                robot_config_path=robot_config_path,
                                progress_queue=progress_queue,
                                logger=logger,
                                fps=fps,
                                progress_start=v30_start,
                                progress_end=v30_end,
                                selected_cameras=selected_cameras,
                                camera_rotations=camera_rotations,
                                image_resize=image_resize,
                                selected_state_topics=selected_state_topics,
                                selected_action_topics=selected_action_topics,
                                selected_joints=selected_joints,
                                source_rosbags=source_folders or [Path(dataset_path).name],
                            )
                            if not success:
                                logger.error(f'Stage {stage_idx} failed: {message}')
                                output_queue.put((
                                    'error',
                                    f'[Stage {stage_idx}/{n_stages} LeRobot v3.0] {message}'))
                                continue
                        elif not convert_v30:
                            logger.info('Skipping LeRobot v3.0 (not selected)')

                        # Cleanup intermediate Stage 1 outputs ({episode}_converted).
                        try:
                            import shutil as _shutil
                            removed = 0
                            for d in Path(dataset_path).rglob('*'):
                                if d.is_dir() and d.name.endswith('_converted'):
                                    _shutil.rmtree(str(d))
                                    removed += 1
                            if removed:
                                logger.info(
                                    f'Cleaned up {removed} *_converted '
                                    f'intermediate folder(s) under {dataset_path}'
                                )
                        except Exception as cleanup_err:
                            logger.warning(
                                f'Failed to remove *_converted folders: {cleanup_err}'
                            )

                        # Cleanup writer-local video stitching caches after
                        # all requested LeRobot versions have finished. These
                        # folders only exist to let v2.1/v3.0 share synced
                        # video work during this conversion run.
                        try:
                            removed = Mp4ConversionWorker._cleanup_lerobot_temp_dirs(
                                outputs.values()
                            )
                            if removed:
                                logger.info(
                                    f'Cleaned up {removed} LeRobot temporary '
                                    f'folder(s) for {dataset_path}'
                                )
                        except Exception as cleanup_err:
                            logger.warning(
                                f'Failed to remove LeRobot temp folders: {cleanup_err}'
                            )

                        # Make the lerobot outputs world-readable. The v3.0
                        # converter runs inside the lerobot container as root
                        # with a restrictive umask (0o077), which leaves files
                        # unreadable from the host filesystem (e.g. VSCode).
                        try:
                            import os as _os
                            for root_dir in outputs.values():
                                if not root_dir.exists():
                                    continue
                                for p in root_dir.rglob('*'):
                                    try:
                                        if p.is_dir():
                                            _os.chmod(p, 0o755)
                                        else:
                                            _os.chmod(p, 0o644)
                                    except Exception:
                                        pass
                                _os.chmod(root_dir, 0o755)
                        except Exception as chmod_err:
                            logger.warning(
                                f'Failed to relax permissions on outputs: {chmod_err}'
                            )

                        logger.info(f'All stages completed for: {dataset_path}')
                        output_queue.put(('success', 'All stages completed successfully'))

                    except queue.Empty:
                        continue

                except Exception as e:
                    error_msg = f'MP4 conversion operation error: {str(e)}'
                    logger.error(error_msg)
                    import traceback
                    logger.error(f'Traceback: {traceback.format_exc()}')
                    output_queue.put(('error', error_msg))

        except Exception as e:
            error_msg = f'MP4 conversion worker initialization error: {str(e)}'
            logger.error(error_msg)
            import traceback
            logger.error(f'Traceback: {traceback.format_exc()}')
            output_queue.put(('error', error_msg))

        logger.info('MP4 conversion worker process shutting down')

    @staticmethod
    def _merge_episodes(
        source_folders: List[str],
        output_path: str,
        progress_queue: multiprocessing.Queue,
        logger: logging.Logger,
    ) -> tuple:
        """
        Merge episodes from multiple source folders using symlinks.

        Creates symlinks in output_path with consecutive episode numbers
        pointing to the original episode directories.

        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            episode_counter = 0
            for src_folder in source_folders:
                src_path = Path(src_folder)
                if not src_path.exists():
                    return False, f'Source folder not found: {src_path}'

                episode_dirs = sorted(
                    [d for d in src_path.iterdir()
                     if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name)
                )

                for ep_dir in episode_dirs:
                    link_path = output_path / str(episode_counter)
                    link_path.symlink_to(ep_dir.resolve())
                    logger.info(f'Symlink: {ep_dir} -> {link_path}')
                    episode_counter += 1

            # Report merge completion (0% ~ 5%)
            progress_queue.put({
                'current': episode_counter,
                'total': episode_counter,
                'percentage': 5.0,
                'current_episode': '',
                'dataset_path': str(output_path),
                'stage': 'merge'
            })

            return True, (
                f'Merged {episode_counter} episodes '
                f'from {len(source_folders)} folders'
            )

        except Exception as e:
            import traceback
            logger.error(f'Merge error: {traceback.format_exc()}')
            return False, f'Merge error: {str(e)}'

    @staticmethod
    def _convert_dataset(
        dataset_path: str,
        robot_type: str,
        robot_config_path: str,
        progress_queue: multiprocessing.Queue,
        logger: logging.Logger,
        fps: int = 15,
        progress_start: float = 0.0,
        progress_end: float = 33.0,
        selected_cameras: Optional[List[str]] = None,
        camera_rotations: Optional[Dict[str, int]] = None,
        image_resize: Optional[tuple] = None,
    ) -> tuple:
        """
        Convert all episodes in a dataset to MP4 format.

        Args:
            dataset_path: Path to the dataset directory.
            progress_queue: Queue for progress updates.
            logger: Logger instance.
            selected_cameras: Camera-name subset to encode (empty = all).
            camera_rotations: Per-camera rotation degrees (0/90/180/270).
            image_resize: Output (height, width) or None for native res.

        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                return False, f'Dataset path does not exist: {dataset_path}'

            episode_dirs = Mp4ConversionWorker._collect_raw_bag_paths(dataset_path)

            if not episode_dirs:
                return False, f'No episode directories found in {dataset_path}'

            total_episodes = len(episode_dirs)
            logger.info(f'Found {total_episodes} episodes to convert')

            converted_count = 0
            failed_episodes = []
            camera_pairs = Mp4ConversionWorker._camera_pairs_from_robot_config(
                robot_type=robot_type,
                robot_config_path=robot_config_path,
                logger=logger,
            )

            from concurrent.futures import ProcessPoolExecutor, as_completed
            from cyclo_data.converter.base_converter import (
                _active_conversion_workers,
                _conversion_worker_init,
                _resolve_conversion_worker_count,
            )

            max_workers = _resolve_conversion_worker_count(total_episodes)
            logger.info(
                f'Starting parallel MP4 conversion with {max_workers} workers'
            )

            # Report initial progress
            progress_queue.put({
                'current': 0,
                'total': total_episodes,
                'percentage': progress_start,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'mp4'
            })

            # Build episode task list
            episode_tasks = []
            for episode_dir in episode_dirs:
                episode_id = Mp4ConversionWorker._episode_display_id(episode_dir)
                output_dir = dataset_path / (
                    Mp4ConversionWorker._converted_dir_name(episode_dir)
                )
                episode_tasks.append((episode_dir, output_dir, episode_id))

            completed_count = 0
            with _active_conversion_workers(max_workers):
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_conversion_worker_init,
                ) as executor:
                    futures = {}
                    for episode_dir, output_dir, episode_id in episode_tasks:
                        future = executor.submit(
                            _convert_single_episode_worker,
                            episode_dir, output_dir,
                            fps, True, True,  # fps from caller, use_hw, enable_smoothing
                            selected_cameras or [],
                            camera_rotations or {},
                            image_resize,
                            camera_pairs,
                        )
                        futures[future] = episode_id

                    for future in as_completed(futures):
                        episode_id = futures[future]
                        completed_count += 1

                        # Update progress
                        stage_progress = completed_count / total_episodes
                        overall_progress = (
                            progress_start
                            + stage_progress * (progress_end - progress_start)
                        )
                        progress_queue.put({
                            'current': completed_count,
                            'total': total_episodes,
                            'percentage': overall_progress,
                            'current_episode': episode_id,
                            'dataset_path': str(dataset_path),
                            'stage': 'mp4'
                        })

                        try:
                            _, success, _ = future.result()
                            if success:
                                converted_count += 1
                                logger.info(
                                    f'Episode {episode_id} converted successfully '
                                    f'({completed_count}/{total_episodes})'
                                )
                            else:
                                failed_episodes.append(episode_id)
                                logger.warning(
                                    f'Episode {episode_id} conversion had issues'
                                )
                        except Exception as e:
                            failed_episodes.append(episode_id)
                            logger.error(
                                f'Error converting episode {episode_id}: {str(e)}'
                            )

            # Final progress update for Stage 1
            progress_data = {
                'current': total_episodes,
                'total': total_episodes,
                'percentage': progress_end,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'mp4'
            }
            progress_queue.put(progress_data)

            # Build result message
            if converted_count == total_episodes:
                return True, (
                    f'Successfully converted all {total_episodes} episodes '
                    f'in {dataset_path}'
                )
            elif converted_count > 0:
                return True, (
                    f'Converted {converted_count}/{total_episodes} episodes. '
                    f'Failed episodes: {", ".join(failed_episodes)}'
                )
            else:
                return False, (
                    f'Failed to convert any episodes. '
                    f'Failed episodes: {", ".join(failed_episodes)}'
                )

        except Exception as e:
            import traceback
            logger.error(f'Conversion error: {traceback.format_exc()}')
            return False, f'Conversion error: {str(e)}'

    @staticmethod
    def _read_episode_info(episode_dir: Path) -> dict:
        try:
            info_path = Path(episode_dir) / 'episode_info.json'
            if info_path.exists():
                return json.loads(info_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        return {}

    @staticmethod
    def _skip_episode_scan_path(path: Path) -> bool:
        return any(
            part.startswith('.')
            or part.endswith('_converted')
            or part in {
                '_stitched_subtasks',
                '_subtask_video_concat',
                'camera_info',
                'meshes',
                'videos',
            }
            for part in path.parts
        )

    @staticmethod
    def _cleanup_lerobot_temp_dirs(output_dirs) -> int:
        import shutil

        removed = 0
        for output_dir in output_dirs:
            for dirname in ('_subtask_video_concat', '_stitched_subtasks'):
                path = output_dir / dirname
                if path.exists():
                    shutil.rmtree(str(path), ignore_errors=True)
                    removed += 1
        return removed

    @staticmethod
    def _is_rosbag_dir(path: Path) -> bool:
        path = Path(path)
        return (
            path.is_dir()
            and (path / 'metadata.yaml').exists()
            and (any(path.glob('*.mcap')) or any(path.glob('*.db3')))
        )

    @staticmethod
    def _episode_sort_key(path: Path):
        info = Mp4ConversionWorker._read_episode_info(path)
        try:
            full_idx = int(info.get('full_episode_index'))
        except (TypeError, ValueError):
            full_idx = None
        try:
            subtask_idx = int(info.get('subtask_index', 0) or 0)
        except (TypeError, ValueError):
            subtask_idx = 0
        try:
            raw_idx = int(info.get('episode_index'))
        except (TypeError, ValueError):
            raw_idx = None

        if full_idx is not None and info.get('recording_mode') == 'subtask':
            return (full_idx, subtask_idx, raw_idx if raw_idx is not None else 0, str(path))
        if raw_idx is not None:
            return (raw_idx, 0, raw_idx, str(path))
        try:
            return (int(path.name), 0, int(path.name), str(path))
        except ValueError:
            return (10**9, 0, 10**9, str(path))

    @staticmethod
    def _collect_raw_bag_paths(dataset_path: Path) -> List[Path]:
        dataset_path = Path(dataset_path)
        candidates: List[Path] = []
        for child in dataset_path.rglob('*'):
            if not child.is_dir():
                continue
            try:
                rel = child.relative_to(dataset_path)
            except ValueError:
                rel = child
            if Mp4ConversionWorker._skip_episode_scan_path(rel):
                continue
            if Mp4ConversionWorker._is_rosbag_dir(child):
                candidates.append(child)
        return sorted(candidates, key=Mp4ConversionWorker._episode_sort_key)

    @staticmethod
    def _episode_display_id(episode_dir: Path) -> str:
        info = Mp4ConversionWorker._read_episode_info(episode_dir)
        if info.get('recording_mode') == 'subtask':
            return (
                f"full_{info.get('full_episode_index', 0)}_"
                f"subtask_{info.get('subtask_index', 0)}"
            )
        if info.get('episode_index') is not None:
            return str(info.get('episode_index'))
        return str(episode_dir.name)

    @staticmethod
    def _converted_dir_name(episode_dir: Path) -> str:
        info = Mp4ConversionWorker._read_episode_info(episode_dir)
        if info.get('recording_mode') == 'subtask':
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
            return (
                f'full_{full_idx:06d}_subtask_{subtask_idx:03d}_'
                f'raw_{raw_idx:06d}_converted'
            )
        try:
            raw_idx = int(info.get('episode_index', episode_dir.name))
            return f'{raw_idx}_converted'
        except (TypeError, ValueError):
            return f'{episode_dir.name}_converted'

    @staticmethod
    def _camera_pairs_from_robot_config(
        robot_type: str,
        robot_config_path: str,
        logger: logging.Logger,
    ) -> Dict[str, tuple[str, str]]:
        """Build ``{camera_name: (image_topic, camera_info_topic)}`` from config."""
        if not robot_type and not robot_config_path:
            logger.warning(
                'No robot_type/robot_config_path supplied; images-in-MCAP '
                'MP4 fallback will not have camera pairs.'
            )
            return {}
        try:
            from cyclo_data.converter.rosbag2mp4 import RosbagToMp4Converter

            pairs = RosbagToMp4Converter.camera_pairs_from_robot_config(
                robot_type,
                robot_config_path or None,
            )
            logger.info(
                f'Loaded {len(pairs)} camera pair(s) from robot_config: '
                f'{list(pairs.keys())}'
            )
            return pairs
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'Failed to build camera pairs from robot_config: {exc!r}')
            return {}

    @staticmethod
    def _collect_converted_bag_paths(dataset_path: Path) -> List[Path]:
        return sorted(
            [
                d for d in dataset_path.rglob('*')
                if d.is_dir()
                and d.name.endswith('_converted')
                and Mp4ConversionWorker._is_rosbag_dir(d)
            ],
            key=Mp4ConversionWorker._episode_sort_key,
        )

    @staticmethod
    def _copy_converter_context(src, dst) -> None:
        """Carry parse-time metadata into a writer-only converter."""
        for attr in (
            '_state_joint_names',
            '_action_joint_names',
            '_joint_order_by_group',
            '_staleness_reports',
            '_quality_reports',
        ):
            if hasattr(src, attr):
                value = getattr(src, attr)
                if isinstance(value, dict):
                    value = dict(value)
                elif isinstance(value, list):
                    value = list(value)
                setattr(dst, attr, value)

    @staticmethod
    def _parse_converted_episodes(
        bag_paths: List[Path],
        config,
        logger: logging.Logger,
    ) -> tuple:
        """Parse _converted episodes once for the dual v2.1/v3.0 path."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from cyclo_data.converter.base_converter import (
            RosbagToLerobotConverterBase,
            _active_conversion_workers,
            _conversion_worker_init,
            _convert_rosbag_worker,
            _resolve_conversion_worker_count,
        )

        parser = RosbagToLerobotConverterBase(config, logger)
        episodes_data = []

        if len(bag_paths) <= 1:
            for idx, bag_path in enumerate(bag_paths):
                episode_data = parser.convert_single_rosbag(Path(bag_path), idx)
                if episode_data is not None:
                    episodes_data.append(episode_data)
            return episodes_data, parser

        config_dict = {
            'repo_id': config.repo_id,
            'output_dir': config.output_dir,
            'fps': config.fps,
            'robot_type': config.robot_type,
            'use_videos': config.use_videos,
            'chunks_size': config.chunks_size,
            'robot_config_path': config.robot_config_path,
            'state_topics': config.state_topics,
            'action_topics': config.action_topics,
            'apply_trim': config.apply_trim,
            'apply_exclude_regions': config.apply_exclude_regions,
            'quality_warning_multiplier': config.quality_warning_multiplier,
            'quality_error_multiplier': config.quality_error_multiplier,
            'selected_cameras': list(config.selected_cameras),
            'camera_rotations': dict(config.camera_rotations),
            'image_resize': (
                tuple(config.image_resize) if config.image_resize else None
            ),
            'selected_state_topics': list(config.selected_state_topics),
            'selected_action_topics': list(config.selected_action_topics),
            'selected_joints': list(config.selected_joints),
            'source_rosbags': list(config.source_rosbags),
        }

        max_workers = _resolve_conversion_worker_count(len(bag_paths))
        logger.info(
            f'Starting shared LeRobot parsing with {max_workers} workers'
        )
        with _active_conversion_workers(max_workers):
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_conversion_worker_init,
            ) as executor:
                futures = {}
                for idx, bag_path in enumerate(bag_paths):
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
                            logger.info(
                                f'Episode {episode_index} parsed successfully'
                            )
                        else:
                            logger.warning(f'Episode {idx} returned no data')
                    except Exception as e:
                        logger.error(f'Error parsing episode {idx}: {e}')

        episodes_data.sort(key=lambda ep: ep.episode_index)
        return episodes_data, parser

    @staticmethod
    def _convert_to_lerobot_both(
        dataset_path: str,
        v21_output_dir: Path,
        v30_output_dir: Path,
        robot_config_path: str,
        progress_queue: multiprocessing.Queue,
        logger: logging.Logger,
        fps: int = 15,
        progress_start: float = 33.0,
        progress_end: float = 100.0,
        selected_cameras: Optional[List[str]] = None,
        camera_rotations: Optional[Dict[str, int]] = None,
        image_resize: Optional[tuple] = None,
        selected_state_topics: Optional[List[str]] = None,
        selected_action_topics: Optional[List[str]] = None,
        selected_joints: Optional[List[str]] = None,
        source_rosbags: Optional[List[str]] = None,
    ) -> tuple:
        """Shared parse path for v2.1 + v3.0 conversion."""
        try:
            from cyclo_data.converter.to_lerobot_v21 import (
                ConversionConfig,
                RosbagToLerobotConverter,
            )
            from cyclo_data.converter.to_lerobot_v30 import (
                V30ConversionConfig,
                RosbagToLerobotV30Converter,
            )
        except ImportError as e:
            return False, f'Failed to import LeRobot converters: {str(e)}'

        try:
            dataset_path = Path(dataset_path)
            LEROBOT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            repo_id = dataset_path.name

            bag_paths = Mp4ConversionWorker._collect_converted_bag_paths(dataset_path)
            if not bag_paths:
                return False, f'No _converted folders found in {dataset_path}'

            progress_queue.put({
                'current': 0,
                'total': len(bag_paths),
                'percentage': progress_start,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21_v30'
            })

            parser_config = ConversionConfig(
                repo_id=repo_id,
                output_dir=v21_output_dir,
                fps=fps,
                robot_config_path=robot_config_path if robot_config_path else None,
                selected_cameras=list(selected_cameras or []),
                camera_rotations=dict(camera_rotations or {}),
                image_resize=tuple(image_resize) if image_resize else None,
                selected_state_topics=list(selected_state_topics or []),
                selected_action_topics=list(selected_action_topics or []),
                selected_joints=list(selected_joints or []),
                source_rosbags=list(source_rosbags or [dataset_path.name]),
            )
            episodes_data, parser = Mp4ConversionWorker._parse_converted_episodes(
                bag_paths, parser_config, logger,
            )
            if not episodes_data:
                return False, 'No episodes were successfully parsed'

            progress_queue.put({
                'current': len(episodes_data),
                'total': len(bag_paths),
                'percentage': progress_start + (progress_end - progress_start) * 0.45,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21_v30'
            })

            v21_converter = RosbagToLerobotConverter(parser_config, logger)
            Mp4ConversionWorker._copy_converter_context(parser, v21_converter)
            if not v21_converter.write_from_episodes(episodes_data):
                return False, f'LeRobot v2.1 writing failed for {dataset_path}'
            _copy_dataset_readme(dataset_path, v21_output_dir, logger)

            progress_queue.put({
                'current': len(episodes_data),
                'total': len(bag_paths),
                'percentage': progress_start + (progress_end - progress_start) * 0.65,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21_v30'
            })

            v30_config = V30ConversionConfig(
                repo_id=repo_id,
                output_dir=v30_output_dir,
                fps=fps,
                robot_config_path=robot_config_path if robot_config_path else None,
                selected_cameras=list(selected_cameras or []),
                camera_rotations=dict(camera_rotations or {}),
                image_resize=tuple(image_resize) if image_resize else None,
                selected_state_topics=list(selected_state_topics or []),
                selected_action_topics=list(selected_action_topics or []),
                selected_joints=list(selected_joints or []),
                source_rosbags=list(source_rosbags or [dataset_path.name]),
            )
            v30_converter = RosbagToLerobotV30Converter(v30_config, logger)
            Mp4ConversionWorker._copy_converter_context(parser, v30_converter)
            if not v30_converter.write_from_episodes(episodes_data):
                return False, f'LeRobot v3.0 writing failed for {dataset_path}'
            _copy_dataset_readme(dataset_path, v30_output_dir, logger)

            progress_queue.put({
                'current': len(episodes_data),
                'total': len(bag_paths),
                'percentage': progress_end,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21_v30'
            })

            return True, (
                f'LeRobot v2.1+v3.0 conversion completed: '
                f'{v21_output_dir}, {v30_output_dir}'
            )
        except Exception as e:
            import traceback
            logger.error(
                f'LeRobot shared conversion error: {traceback.format_exc()}'
            )
            return False, f'LeRobot shared conversion error: {str(e)}'

    @staticmethod
    def _convert_to_lerobot_v21(
        dataset_path: str,
        output_dir: Path,
        robot_config_path: str,
        progress_queue: multiprocessing.Queue,
        logger: logging.Logger,
        fps: int = 15,
        progress_start: float = 33.0,
        progress_end: float = 66.0,
        selected_cameras: Optional[List[str]] = None,
        camera_rotations: Optional[Dict[str, int]] = None,
        image_resize: Optional[tuple] = None,
        selected_state_topics: Optional[List[str]] = None,
        selected_action_topics: Optional[List[str]] = None,
        selected_joints: Optional[List[str]] = None,
        source_rosbags: Optional[List[str]] = None,
    ) -> tuple:
        """
        Stage 2: Convert _converted folders to LeRobot v2.1 format.

        Selection knobs are forwarded into ConversionConfig so the
        converter applies them at parsing / feature-build / output-write
        time. Defaults preserve legacy behaviour.

        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            from cyclo_data.converter.to_lerobot_v21 import (
                ConversionConfig,
                RosbagToLerobotConverter
            )
        except ImportError as e:
            return False, f'Failed to import LeRobot v2.1 converter: {str(e)}'

        try:
            dataset_path = Path(dataset_path)
            LEROBOT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            repo_id = dataset_path.name

            bag_paths = Mp4ConversionWorker._collect_converted_bag_paths(dataset_path)

            if not bag_paths:
                return False, f'No _converted folders found in {dataset_path}'

            logger.info(
                f'Found {len(bag_paths)} converted episodes for LeRobot v2.1'
            )

            # Report stage start
            progress_queue.put({
                'current': 0,
                'total': len(bag_paths),
                'percentage': progress_start,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21'
            })

            config = ConversionConfig(
                repo_id=repo_id,
                output_dir=output_dir,
                fps=fps,
                robot_config_path=robot_config_path if robot_config_path else None,
                selected_cameras=list(selected_cameras or []),
                camera_rotations=dict(camera_rotations or {}),
                image_resize=tuple(image_resize) if image_resize else None,
                selected_state_topics=list(selected_state_topics or []),
                selected_action_topics=list(selected_action_topics or []),
                selected_joints=list(selected_joints or []),
                source_rosbags=list(source_rosbags or [dataset_path.name]),
            )

            converter = RosbagToLerobotConverter(config, logger)
            success = converter.convert_multiple_rosbags(bag_paths)

            # Report stage completion
            progress_queue.put({
                'current': len(bag_paths),
                'total': len(bag_paths),
                'percentage': progress_end,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v21'
            })

            if success:
                _copy_dataset_readme(dataset_path, output_dir, logger)
                return True, f'LeRobot v2.1 conversion completed: {output_dir}'
            else:
                return False, f'LeRobot v2.1 conversion failed for {dataset_path}'

        except Exception as e:
            import traceback
            logger.error(f'LeRobot v2.1 conversion error: {traceback.format_exc()}')
            return False, f'LeRobot v2.1 conversion error: {str(e)}'

    @staticmethod
    def _convert_to_lerobot_v30(
        dataset_path: str,
        output_dir: Path,
        robot_config_path: str,
        progress_queue: multiprocessing.Queue,
        logger: logging.Logger,
        fps: int = 15,
        progress_start: float = 66.0,
        progress_end: float = 100.0,
        selected_cameras: Optional[List[str]] = None,
        camera_rotations: Optional[Dict[str, int]] = None,
        image_resize: Optional[tuple] = None,
        selected_state_topics: Optional[List[str]] = None,
        selected_action_topics: Optional[List[str]] = None,
        selected_joints: Optional[List[str]] = None,
        source_rosbags: Optional[List[str]] = None,
    ) -> tuple:
        """
        Stage 3: Convert rosbag _converted/ folders to LeRobot v3.0 in-process.

        Mirrors Stage 2's structure (also reads _converted/ folders) but
        emits LeRobot v3.0 layout via cyclo_data.converter.to_lerobot_v30.
        Used to shell out to 'docker exec lerobot_server …' against the
        upstream `lerobot.datasets.v30.convert_dataset_v21_to_v30` script
        — that path required the lerobot container to be running and
        coupled this stage to a heavy PyTorch dependency. The in-tree
        RosbagToLerobotV30Converter has no lerobot package import (just
        pandas / pyarrow / numpy + ffmpeg subprocess for video concat),
        so cyclo_intelligence can produce v3.0 datasets standalone.

        Note: parses rosbags a second time (Stage 2 already did once for
        v2.1). Slightly wasteful but trades CPU for self-containment.
        Skip Stage 2 in the future if only v3.0 is needed.

        Args:
            dataset_path: Path to the dataset directory containing _converted/ subdirs.
            robot_config_path: Path to robot config YAML file.
            progress_queue: Queue for progress updates.
            logger: Logger instance.
            fps: Target frame rate written into info.json.
            progress_start, progress_end: Percentage band assigned to
                this stage by the worker loop.

        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            from cyclo_data.converter.to_lerobot_v30 import (
                V30ConversionConfig,
                RosbagToLerobotV30Converter,
            )
        except ImportError as e:
            return False, f'Failed to import LeRobot v3.0 converter: {str(e)}'

        try:
            dataset_path = Path(dataset_path)
            LEROBOT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            repo_id = dataset_path.name

            # Same input as Stage 2 — _converted/ folders from Stage 1.
            bag_paths = Mp4ConversionWorker._collect_converted_bag_paths(dataset_path)

            if not bag_paths:
                return False, f'No _converted folders found in {dataset_path}'

            logger.info(
                f'Found {len(bag_paths)} converted episodes for LeRobot v3.0'
            )

            # Report stage start
            progress_queue.put({
                'current': 0,
                'total': len(bag_paths),
                'percentage': progress_start,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v30'
            })

            config = V30ConversionConfig(
                repo_id=repo_id,
                output_dir=output_dir,
                fps=fps,
                robot_config_path=robot_config_path if robot_config_path else None,
                selected_cameras=list(selected_cameras or []),
                camera_rotations=dict(camera_rotations or {}),
                image_resize=tuple(image_resize) if image_resize else None,
                selected_state_topics=list(selected_state_topics or []),
                selected_action_topics=list(selected_action_topics or []),
                selected_joints=list(selected_joints or []),
                source_rosbags=list(source_rosbags or [dataset_path.name]),
            )

            converter = RosbagToLerobotV30Converter(config, logger)
            success = converter.convert_multiple_rosbags(bag_paths)

            # Report stage completion
            progress_queue.put({
                'current': len(bag_paths),
                'total': len(bag_paths),
                'percentage': progress_end,
                'current_episode': '',
                'dataset_path': str(dataset_path),
                'stage': 'lerobot_v30'
            })

            if success:
                _copy_dataset_readme(dataset_path, output_dir, logger)
                return True, f'LeRobot v3.0 conversion completed: {output_dir}'
            else:
                return False, f'LeRobot v3.0 conversion failed for {dataset_path}'

        except Exception as e:
            import traceback
            logger.error(f'LeRobot v3.0 conversion error: {traceback.format_exc()}')
            return False, f'LeRobot v3.0 conversion error: {str(e)}'

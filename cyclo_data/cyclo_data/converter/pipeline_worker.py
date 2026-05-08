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
):
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Creates a fresh RosbagToMp4Converter instance in each worker process
    and converts a single episode to MP4 format. Selection knobs
    (selected_cameras / camera_rotations / image_resize) drop or
    transform per-camera output during the encode.
    """
    from cyclo_data.converter.rosbag2mp4 import RosbagToMp4Converter
    converter = RosbagToMp4Converter(
        fps=fps,
        use_hardware_encoding=use_hw,
        enable_timestamp_smoothing=enable_smoothing,
        selected_cameras=list(selected_cameras or []),
        camera_rotations=dict(camera_rotations or {}),
        image_resize=tuple(image_resize) if image_resize else None,
    )
    results = converter.convert_episode(str(episode_dir), str(output_dir))
    # Check if any camera conversion succeeded
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

            grace_timeout = min(max(timeout, 0.0), 1.0)
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

                        # Stage 2: LeRobot v2.1 conversion
                        if convert_v21:
                            v21_start, v21_end = ranges['v21']
                            stage_idx = stage_names.index('v21') + 1
                            logger.info(
                                f'=== Stage {stage_idx}/{n_stages}: Converting to LeRobot v2.1 ===')
                            success, message = Mp4ConversionWorker._convert_to_lerobot_v21(
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
                        if convert_v30:
                            v30_start, v30_end = ranges['v30']
                            stage_idx = stage_names.index('v30') + 1
                            logger.info(
                                f'=== Stage {stage_idx}/{n_stages}: Converting to LeRobot v3.0 ===')
                            success, message = Mp4ConversionWorker._convert_to_lerobot_v30(
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
                        else:
                            logger.info('Skipping LeRobot v3.0 (not selected)')

                        # Cleanup intermediate Stage 1 outputs ({episode}_converted).
                        try:
                            import shutil as _shutil
                            removed = 0
                            for d in Path(dataset_path).iterdir():
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

                        # Make the lerobot outputs world-readable. The v3.0
                        # converter runs inside the lerobot container as root
                        # with a restrictive umask (0o077), which leaves files
                        # unreadable from the host filesystem (e.g. VSCode).
                        try:
                            import os as _os
                            v21_dir = LEROBOT_OUTPUT_ROOT / f'{Path(dataset_path).name}_lerobot_v21'
                            v30_dir = LEROBOT_OUTPUT_ROOT / f'{Path(dataset_path).name}_lerobot_v30'
                            for root_dir in (v21_dir, v30_dir):
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

            # Find all episode directories (numeric folders)
            episode_dirs = sorted([
                d for d in dataset_path.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])

            if not episode_dirs:
                return False, f'No episode directories found in {dataset_path}'

            total_episodes = len(episode_dirs)
            logger.info(f'Found {total_episodes} episodes to convert')

            converted_count = 0
            failed_episodes = []

            # Parallel episode conversion using ProcessPoolExecutor
            # Each worker creates its own RosbagToMp4Converter (stateless, picklable args)
            # max_workers=min(4, total_episodes): 2 episodes × 4 cameras = 8 NVENC sessions
            from concurrent.futures import ProcessPoolExecutor, as_completed

            max_workers = min(4, total_episodes)
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
                episode_id = episode_dir.name
                output_dir = dataset_path / f'{episode_id}_converted'
                episode_tasks.append((episode_dir, output_dir, episode_id))

            completed_count = 0
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for episode_dir, output_dir, episode_id in episode_tasks:
                    future = executor.submit(
                        _convert_single_episode_worker,
                        episode_dir, output_dir,
                        fps, True, True,  # fps from caller, use_hw, enable_smoothing
                        selected_cameras or [],
                        camera_rotations or {},
                        image_resize,
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
    def _convert_to_lerobot_v21(
        dataset_path: str,
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
            output_dir = LEROBOT_OUTPUT_ROOT / f'{dataset_path.name}_lerobot_v21'
            repo_id = dataset_path.name

            # Collect _converted folders as bag_paths
            bag_paths = sorted([
                d for d in dataset_path.iterdir()
                if d.is_dir() and d.name.endswith('_converted')
            ])

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
            output_dir = LEROBOT_OUTPUT_ROOT / f'{dataset_path.name}_lerobot_v30'
            repo_id = dataset_path.name

            # Same input as Stage 2 — _converted/ folders from Stage 1.
            bag_paths = sorted([
                d for d in dataset_path.iterdir()
                if d.is_dir() and d.name.endswith('_converted')
            ])

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

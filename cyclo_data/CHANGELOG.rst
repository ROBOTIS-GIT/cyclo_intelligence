^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.0 (2026-06-29)
------------------
* Updated Cyclo release metadata to 1.0.0.
* Contributors: Taehyeong Kim

0.2.2 (2026-06-26)
------------------
* Optimized recording startup by caching existing episode metadata scans while initializing and refreshing recording sessions.
* Preserved segmented recording subtask cache updates across save, rerecord, and discard flows.
* Contributors: Taehyeong Kim

0.2.1 (2026-06-26)
------------------
* None

0.2.0 (2026-06-25)
------------------
* Added segment-aware Replay data extraction with video segment metadata, configured robot state/action topic layouts, and Twist/Odometry motion values.
* Normalized replay recording dates to UTC ISO timestamps.
* Contributors: Taehyeong Kim

0.1.16 (2026-06-23)
-------------------
* None

0.1.15 (2026-06-22)
-------------------
* Normalized inference recording folder metadata with unique timestamp IDs.
* Preserved active recording sessions across robot type refreshes and kept status robot type synchronized.
* Contributors: Taehyeong Kim

0.1.14 (2026-06-17)
-------------------
* Added saved-subtask tracking and full-episode validation for segmented recording finish/discard flows.
* Added camera stream monitoring, recording warnings, and raw MJPEG spool remux handling for recorder video output.
* Hardened recording start/stop cleanup paths and added regression coverage for segmented recording and transcoding behavior.
* Contributors: Taehyeong Kim

0.1.13 (2026-06-11)
-------------------
* None

0.1.12 (2026-06-05)
-------------------
* Restored Hugging Face dataset and model download defaults to workspace rosbag and LeRobot model paths.
* Contributors: Seongwoo Kim

0.1.11 (2026-06-05)
-------------------
* Preserved LeRobot camera frame reuse metadata across multi-subtask stitching so target frame indexes are reported in final episode coordinates.
* Added legacy prepared episode cache handling for LeRobot conversions that restore cached episode objects without frame reuse report fields.
* Contributors: Taehyeong Kim

0.1.10 (2026-06-04)
-------------------
* Switched LeRobot camera and observation state synchronization to publisher ``header.stamp`` timestamps while keeping action data on receive/log time.
* Added ``meta/frame_reuse.parquet`` metadata for LeRobot conversions to record target video frames that reused a previous camera frame.
* Reworked camera recording to append JPEG payloads to raw MJPEG spools during capture and remux them to MP4 on STOP.
* Split video and metadata recorder workers, preserved raw spools on remux or frame-count validation failures, and added recorder diagnostics/stat summaries.
* Added a video diagnostics analysis CLI for callback delay, queue wait, raw write, metadata wait, and flush timing reports.
* Contributors: Taehyeong Kim

0.1.9 (2026-06-02)
------------------
* Improved LeRobot dataset conversion and merge handling for mixed subtask counts, segmented videos, and v3.0 video aggregation.
* Simplified the LeRobot video sync path around the CPU H.264 backend and removed unused hardware encoder code.
* Updated task information sync so SET_TASK_INFO refreshes recording metadata, save paths, and recording status immediately.
* Contributors: Taehyeong Kim

0.1.8 (2026-06-01)
------------------
* Optimized LeRobot conversion by syncing H.264 videos directly instead of materializing decoded PNG frames on disk.
* Added source video cache reuse, fast metadata validation, and tuned worker scheduling for v2.1 and v3.0 exports.
* Kept the default video backend on CPU ``libx264`` after Jetson and desktop benchmark comparisons.
* Contributors: Taehyeong Kim

0.1.7 (2026-05-27)
------------------
* None

0.1.6 (2026-05-27)
------------------
* Added behavior tree XML save support to the data file server API.
* Extended JSON error responses with structured details for tree-save conflicts.
* Contributors: Taehyeong Kim, Seongwoo Kim

0.1.5 (2026-05-26)
------------------
* Canonicalized camera names to ``cam_<side>_<part>`` throughout recording, MP4 conversion, and LeRobot export.
* Persisted record-time camera metadata next to camera calibration files and used it to avoid double-applying rotations during conversion.
* Added frame-index subtask annotations and validation that prevents mixing different subtask counts in one dataset.
* Aligned LeRobot v2.1 and v3.0 camera feature paths and conversion audit metadata with robot configuration keys.
* Fixed episode discard handling so active and partially saved segmented episodes are removed cleanly.
* Decoupled segmented episode archive from H.264 transcoding so long recordings can finish without blocking the service response.
* Contributors: Taehyeong Kim

0.1.4 (2026-05-22)
------------------
* Changed the default Hugging Face model download path to the LeRobot policy checkpoint dropbox.
* Contributors: Dongyun Kim

0.1.3 (2026-05-22)
------------------
* Fixed VideoRecorder MJPEG pipe finalization so the last real camera frame is preserved without muxing a fake gray trailer frame.
* Removed ffmpeg ``+nobuffer`` from the recording pipe because it could drop the final MJPEG packet and leave MP4 frame counts one behind timestamp sidecars.
* Added regression coverage for recorder trailer handling and raw video/sidecar frame-count parity.
* Contributors: Taehyeong Kim

0.1.2 (2026-05-20)
------------------
* Optimized LeRobot video conversion by streaming selected MP4 frames directly into ffmpeg instead of writing temporary JPEG sequences.
* Reused synced-video cache entries with stronger frame-index, source-file, fps, resize, and rotation validation.
* Cached video statistics during streaming sync to avoid later random MP4 sampling when possible.
* Shared parsed episode data when converting LeRobot v2.1 and v3.0 together to avoid duplicate rosbag parsing.
* Added a fast path for single-episode LeRobot v3.0 video aggregation.
* Added timestamp-based replay segments in episode_info.json and restored them in replay data responses.
* Stopped creating robot_config.yaml when saving replay-only segment annotations unless an existing robot config is being updated.
* Simplified new recording episode_info.json metadata by removing unused recorder/transcoding bookkeeping fields.
* Added regression tests for streaming video sync frame counts, duplicate frames, resize/rotation, fallback behavior, and cache invalidation.
* Added regression tests for timestamp-based replay segment loading, legacy frame conversion, malformed segment skipping, and save behavior.
* Contributors: Taehyeong Kim

0.1.1 (2026-05-15)
------------------
* Stabilized VideoRecorder shutdown and resource lifecycle for repeated START/STOP recording.
* Hardened background transcode worker shutdown.
* Fixed LeRobot v3.0 aggregated video timing and frame-count validation.
* Fixed video_sync H.264 frame extraction by regenerating monotonic PTS.
* Added output-adjacent temporary disk controls for video sync.
* Corrected global statistics calculation and standard-deviation flooring for LeRobot normalization.
* Added regression tests for recorder resources, video sync, and v3.0 aggregation.
* Contributors: Taehyeong Kim

0.1.0 (2026-05-15)
------------------
* Initial open-source release of the Cyclo Intelligence data package.
* Added ROS 2 services and CLI tools for recording, conversion, dataset editing, visualization, and Hugging Face operations.
* Contributors: Taehyeong Kim

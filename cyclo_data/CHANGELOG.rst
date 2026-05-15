^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.1 (2026-05-15)
------------------
* Stabilized VideoRecorder shutdown and resource lifecycle for repeated START/STOP recording.
* Hardened background transcode worker shutdown.
* Fixed LeRobot v3.0 aggregated video timing and frame-count validation.
* Fixed video_sync H.264 frame extraction by regenerating monotonic PTS.
* Added output-adjacent temporary disk controls for video sync.
* Corrected global statistics calculation and standard-deviation flooring for LeRobot normalization.
* Added regression tests for recorder resources, video sync, and v3.0 aggregation.
* Contributors: kimtaehyeong99

0.1.0 (2026-05-15)
------------------
* Initial open-source release of the Cyclo Intelligence data package.
* Added ROS 2 services and CLI tools for recording, conversion, dataset editing, visualization, and Hugging Face operations.
* Contributors: kimtaehyeong99

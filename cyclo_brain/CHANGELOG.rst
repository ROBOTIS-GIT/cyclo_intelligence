^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo_brain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.0 (2026-06-29)
------------------
* Updated Cyclo release metadata to 1.0.0.
* Contributors: Taehyeong Kim

0.2.2 (2026-06-26)
------------------
* Updated Cyclo release metadata to 0.2.2.
* Contributors: Taehyeong Kim

0.2.1 (2026-06-26)
------------------
* Updated the Isaac-GR00T submodule to include Cyclo GR00T modality configuration examples.
* Scoped GR00T Blackwell flash-attn source builds to sm_120 with single-job defaults for safer amd64 image validation.
* Contributors: Taehyeong Kim

0.2.0 (2026-06-25)
------------------
* None

0.1.16 (2026-06-23)
-------------------
* Documented the externally managed Zenoh router flow for policy-runtime integration.
* Contributors: Taehyeong Kim

0.1.15 (2026-06-22)
-------------------
* Added async/sync action request modes with updated buffer refill behavior for the shared policy runtime.
* Added GR00T TensorRT DiT acceleration selection, synthetic observation-based engine preparation, and model-local engine reuse.
* Updated GR00T policy image support to ``robotis/groot-zenoh:1.3.2``.
* Contributors: Taehyeong Kim

0.1.14 (2026-06-17)
-------------------
* None

0.1.13 (2026-06-11)
-------------------
* Restored the shared policy runtime architecture around ``main-runtime`` and ``engine-process`` services for LeRobot and GR00T.
* Added simulation-safe action handling so empty action buffers stop publishing stale robot commands.
* Standardized LeRobot and GR00T camera/model IO mapping around dataset keys, including legacy camera aliases.
* Switched policy submodules to ROBOTIS forks, updated LeRobot and GR00T image support to ``robotis/lerobot-zenoh:1.3.0`` and ``robotis/groot-zenoh:1.3.0``, and removed parent-side GR00T training wrappers.
* Fixed amd64 LeRobot and GR00T policy image builds for Blackwell-capable training smoke validation.
* Contributors: Dongyun Kim

0.1.12 (2026-06-05)
-------------------
* Flushed pending policy action chunks on inference stop in LeRobot and GR00T control publishers.
* Contributors: Seongwoo Kim

0.1.11 (2026-06-05)
-------------------
* None

0.1.10 (2026-06-04)
-------------------
* None

0.1.9 (2026-06-02)
------------------
* None

0.1.8 (2026-06-01)
------------------
* None

0.1.7 (2026-05-27)
------------------
* None

0.1.6 (2026-05-27)
------------------
* Updated LeRobot and GR00T policy container runtimes for behavior tree inference workflows.
* Added backend-specific s6 service status reporting for inference and control publisher processes.
* Improved GR00T TensorRT engine build safety and behavior tree inference lifecycle handling.
* Updated action chunk processing and policy runtime wiring for synchronized SendCommand execution.
* Contributors: Taehyeong Kim, Seongwoo Kim

0.1.5 (2026-05-26)
------------------
* Updated LeRobot and GR00T policy IO mapping to use canonical ``cam_<side>_<part>`` camera names.
* Contributors: Taehyeong Kim

0.1.4 (2026-05-22)
------------------
* Added the shared two-process policy runtime with ``main-runtime`` and ``engine-process`` services.
* Refactored LeRobot and GR00T backends onto the shared runtime, replacing the backend-local inference server and control publisher split.
* Added schema-driven ``RobotClient`` runtime configuration for cameras, joint groups, sensor-backed state, and command publishers.
* Added action chunk buffering and command publishing through ``ActionChunkProcessor`` and ``RobotClient.publish_action``.
* Added LeRobot backend loading, IO mapping, preprocessing, prediction, and optimization modules for policy-container inference.
* Added GR00T N1.7 deployment support, camera mapping, smoke checks, and runtime tests.
* Added policy container Docker wiring, s6 service layout, SDK/runtime bind mounts, and checkpoint dropbox conventions.
* Added runtime architecture documentation, policy runtime contracts, fake robot publisher, and inference verification scripts.
* Fixed GR00T ``odometry`` action key routing to the configured ``mobile`` / ``/cmd_vel`` command topic.
* Contributors: Dongyun Kim

0.1.3 (2026-05-22)
------------------
* None

0.1.2 (2026-05-20)
------------------
* None

0.1.1 (2026-05-15)
------------------
* None

0.1.0 (2026-05-15)
------------------
* Initial open-source release of the Cyclo Intelligence policy backend workspace.
* Added policy backend, SDK, and runtime source layout for Cyclo Brain development.
* Contributors: Dongyun Kim

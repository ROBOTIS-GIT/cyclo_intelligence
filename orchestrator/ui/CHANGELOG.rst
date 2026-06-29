^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cyclo-ui
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.0 (2026-06-29)
------------------
* Updated Cyclo UI release metadata to 1.0.0.
* Contributors: Taehyeong Kim

0.2.2 (2026-06-26)
------------------
* Updated Cyclo release metadata to 0.2.2.
* Contributors: Taehyeong Kim

0.2.1 (2026-06-26)
------------------
* Replaced the Training placeholder with a Training Guide for LeRobot and GR00T CLI training handoff.
* Documented that training should use the pinned policy submodules that match Cyclo Inference.
* Contributors: Taehyeong Kim

0.2.0 (2026-06-25)
------------------
* Merged Replay into Data Tools as the Review Episodes workflow.
* Reworked Replay playback with segment-aware camera panels, timeline controls, joint charts, and 3D viewer layout updates.
* Added cache-control rules so rebuilt UI assets and SPA entrypoints refresh predictably.
* Contributors: Taehyeong Kim

0.1.16 (2026-06-23)
-------------------
* Moved the inference ``Action Request`` controls into the runtime timing section so request scheduling sits with the related rate controls.
* Contributors: Taehyeong Kim

0.1.15 (2026-06-22)
-------------------
* Added inference action request mode and acceleration controls, including GR00T TensorRT engine build and status UI.
* Refined inference task and page sync so robot type, policy path, deployment mode, and recording state stay consistent across pages.
* Contributors: Taehyeong Kim

0.1.14 (2026-06-17)
-------------------
* Added camera recording monitor rows, warning toasts, and speech notifications from RecordingStatus diagnostics.
* Synced saved subtask indices from the server and hardened segmented episode finish/discard reset handling.
* Normalized record task info synchronization for policy and inference metadata.
* Contributors: Taehyeong Kim

0.1.13 (2026-06-11)
-------------------
* Added explicit ``3D Sim Deploy`` and ``Real Robot Deploy`` controls with a confirmation dialog before robot publishing.
* Added policy backend image pull controls with streaming progress feedback and explicit stale-container update handling.
* Hid unavailable backend start/restart controls when the required Docker image is missing.
* Contributors: Dongyun Kim

0.1.12 (2026-06-05)
-------------------
* Fixed workspace mount path handling in dataset and Hugging Face tools.
* Restored Hugging Face dataset paths and model backend-specific download targets.
* Added BT Manager controls for the ``bt_node`` lifecycle.
* Refined the BT Manager node-list refresh UI with BT node status gating and clearer update labeling.
* Contributors: Seongwoo Kim

0.1.11 (2026-06-05)
-------------------
* None

0.1.10 (2026-06-04)
-------------------
* Preserved empty planned subtask slots when syncing Record page task information so ``Number of SubTasks`` no longer resets after SET_TASK_INFO echo.
* Renamed the subtask input placeholder to ``Sub Task Instruction``.
* Contributors: Taehyeong Kim

0.1.9 (2026-06-02)
------------------
* Added debounced Record page Task Information sync with the backend.
* Reflected synced server task information across multiple Record pages with conflict handling and manual server/draft resolution actions.
* Contributors: Taehyeong Kim

0.1.8 (2026-06-01)
------------------
* None

0.1.7 (2026-05-27)
------------------
* Removed the UI foot-switch command bridge and synced Record subtask progress from backend recording status.
* Added Inference record session preparation for trigger-driven recordings.
* Reflected backend-triggered inference recording state in the control panel.
* Contributors: Taehyeong Kim

0.1.6 (2026-05-27)
------------------
* Reworked BT Manager with drag-and-drop node palette, dynamic node catalog loading, and parameter editing.
* Added behavior tree XML serialization, save-as handling, tree list modal, undo/redo history, and parser tests.
* Added persistent BT runtime flow and improved node drop positioning.
* Optimized RobotViewer3D ROS subscriptions by sharing rosbridge connections, lowering queue depth, and gating action preview subscriptions.
* Contributors: Taehyeong Kim, Seongoo

0.1.5 (2026-05-26)
------------------
* Added single-task episode recording and save support when no subtask count is configured.
* Preserved numeric task IDs when dispatching recording commands from the UI.
* Enabled episode discard during active single-task and segmented recording flows.
* Contributors: Taehyeong Kim

0.1.4 (2026-05-22)
------------------
* Fixed URDF loading fallbacks for the 3D robot viewer.
* Added LeRobot and GR00T target backend selection for Hugging Face model downloads.
* Routed model download folder browsing to the selected policy checkpoint dropbox.
* Contributors: Dongyun Kim

0.1.3 (2026-05-22)
------------------
* None

0.1.2 (2026-05-20)
------------------
* Added Cyclo Manager shortcuts and navigation.
* Added light/dark theme controls and supporting UI styles.
* Fixed dataset conversion folder selection so nested paths such as Temp/<dataset> are preserved.
* Set Hugging Face upload/download folder browser defaults to /workspace.
* Added replay segment restore/save support backed by timestamp-based episode_info.json segments.
* Contributors: Taehyeong Kim

0.1.1 (2026-05-15)
------------------
* None

0.1.0 (2026-05-15)
------------------
* Initial open-source release of the Cyclo Intelligence web UI.
* Added pages and controls for recording, inference, training, dataset tools, and robot status monitoring.
* Added ROS bridge integration for orchestrator and cyclo_data services.
* Contributors: Taehyeong Kim

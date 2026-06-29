^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package orchestrator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
* Updated Cyclo release metadata to 0.2.1.
* Contributors: Taehyeong Kim

0.2.0 (2026-06-25)
------------------
* Added Replay regression coverage for segmented video metadata and robot-configured joint/action layouts.
* Contributors: Taehyeong Kim

0.1.16 (2026-06-23)
-------------------
* Removed the Cyclo-managed ``zenoh_router`` service from supervisor user controls so Cyclo uses the externally managed router on port 7447.
* Contributors: Taehyeong Kim

0.1.15 (2026-06-22)
-------------------
* Propagated action request mode and TensorRT acceleration options through inference task info, behavior tree SendCommand, and backend load requests.
* Hardened inference recording task sync so trigger and UI recording flows keep inference metadata.
* Added supervisor endpoints for GR00T TensorRT engine status and build requests.
* Contributors: Taehyeong Kim

0.1.14 (2026-06-17)
-------------------
* Serialized recording command forwarding to avoid overlapping UI and trigger commands.
* Avoided forwarding stale idle cancel/rerecord commands that could race with a new recording start.
* Contributors: Taehyeong Kim

0.1.13 (2026-06-11)
-------------------
* Added simulation-first inference mode handling so policy actions can be previewed without publishing robot commands.
* Added Real Robot Deploy command gating and inference-mode propagation through task information.
* Improved policy backend lifecycle handling for Docker Compose recreation and backend process status checks.
* Contributors: Dongyun Kim

0.1.12 (2026-06-05)
-------------------
* Added BT Manager lifecycle control for starting and stopping the ``bt_node`` runtime through supervisor-managed s6 services.
* Disabled behavior tree execution controls while the ``bt_node`` process is stopped and documented the on-demand lifecycle flow.
* Contributors: Seongwoo Kim

0.1.11 (2026-06-05)
-------------------
* Added LeRobot converter regression coverage for multi-subtask frame reuse offsets, legacy prepared episode caches, and frame reuse metadata writing.
* Contributors: Taehyeong Kim

0.1.10 (2026-06-04)
-------------------
* Extended recording service timeouts for STOP, FINISH, cancel, rerecord, and segmented recording operations so post-record remux work can complete.
* Contributors: Taehyeong Kim

0.1.9 (2026-06-02)
------------------
* Synced prepared recording task information from SET_TASK_INFO so robot-button recordings use the latest folder name and metadata.
* Cached UI task information for trigger-started recording sessions and refreshed recording status after task info updates.
* Contributors: Taehyeong Kim

0.1.8 (2026-06-01)
------------------
* None

0.1.7 (2026-05-27)
------------------
* Moved leader trigger recording control into the orchestrator backend using right and left tact trigger events.
* Added prepared Record and Inference recording contexts so trigger input can start, save, and cancel recordings without UI command bridging.
* Kept inference recording state tied to actual recording start and stop responses.
* Contributors: Taehyeong Kim

0.1.6 (2026-05-27)
------------------
* Added dynamic behavior tree node discovery, catalog services, and XML tree listing.
* Added behavior tree node templates and XML serialization support for generated action and control nodes.
* Refactored behavior tree actions around joint control, SendCommand model routing, inference lifecycle handling, and monotonic timing.
* Added tree editing support for save-as workflows, execution completion state, and persistent runtime flow.
* Contributors: Taehyeong Kim, Seongwoo

0.1.5 (2026-05-26)
------------------
* None

0.1.4 (2026-05-22)
------------------
* None

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
* Initial open-source release of the Cyclo Intelligence orchestrator package.
* Added ROS 2 orchestration for recording, inference, training, task commands, file browsing, and behavior tree workflows.
* Added service forwarding boundaries for cyclo_data-owned recording, conversion, editing, and Hugging Face operations.
* Contributors: Taehyeong Kim

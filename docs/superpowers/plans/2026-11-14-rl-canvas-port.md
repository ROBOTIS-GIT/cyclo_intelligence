# RL Canvas and Playground TT-RTC integration plan

Goal: Port feature/rl_playground_simplify into current Autonomy Studio, add RL Canvas, replace old TT-RTC execution and update Action Canvas SendCommand.
Architecture: Keep Studio shell, existing map/navigation/BT and confirmed robot hold. Port source Playground environment, replay and training workflows with their recorder/API/runtime dependencies. Use action_request_mode=tt_rtc and source-clock timeline with control_hz output. Retain ordinary sync/async and VLA-RAIL. Do not copy datasets/checkpoints or overwrite deployment configuration.

- [x] UI: RL Canvas chooser, Environment/Replay Buffer/Training, hooks/API and SendCommand controls; Jest and production build.
- [x] Runtime: source TT-RTC timeline/checkpoint/RLT recording and algorithms; remove legacy execution, preserve confirmed hold/fault latch and RAIL; runtime tests.
- [x] API/data: isolated RL training/data endpoints, intervention, recorder and sidecars; dataset and API tests.
- [x] Integration: interfaces, orchestrator bridges, BT SendCommand migration, Docker mounts/dependencies. Preserve inference 7=CAPABILITIES and 8=FAULT_STOP; RLT commands 20..28.
- [x] Validate protocol, no live old TT-RTC path, BT/runtime regressions, UI/ROS build. Copy verified changes into current branch without committing. Check active execution before restarting.

Review: saved BT migration; fault-stop semantics; session ownership on UI unmount; dataset/checkpoint provenance; consistent generated ROS interfaces.

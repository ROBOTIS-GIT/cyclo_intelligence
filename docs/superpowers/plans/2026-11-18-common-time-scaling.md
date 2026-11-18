# SG2 TT-RTC common time scaling implementation plan

Goal: preserve waypoint geometry and joint coordination without retraining, bounded by 0.7 rad/s, 6 rad/s², and 100 rad/s³.

Architecture: opt-in TTActionTimeline subclass uses one quintic scalar progress for each source segment. Duration is at least the original 1/15 s and satisfies analytic peak velocity/acceleration/jerk limits for all 14 arm joints. Stop-to-stop at waypoint corners is intentionally conservative: position, velocity and acceleration remain continuous, with no dwell. All action components (including gripper timing) share progress; only arm joints determine limits. Disable independent output smoothing in this mode. Preserve unconsumed exact source targets in prefixes. Start from measured feedback; reset on lifecycle changes. Refill after actual source-target consumption, preserving the fixed six-target prefix instead of shortening it with a wall-time trigger. A pending policy switch retains its six-target committed prefix and discards only the later old-policy tail before requesting continuation, avoiding fixed service timeouts while preserving active motion.

Tech stack: Python, NumPy, existing TT-RTC control loop, pytest; no new dependency.

- [x] Add failing tests for coordinated geometry, analytic/discrete limits, queue consumption, reset/refill and active segment protection.
- [x] Implement common-time timeline and integrate opt-in SG2 TT-RTC selection, feedback seeding and source-aware scheduling.
- [x] Test switching/refill integration, old modes and real captured commands offline; report slowdown honestly.
- [x] Enable Compose option, validate runtime dependency and effective config; deploy only with BT stopped, never start robot motion.

The user approved common retiming in the preceding exchange. Existing unrelated workspace changes must be retained. Use inline execution; no subagent delegation.

Validation: 132 tests and 55 subtests pass. Captured four MLP chunks (40 targets) replayed at 100 Hz: all targets reached, common progress error 3.2e-12, maximum sampled velocity 0.699986 rad/s, acceleration 2.92559 rad/s², jerk 79.5055 rad/s³. First target 1.99 s; full fixed replay 9.82 s. No real-robot motion test was initiated.

## Follow-up: MLP-only common timing

User requested common timing only for MLP, with VLA restored to its prior independent limiter. PolicyTimeTimeline tags each queued target with its producing policy. VLA uses the original TTActionTimeline and ArmCommandSmoother. MLP uses common timing; VLA-to-MLP first brakes the existing velocity/acceleration without dropping the queued MLP targets. The policy switches when its targets execute, not when inference returns. Compose enables CYCLO_SG2_COMMON_TIME_RLT_ONLY. Gripper timing is common within MLP and follows the original VLA commands in base mode.

Validation: 114 tests and 55 subtests passed. Deployed-runtime offline replay of captured VLA prefix, four MLP chunks and a VLA continuation yielded 1201 commands, with maximum velocity 0.7 rad/s, acceleration 6 rad/s², jerk 100 rad/s³ (floating-point tolerance). Real robot execution has not been started by the agent.

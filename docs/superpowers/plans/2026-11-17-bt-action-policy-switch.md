# BT Action Policy Switch Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a selectable VLA/MLP BT node with Actor/Critic/RL bundle selection.
**Architecture:** Trees containing SwitchActionPolicy use the existing InferenceSupervisor for all inference commands. Cache the acknowledged LOAD request there; clone it for stopped bundle reloads. The new leaf polls bounded service futures without a background restart worker.
**Tech Stack:** Python ROS2, existing BT registry, React/Jest.
**Spec:** docs/superpowers/specs/2026-11-17-bt-action-policy-switch-design.md

## Global Constraints
Preserve user changes; no robot test or automatic inference start during deployment. No new dependencies or ROS schema changes. Existing ordinary BTs keep their lifecycle path. Critic chooses that bundle's Actor.

## Review Focus
- Cancel while LOAD/START is pending: no late automatic motion.
- Paused inference: selecting MLP stages route until explicit RESUME.
- Same path latest replaced: identity change must reload.
- Missing or invalid bundle: fail before stopping the current session.
- Manual trees containing Docker actions: continue using normal Docker lifecycle and confirmed inference cleanup.

### Task 1: Lifecycle and leaf
- [x] Add fake-service tests in orchestrator/tests/bt/test_switch_action_policy.py for path validation, direct route, reload ordering, identity change, paused selection, errors and cancellation.
- [x] Extend inference/supervisor.py with acknowledged LOAD snapshot, desired action mode and configurable strict completion. Support prepared request submission without bypassing lifecycle guards.
- [x] Add actions/switch_action_policy.py with action_source, training_stage and bundle_name constructor ports; validate GR00T TT-RTC and checkpoint phase; reload via STOP/UNLOAD/LOAD, restore running state and apply route.
- [x] Extend supervised_command.py START path to restore selected route; preserve explicit instruction/pose behavior.
- [x] Select direct ownership for switch-containing XML in bt_node.py; route only inference SendCommands to that owner.
- [x] Run focused BT lifecycle and registry tests.

### Task 2: Canvas and deployment
- [x] Add fallback catalog and parameter dropdowns/help/disabled stage fields; preserve XML serialization.
- [x] Test node options and XML round trip; run UI build.
- [x] Run backend regression suite and review lifecycle/cancellation diff.
- [x] Deploy UI assets preserving cyclo-config.js. Reload only inactive BT service if needed; do not start trees. Verify catalog and served assets.

## Execution notes
- Implemented inline under the user's implementation approval; reviewed by a separate read-only reviewer.
- Ruling: switch-containing manual trees share the existing direct lifecycle owner, because mixing /task asynchronous LOAD with direct switching would leave stale session state. Other manual trees remain unchanged.
- Ruling: manual tree completion confirms STOP automatically; supervised autonomy retains its explicit STOP requirement.
- Review fixes: Docker STOP/RESTART confirms STOP + UNLOAD before container lifecycle; Docker leaves do not break tree Pause. Corrupt artifacts fail before stopping; latest replaced during LOAD fails without START.
- Validation: 269 BT/plumbing/autonomy-pause tests, 41 UI tests; production UI build passed with the existing unrelated InferenceControlPanel hook warning. Live catalog exposes all three ports. No physical action test was run.

## Follow-up: initial LOAD preloading
The user superseded switch-time reload behavior: derive the active tree's single MLP bundle selection before execution, include it in initial GR00T LOAD, and keep SwitchActionPolicy route-only. Conflicting selections fail before execution; latest is pinned until the next LOAD. Regression coverage includes native float serialization and cancellation.

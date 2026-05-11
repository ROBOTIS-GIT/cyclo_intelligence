# cyclo_brain Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Clean up the lerobot inference path in `cyclo_brain/` and split three large files (`control_publisher.py`, `inference_server.py`, `lerobot_engine.py`) into responsibility-scoped mixin files. Zero behavior change.

**Architecture:** Mixin-based responsibility split. New files use `_<owner>_*.py` prefix (`_cp_`, `_is_`, `_le_`). No new directories. External method signatures unchanged — call sites zero-touch. Docker bind-mount paths and Zenoh topics unchanged.

**Tech Stack:** Python 3.12, Zenoh (`zenoh_ros2_sdk`), ROS 2 jazzy (cyclo_intelligence container), Docker Compose, s6-overlay.

**Design doc:** `docs/plans/2026-05-10-cyclo_brain-refactor-design.md` (commit `b82cf21`).

**Branches:**
- Source: `feature-update-lerobot` (current).
- Target: `feature-cyclo-brain-refactor` (created in Task 0).

**PR splitting:**
- PR1 = S1–S4 (cleanup, one commit per S).
- PR2 = S5 (`control_publisher.py` split).
- PR3 = S6 (`inference_server.py` split).
- PR4 = S7 (`lerobot_engine.py` split + docker-compose mount).

**Hard contracts (must not break):**
- Bind-mount paths: `/policy_runtime`, `/app`, `/robot_client_sdk`, `/zenoh_sdk`, `/orchestrator_config`.
- Zenoh topics: `cyclo/policy/lerobot/{configure,lifecycle,run_inference,action_chunk_raw}`.
- `InferenceCommand` srv request/response fields.
- s6 longrun service names: `inference-server`, `control-publisher`, `user`.
- `create_engine()` factory function name and signature.
- Flat numpy `action_chunk` from engine.

---

## Task 0: Set up worktree + persist verification harness

The cleanup/refactor work needs an isolated branch and a re-usable verification harness (this plan refers to it from every PR's gate).

**Files:**
- Create worktree: `~/workspace/cyclo_intelligence-refactor/` on branch `feature-cyclo-brain-refactor`.
- Persist: `docs/scripts/fake_robot_publisher.py` (currently lives only at `/tmp/fake_robot_publisher.py`).
- Persist: `docs/scripts/verify_inference_pipeline.sh` (currently `/tmp/verify_clean.sh` on the cyclo_intelligence container).

**Step 1: Create the worktree**

Run:
```bash
cd /home/rc/workspace/cyclo_intelligence
git worktree add -b feature-cyclo-brain-refactor ../cyclo_intelligence-refactor feature-update-lerobot
cd ../cyclo_intelligence-refactor
git status
```
Expected: clean working tree on `feature-cyclo-brain-refactor`.

**Step 2: Persist fake_robot_publisher.py into the repo**

Copy from the running cyclo_intelligence container:
```bash
mkdir -p docs/scripts
docker cp cyclo_intelligence:/tmp/fake_robot_publisher.py docs/scripts/fake_robot_publisher.py
chmod +x docs/scripts/fake_robot_publisher.py
head -5 docs/scripts/fake_robot_publisher.py
```
Expected: file with `"""Fake observation publisher for ffw_sg2_rev1 inference smoke test."""` docstring.

**Step 3: Persist verify_inference_pipeline.sh**

```bash
docker cp cyclo_intelligence:/tmp/verify_clean.sh docs/scripts/verify_inference_pipeline.sh
chmod +x docs/scripts/verify_inference_pipeline.sh
head -5 docs/scripts/verify_inference_pipeline.sh
```
Expected: shebang + `source /opt/ros/jazzy/setup.bash` lines.

**Step 4: Commit**

```bash
git add docs/scripts/fake_robot_publisher.py docs/scripts/verify_inference_pipeline.sh
git commit -m "chore(docs): persist inference verification harness used by refactor PRs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Step 5: Baseline verification — verify gate passes on current code**

```bash
cd /home/rc/workspace/cyclo_intelligence   # original tree, where containers run
docker exec cyclo_intelligence bash -c '/tmp/verify_clean.sh' 2>&1 | tail -22
```
Expected: STAGE 1 LOAD `success=True`, STAGE 2 `running`, STAGE 6 `/cmd_vel  →  average rate: 100.0xx`.

If baseline fails, stop and resolve before any refactor — the gate cannot validate after.

---

# PR1 — Cleanup (S1, S2, S3, S4)

One PR, four commits.

## Task 1: S1 — Remove `[CP-DBG]` stderr prints (12 sites)

**Files:**
- Modify: `cyclo_brain/policy/common/runtime/control_publisher.py`

The debug prints were added during the trigger-emission diagnosis. Now that the bug is reproduced and fixed, remove them. Keep the structured `logger.*` calls.

**Step 1: List every `[CP-DBG]` site**

```bash
grep -n "CP-DBG" cyclo_brain/policy/common/runtime/control_publisher.py
```
Expected: ~12 lines printed.

**Step 2: Remove each `[CP-DBG]` line**

For every `print(... "[CP-DBG]" ..., file=sys.stderr, flush=True)` line, delete the entire line. Also delete the surrounding `import traceback / traceback.print_exc(file=sys.stderr)` blocks that were added only to surface CP-DBG context (the `logger.error(..., exc_info=True)` already covers it).

Specifically:
- `main()` startup: 5 lines (`main() entered`, `logging configured`, `ControlPublisher instance created`, `setup() returned`, FATAL handler block).
- `_on_configure`: 2 lines + traceback block.
- `_on_lifecycle`: 1 line.
- `_setup_robot_specific_locked`: 1 line.
- `configure`: 2 lines.
- Restore the BaseException handler to only `logger.info("shutdown via SIGINT")` for `KeyboardInterrupt` and the original `finally: publisher.shutdown()` (no separate `except BaseException`).

**Step 3: Verify removal**

```bash
grep -c "CP-DBG" cyclo_brain/policy/common/runtime/control_publisher.py
```
Expected: `0`.

**Step 4: Verify no syntax regression**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/control_publisher.py
echo "compile exit=$?"
```
Expected: `compile exit=0`.

**Step 5: Live verify — restart lerobot container, run gate**

```bash
cd /home/rc/workspace/cyclo_intelligence
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml restart lerobot
sleep 8
ss -tulpn 2>/dev/null | grep 7447 || docker exec -d cyclo_intelligence bash -c 'source /opt/ros/jazzy/setup.bash && exec /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd > /tmp/zenohd.log 2>&1'
sleep 3
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -22
```
Expected: STAGE 6 `/cmd_vel  →  average rate: 100.0xx`. Also confirm:
```bash
docker logs --since 30s lerobot_server 2>&1 | grep CP-DBG | wc -l
```
Expected: `0`.

**Step 6: Commit**

```bash
git add cyclo_brain/policy/common/runtime/control_publisher.py
git commit -m "refactor(control_publisher): drop CP-DBG stderr prints

12 stderr print() sites added during the trigger-emission diagnosis are
no longer needed. The structured logger calls beside them already
surface the same information. No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: S2 — Update `cyclo_brain/README.md`

**Files:**
- Modify: `cyclo_brain/README.md`

The README still documents a per-backend `runtime/` under LeRobot (pre-refactor layout). Today lerobot uses `policy/common/runtime/` via bind mount.

**Step 1: Read current README**

```bash
cat cyclo_brain/README.md
```
Locate the section that describes `policy/lerobot/` and any reference to a per-backend `runtime/` directory.

**Step 2: Apply minimal edits**

Change the `policy/lerobot/` block to:

```
└── policy/
    ├── common/
    │   ├── runtime/          Process A (inference_server.py) + Process B
    │                          (control_publisher.py) + InferenceEngine
    │                          ABC. Bind-mounted into every policy
    │                          container at /policy_runtime, so backends
    │                          share one supervisor pair.
    │   └── s6-services/      Single s6 unit tree used by every policy
    │                          container (inference-server / control-
    │                          publisher / user longruns).
    ├── lerobot/              LeRobot backend container.
    │   ├── Dockerfile.{arm64,amd64}
    │   ├── lerobot_engine.py  Bind-mounted at /app/lerobot_engine.py;
    │                          implements InferenceEngine.
    │   ├── lerobot/          huggingface lerobot submodule.
    │   ├── RESULTS/          Validation outputs from policy load tests.
    │   └── scripts/
    └── groot/                GR00T backend (still owns its own runtime/
                              and s6-services/; common-runtime migration
                              is a separate phase).
```

Also delete any sentence claiming lerobot owns an `entrypoint.sh` or a per-backend `s6-services/` — those moved to `common/`.

**Step 3: Verify**

```bash
git diff cyclo_brain/README.md
```
Expected: only the `policy/` block changed; `sdk/` block untouched.

**Step 4: Commit**

```bash
git add cyclo_brain/README.md
git commit -m "docs(cyclo_brain): README reflects common/runtime + common/s6-services

Update the policy/ subtree description to match the post-refactor
layout — common/runtime is bind-mounted into every backend at
/policy_runtime, and s6 units live under common/. GR00T's own
runtime/ + s6-services/ are called out as a pending migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: S3 — Label the state-padding hack

**Files:**
- Modify: `cyclo_brain/policy/lerobot/lerobot_engine.py` (around lines 447–462; the `flat_state.size < expected` block).

The padding was added so 22-dim observations could feed a 36-dim ACT policy during smoke tests. It is **not** the right long-term fix — the missing 14 dims are EE pose and need either FK or a robot_config schema extension.

**Step 1: Locate the block**

```bash
grep -n "flat_state.size < expected" cyclo_brain/policy/lerobot/lerobot_engine.py
```
Expected: one match around line 458.

**Step 2: Add TODO comment + dim log**

Replace the block:

```python
        flat_state = np.concatenate(state_parts)
        # Some training datasets carry extra state dimensions ...
        try:
            expected = int(
                self._policy.config.input_features[_STATE_KEY].shape[0]
            )
        except Exception:
            expected = flat_state.size
        if flat_state.size < expected:
            pad = np.zeros(expected - flat_state.size, dtype=np.float32)
            flat_state = np.concatenate([flat_state, pad])
        batch[_STATE_KEY] = (
            torch.from_numpy(flat_state).unsqueeze(0).to(self._device)
        )
```

with:

```python
        flat_state = np.concatenate(state_parts)
        # TODO(ROBOTIS): replace zero-padding with real values. Some
        # training datasets carry extra state dimensions (e.g. EE pose
        # from forward-kinematics) that the robot_config's joint
        # topics don't surface. The right fix is one of:
        #   (a) compute EE pose via FK from the upper_body joint state,
        #   (b) extend ffw_sg2_rev1_config.yaml with an EE pose
        #       observation entry and a matching subscriber path,
        #   (c) retrain without the extra dims.
        # Until then, pad zeros so inference can at least run.
        try:
            expected = int(
                self._policy.config.input_features[_STATE_KEY].shape[0]
            )
        except Exception:
            expected = flat_state.size
        if flat_state.size < expected:
            pad = np.zeros(expected - flat_state.size, dtype=np.float32)
            logger.warning(
                "state dim mismatch: got %d, policy expects %d — "
                "padding %d zeros (see TODO in _build_observation)",
                flat_state.size,
                expected,
                expected - flat_state.size,
            )
            flat_state = np.concatenate([flat_state, pad])
        batch[_STATE_KEY] = (
            torch.from_numpy(flat_state).unsqueeze(0).to(self._device)
        )
```

**Step 3: Verify**

```bash
python3 -m py_compile cyclo_brain/policy/lerobot/lerobot_engine.py
grep -n "TODO(ROBOTIS)" cyclo_brain/policy/lerobot/lerobot_engine.py
```
Expected: one TODO match; compile exit 0.

**Step 4: Live verify — LOAD + check the warning fires**

```bash
cd /home/rc/workspace/cyclo_intelligence
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml restart lerobot
sleep 8
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -8
docker logs --since 60s lerobot_server 2>&1 | grep "state dim mismatch"
```
Expected: `/cmd_vel  →  average rate: 100.0xx` AND at least one `state dim mismatch: got 22, policy expects 36 — padding 14 zeros` log line.

**Step 5: Commit**

```bash
git add cyclo_brain/policy/lerobot/lerobot_engine.py
git commit -m "refactor(lerobot_engine): label state-padding as TODO + add dim log

The zero-pad in _build_observation was a smoke-test workaround for
the 36-dim ACT policy vs the 22-dim robot_config observation. Mark
it clearly as temporary and log the actual dimensions every time it
fires, so the next pass can decide between FK, schema extension, or
retraining without re-discovering the gap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: S4 — Document the two hidden contracts

**Files:**
- Modify: `cyclo_brain/policy/common/runtime/control_publisher.py` (top-of-file docstring + `_publish_trajectory_preview_locked` docstring).
- Modify: `cyclo_brain/policy/common/runtime/inference_server.py` (top-of-file docstring + `_cmd_update_instruction` docstring).
- Create: `docs/specs/policy-runtime-contracts.md` (single small spec).

Two contracts are currently implicit and at silent-break risk:
1. `/inference/trajectory_preview` topic — Process B publishes a JointTrajectory preview of each chunk for the UI 3D viz.
2. `CMD_UPDATE_INSTRUCTION=6` — InferenceCommand variant that updates `task_instruction` without re-loading weights.

**Step 1: Add docstring to `_publish_trajectory_preview_locked`**

Open `cyclo_brain/policy/common/runtime/control_publisher.py`, find `def _publish_trajectory_preview_locked` (around line 553). Extend the existing docstring:

```python
    def _publish_trajectory_preview_locked(self, chunk: np.ndarray) -> None:
        """Emit the full predicted action chunk as JointTrajectory.

        Publishes to ``/inference/trajectory_preview``
        (trajectory_msgs/msg/JointTrajectory). Consumer: the UI's 3D
        viewer in orchestrator. This is a UI-only auxiliary; the real
        100 Hz robot commands go out on per-modality topics built in
        ``_setup_robot_specific_locked``. ``chunk`` is (T, D) where D
        matches ``len(self._preview_joint_names)``. Caller holds
        ``_config_lock``.
        """
```

**Step 2: Add `CMD_UPDATE_INSTRUCTION` docstring**

In `cyclo_brain/policy/common/runtime/inference_server.py`, find `_cmd_update_instruction` (around line 354). Replace the existing docstring with:

```python
    def _cmd_update_instruction(self, request):
        """Update the current task_instruction without reloading weights.

        InferenceCommand variant ``CMD_UPDATE_INSTRUCTION=6``. Engines
        with language conditioning (smolvla, pi0, eo1, xvla, wallx)
        read the new instruction on the next ``get_action_chunk``
        call. Engines without language conditioning (act, diffusion,
        multi_task_dit) ignore it. The response carries no action_keys
        and no policy reload occurs.
        """
```

**Step 3: Write the spec file**

Create `docs/specs/policy-runtime-contracts.md`:

```markdown
# policy-runtime contracts

External (non-obvious) wire contracts honoured by
`cyclo_brain/policy/common/runtime/`. Update this file whenever you
add or change a topic/srv that someone outside Process A/B needs.

## Zenoh topics

| Topic | Direction | Producer | Type | Notes |
|---|---|---|---|---|
| `/lerobot/inference_command` (srv) | call | external ROS2 → Process A | `interfaces/srv/InferenceCommand` | LOAD/START/PAUSE/RESUME/STOP/UNLOAD/UPDATE_INSTRUCTION |
| `cyclo/policy/lerobot/configure` | pub→sub | A → B | `std_msgs/String` | robot_type broadcast on LOAD; empty string = deconfigure |
| `cyclo/policy/lerobot/lifecycle` | pub→sub | A → B | `std_msgs/String` | `unloaded` / `loaded` / `running` / `stopped` |
| `cyclo/policy/lerobot/run_inference` | pub→sub | B → A | `std_msgs/UInt64` | trigger seq_id |
| `cyclo/policy/lerobot/action_chunk_raw` | pub→sub | A → B | `interfaces/msg/ActionChunk` | flat float64 buffer + chunk_size + action_dim |
| `/inference/trajectory_preview` | pub | B → orchestrator UI | `trajectory_msgs/msg/JointTrajectory` | per-chunk preview for 3D viz |

The `cyclo/policy/<backend>/...` prefix substitutes `<backend>` =
`POLICY_BACKEND` env (default `lerobot`).

## InferenceCommand enum

| Value | Name | Notes |
|---|---|---|
| 0 | `CMD_LOAD` | Loads weights + brings up RobotClient + broadcasts configure |
| 1 | `CMD_START` | Wires up trigger sub + chunk pub, sets lifecycle to `running` |
| 2 | `CMD_PAUSE` | Stops honoring triggers (chunks already in buffer keep playing) |
| 3 | `CMD_RESUME` | Re-honors triggers; optional `task_instruction` field updates instruction |
| 4 | `CMD_STOP` | Stops honoring; clears buffer |
| 5 | `CMD_UNLOAD` | Drops weights + tears down RobotClient + broadcasts empty configure |
| 6 | `CMD_UPDATE_INSTRUCTION` | Updates `task_instruction` only; weights and lifecycle unchanged |

## Hard contracts (do not break without a coordinated PR)

- Topic names + types in the table above.
- `InferenceCommand` request fields: `command`, `model_path`,
  `embodiment_tag`, `robot_type`, `task_instruction`.
- `InferenceCommand` response fields: `success`, `message`, `action_keys`.
- Bind-mount paths into policy containers: `/policy_runtime`,
  `/app/<backend>_engine.py`, `/zenoh_sdk`, `/robot_client_sdk`,
  `/post_processing_sdk`, `/orchestrator_config`.
- s6 longrun names: `inference-server`, `control-publisher`, `user`.
- Engine entry point: `create_engine() -> InferenceEngine` (module-level
  function in `<backend>_engine.py`).
- `action_chunk` payload: flat numpy `float64` array of length
  `chunk_size * action_dim`.
```

**Step 4: Reference the spec from the runtime docstrings**

At the top of `control_publisher.py`, append to the module docstring:

```
See ``docs/specs/policy-runtime-contracts.md`` for the full topic /
srv / mount contract reference.
```

Same line at the top of `inference_server.py`.

**Step 5: Verify**

```bash
ls docs/specs/policy-runtime-contracts.md
python3 -m py_compile cyclo_brain/policy/common/runtime/control_publisher.py cyclo_brain/policy/common/runtime/inference_server.py
```
Expected: file exists, compile exit 0.

**Step 6: Commit**

```bash
git add docs/specs/policy-runtime-contracts.md cyclo_brain/policy/common/runtime/control_publisher.py cyclo_brain/policy/common/runtime/inference_server.py
git commit -m "docs: pin the policy-runtime wire contracts in a spec + docstrings

Surface two previously implicit contracts:
  - /inference/trajectory_preview (Process B → UI viz),
  - CMD_UPDATE_INSTRUCTION (re-condition without weight reload).
Plus a single spec file enumerating every topic / srv / mount the
runtime promises to outsiders.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: PR1 final gate + open PR

**Step 1: Full pipeline gate**

```bash
cd /home/rc/workspace/cyclo_intelligence
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml restart lerobot
sleep 8
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -22
```
Expected: `success=True` at every service stage, `/cmd_vel  →  average rate: 100.0xx`.

**Step 2: Push branch**

```bash
cd ../cyclo_intelligence-refactor
git log --oneline feature-update-lerobot..HEAD
git push -u origin feature-cyclo-brain-refactor
```
Expected: 4 commits printed.

**Step 3: Open PR1**

```bash
gh pr create --title "refactor(cyclo_brain): cleanup pass (S1-S4)" --body "$(cat <<'EOF'
## Summary
- Drop 12 CP-DBG stderr prints from control_publisher.py (left over from trigger-emission diagnosis)
- Update cyclo_brain/README.md to reflect common/runtime + common/s6-services layout
- Label the state-padding hack in lerobot_engine.py with a TODO + a warning log every time it fires
- Pin two previously implicit wire contracts (/inference/trajectory_preview, CMD_UPDATE_INSTRUCTION) in a spec doc + docstrings

Zero runtime behavior change. Verified with verify_inference_pipeline.sh — /cmd_vel keeps 100 Hz.

## Test plan
- [x] docker exec cyclo_intelligence bash docs/scripts/verify_inference_pipeline.sh — LOAD/START/STOP/UNLOAD pass, /cmd_vel 100 Hz
- [x] grep -c CP-DBG cyclo_brain/policy/common/runtime/control_publisher.py == 0
- [x] docker logs lerobot_server includes the new 'state dim mismatch' warning during inference

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL printed.

---

# PR2 — S5: Split `control_publisher.py` into mixins

Target: `policy/common/runtime/` — directory mount, no compose change.

## Task 6: Add `_cp_lifecycle.py` mixin

**Files:**
- Create: `cyclo_brain/policy/common/runtime/_cp_lifecycle.py`

Methods to move out of `control_publisher.py` (line numbers from current head; will shift as you go):
- `configure` (277-327)
- `deconfigure` (329-347)
- `_setup_robot_specific_locked` (349-392)
- `_teardown_robot_specific_locked` (394-414)
- `_on_configure` (416-432)
- `_on_lifecycle` (434-459)

**Step 1: Create `_cp_lifecycle.py` with all six methods + necessary imports**

The mixin is purely a vertical slice — `self.*` access only, no method calls into other mixins by name beyond what control_publisher.py already does.

Header for the new file:

```python
#!/usr/bin/env python3
"""LifecycleMixin — configure / deconfigure / per-robot setup + teardown.

Pulled out of control_publisher.py (S5 in
docs/plans/2026-05-10-cyclo_brain-refactor.md). Method bodies are
identical to the originals — only the class wrapper is new. Caller
holds ``_config_lock`` for the locked helpers, same as before.
"""
from __future__ import annotations

import logging
from typing import Optional

from post_processing.action_chunk_processor import ActionChunkProcessor
from robot_client import robot_schema

from zenoh_ros2_sdk import (  # noqa: F401  (consistency w/ control_publisher imports)
    ROS2Publisher,
    ROS2Subscriber,
    get_logger,
)

from .control_publisher_helpers import build_action_joint_map  # if needed; see step 2

logger = get_logger("control_publisher")


class LifecycleMixin:
    def configure(self, robot_type: str) -> None:
        ...
    def deconfigure(self) -> None:
        ...
    def _setup_robot_specific_locked(self) -> None:
        ...
    def _teardown_robot_specific_locked(self) -> None:
        ...
    def _on_configure(self, msg) -> None:
        ...
    def _on_lifecycle(self, msg) -> None:
        ...
```

Copy the **bodies** of these six methods verbatim from `control_publisher.py`. Keep all `self.*` references unchanged.

**Step 2: Resolve imports**

Compare imports at the top of `control_publisher.py` with what the six method bodies actually use. Copy the ones you need into `_cp_lifecycle.py`. Notable likely-needed:
- `INFERENCE_HZ`, `CONTROL_HZ`, `CHUNK_ALIGN_WINDOW_S` (currently constants at module top of `control_publisher.py`). Import them via `from .control_publisher import INFERENCE_HZ, CONTROL_HZ, CHUNK_ALIGN_WINDOW_S` is **NOT** ok during the migration (creates a cycle when `control_publisher` imports the mixin). Instead move the three constants into a new file `_cp_constants.py` first, or duplicate them at the top of `_cp_lifecycle.py` with a clear comment. Pick whichever is smaller — duplication is fine for three int constants.
- `build_action_joint_map` is defined in `control_publisher.py` itself. Move it to `_cp_lifecycle.py` if `configure()` is its only caller, or to a tiny shared helper module. Inventory shows it's only called from `configure`, so move it into `_cp_lifecycle.py`.

**Step 3: Sanity-compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/_cp_lifecycle.py
```
Expected: exit 0. If errors: missing import → resolve.

**Step 4: Commit (intermediate, keeps PR2 reviewable)**

```bash
git add cyclo_brain/policy/common/runtime/_cp_lifecycle.py
git commit -m "refactor(control_publisher): introduce LifecycleMixin

Step 1/4 of the control_publisher split. The mixin is not yet wired
into ControlPublisher — that happens in the final commit so the
intermediate state remains compilable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add `_cp_pipeline.py` mixin

**Files:**
- Create: `cyclo_brain/policy/common/runtime/_cp_pipeline.py`

Methods to move:
- `_on_chunk` (505-530)
- `_send_trigger_locked` (532-543)

Plus the `REQUEST_TIMEOUT_S` constant referenced in `_tick` (stays in `control_publisher.py`) but also in `_send_trigger_locked` — leave it in `control_publisher.py` and import it.

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""PipelineMixin — chunk reception + trigger emission."""
from __future__ import annotations

import time

import numpy as np

from zenoh_ros2_sdk import get_logger

logger = get_logger("control_publisher")


class PipelineMixin:
    def _on_chunk(self, msg) -> None:
        ...
    def _send_trigger_locked(self) -> None:
        ...
```

Body verbatim from `control_publisher.py`.

**Step 2: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/_cp_pipeline.py
```
Expected: exit 0.

**Step 3: Commit**

```bash
git add cyclo_brain/policy/common/runtime/_cp_pipeline.py
git commit -m "refactor(control_publisher): introduce PipelineMixin

Step 2/4 of the split. _on_chunk + _send_trigger_locked extracted
verbatim. Not yet wired in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Add `_cp_publishers.py` mixin

**Files:**
- Create: `cyclo_brain/policy/common/runtime/_cp_publishers.py`

Methods to move:
- `_publish_trajectory_preview_locked` (553-592)
- `_publish_action_locked` (594-620)
- `_publish_twist` (622-634)
- `_publish_joint_trajectory` (636-649)

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""PublishersMixin — emit joint trajectories / twist / preview."""
from __future__ import annotations

import numpy as np

from zenoh_ros2_sdk import get_logger

logger = get_logger("control_publisher")


class PublishersMixin:
    def _publish_action_locked(self, action: np.ndarray) -> None:
        ...
    def _publish_twist(self, pub, values: np.ndarray) -> None:
        ...
    def _publish_joint_trajectory(self, ...) -> None:
        ...
    def _publish_trajectory_preview_locked(self, chunk: np.ndarray) -> None:
        ...
```

Body verbatim.

**Step 2: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/_cp_publishers.py
```
Expected: exit 0.

**Step 3: Commit**

```bash
git add cyclo_brain/policy/common/runtime/_cp_publishers.py
git commit -m "refactor(control_publisher): introduce PublishersMixin

Step 3/4 of the split. Four publish_* methods extracted verbatim.
Not yet wired in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire mixins into `ControlPublisher` + delete originals

**Files:**
- Modify: `cyclo_brain/policy/common/runtime/control_publisher.py`

**Step 1: Add the three mixins to the class declaration**

Change:

```python
class ControlPublisher:
```

to:

```python
from ._cp_lifecycle import LifecycleMixin
from ._cp_pipeline import PipelineMixin
from ._cp_publishers import PublishersMixin


class ControlPublisher(LifecycleMixin, PipelineMixin, PublishersMixin):
```

Place the imports near the other `from .` and `from zenoh_ros2_sdk` imports at the top of the file.

**Step 2: Delete the original method bodies**

Delete the in-place definitions of all 10 methods moved in Tasks 6–8 (the bodies inside `ControlPublisher`). Leave the class skeleton with only the methods that stay:
- `__init__`
- `_common_kwargs`
- `setup`
- `shutdown`
- `run`
- `_tick`

Plus the module-level `_try_rt_priority` and `main` stay.

If `build_action_joint_map` was a free function in `control_publisher.py` and you moved it to `_cp_lifecycle.py` in Task 6, delete it here.

**Step 3: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/control_publisher.py
```
Expected: exit 0.

**Step 4: Line-count check (regression sanity)**

```bash
wc -l cyclo_brain/policy/common/runtime/_cp_lifecycle.py cyclo_brain/policy/common/runtime/_cp_pipeline.py cyclo_brain/policy/common/runtime/_cp_publishers.py cyclo_brain/policy/common/runtime/control_publisher.py
```
Expected: the four numbers sum to ~700 (original 683 + a few lines of new mixin class headers + import additions).

**Step 5: Live gate**

```bash
cd /home/rc/workspace/cyclo_intelligence
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml restart lerobot
sleep 8
ss -tulpn 2>/dev/null | grep 7447 || docker exec -d cyclo_intelligence bash -c 'source /opt/ros/jazzy/setup.bash && exec /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd > /tmp/zenohd.log 2>&1'
sleep 3
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -22
```
Expected: `success=True` at LOAD/START/STOP/UNLOAD; `/cmd_vel  →  average rate: 100.0xx`.

**Step 6: Commit**

```bash
git add cyclo_brain/policy/common/runtime/control_publisher.py
git commit -m "refactor(control_publisher): wire mixins, delete moved method bodies

Step 4/4 of the split. ControlPublisher now inherits from
LifecycleMixin, PipelineMixin, PublishersMixin. External signatures
unchanged; verify_inference_pipeline.sh keeps /cmd_vel at 100 Hz.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**Step 7: Push + open PR2**

```bash
git push
gh pr create --title "refactor(control_publisher): split into LifecycleMixin / PipelineMixin / PublishersMixin (S5)" --body "$(cat <<'EOF'
## Summary
Splits cyclo_brain/policy/common/runtime/control_publisher.py (683 lines) into four files via mixin inheritance:
- control_publisher.py — ControlPublisher core (__init__, setup, shutdown, run, _tick) + main()
- _cp_lifecycle.py — LifecycleMixin (configure, deconfigure, setup/teardown_robot_specific_locked, _on_configure, _on_lifecycle)
- _cp_pipeline.py — PipelineMixin (_on_chunk, _send_trigger_locked)
- _cp_publishers.py — PublishersMixin (_publish_action_locked, _publish_twist, _publish_joint_trajectory, _publish_trajectory_preview_locked)

External signatures unchanged. Directory bind-mount (/policy_runtime) auto-picks up the new files; no docker-compose change.

## Test plan
- [x] docker exec cyclo_intelligence bash docs/scripts/verify_inference_pipeline.sh — LOAD/START/STOP/UNLOAD pass, /cmd_vel 100 Hz
- [x] python3 -m py_compile on all four files exits 0
- [x] wc -l sums to ~700 (original 683 + mixin headers)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR3 — S6: Split `inference_server.py` into mixins

Target: `policy/common/runtime/` — no compose change.

## Task 10: Add `_is_commands.py` mixin

**Files:**
- Create: `cyclo_brain/policy/common/runtime/_is_commands.py`

Methods to move from `inference_server.py`:
- `_handle_command` (254-276)
- `_cmd_load` (278-311)
- `_cmd_start` (313-320)
- `_cmd_pause` (322-328)
- `_cmd_resume` (330-338)
- `_cmd_stop` (340-345)
- `_cmd_unload` (347-352)
- `_cmd_update_instruction` (354-373) — note that the docstring was already updated in PR1, keep it
- `_make_response` (470-485) — used by every `_cmd_*` so it moves with them.

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""CommandsMixin — InferenceCommand dispatch (LOAD / START / ... / UPDATE_INSTRUCTION)."""
from __future__ import annotations

from zenoh_ros2_sdk import get_logger

logger = get_logger("inference_server")

CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6


class CommandsMixin:
    def _handle_command(self, request):
        ...
    def _cmd_load(self, request):
        ...
    def _cmd_start(self):
        ...
    def _cmd_pause(self):
        ...
    def _cmd_resume(self, request):
        ...
    def _cmd_stop(self):
        ...
    def _cmd_unload(self):
        ...
    def _cmd_update_instruction(self, request):
        ...
    def _make_response(self, ...):
        ...
```

Body verbatim.

**Step 2: Decide where the CMD_* constants live**

Originally at the top of `inference_server.py`. Move them to `_is_commands.py` (above). Then in `inference_server.py` re-import for downstream readers:

```python
from ._is_commands import (
    CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME,
    CMD_STOP, CMD_UNLOAD, CMD_UPDATE_INSTRUCTION,
)
```

**Step 3: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/_is_commands.py
```
Expected: exit 0.

**Step 4: Commit**

```bash
git add cyclo_brain/policy/common/runtime/_is_commands.py
git commit -m "refactor(inference_server): introduce CommandsMixin

Step 1/3 of the split. Eight _cmd_* handlers + _handle_command +
_make_response + CMD_* constants extracted verbatim. Not yet wired in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Add `_is_zenoh.py` mixin

**Files:**
- Create: `cyclo_brain/policy/common/runtime/_is_zenoh.py`

Methods to move:
- `_publish_configure` (233-241)
- `_publish_lifecycle` (243-252)
- `_setup_zenoh_io` (375-390)
- `_on_trigger` (392-413)
- `_publish_chunk` (415-434)
- `_teardown_runtime` (436-468)

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""ZenohIOMixin — configure/lifecycle pub, trigger sub, chunk pub, teardown."""
from __future__ import annotations

from zenoh_ros2_sdk import (
    ROS2Publisher,
    ROS2Subscriber,
    get_logger,
)

logger = get_logger("inference_server")


class ZenohIOMixin:
    def _publish_configure(self, robot_type: str) -> None:
        ...
    def _publish_lifecycle(self, state: str) -> None:
        ...
    def _setup_zenoh_io(self) -> None:
        ...
    def _on_trigger(self, msg) -> None:
        ...
    def _publish_chunk(self, seq_id: int, result: dict) -> None:
        ...
    def _teardown_runtime(self) -> None:
        ...
```

Body verbatim.

**Step 2: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/_is_zenoh.py
```
Expected: exit 0.

**Step 3: Commit**

```bash
git add cyclo_brain/policy/common/runtime/_is_zenoh.py
git commit -m "refactor(inference_server): introduce ZenohIOMixin

Step 2/3 of the split. Six Zenoh-side I/O methods extracted verbatim.
Not yet wired in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Wire mixins into `InferenceServer` + delete originals

**Files:**
- Modify: `cyclo_brain/policy/common/runtime/inference_server.py`

**Step 1: Inherit the two mixins**

```python
from ._is_commands import (
    CommandsMixin,
    CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME,
    CMD_STOP, CMD_UNLOAD, CMD_UPDATE_INSTRUCTION,
)
from ._is_zenoh import ZenohIOMixin


class InferenceServer(CommandsMixin, ZenohIOMixin):
```

**Step 2: Delete original method bodies**

Delete the in-place definitions of all 15 methods moved in Tasks 10 + 11. Methods remaining inside `InferenceServer`:
- `__init__`
- `_common_kwargs`
- `start_service`
- `shutdown`

Plus module-level `_resolve_engine` and `main` stay. Also remove the duplicate CMD_* constants (now imported from `_is_commands`).

**Step 3: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/common/runtime/inference_server.py
```
Expected: exit 0.

**Step 4: Live gate**

```bash
cd /home/rc/workspace/cyclo_intelligence
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml restart lerobot
sleep 8
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -22
```
Expected: full success, `/cmd_vel  →  average rate: 100.0xx`.

**Step 5: Commit + push + open PR3**

```bash
git add cyclo_brain/policy/common/runtime/inference_server.py
git commit -m "refactor(inference_server): wire mixins, delete moved method bodies

Step 3/3 of the split. InferenceServer inherits CommandsMixin +
ZenohIOMixin. External signatures unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push
gh pr create --title "refactor(inference_server): split into CommandsMixin / ZenohIOMixin (S6)" --body "$(cat <<'EOF'
## Summary
Splits cyclo_brain/policy/common/runtime/inference_server.py (509 lines) into three files via mixin inheritance:
- inference_server.py — InferenceServer core (__init__, start_service, shutdown) + _resolve_engine + main()
- _is_commands.py — CommandsMixin (_handle_command, _cmd_load/start/pause/resume/stop/unload/update_instruction, _make_response) + CMD_* constants
- _is_zenoh.py — ZenohIOMixin (_publish_configure/_publish_lifecycle/_setup_zenoh_io/_on_trigger/_publish_chunk/_teardown_runtime)

External signatures unchanged. Directory bind-mount (/policy_runtime) auto-picks up new files.

## Test plan
- [x] docker exec cyclo_intelligence bash docs/scripts/verify_inference_pipeline.sh — LOAD/START/STOP/UNLOAD pass, /cmd_vel 100 Hz
- [x] python3 -m py_compile on all three files exits 0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR4 — S7: Split `lerobot_engine.py` + extend compose mount

Target: `policy/lerobot/` — single-file mount → 4-file mount. **Highest-risk PR**, do this last.

## Task 13: Add `_le_loading.py` mixin

**Files:**
- Create: `cyclo_brain/policy/lerobot/_le_loading.py`

Methods to move from `lerobot_engine.py`:
- `_resolve_model_dir` (215-227, static)
- `_load_policy_assets` (230-283, static, includes the HF Hub policy_type fix from this session — keep verbatim)
- `_infer_image_resize` (285-308)

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""LoadingMixin — model_path resolution + policy weights + processors."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from lerobot.policies import PreTrainedPolicy, get_policy_class, make_pre_post_processors

logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        ...
    @staticmethod
    def _load_policy_assets(model_path: str, device) -> tuple:
        ...
    def _infer_image_resize(self, policy) -> Optional[tuple]:
        ...
```

Body verbatim. Note: `_load_policy_assets` keeps the HF Hub `huggingface_hub.hf_hub_download(filename="config.json")` shim added in this session.

**Step 2: Compile + commit**

```bash
python3 -m py_compile cyclo_brain/policy/lerobot/_le_loading.py
git add cyclo_brain/policy/lerobot/_le_loading.py
git commit -m "refactor(lerobot_engine): introduce LoadingMixin

Step 1/4 of the lerobot_engine split. Model path resolution + asset
loading (including the HF Hub policy_type detection shim) extracted
verbatim. Not yet wired in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Add `_le_io_mapping.py` mixin

**Files:**
- Create: `cyclo_brain/policy/lerobot/_le_io_mapping.py`

Methods to move:
- `_init_robot` (310-372) — owns camera + state modality + action_keys mapping
- `_teardown_robot` (374-380)
- `_policy_image_keys` (382-387)

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""IoMappingMixin — RobotClient bring-up + cam/state mapping."""
from __future__ import annotations

from typing import Dict, List
import logging

from robot_client import RobotClient

logger = logging.getLogger("lerobot_engine")


class IoMappingMixin:
    def _init_robot(self, robot_type: str) -> None:
        ...
    def _teardown_robot(self) -> None:
        ...
    def _policy_image_keys(self) -> set:
        ...
```

Body verbatim. Note: `_init_robot` is where the camera-key mismatch errors get raised — preserve the exact error messages so logs stay greppable.

**Step 2: Compile + commit**

```bash
python3 -m py_compile cyclo_brain/policy/lerobot/_le_io_mapping.py
git add cyclo_brain/policy/lerobot/_le_io_mapping.py
git commit -m "refactor(lerobot_engine): introduce IoMappingMixin

Step 2/4 of the split. _init_robot + _teardown_robot + _policy_image_keys
extracted verbatim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Add `_le_observation.py` mixin

**Files:**
- Create: `cyclo_brain/policy/lerobot/_le_observation.py`

Methods to move:
- `_build_observation` (393-457) — keep the PR1 state-padding TODO/log block
- `_predict_chunk` (478-502)
- `_to_numpy_chunk` (504-519, static)

**Step 1: Write the mixin**

```python
#!/usr/bin/env python3
"""ObservationMixin — sensor reads → policy batch → chunk tensor."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger("lerobot_engine")

_IMAGE_KEY_PREFIX = "observation.images."
_STATE_KEY = "observation.state"


class ObservationMixin:
    def _build_observation(self, task_instruction: str) -> Dict[str, Any]:
        ...
    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @staticmethod
    def _to_numpy_chunk(action: torch.Tensor) -> np.ndarray:
        ...
```

Body verbatim. The `_IMAGE_KEY_PREFIX` / `_STATE_KEY` constants need to be visible to both this mixin and `_le_io_mapping.py` (used in `_policy_image_keys`). Either duplicate them in both files (cheap, two lines each) or move to a tiny `_le_constants.py`. Duplication is simpler — keep it.

**Step 2: Compile + commit**

```bash
python3 -m py_compile cyclo_brain/policy/lerobot/_le_observation.py
git add cyclo_brain/policy/lerobot/_le_observation.py
git commit -m "refactor(lerobot_engine): introduce ObservationMixin

Step 3/4 of the split. _build_observation (incl. state-padding TODO
block) + _predict_chunk + _to_numpy_chunk extracted verbatim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Wire mixins into `LeRobotEngine` + delete originals

**Files:**
- Modify: `cyclo_brain/policy/lerobot/lerobot_engine.py`

**Step 1: Inherit**

```python
from _le_loading import LoadingMixin
from _le_io_mapping import IoMappingMixin
from _le_observation import ObservationMixin

class LeRobotEngine(LoadingMixin, IoMappingMixin, ObservationMixin, InferenceEngine):
```

The container's `/app/` has no `__init__.py` (it's a flat directory of bind-mounted files), so the import path is bare `from _le_loading import ...` rather than dotted. Verify this matches how `lerobot_engine.py` is currently imported by `inference_server.py` (`importlib.import_module("lerobot_engine")` — yes, bare).

**Step 2: Delete the moved method bodies**

Remaining inside `LeRobotEngine`:
- `__init__`
- `is_ready` (property)
- `load_policy`
- `get_action_chunk`
- `cleanup`
- `_fail` (static)

Plus module-level `create_engine()` stays.

**Step 3: Compile**

```bash
python3 -m py_compile cyclo_brain/policy/lerobot/lerobot_engine.py cyclo_brain/policy/lerobot/_le_loading.py cyclo_brain/policy/lerobot/_le_io_mapping.py cyclo_brain/policy/lerobot/_le_observation.py
```
Expected: all exit 0.

**Step 4: Commit (still untested in container — compose mount missing)**

```bash
git add cyclo_brain/policy/lerobot/lerobot_engine.py
git commit -m "refactor(lerobot_engine): wire LoadingMixin / IoMappingMixin / ObservationMixin

Step 4/4 of the lerobot_engine split. LeRobotEngine now inherits the
three mixins; the four files together preserve every original method.
docker-compose mount still references only lerobot_engine.py — the
helpers are bind-mounted in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Extend docker-compose volumes for the three helpers

**Files:**
- Modify: `docker/docker-compose.yml`

**Step 1: Locate the current single-file mount**

```bash
grep -n "lerobot_engine.py" docker/docker-compose.yml
```
Expected: one line, under the `lerobot` service `volumes:` list.

**Step 2: Add three new mount lines**

After the existing line:

```yaml
      - ../cyclo_brain/policy/lerobot/lerobot_engine.py:/app/lerobot_engine.py:ro
```

add:

```yaml
      - ../cyclo_brain/policy/lerobot/_le_loading.py:/app/_le_loading.py:ro
      - ../cyclo_brain/policy/lerobot/_le_io_mapping.py:/app/_le_io_mapping.py:ro
      - ../cyclo_brain/policy/lerobot/_le_observation.py:/app/_le_observation.py:ro
```

Keep indentation identical to neighboring lines (4 spaces under `volumes:` per YAML structure already in the file).

**Step 3: Verify YAML parses**

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml config lerobot 2>&1 | grep -A1 _le_
```
Expected: three mount entries echoed.

**Step 4: Recreate the container with new mounts**

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.override.yml up -d --force-recreate lerobot
sleep 8
ss -tulpn 2>/dev/null | grep 7447 || docker exec -d cyclo_intelligence bash -c 'source /opt/ros/jazzy/setup.bash && exec /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd > /tmp/zenohd.log 2>&1'
sleep 3
docker exec lerobot_server ls /app/
```
Expected: `_le_loading.py`, `_le_io_mapping.py`, `_le_observation.py`, `lerobot_engine.py` all present.

**Step 5: Live gate**

```bash
docker exec cyclo_intelligence bash /tmp/verify_clean.sh 2>&1 | tail -22
```
Expected: `success=True` everywhere, `/cmd_vel  →  average rate: 100.0xx`, `action_keys=['arm_left', 'arm_right', 'head', 'lift', 'mobile']`.

**Step 6: Commit + push + open PR4**

```bash
git add docker/docker-compose.yml
git commit -m "build(compose): bind-mount lerobot_engine helper files into /app

Extends the lerobot service volumes to also mount _le_loading.py,
_le_io_mapping.py, _le_observation.py alongside lerobot_engine.py.
Without these the previous commit's mixin inheritance would crash on
ImportError at container start. Verified end-to-end with
verify_inference_pipeline.sh.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push
gh pr create --title "refactor(lerobot_engine): split into LoadingMixin / IoMappingMixin / ObservationMixin (S7)" --body "$(cat <<'EOF'
## Summary
Splits cyclo_brain/policy/lerobot/lerobot_engine.py (531 lines) into four files via mixin inheritance, and extends docker-compose volumes to bind-mount the three new helpers:
- lerobot_engine.py — LeRobotEngine core (__init__, is_ready, load_policy, get_action_chunk, cleanup) + create_engine()
- _le_loading.py — LoadingMixin (_resolve_model_dir, _load_policy_assets incl. HF Hub policy_type shim, _infer_image_resize)
- _le_io_mapping.py — IoMappingMixin (_init_robot, _teardown_robot, _policy_image_keys)
- _le_observation.py — ObservationMixin (_build_observation, _predict_chunk, _to_numpy_chunk)
- docker/docker-compose.yml — adds 3 mount lines under lerobot.volumes

External signatures unchanged. /app/ contents listed inside container now include the three helpers.

## Test plan
- [x] docker compose ... config lerobot | grep _le_ shows three new mounts
- [x] docker exec lerobot_server ls /app shows _le_*.py + lerobot_engine.py
- [x] docker exec cyclo_intelligence bash docs/scripts/verify_inference_pipeline.sh — LOAD/START/STOP/UNLOAD pass, /cmd_vel 100 Hz, action_keys=['arm_left',...]
- [x] python3 -m py_compile on all four engine files exits 0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# After all PRs merge

## Task 18: Remove the worktree

```bash
cd /home/rc/workspace/cyclo_intelligence
git worktree remove ../cyclo_intelligence-refactor
git branch -d feature-cyclo-brain-refactor
```

The persisted `docs/scripts/verify_inference_pipeline.sh` + `docs/scripts/fake_robot_publisher.py` stay in the main branch for future regressions.

---

## Notes for the engineer executing this plan

1. **TDD doesn't quite apply here.** This is a behavior-preserving refactor of working code. The regression gate is `verify_inference_pipeline.sh` after each PR, not new unit tests. Don't invent tests just to satisfy TDD habit.

2. **Each "extract a mixin" task is a separate commit but a single intermediate state** — the original file still has the method bodies until the wire-up commit. That's deliberate so a reviewer can read just the wire-up commit and see exactly what changed.

3. **Mixin MRO.** All four mixins use only `self.*` attributes; none of them call methods defined in another mixin by name (they share `self` state but no cross-method calls). Multiple-inheritance order therefore doesn't matter. Put them in the order shown for readability only.

4. **Don't merge PR4 first.** It's the highest-risk one (compose volume change + container recreate). Land PR1 (lowest risk, smallest surface), then PR2/PR3 in either order, then PR4.

5. **If a gate fails after a commit:** `git reset --soft HEAD~1`, fix the issue, re-commit. Do not advance to the next PR until the current PR's gate is green end-to-end.

6. **For S7 specifically:** the container must be `up -d --force-recreate` (not just `restart`) so the new bind-mounts attach. A plain restart keeps the old volume set.

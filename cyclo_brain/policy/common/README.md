# Common Policy Runtime

This directory contains the two framework-independent halves of Cyclo policy
inference. They are deployed in different images.

```text
cyclo_intelligence image                  Worker image
┌──────────────────────────────┐          ┌──────────────────────────┐
│ launch-managed runtime       │  Zenoh   │ engine-process s6 service│
│                              │          │                          │
│ ServiceHandler               │ request  │ EngineWorker             │
│ SessionState                 │─────────>│ <backend>_engine          │
│ WorkerRegistry               │<─────────│ model + observations     │
│ ControlLoop                  │ actions  │                          │
│ ActionChunkProcessor         │          └──────────────────────────┘
│ RobotClient command path     │
└──────────────────────────────┘
```

## Directory Ownership

- `catalog/`: strict loader for `policy/<runtime>/manifest.yaml`.
- `runtime/main_runtime/`: central Policy Runtime. The package name is kept for
  compatibility; it is not installed as a Worker `main-runtime` service.
- `runtime/engine_process/`: framework-neutral Worker service and Engine wire
  protocol.
- `runtime/engine.py`: `InferenceEngine` contract implemented by each adapter.
- `s6-services/engine-process/`: the only application longrun copied into a
  Worker image.

## InferenceEngine Contract

```python
from engine import InferenceEngine

class MyEngine(InferenceEngine):
    def load_policy(self, request): ...
    def get_action_chunk(self, request): ...
    def cleanup(self): ...

    @property
    def is_ready(self): ...

def create_engine() -> InferenceEngine:
    return MyEngine()
```

The adapter subscribes to model observations through `RobotClient` and returns
one `(T, D)` action chunk. It must not publish robot commands.

## Runtime Contracts

### Action Steps Availability

Manifests declare `runtime.capabilities.execution_mode: chunk` or `step`.
A model may override it with `models[].execution_mode`. The catalog exposes the
resolved mode per model without importing model frameworks. An absent declaration
is unknown, not implicitly chunk-compatible.

Only chunk policies allow a numeric Action Steps selection. Step policies own
their action queue and use All (`0`). Orchestrator canonicalizes stale numeric
settings for step policies to `0` and publishes the shared setting to all UIs.
Runtime rejects unsupported numeric LOAD requests before loading weights and
checks the actual Worker contract against the declaration after LOAD, rolling
back on mismatch. Adapter regression tests must keep declarations and contracts
in agreement; never enable numeric selection by adding UI model-name exceptions.

After changing declarations, restart Supervisor API, Orchestrator and Policy
Runtime so their process-local catalogs agree, and refresh the UI. No ROS field
changes are required for this metadata.

External callers use `interfaces/srv/InferenceCommand` at
`/policy/inference_command`. The Runtime chooses a Worker from the namespaced
`policy_id` and calls `interfaces/srv/EngineCommand` at
`/<runtime>/engine_command`.

`EngineCommand` supports:

- `DESCRIBE`: protocol version, runtime ID, instance ID, policies, capabilities.
- `LOAD`: checkpoint and validated policy parameters.
- `GET_ACTION`: one action chunk.
- `UNLOAD`: release model resources.
- `STATUS`: current Worker metadata and engine state.

Worker heartbeats are published at `/<runtime>/worker_heartbeat`. During an
active session, a stale heartbeat, Worker instance change, Orchestrator
heartbeat loss, or GET_ACTION failure causes the Runtime to clear buffered
actions and enter `error`. In robot mode it also attempts zero Twist and a
fresh-current-joint hold, provided Cyclo still owns physical output authority.

The Supervisor uses `/run/cyclo/policy-runtime.sock` to reserve Worker
start/stop/recreate operations. A Worker used by a loaded session, or one with
a pending hold, cannot be changed through the Supervisor API.

### Teleoperation Handoff

SG2, BG2, W1 and W2 profiles opt in through `inference_handoff` in the robot YAML.
Profiles declare the source/status topics and allowed control modes; timing
parameters use the common defaults unless explicitly overridden.
An explicit physical Start/Resume can select Cyclo's internal `local_model`
state when no external control publisher or state has been observed by the
loaded RobotClient. This does not send a controller mode service request or
publish synthetic `/source` or `/control/status` messages. Fresh, valid configured
JointState feedback is still required; loading or polling never selects local mode.

Once any source/status publisher or state is observed, the client uses external
handoff for the remainder of its lifetime. Discovery revokes local publication
before waiting for a state event. Publisher loss, errors, and a known teleop mode
cannot be overridden by Start or silently fall back to local mode. The controller's
actual mode is not changed by Cyclo's internal execution state.

External handoff requires `/source` and configured JointState feedback within
0.5 seconds and allows absolute-joint model mode 1. `/control/status` is event-driven: Cyclo
rechecks its latest transient-local cached state without issuing controller
commands. A confirmation must belong to the current source epoch and controller
publisher. Old replies must not overwrite a newer error or source transition.
Other robots and preview-only execution do not participate in automatic handoff.

`status_confirmation_interval_s` defaults to 0.1 seconds and
`status_confirmation_timeout_s` to 0.5 seconds. The latter limits time since a
successful cache confirmation, not time since a controller status event. The
interval must be less than one third of the confirmation timeout to accommodate
both bounded queries and the next poll. Source/status publishers must be unique
and belong to the same discovered node/session.

Physical LOAD waits up to `initial_confirmation_timeout_s` (default 2 seconds) for the new client's
first validated control snapshot. This avoids an immediate START racing initial
publisher discovery after Clear. Timeout does not fail LOAD or grant authority:
START still checks the current source, feedback and confirmation before arming.
Preview LOAD does not wait, and confirmed teleop still rejects START.
This discovery budget is independent of the 0.5-second ongoing confirmation
freshness limit; extending startup discovery does not extend output authority.

Isolated native ROS/Zenoh tests cover single-event stable status, late joining,
activation-before-source ordering, publisher restart and command gating for
SG2/BG2 profiles. SDK profile tests also cover W1/W2 subscriptions, teleop output
blocking and rejection of old publication leases. Actual leader-button simulation and physical robot validation remain
separate deployment checks.

While a previously started physical session is active, `teleop` or `switching`
revokes its publication lease, clears buffered actions and pauses inference
without publishing hold/zero commands. A ready `model_action` source can resume
that same loaded session, including AI Worker's intentional leader-loss fallback.
This does not distinguish a button press from that fallback. UI draft paths and
instructions are not used for automatic resume; merely loading a model never
starts it. UI Pause/Stop, Unload, faults and shutdown cancel automatic resume.
Missing/stale feedback, lost status confirmation, publisher replacement or an
unsupported control mode requires explicit Start after recovery. Paused status
messages explain the reason.

Resume waits for any old prediction to finish. Updated Workers reset their
execution queues/history and require new observations received on the Worker
after accepting the reset. Old-generation results are discarded. Completed slow
start is retained; an interrupted slow start is restarted. These transitions do
not start or stop recording.

Deploy the Runtime, RobotClient and selected Worker adapters together and restart
their processes after stopping/clearing inference. An older Worker without reset
context support is rejected at physical Start. Handoff itself requires no new
ROS message fields; recording receipt recovery requires rebuilding the
`RecordingCommand` interface as described in the recording documentation.
This change does not implement intervention labeling or DAgger training.

Publication gating cannot recall commands already sent or follower trajectories
already scheduled. Bounded cache reconciliation cannot reconstruct every missed
intermediate controller event or provide controller-side atomic arbitration.
Validate each robot's source switching in simulation, including switching during slow
start, before physical use.

### Worker Status Snapshots

Supervisor's `worker_status` request reads a shared, nonblocking snapshot from
WorkerRegistry. Network `DESCRIBE` probes run on a separate status client, at most
one probe loop per runtime, approximately once per second while readers remain
interested. Ten seconds without a read stops that loop. A slow or disconnected
Worker must not occupy the serial Unix lifecycle handler or block another Worker.
SDK client construction also occurs outside the heartbeat/status lock; concurrent
requests for the same client share one construction.

Descriptor snapshots expire after three seconds. Reading a snapshot does not
extend its lifetime. Heartbeat age and engine state are read from the latest
matching Worker instance; a new or unidentified instance cannot reuse old
compatibility results. Expired/unavailable snapshots report waiting, and a probe's
compatibility failure replaces any previous ready result.

These snapshots are UI diagnostics, not authority to execute or mutate a Worker.
LOAD still performs a fresh descriptor handshake. The active-session heartbeat
watchdog and mutation reservation/hold checks retain their independent paths.
No frontend polling request directly triggers a second simultaneous Worker probe.

## Environment

The common `cyclo_intelligence` ROS launch starts Runtime first and starts its
ROS clients only after the ready marker identifies the new Runtime PID.
Runtime and Orchestrator remain separate processes but inherit the same launch
environment. The s6-managed common launch and Worker services use interactive
bash; manual launch inherits the current shell's environment:

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
```

After editing `/root/.bashrc`, use a new shell (or source it) and restart the
**common launch**, not only `orchestrator_node`. The component-only
`orchestrator` and `cyclo_data` launch commands do not start Policy Runtime.
All policy Workers, including ABot, LingBot-VLA, and RLDX, also read
`/root/.bashrc` when their engine service starts. Configure ROS/Zenoh there,
not in a second Compose environment override. Restart the Worker after editing
its configuration, once inference is stopped and cleared. A container restart
preserves the edits; recreating the container restores the image defaults.
Model asset paths such as `ABOT_BASE_VLM` and `QWEN3VL_PATH` remain separate
from the ROS/Zenoh settings.
Do not run the manual common launch while the s6 `cyclo_intelligence` unit is
already running. A process lock rejects a second Runtime before it can replace
the existing control socket or publish robot commands.

Runtime exit shuts down the common launch; Orchestrator exit also shuts down
its launch. There is no Runtime-only respawn or automatic inference resume.
Runtime shutdown rejects new lifecycle requests and Worker mutations, clears
the action plan, and retries failed holds while retaining robot I/O. Launch
allows 30 seconds before SIGTERM and another 5 before SIGKILL. A controller
command watchdog is mandatory: SIGKILL, missing joint feedback and power loss
cannot guarantee a successful software hold.

Docker health checks cover the always-on UI/management services; they do not
mean inference is ready. With the common launch stopped, Supervisor reports
Runtime unavailable and still denies Worker mutations whose safety cannot be
verified. Start the common launch before managing Workers through the UI.

Runtime fallbacks include:

| Variable | Default | Owner |
|---|---:|---|
| `GET_ACTION_TIMEOUT_S` | `5.0` | Runtime -> Worker request |
| `LOAD_POLICY_TIMEOUT_S` | `7200.0` | Runtime -> Worker LOAD |
| `CONTROL_HZ` | `100.0` | command loop |
| `INFERENCE_HZ` | `15.0` | source action waypoint rate |
| `CHUNK_ALIGN_WINDOW_S` | `0.3` | chunk alignment |
| `INITIAL_POSE_SYNC_STATE_MAX_AGE_S` | `1.0` | joint-state freshness |
| `WORKER_HEARTBEAT_TIMEOUT_S` | `2.0` | active Worker watchdog |
| `WORKER_READY_TIMEOUT_S` | `120.0` | Continuous unanswered readiness probes before showing an error; not a startup delay. A fresh `loading` heartbeat uses `LOAD_POLICY_TIMEOUT_S` instead. |
| `ORCHESTRATOR_HEARTBEAT_TIMEOUT_S` | `3.0` | active owner watchdog |

## Adding A Runtime

1. Add `policy/<runtime>/manifest.yaml`.
2. Implement `<runtime>_engine/create_engine()`.
3. Build an Engine-only image with the common `engine_process` package.
4. Add one explicit Compose service with the same runtime ID.
5. Add adapter and image smoke tests.

The catalog then exposes the runtime to UI, BT, Supervisor, and central
Runtime validation without adding another model list to those components.

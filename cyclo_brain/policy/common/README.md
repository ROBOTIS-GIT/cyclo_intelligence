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
fresh-current-joint hold.

The Supervisor uses `/run/cyclo/policy-runtime.sock` to reserve Worker
start/stop/recreate operations. A Worker used by a loaded session, or one with
a pending hold, cannot be changed through the Supervisor API.

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

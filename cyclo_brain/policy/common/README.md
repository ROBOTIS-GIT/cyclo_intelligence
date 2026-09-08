# Common Policy Runtime

This directory contains the two framework-independent halves of Cyclo policy
inference. They are deployed in different images.

```text
cyclo_intelligence image                  Worker image
┌──────────────────────────────┐          ┌──────────────────────────┐
│ policy-runtime s6 service    │  Zenoh   │ engine-process s6 service│
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

## Environment

The s6 services run through interactive bash so `/root/.bashrc` supplies the
ROS/Zenoh settings:

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
```

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

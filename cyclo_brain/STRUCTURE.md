# cyclo_brain Runtime Structure

This is the textual reference for Cyclo policy inference. The visual reference
is [`docs/architecture.html`](docs/architecture.html).

## 1. Deployment Topology

```text
┌──────────────────────── cyclo_intelligence ────────────────────────┐
│ UI -> Orchestrator -> /policy/inference_command                    │
│                              │                                     │
│                    policy-runtime (s6)                             │
│                    - one global session                            │
│                    - lifecycle and safety                          │
│                    - ControlLoop / ActionChunkProcessor             │
│                    - RobotClient command publisher                 │
│                              │                                     │
│             /<runtime>/engine_command over ROS2/Zenoh              │
└──────────────────────────────┼─────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              v                                 v
┌──────── lerobot_server ────────┐  ┌──────── groot_server ──────────┐
│ engine-process (s6)            │  │ engine-process (s6)            │
│ LeRobot model + observations   │  │ GR00T model + observations     │
│ /lerobot/engine_command        │  │ /groot/engine_command          │
│ /lerobot/worker_heartbeat      │  │ /groot/worker_heartbeat        │
└────────────────────────────────┘  └────────────────────────────────┘
```

The Engine reads camera, state, and sensor topics directly. The Policy Runtime
receives only action chunks from the selected Engine. There is no additional
observation relay.

## 2. Module Responsibilities

### PolicyRuntime

Path: `policy/common/runtime/main_runtime/main.py`

Creates the catalog, Worker registry, one SessionState, ControlLoop, external
services, heartbeat subscribers, and the local Supervisor control socket. It
hosts the canonical `/policy/inference_command` service and temporary
`/<runtime>/inference_command` aliases. It also monitors the active Worker and
Orchestrator.

### ServiceHandler

Path: `policy/common/runtime/main_runtime/service_handler.py`

Serializes all lifecycle commands with one re-entrant lock.

- LOAD resolves and validates `policy_id` and parameters, verifies Worker
  `DESCRIBE`, asks the Worker to load, then configures the robot control path.
- START/RESUME start initial-pose sync or the normal control loop.
- PAUSE/STOP require a successful current-pose hold when one is needed.
- UNLOAD is rejected while a hold is pending and clears the session only after
  the Worker confirms unload.
- STATUS returns the central session snapshot.
- Faults clear action output and move the session to `error`.

The same lock protects Worker mutation reservations, closing the race between a
LOAD request and a Supervisor start/stop/recreate operation.

### SessionState

Path: `policy/common/runtime/main_runtime/session_state.py`

Stores only logical state: loaded/running/paused/error, runtime and policy IDs,
checkpoint, canonical parameters, instruction, action keys, and publish mode.
It does not perform I/O.

### WorkerRegistry

Path: `policy/common/runtime/main_runtime/worker_registry.py`

Creates one Engine client per runtime and routes by namespaced policy ID. Before
LOAD it compares Worker `DESCRIBE` with the repository catalog:

- protocol major version;
- runtime ID;
- complete supported policy ID set;
- complete capability object;
- selected policy support.

It records heartbeat timestamps and Worker instance IDs. A changed instance,
missing/stale heartbeat, or stale Orchestrator heartbeat becomes a fail-safe
reason during an active session.

### InferenceRequester

Path: `policy/common/runtime/main_runtime/inference_requester.py`

Builds EngineCommand requests and owns LOAD/GET_ACTION/UNLOAD/DESCRIBE/STATUS
timeouts. Sequence IDs reject late responses after a timeout. Only one
GET_ACTION request may be active at a time.

### ControlLoop

Path: `policy/common/runtime/main_runtime/control_loop.py`

Owns command timing and action buffering.

- Requests action chunks using sync or async-prefetch scheduling.
- Pushes chunks into `ActionChunkProcessor`.
- Pops one processed action per output tick.
- Always publishes trajectory preview when available.
- Publishes robot commands only in robot mode.
- Runs initial-pose synchronization before normal inference when enabled.
- Clears buffered and in-flight generations on pause, stop, mode changes, and
  faults.
- On a safety stop, publishes zero Twist and holds fresh current joint values.
  Missing or stale joint state does not count as a successful hold.

### RuntimeControlServer

Path: `policy/common/runtime/main_runtime/runtime_control.py`

Small JSON API over `/run/cyclo/policy-runtime.sock`. It is local to the Cyclo
container and is used by Supervisor to atomically reserve Worker lifecycle
changes. It is not a public network API.

### RobotClient

Path: `sdk/robot_client/robot_client/robot_client.py`

Maps shared robot config to ROS2/Zenoh I/O. Subscription groups can be enabled
independently.

- Engine: images + state + sensors.
- Policy Runtime, simulation: no observation subscriptions.
- Policy Runtime, robot mode: state only for initial sync and safety hold.

Robot configs remain the owner of camera topics/names and joint order.

### ActionChunkProcessor

Path: `sdk/action_chunk_processing/`

Resamples, aligns, buffers, and pops model actions according to Dataset FPS,
control rate, and align window. It is installed only in the Cyclo image after
this refactor.

### EngineWorker

Path: `policy/common/runtime/engine_process/worker.py`

Framework-neutral Worker process. It exposes DESCRIBE, LOAD, GET_ACTION,
UNLOAD, and STATUS, publishes heartbeat payloads, and manages the ready marker.
It loads the concrete adapter through `POLICY_ENGINE_MODULE` and
`create_engine()`.

### Backend Engine

Paths:

- `policy/lerobot/lerobot_engine/`
- `policy/groot/groot_engine/`

Owns model-specific loading, optimization, preprocessing, inference, and action
shape. It receives validated policy parameters. It must return a finite
`(chunk_size, action_dim)` array and must never command the robot.

## 3. Lifecycle Flow

### LOAD

```text
UI / BT
  -> TaskInfo(policy_id, parameters, checkpoint, timing)
  -> Orchestrator canonical JSON check
  -> /policy/inference_command LOAD
  -> ServiceHandler catalog validation
  -> WorkerRegistry DESCRIBE compatibility check
  -> /<runtime>/engine_command LOAD
  -> Engine configures observation subscriptions and loads checkpoint
  <- action_keys
  -> Policy Runtime configures command-only RobotClient and action processor
  -> central session = loaded
```

A failed Worker LOAD does not create a central loaded session.

### START And Action Flow

```text
START
  -> optional initial-pose target
  -> central session = running/syncing

ControlLoop refill
  -> Worker GET_ACTION
  -> Engine reads latest model observations directly
  -> model returns action chunk
  -> ActionChunkProcessor align/resample/buffer
  -> ControlLoop preview
  -> optional robot command
```

Only one global inference session exists, so a second model cannot LOAD until
the current policy is unloaded.

### PAUSE, STOP, And UNLOAD

PAUSE and STOP immediately stop the loop, clear the buffer, invalidate in-flight
results, and perform a required hold. A failed hold leaves the session in a
retryable safety state. UNLOAD is blocked until the hold succeeds. The Worker
and central state are cleared only after Worker UNLOAD succeeds.

### Faults

During an active session, these faults are fail-closed:

- GET_ACTION failure or timeout;
- Worker heartbeat missing/stale;
- Worker instance ID changes after restart;
- Orchestrator heartbeat stale.

The Runtime clears the action buffer, stops publication, attempts zero Twist
and current-pose hold, records the error, and rejects unsafe Worker mutation.

## 4. Catalog And UI Flow

```text
policy/<runtime>/manifest.yaml
  -> strict catalog validation
  -> Supervisor GET /api/policies/catalog
  -> Inference UI and BT selector
  -> TaskInfo policy_id + parameters
  -> Runtime re-validates before Engine LOAD
```

Manifest data owns model names and capabilities. Compose owns images,
containers, and mounts. Catalog loading fails when a runtime references a
missing Compose service or violates the schema.

## 5. Container Lifecycle

- `cyclo_intelligence`: s6 starts `policy-runtime` automatically. Health
  requires the service and `/run/cyclo/policy-runtime.ready`.
- Worker containers: s6 starts only `engine-process`. The old Worker
  `main-runtime` service is removed.
- Worker container health checks s6 and the Engine ready marker.
- Supervisor additionally checks Worker DESCRIBE compatibility and heartbeat
  freshness before the UI reports `Backend ready`.
- Supervisor start, stop, restart, recreate, and pull paths acquire a local
  Runtime mutation lease.
- A normal Cyclo shutdown runs the safety stop before closing Runtime resources.
  Power loss still requires the robot controller command watchdog.

## 6. Container Deployment

`docker/docker-compose.yml` is the single deployment definition. Runtime, Worker adapter, SDK,
Supervisor, UI, robot config, and URDF files come from the image. Only data,
model cache, devices, and operational sockets are mounted.

```bash
./docker/container.sh start --build
./docker/container.sh start-policy lerobot --build
./docker/container.sh build-ui
```

## 7. Source Layout

```text
cyclo_brain/
├── policy/
│   ├── common/
│   │   ├── catalog/
│   │   ├── runtime/
│   │   │   ├── main_runtime/      # central Policy Runtime implementation
│   │   │   ├── engine_process/    # Worker implementation
│   │   │   └── engine.py          # adapter ABC
│   │   └── s6-services/
│   │       └── engine-process/    # Worker longrun only
│   ├── lerobot/
│   │   ├── manifest.yaml
│   │   ├── lerobot/
│   │   └── lerobot_engine/
│   └── groot/
│       ├── manifest.yaml
│       ├── Isaac-GR00T/
│       └── groot_engine/
└── sdk/
    ├── action_chunk_processing/
    ├── robot_client/
    └── zenoh_ros2_sdk/

docker/
├── s6-services/policy-runtime/    # central longrun
├── supervisor_api/
└── docker-compose.yml             # deployment definition
```

## 8. Extension Rule

A LeRobot policy already supported by the pinned fork stays in the LeRobot
Worker image. Add its manifest entry and adapter logic only if needed.

A framework with incompatible Python, CUDA, JAX, or system dependencies gets a
new Engine-only Worker image:

1. add `policy/<runtime>/manifest.yaml`;
2. implement `<runtime>_engine/create_engine()`;
3. add AMD64/ARM64 Dockerfiles;
4. add one explicit Compose service;
5. validate catalog, DESCRIBE, heartbeat, LOAD, and inference.

No new model list should be added to UI, Supervisor, BT, or Orchestrator.

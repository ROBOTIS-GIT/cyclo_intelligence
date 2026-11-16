# Architecture - cyclo_intelligence

As-built inference topology after `docker/container.sh start`.

- Detailed code map: [`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md)
- Visual map: [`cyclo_brain/docs/architecture.html`](../cyclo_brain/docs/architecture.html)
- Refactoring report: [`cyclo_brain/docs/policy_runtime_refactoring_implementation.html`](../cyclo_brain/docs/policy_runtime_refactoring_implementation.html)

## Container Topology

```text
cyclo_intelligence
├── UI / nginx (Autonomy Studio: Mission Canvas, Action Canvas)
├── supervisor_api (service lifecycle, navigation, maps/missions/trees)
├── orchestrator
├── bt_node (on demand)
├── standalone CLI
├── cyclo_data
└── policy-runtime
    ├── lifecycle and global session
    ├── action chunk processing
    ├── safety and initial pose sync
    └── robot command publishers

lerobot_server                   groot_server
└── engine-process               └── engine-process
    ├── observation subscribers      ├── observation subscribers
    ├── LeRobot policy               ├── GR00T policy
    └── action chunk inference       └── action chunk inference

ai_worker (externally managed)
└── SLAM / Nav2 navigation runtime
```

The central Policy Runtime owns control behavior once. Worker containers own
only framework-specific model dependencies, observation preprocessing, and
inference.

## Task Execution

```text
Autonomy Studio (browser)
  │ /api/services/bt_node/start        supervisor_api starts the s6 service
  │ /api/bt/trees, /api/bt/support     saved trees + supported robots
  │ /bt/load_and_run, /bt/set_running  via rosbridge
  ▼
bt_node  ──/task/command──▶ orchestrator_node ──InferenceCommand──▶ central Policy Runtime
  │
  │ joint / velocity commands
  ▼
Robot
```

`bt_node` and `orchestrator_node` live in the same ROS package but share no
Python; the boundary is enforced by
`orchestrator/tests/bt/test_package_boundary.py`. Mission sequencing
(Nav2 goal per waypoint, then that waypoint's tree) currently runs in the
browser (`orchestrator/ui/src/hooks/useMissionRunner.js`) against the
supervisor's navigation endpoints and the `/bt/*` services. Saved trees are
user data under `CYCLO_BT_TREES_DIR` (default `/workspace/bt/trees`), and the
robots the engine supports are listed once in
`shared/shared/robot_configs/schema.py`.

Navigation maps and missions use `CYCLO_NAVIGATION_DATA_DIR` (default
`/workspace/navigation`). Both navigation and saved trees persist in the
`docker/workspace` bind mount. `CYCLO_BT_EXAMPLE_TREES_DIR` points to packaged
examples used to seed the saved-tree directory, not to the AI Worker sources.

Policy containers are backend-isolated so each opensource model can own its
Dockerfile, Python dependencies, and upstream submodule.

## Data Flow

```text
UI / BT
  -> Orchestrator
  -> /policy/inference_command
  -> Policy Runtime
  -> /<runtime>/engine_command
  -> selected Engine Worker
  -> action chunk
  -> Policy Runtime buffer/control loop
  -> RobotClient command publishers
  -> Robot
```

Workers subscribe to camera, state, and sensor topics directly over
ROS2/Zenoh. Images are not relayed through the Cyclo container. Policy Runtime
subscribes only to joint state when robot command safety or initial pose sync
requires it.

## Contracts

| Contract | Owner | Purpose |
|---|---|---|
| `/policy/inference_command` | Policy Runtime | Public policy lifecycle and session control |
| `/<runtime>/inference_command` | Policy Runtime | Temporary compatibility aliases |
| `/<runtime>/engine_command` | Engine Worker | Internal model load, action, describe, status, unload |
| `policy/<runtime>/manifest.yaml` | Catalog | Policy IDs, capabilities, UI parameters, checkpoint root |

`EngineCommand` echoes `seq_id` so late responses can be discarded after a
timeout. Its protocol and each Worker descriptor are checked before LOAD.

## Deployment Boundary

UI, Orchestrator, control loop, safety, and action processing changes rebuild
the Cyclo image. A model adapter or framework dependency change rebuilds only
its Worker image. Wire-protocol changes require Cyclo and affected Workers to
be released as one compatible set.

All image builds use the repository root as context. Worker Build is a local
operation that does not pull Worker images or change running containers; the
main image can still be pulled by `container.sh start`. The release defaults
are Cyclo `1.5.0` and the local LeRobot build tag `1.4.2-<arch>`. UI build/test commands use an
external Node container with the host user's UID/GID, not npm in the main
container.

## Code Map

| Concern | Source |
|---|---|
| Target structure | [`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md) |
| Common runtime | [`cyclo_brain/policy/common/runtime/`](../cyclo_brain/policy/common/runtime/) |
| Central Policy Runtime | [`main_runtime`](../cyclo_brain/policy/common/runtime/main_runtime/) |
| Engine process | [`engine_process`](../cyclo_brain/policy/common/runtime/engine_process/) |
| LeRobot engine | [`lerobot_engine`](../cyclo_brain/policy/lerobot/lerobot_engine/) |
| GR00T engine | [`groot_engine`](../cyclo_brain/policy/groot/groot_engine/) |
| Robot client | [`robot_client`](../cyclo_brain/sdk/robot_client/) |
| Action processing | [`action_chunk_processing`](../cyclo_brain/sdk/action_chunk_processing/) |
| Behavior-tree engine | [`orchestrator/orchestrator/bt/`](../orchestrator/orchestrator/bt/) |
| Supervisor API | [`docker/supervisor_api/`](../docker/supervisor_api/) |
| Interfaces | [`interfaces`](../interfaces/) |
| Compose | [`docker/docker-compose.yml`](../docker/docker-compose.yml) |

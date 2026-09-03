# Architecture - cyclo_intelligence

As-built runtime topology after `docker/container.sh start`.

Visual map: [`cyclo_brain/docs/architecture.html`](../cyclo_brain/docs/architecture.html)

When runtime structure changes, update both
[`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md) and the visual map in
`cyclo_brain/docs/`.

## Container Topology

```text
Host
├── cyclo_intelligence container
│   ├── UI / nginx                 Autonomy Studio (Mission Canvas, Action Canvas)
│   ├── supervisor_api             service lifecycle, maps/missions/trees data,
│   │                              Nav2 goals, robot capability checks
│   ├── orchestrator_node          control plane (record / train / infer)
│   ├── bt_node                    behavior-tree engine, started on demand
│   ├── standalone CLI
│   └── cyclo_data
│
├── ai_worker container            SLAM / Nav2 navigation runtime
│
└── policy container per backend
    ├── main-runtime
    └── engine-process
```

## Task Execution

```text
Autonomy Studio (browser)
  │ /api/services/bt_node/start        supervisor_api starts the s6 service
  │ /api/bt/trees, /api/bt/support     saved trees + supported robots
  │ /bt/load_and_run, /bt/set_running  via rosbridge
  ▼
bt_node  ──/task/command──▶ orchestrator_node ──InferenceCommand──▶ policy container
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

Policy containers are backend-isolated so each opensource model can own its
Dockerfile, Python dependencies, and upstream submodule.

## Policy Container

```text
UI / orchestrator or standalone CLI
  │
  │ command args or /<backend>/inference_command
  ▼
main-runtime
  ├── ServiceHandler
  ├── SessionState
  ├── InferenceRequester
  ├── ActionChunkProcessor
  ├── ControlLoop
  └── RobotClient command publishers
       │
       │ /cmd_vel, /leader/*/joint_trajectory
       ▼
     Robot

main-runtime
  │
  │ /<backend>/engine_command
  ▼
engine-process
  ├── PolicyLoader
  ├── optional Optimizer
  ├── Preprocessor
  ├── Predictor
  └── RobotClient observation subscribers
```

Main owns session flow and robot command publishing. Engine owns model loading,
sensor/state reads, preprocessing, and inference.

## Key Services

| Service | Owner | Purpose |
|---|---|---|
| `/<backend>/inference_command` | Main | External LOAD/START/PAUSE/RESUME/STOP/UNLOAD |
| `/<backend>/engine_command` | Engine | Internal LOAD_POLICY/GET_ACTION/UNLOAD_POLICY |

`EngineCommand` echoes `seq_id`. Main uses it to discard stale responses after
timeout.

## Code Map

| Concern | Source |
|---|---|
| Target structure | [`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md) |
| Common runtime | [`cyclo_brain/policy/common/runtime/`](../cyclo_brain/policy/common/runtime/) |
| Main process | [`main_runtime`](../cyclo_brain/policy/common/runtime/main_runtime/) |
| Engine process | [`engine_process`](../cyclo_brain/policy/common/runtime/engine_process/) |
| LeRobot engine | [`lerobot_engine`](../cyclo_brain/policy/lerobot/lerobot_engine/) |
| GR00T engine | [`groot_engine`](../cyclo_brain/policy/groot/groot_engine/) |
| Robot client | [`robot_client`](../cyclo_brain/sdk/robot_client/) |
| Action processing | [`action_chunk_processing`](../cyclo_brain/sdk/action_chunk_processing/) |
| Behavior-tree engine | [`orchestrator/orchestrator/bt/`](../orchestrator/orchestrator/bt/) |
| Supervisor API | [`docker/supervisor_api/`](../docker/supervisor_api/) |
| Interfaces | [`interfaces`](../interfaces/) |
| Compose | [`docker/docker-compose.yml`](../docker/docker-compose.yml) |

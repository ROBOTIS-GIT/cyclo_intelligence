# cyclo_brain

`cyclo_brain` contains policy inference adapters, the shared Policy Runtime,
and the SDKs used to connect models to Cyclo robot I/O.

## Runtime Architecture

The robot-facing runtime is part of the `cyclo_intelligence` image. Model
frameworks remain isolated in Engine-only worker images.

```text
UI / BT
   |
   v
Orchestrator
   |  /policy/inference_command
   v
Policy Runtime (cyclo_intelligence)
   |  /<runtime>/engine_command
   v
Engine Process (lerobot_server or groot_server)
   |  direct ROS2/Zenoh observation subscriptions
   v
Model inference -> action chunk -> Policy Runtime -> robot command
```

The Policy Runtime owns the single global inference session, lifecycle,
`ControlLoop`, `ActionChunkProcessor`, initial-pose synchronization, command
publishing, and safety stops. An Engine Process owns model dependencies,
checkpoint loading, observation preprocessing, and action-chunk inference. It
never publishes robot commands.

## Layout

```text
cyclo_brain/
├── policy/
│   ├── common/
│   │   ├── catalog/               # manifest loading and validation
│   │   └── runtime/
│   │       ├── main_runtime/      # central Policy Runtime package
│   │       └── engine_process/    # code copied into each worker image
│   ├── lerobot/
│   │   ├── manifest.yaml
│   │   ├── lerobot/               # upstream fork submodule
│   │   └── lerobot_engine/        # InferenceEngine adapter
│   └── groot/
│       ├── manifest.yaml
│       ├── Isaac-GR00T/           # upstream submodule
│       └── groot_engine/          # InferenceEngine adapter
└── sdk/
    ├── action_chunk_processing/
    ├── robot_client/
    └── zenoh_ros2_sdk/
```

The Python package is still named `main_runtime` for source compatibility,
but it runs only as the central `policy-runtime` s6 service in the Cyclo
container. Worker images contain only `engine-process`.

## External And Worker APIs

- `/policy/inference_command`: canonical external lifecycle service.
- `/lerobot/inference_command` and `/groot/inference_command`: temporary
  compatibility aliases backed by the same global session.
- `/<runtime>/engine_command`: internal Worker API with `DESCRIBE`, `LOAD`,
  `GET_ACTION`, `UNLOAD`, and `STATUS`.
- `/<runtime>/worker_heartbeat`: Worker instance and engine-state heartbeat.

`policy_id` uses a namespaced value such as `lerobot:act` or `groot:n17`.
The Runtime validates the repository manifest against Worker `DESCRIBE` before
LOAD. Incompatible protocol major versions, policy sets, or capabilities are
rejected.

## Policy Catalog

Each independently deployed runtime owns one `manifest.yaml`. It declares its
runtime ID, Compose service, checkpoint root, supported policies,
capabilities, and model-specific UI parameters. The catalog is the common
source for the Inference UI, BT model selector, Supervisor API, and Runtime
request validation. Docker image names and mounts remain owned by Compose.

## Deployment Units

| Change | Images to rebuild |
|---|---|
| UI or Orchestrator | Cyclo |
| Control loop, safety, sync/async, or Runtime catalog | Cyclo |
| LeRobot model adapter or dependency | LeRobot Worker |
| GR00T model adapter or dependency | GR00T Worker |
| Engine wire protocol or shared Worker SDK | Cyclo and affected Workers |

Compose runs image contents without source bind mounts. Rebuild the affected
image with `--build`; for React-only iteration, use `docker/container.sh build-ui`.

See [`STRUCTURE.md`](STRUCTURE.md) for module ownership and
[`docs/architecture.html`](docs/architecture.html) for the visual architecture.

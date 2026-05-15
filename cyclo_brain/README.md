# cyclo_brain

Training and inference backend code lives here.

- `sdk/`: shared Python SDKs mounted into policy containers.
- `policy/`: one Docker container per opensource/backend policy.

This directory is not a colcon target. `sdk/` is mounted into policy
containers at runtime, and `policy/` is built by Docker.

```text
cyclo_brain/
├── sdk/
│   ├── action_chunk_processing/  # action_list buffering/post-processing
│   ├── robot_client/             # robot observation + command topic client
│   └── zenoh_ros2_sdk/           # ROS2-over-Zenoh SDK submodule
└── policy/
    ├── common/
    │   ├── runtime/
    │   │   ├── engine.py         # InferenceEngine ABC
    │   │   ├── main_runtime/     # Main process
    │   │   └── engine_process/   # Engine process
    │   └── s6-services/
    │       ├── main-runtime/
    │       └── engine-process/
    ├── lerobot/
    │   └── lerobot_engine/
    └── groot/
        └── groot_engine/
```

## Runtime Shape

Each policy container runs two Python processes:

- `main-runtime`: hosts `/<backend>/inference_command`, owns session state,
  owns the control loop, and publishes robot commands through `RobotClient`.
- `engine-process`: hosts `/<backend>/engine_command`, owns policy loading,
  reads sensor/state topics through `RobotClient`, and runs inference.

Main and Engine exchange `interfaces/srv/EngineCommand` requests. `GET_ACTION`
responses carry a flat `action_list` plus `chunk_size`, `action_dim`, and
`seq_id`. Main discards stale responses by `seq_id` after timeout.

`ActionChunkProcessor` is owned by Main. It can either post-process model
actions into control-rate actions, or buffer the raw action list directly.

## Adding A Backend

1. Create `policy/<backend>/<backend>_engine/`.
2. Implement `create_engine() -> InferenceEngine`.
3. Add backend-specific dependency isolation in `Dockerfile.{arm64,amd64}`.
4. Mount `policy/common/runtime/` at `/policy_runtime` and the engine package
   at `/app/<backend>_engine`.
5. Set `POLICY_BACKEND=<backend>` and, if needed,
   `POLICY_ENGINE_MODULE=<backend>_engine`.

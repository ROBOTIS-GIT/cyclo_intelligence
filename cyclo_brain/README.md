# cyclo_brain

Everything related to training and inference lives under this folder.
Two axes:

- **`sdk/`** — host- and container-shared Python assets that aren't
  specific to any one policy backend.
- **`policy/`** — a Docker container per backend. Each container runs
  two independent OS processes that talk over Zenoh.

> **Not a colcon target.** `sdk/` is mounted into policy containers at
> runtime; `policy/` is built by `docker build`, not `colcon build`.
> A top-level `COLCON_IGNORE` short-circuits colcon's package walker
> so a default `colcon build` from the workspace root doesn't descend
> into the upstream submodules (`lerobot/`, `Isaac-GR00T/`,
> `zenoh_ros2_sdk/`). The 5 actual ROS 2 packages should be built by
> pinning `--paths` explicitly.

```
cyclo_brain/
├── sdk/
│   ├── post_processing/       ActionChunkProcessor — L2 align /
│   │                           interpolate / blend / buffered pop.
│   │                           Pure numpy, no ROS2 / Zenoh imports,
│   │                           so Process B can own it + unit tests
│   │                           run outside the container.
│   ├── robot_client/          In-package module — hosts
│   │                           RobotServiceServer (training-only
│   │                           framework after Step 4-E cleanup)
│   │                           and message_definitions consumed by
│   │                           inference_server.py / control_publisher.py.
│   └── zenoh_ros2_sdk/        Submodule (ROBOTIS-GIT). Mounted into
│                               policy containers at /zenoh_sdk.
└── policy/
    ├── common/
    │   ├── runtime/          Process A (inference_server.py) + Process B
    │   │                      (control_publisher.py) + InferenceEngine
    │   │                      ABC. Bind-mounted into every policy
    │   │                      container at /policy_runtime, so backends
    │   │                      share one supervisor pair.
    │   └── s6-services/      Single s6 unit tree used by every policy
    │                          container (inference-server / control-
    │                          publisher / user longruns).
    ├── lerobot/              LeRobot backend container.
    │   ├── Dockerfile.{arm64,amd64}
    │   ├── lerobot_engine.py  Bind-mounted at /app/lerobot_engine.py;
    │   │                      implements InferenceEngine.
    │   ├── lerobot/          huggingface lerobot submodule.
    │   ├── RESULTS/          Validation outputs from policy load tests.
    │   └── scripts/
    └── groot/                GR00T backend (still owns its own runtime/
                              and s6-services/; common-runtime migration
                              is a separate phase).
```

## Two-process runtime contract

Every policy backend follows the same shape:

- **Process A (`inference_server.py`)** hosts
  `/<backend>/inference_command` (`interfaces/srv/InferenceCommand`)
  with the LOAD / START / PAUSE / RESUME / STOP / UNLOAD enum.
  Loading the model and subscribing to observations
  (`ROS2Subscriber` from `zenoh_ros2_sdk`) happen inside LOAD.
  After START, Process A listens on Zenoh topic
  `cyclo/policy/<backend>/run_inference` and publishes raw action
  chunks (`interfaces/msg/ActionChunk`) on
  `cyclo/policy/<backend>/action_chunk_raw`.

- **Process B (`control_publisher.py`)** boots with the container,
  reads `orchestrator/config/<robot_type>_config.yaml` for
  `command_topic_list` and `joint_order`, and runs a monotonic
  100 Hz loop. On each tick it pops one interpolated action from
  `ActionChunkProcessor` and publishes the per-group JointTrajectory
  (or Twist for mobile). When the buffer falls below the refill
  threshold it publishes a trigger on the Zenoh topic above; Process
  A answers with a fresh chunk.

The control loop *never* leaves the container. The orchestrator is a
command dispatcher, not a real-time actor — that's the main win over
the prior design.

## Naming the modality keys

`action_keys` is computed deterministically from the robot config's
follower groups — sorted group names with the `follower_` prefix
stripped. Both Process A and Process B read the same YAML, so they
arrive at the same ordering without a side channel.

Example for `ffw_sg2_rev1`:
```
follower_arm_left, follower_arm_right, follower_head,
follower_lift, follower_mobile
    ↓
action_keys = ['arm_left', 'arm_right', 'head', 'lift', 'mobile']
```

`ActionChunkProcessor.split_action` slices a flat action vector by
this ordering, mapping each slice to `leader_<key>` for the output
publisher (leader / follower topic separation).

## Adding a backend

1. Copy `cyclo_brain/policy/lerobot/` to
   `cyclo_brain/policy/<new>/`. The runtime + s6 units come from
   the shared `policy/common/` tree via bind mount — only the
   backend-specific engine file lives here.
2. Point the backend's `Dockerfile.arm64` at whatever upstream
   container image makes sense for its dependencies.
3. Implement `<new>_engine.py` with `create_engine()` returning an
   `InferenceEngine` subclass; Process A (`inference_server.py` in
   `policy/common/runtime/`) imports it by name.
4. `control_publisher.py` in `policy/common/runtime/` typically
   doesn't need changes — it's backend-agnostic.
5. Add a compose service entry (see
   [`docker/docker-compose.yml`](../docker/docker-compose.yml)),
   mounting `policy/common/runtime/` at `/policy_runtime` and the
   engine file at `/app/<new>_engine.py`.
6. Register the backend in `supervisor_api` for on-demand pull.

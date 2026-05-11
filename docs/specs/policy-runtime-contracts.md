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

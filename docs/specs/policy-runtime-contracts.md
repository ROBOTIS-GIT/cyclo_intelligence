# policy-runtime contracts

External and internal wire contracts honoured by
`cyclo_brain/policy/common/runtime/`. Update this file whenever you
add or change a topic/srv.

## Zenoh services

| Service | Direction | Type | Notes |
|---|---|---|---|
| `/lerobot/inference_command` | external -> Main | `interfaces/srv/InferenceCommand` | LOAD/START/PAUSE/RESUME/STOP/UNLOAD/UPDATE_INSTRUCTION |
| `/lerobot/engine_command` | Main -> Engine | `interfaces/srv/EngineCommand` | LOAD_POLICY/GET_ACTION/UNLOAD_POLICY, seq_id echoed |

The `/<backend>/...` prefix substitutes `<backend>` = `POLICY_BACKEND`.

## InferenceCommand enum

| Value | Name | Notes |
|---|---|---|
| 0 | `CMD_LOAD` | Engine loads weights; Main configures RobotClient command path |
| 1 | `CMD_START` | Main control loop starts publishing/actions refilling |
| 2 | `CMD_PAUSE` | Main stops publishing robot commands; model stays loaded |
| 3 | `CMD_RESUME` | Resumes command publishing; optional `task_instruction` updates instruction |
| 4 | `CMD_STOP` | Stops honoring; clears buffer |
| 5 | `CMD_UNLOAD` | Main closes command RobotClient; Engine cleanup releases observation RobotClient |
| 6 | `CMD_UPDATE_INSTRUCTION` | Updates `task_instruction` only; weights and lifecycle unchanged |

`paused` and `stopped` both stop command publishing. `STOP` also clears the
action buffer.

## Hard contracts (do not break without a coordinated PR)

- Service names + types in the table above.
- `InferenceCommand` request fields: `command`, `model_path`,
  `embodiment_tag`, `robot_type`, `task_instruction`.
- `InferenceCommand` response fields: `success`, `message`, `action_keys`.
- Bind-mount paths into policy containers: `/policy_runtime`,
  `/app/<backend>_engine`, `/zenoh_sdk`, `/robot_client_sdk`,
  `/action_chunk_processing_sdk`, `/orchestrator_config`.
- s6 longrun names: `main-runtime`, `engine-process`, `user`.
- Engine entry point: `create_engine() -> InferenceEngine` (module-level
  function in `<backend>_engine`).
- `EngineCommand` GET_ACTION response: flat `float64[] action_list` of length
  `chunk_size * action_dim`, with matching `seq_id`.

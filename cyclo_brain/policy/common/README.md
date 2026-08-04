# Common Policy Runtime

Policy-agnostic two-process container runtime. Each opensource policy backend
(LeRobot, GR00T, OpenVLA, ...) plugs in by providing a backend engine package
such as `<policy>_engine`. The Main process, Engine process, and s6 supervisor
are shared.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Container                                                        │
│                                                                  │
│  ┌──────────────────────┐  EngineCommand srv  ┌────────────────┐│
│  │ main-runtime         │ ───────────────────▶│ engine-process ││
│  │ external service     │                     │ policy deps    ││
│  │ control loop         │◀────────────────────│ RobotClient obs││
│  │ RobotClient command  │     action_list     │ inference      ││
│  └──────────────────────┘                     └────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

`main_runtime` and `engine_process` are never edited per policy. The
per-policy code lives in the backend engine package.

## Engine contract

Implement `cyclo_brain.policy.common.runtime.engine.InferenceEngine`:

```python
from engine import InferenceEngine

class MyEngine(InferenceEngine):
    def load_policy(self, request): ...        # weights + RobotClient
    def get_action_chunk(self, request): ...   # one (T, D) chunk
    def cleanup(self): ...
    @property
    def is_ready(self): ...

def create_engine() -> InferenceEngine:
    return MyEngine()
```

See `cyclo_brain/policy/lerobot/lerobot_engine/` for a worked example.

## Container layout

| Path | Source | Mount mode |
|---|---|---|
| `/policy_runtime/` | `cyclo_brain/policy/common/runtime/` | bind, ro |
| `/app/<policy>_engine/` | `cyclo_brain/policy/<policy>/<policy>_engine/` | bind, ro |
| `/etc/s6-overlay/s6-rc.d/` | `cyclo_brain/policy/common/s6-services/` | baked in image |
| `/zenoh_sdk/`, `/robot_client_sdk/`, `/action_chunk_processing_sdk/` | `cyclo_brain/sdk/...` | bind, ro |
| `/orchestrator_config/` | `shared/shared/robot_configs/` | bind, ro |
| `/policy_checkpoints/<policy>/` | `cyclo_brain/policy/<policy>/checkpoints/` | bind, rw |

For LeRobot, user-trained models can be placed under
`cyclo_brain/policy/lerobot/checkpoints/` on the host and loaded from
`/policy_checkpoints/lerobot/...` inside the container.

## Required environment

| Variable | Required | Default | Used by |
|---|---|---|---|
| `POLICY_BACKEND` | yes | - | both processes |
| `POLICY_ENGINE_MODULE` | no | `${POLICY_BACKEND}_engine` | Engine process |
| `POLICY_ENGINE_FACTORY` | no | `create_engine` | Engine process |
| `GET_ACTION_TIMEOUT_S` | no | `5.0` | Main -> Engine request |
| `LOAD_POLICY_TIMEOUT_S` | no | `7200.0` | Main -> Engine request |
| `INFERENCE_HZ` | no | `15.0` | Main action waypoint timing |
| `CONTROL_HZ` | no | `100.0` | Main robot command loop |
| `TARGET_CHUNK_SIZE` | no | `none` | Fixed-size resampling override; `none` keeps chunk duration |
| `REFILL_MARGIN_S` | no | `0.2` | Extra buffer time after observed GET_ACTION latency |
| `REFILL_LATENCY_WARMUP_SAMPLES` | no | `1` | Initial GET_ACTION latency samples ignored for warmup |
| `REFILL_LATENCY_SAMPLE_MAX_S` | no | `2.0` | Ignore longer latency samples; `none` disables filtering |
| `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` / `ZENOH_CONFIG_OVERRIDE` | no | set in `/root/.bashrc` | both |

`main-runtime` and `engine-process` source `/root/.bashrc` before applying
these defaults. Enter the policy container, edit the Cyclo ROS/Zenoh block near
the top of `/root/.bashrc` when the robot's Zenoh router or ROS domain changes.
For a remote router, comment the local `ZENOH_CONFIG_OVERRIDE` line and uncomment
the remote example with the router's IP, then restart the policy container so s6
processes read the new values.
`docker restart` preserves the edit; recreating or updating the container
resets `/root/.bashrc` to the image default.

For GR00T N1.7, the trained checkpoint may reference the gated
`nvidia/Cosmos-Reason2-2B` backbone instead of vendoring those weights. Register
a Hugging Face token for an approved account before first inference, or pre-cache
the Cosmos files under the shared Hugging Face cache. Policy containers sync the
Cyclo endpoint token store to the standard Hugging Face token file on startup.

## Adding a new policy

1. Create `cyclo_brain/policy/<policy>/<policy>_engine/` implementing the ABC.
2. Create `cyclo_brain/policy/<policy>/Dockerfile.{amd64,arm64}` — install
   the policy's deps; **do not** copy `runtime/` (it's bind-mounted).
   Copy `common/s6-services/` into `/etc/s6-overlay/s6-rc.d/`.
3. Add a service to `docker/docker-compose.yml` mounting `common/runtime/`
   at `/policy_runtime` and `<policy>_engine/` at `/app/`. Set
   `POLICY_BACKEND` env.
4. The same orchestrator yaml (`shared/shared/robot_configs/<robot>_config.yaml`)
   is reused for any backend — no per-policy yaml required.

## Main <-> Engine contract

- `/<backend>/inference_command` (interfaces/srv/InferenceCommand) - external -> Main.
- `/<backend>/engine_command` (interfaces/srv/EngineCommand) - Main -> Engine.
- `EngineCommand.seq_id` is echoed in the response so Main can discard stale
  responses after timeout.

These are stable across policies; the engine never sees them.

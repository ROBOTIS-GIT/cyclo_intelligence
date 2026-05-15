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
| `LOAD_POLICY_TIMEOUT_S` | no | `300.0` | Main -> Engine request |
| `ZENOH_ROUTER_IP` / `ZENOH_ROUTER_PORT` / `ROS_DOMAIN_ID` | no | `127.0.0.1 / 7447 / 30` | both |

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

# Common Policy Runtime

Policy-agnostic two-process container runtime. Each opensource policy
backend (LeRobot, GR00T, OpenVLA, …) plugs in by providing **one Python
file**: `<policy>_engine.py`. The server, control publisher, and s6
supervisor are shared.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Container                                                        │
│                                                                  │
│  ┌──────────────────────┐    Zenoh    ┌──────────────────────┐  │
│  │ Process A            │             │ Process B            │  │
│  │ inference-server     │ ───────────▶│ control-publisher    │  │
│  │  (PyTorch / GPU)     │   chunks    │  (100 Hz, SCHED_FIFO)│  │
│  │                      │ ◀───────────│                      │  │
│  │                      │   triggers  │                      │  │
│  └──────────────────────┘             └──────────────────────┘  │
│           │                                                      │
│           │ delegates to                                         │
│           ▼                                                      │
│  ┌──────────────────────┐                                        │
│  │ <policy>_engine.py   │  ← per-policy: load + inference only   │
│  │  (mounted at /app)   │                                        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

`inference_server.py` and `control_publisher.py` are **never edited per
policy**. The only per-policy code is the `InferenceEngine` subclass.

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

See `cyclo_brain/policy/lerobot/lerobot_engine.py` for a worked example.

## Container layout

| Path | Source | Mount mode |
|---|---|---|
| `/policy_runtime/` | `cyclo_brain/policy/common/runtime/` | bind, ro |
| `/app/<policy>_engine.py` | `cyclo_brain/policy/<policy>/<policy>_engine.py` | bind, ro |
| `/etc/s6-overlay/s6-rc.d/` | `cyclo_brain/policy/common/s6-services/` | baked in image |
| `/zenoh_sdk/`, `/robot_client_sdk/`, `/post_processing_sdk/` | `cyclo_brain/sdk/...` | bind, ro |
| `/orchestrator_config/` | `shared/shared/robot_configs/` | bind, ro |

## Required environment

| Variable | Required | Default | Used by |
|---|---|---|---|
| `POLICY_BACKEND` | yes | — | both processes (Zenoh topic prefix) |
| `POLICY_ENGINE_MODULE` | no | `${POLICY_BACKEND}_engine` | Process A |
| `POLICY_ENGINE_FACTORY` | no | `create_engine` | Process A |
| `REQUEST_TIMEOUT_S` | no | `5.0` | Process B (raise for slow VLAs) |
| `ZENOH_ROUTER_IP` / `ZENOH_ROUTER_PORT` / `ROS_DOMAIN_ID` | no | `127.0.0.1 / 7447 / 30` | both |

## Adding a new policy

1. Create `cyclo_brain/policy/<policy>/<policy>_engine.py` implementing the ABC.
2. Create `cyclo_brain/policy/<policy>/Dockerfile.{amd64,arm64}` — install
   the policy's deps; **do not** copy `runtime/` (it's bind-mounted).
   Copy `common/s6-services/` into `/etc/s6-overlay/s6-rc.d/`.
3. Add a service to `docker/docker-compose.yml` mounting `common/runtime/`
   at `/policy_runtime` and `<policy>_engine.py` at `/app/`. Set
   `POLICY_BACKEND` env.
4. The same orchestrator yaml (`shared/shared/robot_configs/<robot>_config.yaml`)
   is reused for any backend — no per-policy yaml required.

## Process A ↔ Process B contract (Zenoh)

- `cyclo/policy/<backend>/configure` (String) — A → B; robot_type. `""`=deconfigure.
- `cyclo/policy/<backend>/lifecycle` (String) — A → B; `loaded|running|paused|stopped|unloaded`.
- `cyclo/policy/<backend>/run_inference` (UInt64) — B → A; trigger seq_id.
- `cyclo/policy/<backend>/action_chunk_raw` (interfaces/msg/ActionChunk) — A → B; chunk.
- `/<backend>/inference_command` (interfaces/srv/InferenceCommand) — orchestrator → A.

These are stable across policies; the engine never sees them.

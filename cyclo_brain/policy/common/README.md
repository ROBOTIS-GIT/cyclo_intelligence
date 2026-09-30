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
| `INITIAL_POSE_SYNC_STATE_MAX_AGE_S` | no | `1.0` | Maximum joint-state age allowed for initial pose sync and interruption hold |
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
Add an `INITIAL_POSE_SYNC_STATE_MAX_AGE_S` export there only when the one-second
joint-state freshness limit needs to be adjusted for the target robot.
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

## LeRobot GPU startup across JetPack versions

LeRobot keeps its CUDA/PyTorch image dependencies and selects the CUDA driver
at process startup using `runtime/gpu_runtime.py`. The common s6 runners invoke
it **after** the interactive shell reads `.bashrc` and **before** importing any
policy libraries. No model is loaded and no robot commands are sent by the probe.

On Jetson (`/etc/nv_tegra_release` exists), `auto` first tries the mounted host
driver, excluding CUDA `compat` directories from `LD_LIBRARY_PATH`. It discovers
`libcuda.so.1` under the host's `nvgpu`, `tegra` or `nvidia` library directories,
while preserving other library paths. If that probe fails, it tries the image's
compat driver from the original path or `/usr/local/cuda/compat`. Each attempt
imports torch in a **fresh subprocess** and checks a matrix multiplication and
CUDA synchronization; GPU enumeration alone is insufficient. Non-Jetson hosts
retain the NVIDIA Container Toolkit environment in `auto` mode.

| Variable | Default | Meaning |
|---|---|---|
| `CYCLO_CUDA_DRIVER` | `auto` | `auto`, `host`, or `compat`; an explicit driver disables fallback |
| `CYCLO_POLICY_DEVICE` | `cuda` | Require CUDA; set `cpu` explicitly for CPU execution |
| `CYCLO_GPU_PROBE_TIMEOUT_S` | `30` | Positive timeout per driver probe, in seconds |

These variables are passed to LeRobot by Compose. For example, from the repository
root, recreate only that service using the same Compose project as the installer:

```bash
docker compose -p cyclo_intelligence \
  -f docker/docker-compose.yml -f docker/docker-compose.override.yml \
  up -d --no-deps --no-build --force-recreate lerobot
docker logs --tail 100 lerobot_server
```

Remove any earlier board-specific `LD_LIBRARY_PATH` override when deploying the
automatic selector. The checked-in development override does not set that path.
The `[gpu-runtime] Selected` log includes the selected profile, loaded driver
library, torch version, torch CUDA build version, and GPU name. A shell opened
with `docker exec` does **not** inherit the selected service environment; inspect
these logs rather than interpreting a bare shell's torch check as the service
configuration.

If all probes fail, the service does not start and Docker health remains failing.
An explicit CPU setting runs a CPU probe and is reported as such. LeRobot also
checks the requested device when loading weights, so directly launching the
engine cannot silently fall back to CPU. FastWAM's existing selective CPU offload
is preserved.

Health combines the s6 process checks with successful probe records in
`/run/cyclo-gpu/`. Records include PID and process start time so exited or restarted
processes cannot leave a stale healthy result. This is a **startup** device check,
not continuous GPU monitoring or a model-readiness/latency guarantee. Both service
runners and the common runtime are bind-mounted, so existing LeRobot images can
use the change without rebuilding. The image healthchecks are updated too for
subsequent builds. GR00T uses separate runners and is not changed by this wiring.

Before declaring another JetPack/image combination supported, validate on both
the old and upgraded boards: selected driver, real model load, repeated inference
latency, container restart/recreation, and any TensorRT or custom CUDA extensions.
CPU-only unit tests cover selection and failure handling; they do not establish
JetPack compatibility. Keep TensorRT engine caches specific to the tested runtime.

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

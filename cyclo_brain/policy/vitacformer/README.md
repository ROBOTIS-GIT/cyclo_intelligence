# ViTacFormer inference backend

This backend loads the SH5 ViTacFormer checkpoint format and uses Cyclo's
standard two-process policy runtime. It supports inference on `ffw_sh5_rev1`;
training and other robot/checkpoint layouts are not implemented here.

| Setting | Value |
| --- | --- |
| Compose service / UI backend | `vitacformer` |
| Container | `vitacformer_server` |
| Image | `robotis/vitacformer-zenoh:1.0.0-{arm64,amd64}` |
| Engine module | `vitacformer_engine` |
| External service | `/vitacformer/inference_command` |
| Host model directory | `docker/workspace/model/vitacformer` |
| Container model directory | `/workspace/model/vitacformer` |

## Docker layout

LeRobot policies share the `lerobot` backend and its image; GR00T has a separate
backend/image. ViTacFormer follows the latter pattern and shares the same
`common/runtime`, s6 services, RobotClient and action-chunk SDK interfaces.

Like the existing policy images, the ViTacFormer image contains its Python
dependencies and s6 units. Compose mounts the engine, common runtime, SDKs and
robot configuration read-only. An image build alone does **not** embed this
checkout's Python engine code. Keep the checkout and image together when
deploying, and restart the backend after changing mounted source.

The ARM64 base is the same Jetson CUDA/PyTorch base used by LeRobot; AMD64 uses
NVIDIA PyTorch. Sharing a base layer does not share a container or policy
runtime. ViTacFormer does not install or import the LeRobot policy package.

SH5 tactile message definitions are supplied in this backend's `messages/`
directory and mounted at `/zenoh_sdk/messages`. The upstream SDK's committed
message directory has no ROS message definitions. Standard ROS Jazzy messages
are included in the image at `/opt/vitacformer/ros_messages`, selected by
`ZENOH_ROS2_SDK_CACHE`. `prepare_message_cache.sh` pins `common_interfaces` and
`rcl_interfaces` to exact commits and keeps the SDK's local `jazzy` refs.
Runtime message registration needs neither a host cache nor a GitHub download.
No modified SDK submodule is needed.

### Independence from ai_worker

Build and deployment require only this Cyclo checkout and its pinned
`cyclo_brain/sdk/zenoh_ros2_sdk` submodule. No `ai_worker` checkout, image,
container, Python package, ROS install overlay or startup script is used.
The SH5 wire definitions above are local `.msg` files loaded by the Python
Zenoh SDK; an installed `robotis_interfaces` ROS package is not required.
Neither LeRobot nor GR00T submodules need to be initialized for this backend.

The normal runtime still needs a Zenoh router and, after LOAD, compatible
camera, joint-state and tactile publishers on the topics in Cyclo's
`shared/shared/robot_configs/ffw_sh5_rev1_config.yaml`. These are message-level
interfaces: the policy does not start or import a particular robot driver.
Weights remain external user data under `/workspace/model/vitacformer`.

## Build and select

From the repository root:

```bash
git submodule update --init cyclo_brain/sdk/zenoh_ros2_sdk
docker/container.sh start-vitacformer --build
```

The helper detects ARM64 or AMD64. To build without starting a policy service:

```bash
ARCH=arm64 docker compose -f docker/docker-compose.yml build vitacformer
# On an x86 NVIDIA machine, use ARCH=amd64 instead.
```

Use a local build while the release image is unavailable in the registry.
The existing supervisor auto-provisioning path also falls back to a local
build if pulling the image fails. Publishing images to the `robotis` registry
is a separate maintainer release step.

The ViTacFormer image does not require `docker/scripts/init_zenoh_cache.sh`.
Start/build the main Cyclo container and rebuild the UI using the normal
repository workflow so its services and frontend match this checkout.

In Inference, select **ViTacFormer (SH5)**, select a compatible run directory,
and use the standard LOAD / START / STOP / UNLOAD controls. Selection defaults
to 30 Hz action playback and asynchronous requests. Initial verification can
use Simulation mode; this publishes previews rather than robot commands.
For Action Canvas, use `model="vitacformer:vitacformer"`, `inference_hz="30"`
and `action_request_mode="async"`. The legacy `lerobot:vitacformer` model name
routes to the dedicated backend as well.

## Checkpoint and sensor contract

```text
run/
  train_config.json
  normalization_stats.pt
  checkpoints/
    best_model.pt
    1000/model.pt              # optional selected training step
```

A run directory or its `checkpoints` directory selects the best checkpoint.
A numbered checkpoint directory selects its `model.pt`; an explicit `.pt`
file is also supported. Model files can remain in an existing workspace
location if that path is supplied explicitly.

The loader validates the source commit and data contract before loading the
state dictionary strictly with `weights_only=True`. It expects:

- one RGB head camera (`cam_left_head`), resized to 188 × 336;
- six 54-joint state frames sampled causally at 10 Hz;
- eighteen tactile frames at 30 Hz, with 45 taxels per hand;
- a baseline from the first 20 raw frames after each LOAD;
- 180 tactile features: corrected left/right pressures followed by their
  differences from the oldest frame in the history;
- 100 predicted action rows at 30 Hz, ordered as left arm, right arm,
  left hand, right hand (7 + 7 + 20 + 20 joints).

Keep hands unloaded during baseline collection. An incomplete or stale
history fails inference. UNLOAD and LOAD start a new baseline collection.
The engine uses only observation subscriptions; the standard Main runtime
owns command publication and the simulation/robot mode gate.

The SH5 decoder preserves the checkpoint's joint bounds, 16-row arm warm-start
ramp and right tactile persistence conditioning. By default the engine sends
the first 20 of its 100 predicted rows to the common runtime before replanning
(`VITACFORMER_EXECUTION_HORIZON`, range 1–100). Keep action playback at 30 Hz;
the model's trained time base is independent of its forward-pass frequency.

This integration uses upstream lifecycle and action processing. Local Cycle
Home commands, experimental command-safety tuning, recording/conversion
changes and other policy experiments are outside this backend. Existing
experimental deployment behavior therefore requires separate validation
before using this branch for a physical task.

## Verification

After building the image, run this from the repository root on the Docker host:

```bash
python3 cyclo_brain/policy/vitacformer/scripts/check_image.py --gpu
```

This creates and removes a dedicated test container with `--network none`,
private IPC, an empty workspace/HF cache and only read-only Cyclo source
mounts. A temporary loopback-only Zenoh router runs inside that container.
It verifies CUDA computation, the actual SH5 sensor subscriptions and message
definitions, EngineCommand ping and an InferenceCommand response. It does not
mount `/dev`, access the host's robot network, LOAD a policy or publish commands.
Omit `--gpu` for a CPU-only check on AMD64. This check also works while
`ai_worker` is completely absent; no existing robot services need to be stopped.

To additionally load a supported run directory and check a synthetic forward
pass, add `--checkpoint /absolute/path/to/run`. The checkpoint is mounted
read-only; this still does not consume live sensor data or start robot motion.

The policy tests use the real upstream RobotClient with transport doubles;
they do not connect to robot topics:

```bash
python3 -m pytest cyclo_brain/policy/vitacformer/tests
```

They require PyTorch, torchvision, NumPy, OpenCV, PyYAML and pytest; the CI
workflow installs these dependencies. Docker/supervisor and UI tests cover
the dedicated backend, mounts, model selection and provisioning fallback.

To check a real checkpoint without sensor subscriptions or command publishing,
run inside the policy environment:

```bash
python3 -m vitacformer_engine.check_checkpoint /workspace/model/vitacformer/run --device cpu
```

This performs a synthetic forward pass and prints the output shape. It does
not validate live sensor timing, GPU latency or physical robot behavior.
Source attribution is recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

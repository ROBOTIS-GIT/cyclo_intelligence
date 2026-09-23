# LingBot-VLA 2.0

Independent Cyclo Worker for [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2).
This is **not** LingBot-VA. Upstream is pinned by the submodule gitlink at
`ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8`; its source is not patched.

## Scope

- Catalog ID: `lingbot_vla:v2`; label: `LingBot-VLA 2.0`.
- Service/container: `lingbot_vla` / `lingbot_vla_server`.
- Checkpoint root: `/workspace/model/lingbot_vla`.
- NVIDIA AMD64 environment. ARM64 is not implemented or validated.
- Current observations, full action chunks, Async/Sync through the existing Runtime.
  No new ROS messages, command publishers, websocket inference service, or control loop.
- Joint-position/Twist channels only, mapped by training channel names. Whole command
  groups may be omitted; partial groups, unknown names, and synthesized values are rejected.
  Cartesian/IK execution, external history, RTC, and latent-memory feedback are not implemented.

The embedded official inference class owns image resizing, canonical feature padding,
normalization, tokenization, and relative-to-absolute action restoration. Cyclo supplies
RGB uint8 HWC images after the robot configuration's rotation and raw named state values.
Images from different cameras may have different sizes before upstream resizes them.
Do not add another resize/normalization stage without matching training preprocessing.
Only required topics are subscribed; named joint views prevent reliance on JointState array order.

`configs/inference.yaml` is read on LOAD. FP32 is the default to match upstream evaluation;
BF16 is an explicit opt-in that needs model-quality validation. CUDA warmup happens during
LOAD and never publishes robot commands. Clear releases subscriptions/model references.

## Checkpoint Bundle

A raw pretrained model is not a robot-specific deployment checkpoint. Fine-tune using
upstream's training pipeline, explicitly defining the canonical robot mapping and
normalization statistics. In particular, do not guess that a robot's lift is the same
physical quantity as a pretrained waist channel. No fixed FFW mapping is built into this Worker.

Use the **exact mapping and statistics used during training**, plus ordered raw robot
channel names. Export metadata is a JSON object:

```json
{
  "robot_type": "your_robot_type",
  "state_key": "observation.state",
  "state_names": ["joint_1", "joint_2"],
  "action_key": "action",
  "action_names": ["joint_1", "joint_2"],
  "cameras": [
    "observation.images.rgb.cam_left_head",
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_right_wrist"
  ]
}
```

Names must describe the raw training vectors, in their exact order, not the padded
canonical vectors. State and action dimensions do not have to match. Camera names are
resolved using Cyclo's existing camera resolver; the names above are examples only.
The first adapter supports one raw state vector and one raw action vector with explicit
`origin_keys` start/end slices. Every source channel must be covered exactly once.
Relative joint actions require matching named state slices and explicit `relative_type: null`
in the training robot mapping. Upstream otherwise defaults to quaternion-relative poses;
Cyclo does not silently change that training choice. Additional upstream geometric
conversions are rejected instead of silently treating Cartesian outputs as joints.

```bash
python3 cyclo_brain/policy/lingbot_vla/scripts/export_checkpoint.py \
  --weights /path/to/run/checkpoints/step_10000/hf_ckpt \
  --training-config /path/to/run/lingbotvla_cli.yaml \
  --robot-config /path/to/training/robot_config.yaml \
  --norm-stats /path/to/training/norm_stats.json \
  --metadata /path/to/cyclo_input_metadata.json \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --output docker/workspace/model/lingbot_vla/my_checkpoint
```

The exporter copies weights without modifying the training checkpoint, refuses to overwrite
an existing destination, and validates a temporary bundle before renaming it into place.
`--base-model` replaces a training-server-only tokenizer path in the exported copy; it must
refer to the same backbone/processor used for training. It does not convert incompatible weights.
The base processor assets must be cached/downloadable or mounted on the deployment machine.

```text
my_checkpoint/
  lingbotvla_cli.yaml
  robot_config.yaml
  norm_stats.json
  cyclo_input_metadata.json
  checkpoints/export/hf_ckpt/*.safetensors
```

This layout preserves the native loader's lookup of `lingbotvla_cli.yaml` three directories
above `hf_ckpt`. The adapter binds the exported FeatureTransform directly instead of using
upstream's CWD-relative `reset(robot_name)` file lookup. Missing stats and stale paths fail
LOAD; there are no identity-statistics or default-robot fallbacks.

## Deployment

```bash
git submodule update --init cyclo_brain/policy/lingbot_vla/lingbot-vla-v2
./docker/container.sh start-lingbot-vla --build
```

The inference Docker image isolates upstream Torch 2.8 / Transformers 4.57.3 from LeRobot/RLDX.
Upstream queries CUDA capability during import; build without a GPU, then run import
and native-transform checks in a GPU-enabled container. No GPU-name monkey patches are used.
It does not install the optional depth-teacher training stack, whose MLflow/PyArrow pins conflict.
FlashAttention compilation defaults to SM120; set `LINGBOT_VLA_FLASH_ATTN_CUDA_ARCHS`
for a different supported NVIDIA architecture before building. Cyclo-owned engine/config
files are bind-mounted; restart this Worker after edits while inference is stopped.

The main Cyclo installation must discover the new manifest (rebuild its image for packaged
deployments, or restart the source-mounted installation). Rebuild UI assets to expose
the Hugging Face upload/download backend. Inference and BT lists use the dynamic Catalog.
Select the exported bundle root, not its inner `hf_ckpt`, as Policy Path.

Upstream code and the published model card declare Apache-2.0. Review separately downloaded
backbones, training teachers, dependencies, and datasets under their own licenses.

## Verification

```bash
python3 -m pytest cyclo_brain/policy/lingbot_vla/tests
```

Tests cover metadata, exact channel coverage/order, relative-action prerequisites,
different camera sizes, action shape/finiteness, lifecycle resets, stale inputs, export
roundtrips, and Catalog/Docker integration. The lifecycle test mocks the large model and
RobotClient. Passing these tests is not a claim of real checkpoint quality or robot safety.
Robot-specific fine-tuning and physical task success require separate validation.

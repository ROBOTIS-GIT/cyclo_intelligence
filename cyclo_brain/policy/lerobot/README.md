# LeRobot Worker

This directory provides the LeRobot Engine-only Worker used by Cyclo.
Training remains available through the LeRobot CLI, while inference lifecycle,
action processing, safety, and robot command publication are owned by the
central Policy Runtime in `cyclo_intelligence`.

## Layout

```text
policy/lerobot/
├── manifest.yaml              # policies and runtime capabilities
├── lerobot/                   # ROBOTIS LeRobot fork submodule
├── lerobot_engine/            # InferenceEngine adapter
├── configs/image_preprocessing/ # editable per-policy spatial transforms
├── Dockerfile.amd64
├── Dockerfile.arm64
└── tests/
```

The container starts only the `engine-process` s6 service. It hosts
`/lerobot/engine_command`, publishes `/lerobot/worker_heartbeat`, subscribes
to the observations required by the selected checkpoint, and returns action
chunks. It does not own `ControlLoop` and does not publish robot commands.

## Build And Start

From the repository root:

```bash
./docker/container.sh start-policy lerobot --build
./docker/container.sh enter-policy lerobot
```

The compatibility commands remain available:

```bash
./docker/container.sh start-lerobot --build
./docker/container.sh enter-lerobot
```

The container executes files baked into the image. Use `--build` after Worker
source or dependency changes.

## Models And Data

The shared host directory `docker/workspace` is mounted at `/workspace`.
LeRobot checkpoints are selected under `/workspace/model/lerobot`. The list of
policies shown by Cyclo comes from `manifest.yaml`; the Worker `DESCRIBE`
response must match that manifest before LOAD is accepted.

Inference is called externally through `/policy/inference_command` with a
namespaced policy ID such as `lerobot:act`. `/lerobot/inference_command` is a
temporary compatibility alias hosted by the central Runtime, not by this
Worker.

## ROS And Zenoh

Edit the Cyclo block near the top of the Worker's `/root/.bashrc`, then restart
the container so s6 starts `engine-process` with the new settings:

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
```

For a remote router, use the commented client endpoint example already present
in the image bashrc. `docker restart` preserves a container-local edit;
recreate/update resets it to the image default.

## Training

Enter the Worker and use the version-pinned LeRobot CLI. For example:

```bash
lerobot-train \
  --dataset.repo_id=<dataset_repo_id> \
  --dataset.root=/workspace/lerobot/<dataset_folder> \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/workspace/model/lerobot/<output_folder>
```

Training output placed below `/workspace/model/lerobot` is immediately visible
to the Cyclo model browser.

## Camera Preprocessing

Edit `cyclo_brain/policy/lerobot/configs/image_preprocessing/<policy_type>.yaml`.
The type comes from the checkpoint's `config.json`, not the UI label. Compose
mounts this directory read-only at `/app/configs/image_preprocessing`; standalone
images contain the same defaults. No additional file is required in model folders.

Processing order:

```text
RGB camera -> robot-config rotation -> Cyclo YAML spatial operations
-> saved LeRobot preprocessor -> original policy -> saved postprocessor
```

`identity` skips ONLY the additional Cyclo spatial operations. Rotation and
conversion to batched RGB float32 CHW still happen. Model-internal transforms
are never disabled or rewritten. Checkpoint dimensions alone do not describe
the interpolation/crop used to create the training dataset.

| YAML | Shipped Cyclo operation | Basis / limitation |
| --- | --- | --- |
| ACT | identity | Separate camera backbones; external training transforms must be configured explicitly. |
| SmolVLA, Pi0, Pi0-FAST, Pi0.5 | identity | Preserve native aspect ratio for policy-owned resize/padding. |
| XVLA | identity | Training and inference share internal resize + padding when checkpoint `resize_imgs_with_padding` is set. With `None`, both require equal camera sizes before stacking; reproduce any external training transform in YAML. |
| MolmoAct2 | identity | Retain the checkpoint image processor. |
| FastWAM | identity | Retain internal per-view resize/concatenation. |
| Diffusion | OpenCV uint8 bilinear to each checkpoint feature size | Preserves previous Cyclo behavior to align cameras before stacking. This is NOT a verified training interpolation default. |
| VLA-JEPA | identity | Standard training has no external area resize. Saved processor/model transforms, including an explicitly configured inference resize_images_to, remain active. |

Only if a VLA-JEPA checkpoint was trained with an additional Torch area
resize to 224x224, replace `identity` in `vla_jepa.yaml` with
`{type: resize, size: [224, 224], interpolation: area}` and keep `backend: torch`.
That is a checkpoint-specific training recipe, not the shipped default.

To reproduce a training pipeline that used Torch bilinear antialiased resize:

```yaml
backend: torch
operations:
  - type: resize
    size: [224, 224] # [height, width], NOT OpenCV's (width, height)
    interpolation: bilinear
    antialias: true
```

Supported operations are `identity`, `resize`, `center_crop`, and `letterbox`.
Operations execute in list order. `size: checkpoint` explicitly selects each
camera's input-feature height/width instead of a literal size. It does not infer
an interpolation method. Resize/letterbox require `interpolation` (`nearest`,
`bilinear`, `bicubic`, or `area`). Torch bilinear/bicubic use
`align_corners=False`; `antialias` defaults to false and is a Torch-only option.

`backend: opencv` operates on RGB uint8, then converts to float32 / 255.
`backend: torch` converts to float32 / 255 BEFORE spatial operations and does
not quantize back to uint8 or clamp bicubic overshoot. These are intentionally
different numeric pipelines, not interchangeable names for the same resize.

`center_crop` rejects targets larger than the input and rounds the half-offsets
like torchvision CenterCrop. `letterbox` fits with `min(target_h/h, target_w/w)`, rounds the resized
dimensions, and pads using `placement: center` (default) or `top_left` (image
anchored top-left, padding on bottom/right). `fill` is an integer RGB gray level
from 0 to 255, default 0. This optional Cyclo operation is NOT a replacement
for policy-specific padding implementations.

Optional camera overrides use exact checkpoint keys and REPLACE the default
operations for that camera:

```yaml
cameras:
  observation.images.rgb.cam_left_wrist:
    - type: center_crop
      size: [200, 200]
    - type: resize
      size: [224, 224]
      interpolation: area
```

One file applies to all checkpoints of that policy type. If their training
recipes differ, update the YAML before loading each checkpoint. Unknown keys,
missing files, invalid sizes/options, and duplicate YAML keys fail LOAD rather
than silently choosing a transform. Resolved operations are logged on LOAD.

After initially building/recreating the LeRobot Worker with these changes,
YAML-only edits require **Clear/UNLOAD then LOAD**, not an image rebuild. Edits
do not affect a running session; START/RESUME alone does not reload YAML. Even
a cached-model LOAD reads a new snapshot. The original checkpoint files are
never modified. Rebuilding only Cyclo does not update this Worker adapter.
The new mount is also part of Cyclo's baked Supervisor Compose definition:
update that Cyclo image before using UI-driven Worker recreate/update, otherwise
an old Supervisor can recreate the Worker without the editable YAML mount.
Supervisor checks the YAML mount source against Compose as well as its destination,
so a Worker using another checkout's config directory is marked stale. Editing
the contents of the correctly mounted YAML does not require container recreation.

Compare the SAME decoded RGB frame through training and inference transforms,
including rotation, dtype conversion, interpolation, antialias, padding/crop,
and the saved processor/model. Equal shapes alone are not a parity test.
External training hooks are not automatically serialized into saved processors.
Diffusion temporal observation history remains outside this change: matching
image sizes does not fix an `n_obs_steps` mismatch.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  cyclo_brain/policy/lerobot/tests \
  cyclo_brain/policy/lerobot/lerobot_engine/tests
```

Container readiness requires `engine-process` and its ready marker. Cyclo's
Supervisor additionally verifies fresh heartbeat and a compatible `DESCRIBE`
response before reporting `Backend ready`.

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [Cyclo architecture](../../docs/architecture.html)

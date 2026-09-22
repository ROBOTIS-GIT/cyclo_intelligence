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
├── configs/inference_inputs/ # user settings for additional preprocessing only
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

Compose mounts Cyclo's adapter, common runtime, SDK, and configuration sources
from this checkout. After the initial container recreation, Python edits require
a Worker process restart, not an image rebuild. Dependencies and the upstream
LeRobot installation remain image-owned; use `--build` when those change.
See [source-mounted containers](../../../docker/README.md) for application steps.

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

## State And Action Channels

The optional `cyclo_channels` training adapter selects actual dataset channels
and exports `cyclo_io_mapping.json` beside each checkpoint's `config.json`,
including intermediate checkpoints and Hub uploads. Without the adapter,
training remains unchanged. The mapping records the ordered external channels:

```json
{
  "version": 1,
  "state_names": ["joint_a", "joint_b", "head_joint"],
  "action_names": ["joint_b", "joint_a"],
  "dataset": {"repo_id": "owner/dataset", "revision": null}
}
```

Names must match the robot configuration's joint names (or velocity channels
such as `linear_x`, `linear_y`, `angular_z`). Input and output lists are
independent. Cyclo selects state before normalization, then rearranges actions
after the saved postprocessor into command-group order. Camera and temporal
preprocessing remain in the existing YAML; channel lists are not duplicated there.
Name matching does not convert units, coordinate frames, or action representations.

Complete omitted command groups receive no inference commands, including during
Slow Start and inference Stop. Selecting only part of a command group is rejected
at LOAD. Saved Pose remains a separate feature. Relative-action processors that
require matching state/action prefixes retain that constraint. Native GR00T
relative groups are checked against their saved state-group references; EEF or
additional relative action transforms require a separate contract and are rejected.

Training selection uses a separate YAML with `state_names` and `action_names`:
each is an ordered list of dataset channel names or `all`. Both are required.

```bash
lerobot-train \
  --training_adapter.name=cyclo_channels \
  --training_adapter.config_path=/workspace/channels.yaml \
  --policy.type=act --dataset.repo_id=owner/dataset \
  --policy.push_to_hub=false
```

This selects the last tensor axis and corresponding features/statistics without
rewriting the dataset. Names are mandatory; invalid or contradictory metadata
fails before model construction. The resolved selection is saved for portable
resume even without the original YAML. Changed selections cannot resume a run.
The former `io_mapping_path` label-only option is not a selection option and is
no longer supported. See [Training Extension](extensions/README.md) for installation,
supported paths, resume commands and migration details.

Existing or externally trained models do not require retraining. Generate a
draft from the dataset actually used for training:

```bash
cyclo-io-mapping \
  --checkpoint /workspace/model/lerobot/owner/model \
  --dataset-info /workspace/lerobot/owner/dataset/meta/info.json
```

Check the actual training selection/order, then repeat with `--write` to add the
file. Use `--mapping /path/to/verified.json` instead of `--dataset-info` when
training changed the dataset layout. Existing files are never overwritten.
The tool does not modify weights, processor statistics, or model configuration.

Without this file, only exact state/action dimension matches are accepted using
legacy robot order, with a warning; equal sizes alone do not prove semantic
compatibility. There is no implicit truncation or zero-padding. Invalid mapping
files never fall back to legacy mode. Clear and LOAD reload the mapping even
when weights are cached. Install the updated LeRobot generic hooks and external
`cyclo-lerobot-io` package together. Rebuild the LeRobot image once for these
installation changes; subsequent extension edits use the read-only source mount.

## Inference Inputs And Camera Preprocessing

Edit `cyclo_brain/policy/lerobot/configs/inference_inputs/<policy_type>.yaml`.
The type comes from the checkpoint's `config.json`, not the UI label. Compose
mounts this directory read-only at `/app/configs/inference_inputs`; standalone
images contain the same defaults. No additional file is required in model folders.

The YAML declares only additional Cyclo preprocessing. With no extra transforms:

```yaml
preprocessing: identity
```

Input keys, tensor packaging, processor stages and the internal graph are owned
by Python, not user YAML. See the [configuration guide](configs/inference_inputs/README.md)
for ordered transforms, camera overrides and registered custom handlers.
The [developer guide](../../docs/inference_input_design.md) covers temporal inputs,
execution feedback and memory contracts inside those handlers.

Default processing order (adapter-owned):

```text
RGB camera -> rotation -> spatial/tensor conversion -> device placement
-> saved LeRobot preprocessor -> after graph -> original policy
-> saved postprocessor -> optional result graph
```

`identity` skips additional Cyclo preprocessing. Robot-config rotation and basic
conversion to batched RGB float32 CHW remain part of the adapter. Model-internal transforms
are never disabled or rewritten. Checkpoint dimensions alone do not describe
the interpolation/crop used to create the training dataset.

| YAML | Shipped Cyclo operation | Basis / limitation |
| --- | --- | --- |
| ACT | identity | Separate camera backbones; external training transforms must be configured explicitly. |
| SmolVLA, Pi0, Pi0.5 | identity | Preserve native aspect ratio for policy-owned resize/padding. |
| XVLA | identity | Training and inference share internal resize + padding when checkpoint `resize_imgs_with_padding` is set. With `None`, both require equal camera sizes before stacking; reproduce any external training transform in YAML. |
| MolmoAct2 | identity | Retain the checkpoint image processor. |
| FastWAM | identity | Retain internal per-view resize/concatenation. |
| Diffusion | OpenCV uint8 bilinear to each checkpoint feature size | Preserves previous Cyclo behavior to align cameras before stacking. This is NOT a verified training interpolation default. |
| VLA-JEPA | identity | Standard training has no external area resize. Saved processor/model transforms, including an explicitly configured inference resize_images_to, remain active. |

Only if a VLA-JEPA checkpoint was trained with an additional Torch area
resize to 224x224, declare a resize under `preprocessing.images` with
`size: [224, 224]`, `interpolation: area`, and `backend: torch`.
That is a checkpoint-specific training recipe, not the shipped default.

To reproduce a training pipeline that used Torch bilinear antialiased resize:

```yaml
preprocessing:
  images:
    - resize:
        size: [224, 224] # [height, width], NOT OpenCV's (width, height)
        backend: torch
        interpolation: bilinear
        antialias: true
```

Use `images: identity` (or `[]`) for no spatial transforms. Supported list operations
are `resize`, `center_crop`, and `letterbox`.
Operations execute in list order. `size: checkpoint` explicitly selects each
camera's input-feature height/width instead of a literal size. It does not infer
an interpolation method. Resize/letterbox require `interpolation` (`nearest`,
`bilinear`, `bicubic`, or `area`). Torch bilinear/bicubic use
`align_corners=False`; `antialias` defaults to false and is a Torch-only option.

OpenCV operates on RGB uint8. Torch operates on float32 / 255. The adapter makes
that conversion once, before the first Torch operation (or at the end of an
OpenCV-only sequence). Torch followed by OpenCV is rejected, not silently
quantized or reordered. No tensor or graph wiring needs to be edited in YAML.

`center_crop` rejects targets larger than the input and rounds the half-offsets
like torchvision CenterCrop. `letterbox` fits with `min(target_h/h, target_w/w)`, rounds the resized
dimensions, and pads using `placement: center` (default) or `top_left` (image
anchored top-left, padding on bottom/right). `fill` is an integer RGB gray level
from 0 to 255, default 0. This optional Cyclo operation is NOT a replacement
for policy-specific padding implementations.

Optional camera overrides use exact checkpoint keys and REPLACE the default
operations for that camera:

```yaml
preprocessing:
  cameras:
    observation.images.rgb.cam_left_wrist:
      - center_crop:
          size: [200, 200]
          backend: torch
      - resize:
          size: [224, 224]
          backend: torch
          interpolation: area
```

One file applies to all checkpoints of that policy type. If their training
recipes differ, update the YAML before loading each checkpoint. Unknown keys,
missing files, invalid sizes/options, and duplicate YAML keys fail LOAD rather
than silently choosing a transform. The selected configuration path is logged on LOAD.

After initially building/recreating the LeRobot Worker with these changes,
YAML-only edits require **Clear/UNLOAD then LOAD**, not an image rebuild. Edits
do not affect a running session; START/RESUME alone does not reload YAML. Even
a cached-model LOAD reads a new snapshot. The original checkpoint files are
never modified. The Worker adapter is now source-mounted as well, but running
Python processes still need a restart after code edits.
Apply the source-mounted Cyclo image layout once so Supervisor reads this
checkout's Compose definition. Supervisor checks source/config mount paths as
well as their destinations, so a Worker using another checkout is marked stale.
Editing files inside a correctly mounted directory does not require recreation.

Compare the SAME decoded RGB frame through training and inference transforms,
including rotation, dtype conversion, interpolation, antialias, padding/crop,
and the saved processor/model. Equal shapes alone are not a parity test.
External training hooks are not automatically serialized into saved processors.
Image sizing and temporal history are separate: matching image sizes does not
fix an `n_obs_steps` mismatch. See the online Diffusion adapter below.

## Additional Policies

### Diffusion Online Execution

Diffusion uses the same publication-paced public-step infrastructure as
Multi-Task DiT. Each request supplies one current observation to the saved
processor and `select_action()`. LeRobot owns both its `n_obs_steps` observation
queue and its `n_action_steps` action queue; it calls the neural model only when
the latter is empty. Cyclo does not populate private model queues or duplicate
them in a callback history buffer. Initial observation repetition is LeRobot's
documented bootstrap, not fabricated robot reception by Cyclo.

This replaces the broken latest-only `predict_action_chunk()` call, which lacked
the temporal state axis. It does not alter ACT or other existing chunk adapters.
Diffusion now bypasses Cyclo chunk interpolation/alignment and async prefetch.
Use the training FPS for Dataset FPS; it sets a minimum step period, not a
guarantee that network, camera and inference latency achieve that frequency.
Only successfully published steps advance the model. Preview-only is unsupported;
a simulator receiving real command topics is supported by the publication contract.
Stop/reset and cached LOAD clear the policy queues without reloading weights.

Saved processors still run before/after the public model API. Mixed camera sizes
are checked after preprocessing and before the policy stacks images. Enabled
LeRobot relative-action processors are rejected at LOAD: reanchoring cached
actions against every new observation is not a valid chunk-relative execution
contract. Do not remove a training transform to bypass that error.

See [input design decision](../../docs/inference_input_design.md) for the choice
between policy-owned queues, explicit temporal input plans and execution feedback.

### Multi-Task DiT

Select `Multi-Task DiT` (`lerobot:multi_task_dit`) in Inference or BT. It uses
the existing LeRobot Worker and a publication-paced step adapter around the
model's public `select_action` API. The model owns its observation/action queues;
step execution bypasses chunk interpolation and asynchronous prefetch. Actual
command publication is required; preview-only execution is unsupported.

The current `multi_task_dit.yaml` is the test preset for the local 1,000-step
checkpoint: external Torch bilinear resize to 224x224 with antialias enabled.
It is not a universal model default. For that checkpoint, use Dataset FPS 30 and
instruction `pick up the bottle and place it into basket`. Its state/action
dimensions are both 22. Checkpoint path inside the standard workspace mount:

```text
/workspace/inference_context_campaign_20260910/models/multi_task_dit
```

Match preprocessing and state/action ordering before selecting another checkpoint.
LingBot-VA remains excluded pending action-space integration.

### WALL-X And GR00T

WALL-X and GR00T N1.7 (LeRobot) use the same LeRobot
Worker. Model settings come from the checkpoint, not new UI overrides.
Their dependency extras are installed on both AMD64 and ARM64; spatial
preprocessing defaults preserve RGB sizes and leave model-owned transforms active.

`lerobot:groot` loads **LeRobot-format** checkpoints (`config.json` with
`type: groot`), saved pre/post processors, statistics and the referenced base
model assets. It is separate from `groot:n17` in `groot_server`, including its
TensorRT options. The legacy `groot` alias still selects that independent Worker.
Missing processor files, incompatible embodiment metadata and observation history
beyond one frame are rejected, not reconstructed using generic defaults. Raw
NVIDIA checkpoints are not automatically converted to LeRobot checkpoints.
N1.7 requires equal camera H/W at its pack step, not necessarily at Cyclo input.
A saved `image_crop_resize_processor` before packing can align different raw
sizes. Keep `groot.yaml` at identity in that case; the saved processor owns the
transform. Use YAML only to reproduce training transforms not saved in that
pipeline. Cyclo does not invent an automatic resize recipe.

WALL-X's pinned core has a fixed 20-dimensional state/action limit. Cyclo checks
each dimension independently and requires the effective robot layout to match
the checkpoint. A 22-channel robot can run a mapped 16-channel checkpoint when
the selected action groups are complete; an actual 22-channel WALL-X input or
output is still rejected rather than truncated.
Matching dimensions alone does not prove semantic joint ordering compatibility.

EO1, Evo1 and Pi0-FAST are excluded from Cyclo's inference Catalog and its
preprocessing presets. Saved selections for these models are unavailable; select
a supported model and matching checkpoint. Pi0 and Pi0.5 remain supported.
The pinned upstream repository and existing user checkpoints are unchanged.

New policy integration tests do not prove trained-checkpoint or robot performance.
No new-model weights are downloaded by the test suite. Real dependency checks
run separately from mocked Engine tests inside the candidate LeRobot environment:

```bash
CYCLO_TEST_POLICY_DEPENDENCIES=1 python -m pytest \
  cyclo_brain/policy/lerobot/tests/test_new_policy_processors.py
```

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

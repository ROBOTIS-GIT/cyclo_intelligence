# ABot-M0 Worker

Independent backend for the existing Cyclo Engine protocol, not a LeRobot policy
or a second WebSocket server. Policy ID: `abot:m0`; runtime: `abot`;
checkpoint root: `/workspace/model/abot`; container: `abot_server`.

Upstream: [ABot-M0 branch](https://github.com/amap-cvlab/ABot-Manipulation/tree/ABot-M0),
pinned by the submodule to `ee9013cd8341d5424aa722a8cf25e8985a6e673b`.
The main branch contains M0.5, a different model. Do not update the submodule to
main to upgrade M0.

## Scope

- Current RGB cameras, instruction, and optionally one current state vector.
- Strict full-state-dict loading; optional VGGT follows the saved model config.
- Absolute action chunks, reordered by exported channel names into the robot
  layout. All robot action channels must be covered; no silent truncation or
  fabricated commands for omitted channels.
- Explicit state padding/action column selection and per-channel normalization.
- Existing async/sync chunk execution, Stop, Clear and Slow Start are unchanged.
- No observation history, delta/relative action accumulation, action ensemble,
  end-effector-to-joint IK, or LoRA-only checkpoint loading in this adapter.
- AMD64 NVIDIA only. ARM64 deployment is not implemented.

An existing benchmark checkpoint is not sufficient for a new robot: fine-tune
for its channels and export the corresponding training-time metadata first.
This integration does not start training, download large weights, or operate a
robot. Unit tests mock the large policy; they do not establish task performance.

## Build

```bash
git submodule update --init cyclo_brain/policy/abot/ABot-Manipulation
./docker/container.sh start-abot --build
```

The default FlashAttention compilation target is RTX 5090 (SM120). Override
`ABOT_FLASH_ATTN_CUDA_ARCHS` for another supported GPU. The image uses Python 3.10,
Torch 2.8/CUDA 12.8, and the inference dependencies listed in
`requirements-inference.txt`; it does not install upstream's training-only
DeepSpeed, dataset tools or Torch 2.6 pin. No LeRobot image change is needed.

The engine, configuration and common runtime are bind-mounted like RLDX.
Restart the Worker after engine/config edits; rebuild for dependencies or
upstream code. Refresh the main Cyclo catalog/runtime to expose the new policy;
update the UI build for the Hugging Face backend option. Existing processes are
not automatically restarted by a source edit.

## Checkpoint Bundle

```text
my_run/
  config.yaml
  dataset_statistics.json
  cyclo_input_metadata.json
  checkpoints/model.pt
```

UI Policy Path selects `my_run`, not the `.pt` file. Configuration and statistics
come from the same training run as the selected checkpoint. No directory scan
chooses a newest checkpoint or a first statistics entry implicitly.

The metadata must be exported from the actual training channel ordering and
transforms. This small example is a schema illustration, NOT an FFW preset:

```json
{
  "policy_id": "abot:m0",
  "robot_type": "your_robot_type",
  "checkpoint": "checkpoints/model.pt",
  "cameras": ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"],
  "include_state": true,
  "state_names": ["joint_b", "joint_a"],
  "state_indices": [0, 1],
  "state_normalization": ["min_max", "min_max"],
  "action_names": ["joint_b", "joint_a"],
  "action_indices": [0, 1],
  "action_normalization": ["min_max", "min_max"],
  "statistics_key": "your_training_statistics_key",
  "action_mode": "absolute",
  "observation_offsets": [0]
}
```

- Names are ordered physical channels, in the same order as the concatenated
  statistics vectors. They are never inferred from dimension alone.
- Indices explicitly map physical channels to model columns, allowing models
  with padded dimensions. State padding is zero AFTER normalization; unused
  action columns are discarded only where the export explicitly declares them.
- `include_state: false` requires empty state names, indices and normalization
  lists and must match training `datasets.vla_data.include_state`.
- Modes are `identity`, `min_max`, `q99`, `mean_std`. Statistics must have one
  value per named physical channel. Constant-channel behavior follows upstream
  `Normalizer`. No blanket action clipping or gripper binarization is added.
  Binary/sin-cos/relative representations require a separately validated adapter.
- Camera feature names use RobotClient's existing camera alias resolver. Only
  these cameras and required state sources are subscribed. No FFW camera names,
  joint counts, or benchmark-specific reorder indices are hardcoded.

```bash
python3 cyclo_brain/policy/abot/scripts/export_checkpoint.py \
  --checkpoint /path/to/run/checkpoints/step_pytorch_model.pt \
  --run-dir /path/to/run \
  --metadata /path/to/reviewed_cyclo_input_metadata.json \
  --output docker/workspace/model/abot/my_run
```

Exporter needs NumPy, OpenCV and PyYAML, but not Torch. It copies files and refuses
to replace existing output. It never changes training weights or source configs.

## Input and Assets

`configs/inference.yaml` applies no extra resize. Upstream
`ABot_M0.predict_action` resizes according to saved `datasets.vla_data.image_size`
before Qwen/VGGT transforms. Camera rotation follows the robot config and must
match dataset conversion. State normalization is performed once in the adapter;
the public model API returns normalized actions which are decoded once here.

Upstream constructs Qwen from `framework.qwenvl.base_vlm` before loading the full
checkpoint. Keep that asset available (local or Hub), including its processor.
`ABOT_BASE_VLM` in the Worker environment can relocate it without changing the
checkpoint. If `use_vggt` is enabled, upstream also loads `facebook/VGGT-1B`;
prefetch its assets into the shared HF cache for offline use. These assets are
not bundled into the Docker image. LOAD warms up the policy without publishing
any commands. Missing/stale inputs or nonfinite output fail the request.

## Licensing and Validation

The pinned M0 branch's `pyproject.toml` declares an MIT classifier but references
a missing LICENSE file. M0.5 main's Apache-2.0 file is not evidence of the license
for this pinned M0 source. Clarify M0 code/weight permissions with the authors
before redistribution or commercial deployment. Qwen and optional VGGT assets
also have their own terms. This repository does not relabel upstream licenses.

```bash
python3 -m pytest cyclo_brain/policy/abot/tests
```

Tests cover metadata validation, named channel mapping, normalization, camera
rotation, stale input rejection, LOAD/Clear, and checkpoint export. The model is
mocked; actual checkpoint inference, deployment GPU latency, and physical robot
performance require separate validation. Upstream dataset transforms are
robot-specific and must match the exported metadata, not be copied blindly
from the RoboTwin example.

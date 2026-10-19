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

## Native Training Adapter

`abot_training/` is a Cyclo-owned adapter with a separate locked training image
and validate/train/reload wrapper. It is not yet an enabled Bench trainer.
No upstream files or inference services are changed. See
[the Bench handoff](training/BENCH_HANDOFF.md) for the build, Run configuration,
channel contract and publication gates. Qualification results must name the
tested image and commit; unit tests alone do not enable a training Run.

The implemented input contract accepts prepared, train-only LeRobot v2.1 data.
Bench owns dataset version inspection and conversion. The adapter validates
all ordered state/action names against `meta/info.json`, reads current RGB and
state with absolute action targets, and excludes episode-ending start positions
that cannot supply a complete chunk. It does not subtract state or interpret
channel names as physical units. Statistics are computed from the training
vectors and are supplied explicitly on reload; quantiles use the full vectors,
not averaged episode quantiles. Quantile calculation therefore needs memory
proportional to the numeric dataset, not video size.

### Selecting Training Channels

The robot profile describes the full raw dataset, not the subset to train.
Pass a separate `channel_selection` object to `PreparedDataset`, or load that
object with `abot_training.channels.read_selection(path)` from YAML. The keys
match Cyclo's LeRobot adapter. For example, a dataset containing these channels
can use only the two gripper states while retaining every action target:

```yaml
state_names: [gripper_l_joint1, gripper_r_joint1]
action_names: all
```

Both fields are required when supplying a selection. Each accepts `all` or a
nonempty list of unique names. An omitted selection uses all channels. Explicit
lists preserve their specified order, independently for state and action;
unknown names, duplicates and empty lists fail. Selection is applied before
statistics and normalization, without modifying the source dataset or profile.

Profile padding dimensions may be `selected` to derive model dimensions from
the selection, or positive integers at least as large as the selected vectors.
Use `dataset.configure_model(config)` before constructing the model and loading
pretrained weights. No robot channel names or dimensions are built into the
loader. The Task_000759 profile uses `selected` for both dimensions.

`dataset.checkpoint_metadata(run_config, model_config, initialization)` records
the source and selected channels in `train_config.json`, the selected names in
`cyclo_io_mapping.json`, and the actual pretrained initialization report in both
`train_config.json` and `initialization.json`. Statistics include their channel
order, normalization and dataset-metadata hash. `PreparedDataset.from_checkpoint`
verifies checksums and restores this input contract; it rejects mismatched
mapping, model dimensions or initialization metadata. This helper is not a
full-model reload or optimizer-resume implementation.

Selecting fewer action channels for training does not authorize partial robot
commands. The current inference adapter still requires all robot action channels
to be covered by an explicitly exported inference bundle. No omitted channel is
silently replaced with zero or the current pose.

Bench integration remains pending: its current native submission validator
rejects partial selections and requires integer padding. It must accept this
selection contract and `selected` dimensions before enabling ABot runs. Its
channel resolver currently uses dataset order, whereas Cyclo's LeRobot YAML
adapter and this adapter preserve explicit list order; the submitted resolved
selection and exported mapping must agree. Bench was not modified here.

Profile v2 adds these explicit Cyclo requirements to the Bench handoff:

- `camera_order`: ordered array of camera role names. JSON object ordering cannot
  specify tensor order because the profile hash sorts object keys.
- `use_vggt: false`.
- `head_initialization: reinitialize_mismatched`.
- `vocabulary_initialization: restore_checkpoint`.
- Each asset declares `provider: modelscope` or `provider: huggingface`.

Only the state/action input dimensions and action output dimensions may differ
from pretrained weights. An affected encoder/decoder is kept freshly initialized
in full. Other missing keys, unexpected keys and hidden-size changes fail.
Disabled VGGT keys are explicitly reported as discarded. With explicit
`load_pretrained(..., restore_vocabulary=True)`, the Qwen input embedding and LM
head grow to the checkpoint's row count before strict loading. Every new row is
overwritten by pretrained weights. Tokenizer IDs and special tokens must fit
the original vocabulary; no token strings are invented and the tokenizer is not
modified. Tied embeddings remain tied only when the checkpoint's input and
output tensors are exactly equal; differing tensors are rejected. Shrinking,
inconsistent row counts and unrelated shape changes fail before resizing.
Without opt-in, vocabulary mismatches fail.
Record the returned initialization report and save the expanded Qwen config for
reload. Initialize before constructing the optimizer or moving to the GPU.
No vocabulary trimming is performed; earlier peanut training's trimming is not
carried over. This implementation restores weights for continuous action
training; it does not reconstruct a discrete action tokenizer.

`PreparedDataset` returns upstream `image`, `lang`, `state`, `action`, and boolean
`action_mask` fields. Padding is zero after normalization; padded action channels
are excluded by the existing upstream masked loss. `image_size` is
`[width, height]`, matching upstream PIL resizing. Video timestamps must match
the dataset timeline within half a frame. Source data is never modified.

Checkpoint helpers save synchronously into a temporary directory and expose the
completed directory only after writing checksums. A native snapshot contains:

```text
checkpoint-<optimizer_step>/
  checkpoints/steps_<optimizer_step>_pytorch_model.pt
  processor/
  profile.json
  model_config.json
  train_config.json
  dataset_statistics.json
  cyclo_io_mapping.json
  initialization.json
  checkpoint.json
```

This is a native training snapshot, not an inference bundle or a resumable
optimizer snapshot. `save_total_limit` is the number of completed snapshots to
retain, distinct from an optimizer-step save interval. The wrapper owns the
save schedule; invoke its reload phase in a new process before publication.
Existing checkpoints are never overwritten; failed saves keep prior snapshots.

CPU tests cover profile mismatch, data/timestamp errors, camera decoding,
normalization and padding, dimension-only initialization, checksum verification,
retention, and a small native Torch checkpoint reloaded in a fresh process. The
upstream loss test runs the pinned loss implementation with small feature modules
and verifies zero gradient on padded output channels. These are not full ABot
GPU training or checkpoint reload qualification.

```bash
python -m pytest cyclo_brain/policy/abot/tests -q
```

Training tests require Torch, PyArrow, PyAV, Transformers and Diffusers in
addition to the existing lightweight inference test dependencies. They skip
optional training tests when those packages are absent; inspect skip counts.

### Initial Input Qualification

On 2026-10-19 the local prepared
`Task_000759_Move_Vegetable_Whiteroom_Leardboard_lerobot` metadata reported
`ffw_sg2_rev1`, 98 episodes, 10,578 frames, 15 FPS, 22 state channels, 22 action
channels and four RGB cameras. A read-only audit with a 16-step horizon produced
9,108 complete chunks. First/last chunk starts from every episode decoded 784
camera images successfully. This was a dataset audit with test-only asset
identities, not a deployment profile or a model training run. No audit fixture
is installed in the profile catalog. The reviewed profile
`training_profiles/ffw-sg2-task-000759-3cam-v1.json` selects left head, left wrist,
then right wrist, preserving all 22 channels in dataset order. A second audit
with that profile decoded 588 images from 196 samples. The profile now resolves
dimensions from the channel selection instead of fixed 22-dimensional padding;
its new hash and contract must be adopted by Bench. The profile's existence does
not qualify the image, pretrained assets, or GPU training.

A subsequent read-only selection audit excluded `linear_x`, `linear_y` and
`angular_z` through explicit name lists: state-only exclusion resolved to
19 state / 22 action dimensions, and exclusion from both resolved to 19 / 19.
Each case kept 9,108 chunks and decoded 588 images from 196 boundary samples.
Selected statistics and normalized vectors matched the corresponding columns
of the full-channel dataset. The updated profile canonical SHA-256 is
`7f4da847ee427f65dd4a8196b21cd27b5b1e8e3d87bd27fc1f1f9911cb204970`.

At the input-selection stage, the CPU environment passed all 104 ABot tests. The existing
Python 3.10 inference image passed 12 vocabulary tests, but skipped 26 selection
tests because it lacks PyArrow. This does not qualify a training image; its
dependency installation and full-model training/reload require separate tests.

Prior training receipts and the official repository metadata identify the base
asset as ModelScope `amap_cvlab/ABot-M0-Pretrain`, revision
`92026d40d7d03ea54b80eed636e6da1e444170cb`. Its `config.yaml` SHA-256 is
`297b0a6255c0936d46ddb42274df6291334c0daf42bc6f9f292c03acfed2bd08`, and
`checkpoints/ABot_M0_Pretrain.pt` SHA-256 is
`94478682b5c9eecf6f02179ba67ae47ea41257ca059bea6dd20e161716f5e16b`.
The prior Qwen asset is Hugging Face `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. These repository revisions were
queried successfully; the initial input audit did not download or rehash the
large base weights. Full-model qualification additionally verifies the declared
checkpoint SHA-256 before loading it. Bench asset provisioning must distinguish these two providers,
not interpret the ModelScope repository ID as a Hugging Face repository.

The base config references a Qwen `-Action` variant. Previous initialization
records show 153,984 vocabulary rows versus 151,936 in the standard Qwen asset.
Restoration is tested on small real Qwen3-VL models with Transformers 4.57.0 in
the existing ABot image, as well as the CPU development environment. Tests check
all restored tensors, unchanged head initialization and tokenizer, invalid input
rejection, serialized vocabulary size, and forward output after reload. Full-size
qualification uses the wrapper's train and fresh-process reload phases with
real data; do not enable Runs based on these CPU checks alone.

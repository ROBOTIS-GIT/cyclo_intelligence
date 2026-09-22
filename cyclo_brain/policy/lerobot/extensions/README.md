# Channel Selection Training

`cyclo-lerobot-io` is an optional training adapter and a lightweight checkpoint
mapping validator. Cyclo-specific behavior lives here, outside the LeRobot
submodule. LeRobot only discovers adapters, calls their preparation/validation
hooks, and atomically saves their JSON artifacts. Disabled adapters leave
ordinary `lerobot-train` behavior unchanged.

## Installation

The AMD64 and ARM64 LeRobot Dockerfiles install this package. Outside Docker,
install the updated LeRobot fork (including its training dependencies), then:

```bash
pip install -e cyclo_brain/policy/lerobot/extensions
```

Deploy the generic LeRobot hooks and this package together. Rebuild the LeRobot
image for the initial installation; Compose mounts the extension source at
`/app/extensions` read-only. No Torch/CUDA version change is required.

## Selection

The names below are illustrative; use the exact strings in your dataset's
`meta/info.json` features, not positional indices or guessed robot groups.

```yaml
state_names: all
action_names:
  - joint_b
  - joint_a
```

Both fields are required. `all` resolves to the dataset order. Lists select and
reorder the last vector axis; state and action lengths may differ. Corresponding
feature names/shapes and normalization statistics are selected in memory. Images,
instructions, temporal axes and padding masks remain unchanged. The original
dataset is not rewritten or copied. Selection is not added to saved processors.

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=owner/dataset \
  --dataset.root=/workspace/lerobot/owner/dataset \
  --training_adapter.name=cyclo_channels \
  --training_adapter.config_path=/workspace/channels.yaml \
  --output_dir=/workspace/model/lerobot/selection_run \
  --policy.push_to_hub=false
```

[ffw_sg2_channels_no_base.yaml](examples/ffw_sg2_channels_no_base.yaml) is a concrete example for
`Dongkkka/cyclo_dashboard_0904_test_v30`: it excludes `linear_x`, `linear_y` and
`angular_z` from both vectors, leaving 19 state/action channels including the
head and lift. It is not a default for other datasets or robots.
In the updated LeRobot container, select it with:

```bash
--training_adapter.name=cyclo_channels \
--training_adapter.config_path=/app/extensions/examples/ffw_sg2_channels_no_base.yaml
```

Outside Docker, use the example's local filesystem path instead. Training
selection examples live here rather than in inference-input configuration or
dataset metadata; they only take effect when explicitly selected for a run.

This does not resize pretrained weight tensors or increase internal policy
capacity. An incompatible pretrained weight shape remains an error. Relative
actions require matching state/action reference channels except explicitly
excluded absolute channels. Unknown names, duplicate names, missing names,
empty lists, incompatible statistics and invalid processor dimensions fail.

## Checkpoints And Resume

Each checkpoint includes `cyclo_io_mapping.json` (format version 1), describing
the state order before the processor and action order after postprocessing.
`train_config.json` embeds both original and selected names plus dataset identity.
Only the main process writes the additional JSON; the normal checkpoint and
final Hub export paths include it. Failed artifact writes stop checkpoint
completion/upload, and cannot overwrite core model/processor files.

```bash
lerobot-train \
  --config_path=/workspace/model/lerobot/selection_run/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=20000 \
  --dataset.root=/new/location/of/the/same/dataset
```

The original selection YAML may be absent. If present, it must resolve to the
saved selection. Changing source channel definitions, dataset identity, adapter,
or selection is rejected. Moving the dataset directory is allowed. As with
ordinary LeRobot resume, keep the original dataset content unchanged.

Older `io_mapping_path: null` fields are ignored when reading training configs.
A non-null value stops with migration instructions: that option labelled data
and never selected it. Verify what was actually trained and start a new run with
the selection adapter, or attach verified metadata to the existing checkpoint.

## Existing Checkpoints

```bash
cyclo-io-mapping --checkpoint /path/to/pretrained_model \
  --dataset-info /path/to/actual/training/dataset/meta/info.json
cyclo-io-mapping --checkpoint /path/to/pretrained_model \
  --mapping /path/to/verified/cyclo_io_mapping.json --write
```

The default prints a validated draft. `--write` is explicit confirmation of the
training order and never overwrites existing metadata. No weights, statistics
or checkpoint configuration are changed. The lightweight mapping module has no
Torch dependency and is shared by the inference engine.

## Supported Scope

Supported: one non-streaming dataset, held-out offline evaluation, single-device
training and DDP. Unsupported combinations fail before model creation: streaming,
multiple datasets, environment rollout evaluation, reward training, HF Jobs,
FSDP, state/action feature renaming, and sample weighting with independent raw
dataset readers. GR00T relative-action statistics also read the original dataset
and are explicitly unsupported. No policy-specific statistics recalculation is
silently substituted.

Channel names do not encode units, coordinate frames, or new action representations.
Training selection does not enforce Cyclo command groups; inference still rejects
partial command groups and leaves fully omitted groups uncommanded.

## Verification

```bash
python -m pytest cyclo_brain/policy/lerobot/extensions/tests
python -m pytest cyclo_brain/policy/lerobot/lerobot/tests/utils/test_training_adapter.py
```

`test_training_roundtrip.py` uses temporary local image datasets and real small
ACT/Diffusion models on CPU, including resume, held-out evaluation and a two-rank
CPU DDP run. It downloads no weights and publishes no robot commands. WALL-X
tests use its real processors with fixture tensors, not the large backbone.

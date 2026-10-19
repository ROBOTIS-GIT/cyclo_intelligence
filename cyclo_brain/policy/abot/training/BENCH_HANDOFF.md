# ABot-M0 Native Training Contract

This adapter is owned by Cyclo. Bench owns dataset preparation, provisioning,
Run scheduling, immutable source/asset fetching, packaging and publication.
No dataset-version conversion occurs here. The prepared dataset must be v2.1,
train-only, with named vectors and timestamped videos. No upstream files are
patched. LingBot is not part of this delivery.

## Build and Execution

Build context is the Cyclo repository root with ABot submodule initialized at
`ee9013cd8341d5424aa722a8cf25e8985a6e673b`. Do not independently fetch a moving
upstream branch. Pin the delivered Cyclo commit and its gitlink together.

```bash
docker build -f cyclo_brain/policy/abot/Dockerfile.training \
  --build-arg CYCLO_COMMIT="$(git rev-parse HEAD)" \
  --build-arg FLASH_ATTN_CUDA_ARCHS=120 --build-arg MAX_JOBS=2 \
  -t cyclo/abot-training:<cyclo-commit> .
```

The Python path is `/opt/abot-env/bin/python` (3.10.18); working directory is
`/abot`. The entry point is `/opt/cyclo-training/train_run.py`:

```bash
/opt/abot-env/bin/python /opt/cyclo-training/train_run.py --config /run/run.json --phase validate
/opt/abot-env/bin/python /opt/cyclo-training/train_run.py --config /run/run.json --phase train
/opt/abot-env/bin/python /opt/cyclo-training/train_run.py --config /run/run.json --phase reload \
  --checkpoint /output/checkpoint-1
```

Run reload in a NEW process/container after the training process exits. Do not
publish a checkpoint solely because `train` exited successfully. Capture
`CYCLO_RELOAD_RESULT` and verify its checkpoint receipt SHA-256. The image does
not launch ROS, Zenoh, inference services or download model assets at runtime.
Only single-process, single-GPU training is supported; distributed/resume and
merged datasets are not qualified.

## Dependency and Source Evidence

`training/requirements.in` extends `requirements-inference.txt`; the committed
`requirements.lock` pins transitive Python packages with hashes. The lock-input
receipt rejects a changed declaration or lock. To regenerate, use uv 0.12.21:

```bash
uv pip compile cyclo_brain/policy/abot/training/requirements.in \
  --python-version 3.10 --python-platform x86_64-unknown-linux-gnu \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match --generate-hashes \
  --output-file cyclo_brain/policy/abot/training/requirements.lock
```

Review changes, update `lock-inputs.json` hashes, rebuild and rerun qualification.
Torch is 2.8.0+cu128; the CUDA base and uv images have pinned digests.
FlashAttention is built from `flash_attn-2.8.3.tar.gz`, SHA-256
`1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d`, with
`FLASH_ATTENTION_FORCE_BUILD=TRUE`, `FLASH_ATTN_CUDA_ARCHS` and `MAX_JOBS` recorded.
VGGT is imported but disabled, source commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`, installed without resolving dependencies.
The image receipt `/opt/abot-build/receipt.json` records installed packages,
Python path, PyArrow parquet roundtrip, upstream and adapter file hashes, and
build parameters. Bench must additionally retain the final image ID/digest;
an image cannot contain its own final content digest. OS apt packages are not
a bit-for-bit rebuild contract; pin the final image for Run retries.

Transitive COPY inventory (relative to repository root):

- `cyclo_brain/policy/abot/Dockerfile.training` and its `.dockerignore`.
- `cyclo_brain/policy/abot/requirements-inference.txt` and `training/`.
- `cyclo_brain/policy/abot/ABot-Manipulation/` at the gitlink, excluding `.git`.
- `cyclo_brain/policy/abot/abot_engine/` and `abot_training/`.
- `cyclo_brain/policy/abot/training_profiles/` and `scripts/train_run.py`.

## Channel Selection

Either `channel_selection` as a JSON object in the Run config, or
`channel_selection_yaml` as a local filename. Never both. YAML has exactly:

```yaml
state_names: all
action_names: all
```

Each value is `all` or a nonempty unique list of dataset feature names. State
and action selections are independent. `all` uses original metadata order;
explicit lists retain THEIR order, matching the Cyclo LeRobot YAML adapter.
Do not silently sort selections into dataset order. Reject unknown names,
duplicates or empty lists before GPU allocation. The raw robot profile remains
the full dataset contract. `training/ffw_sg2_no_base.yaml` is a dataset-specific
example, not a global default.

For Task_000759, both raw vectors have this order:

```text
arm_l_joint1, arm_l_joint2, arm_l_joint3, arm_l_joint4, arm_l_joint5,
arm_l_joint6, arm_l_joint7, gripper_l_joint1,
arm_r_joint1, arm_r_joint2, arm_r_joint3, arm_r_joint4, arm_r_joint5,
arm_r_joint6, arm_r_joint7, gripper_r_joint1,
head_joint1, head_joint2, lift_joint, linear_x, linear_y, angular_z
```

All-channel selection retains 22 state and 22 action columns. The no-base
example retains the first 19 names in each vector, ending at `lift_joint`.
It omits exactly `linear_x`, `linear_y`, `angular_z`; no name heuristic in the
loader infers what a base, joint, pose or velocity means. Cameras are explicitly
left head, left wrist, right wrist in that order in the dataset profile.

The profile padding fields accept `selected` or explicit integer dimensions
at least as large as the selected vectors. Numeric slicing happens BEFORE
statistics and normalization. Min/max, pooled mean/std and optional exact
quantiles use selected raw training columns. Padding is zero AFTER normalization.
The loss mask is a boolean vector of action model width: selected columns true,
padding false. Upstream expands it over batch/time and excludes padded outputs
from loss. Episode starts without a full future chunk are excluded, not copied
across boundaries. Model action horizon is the requested complete chunk length.

`dataset_statistics.json` records version, original/selected names and indices,
normalization, dataset-info hash and state/action statistics. It cannot be
reused with another selection/order, even when dimensions match.

## Run JSON

Required root fields: `version:1`, `model_id:"abot-m0"`, `policy_id:"abot:m0"`,
`profile:{id,sha256}`, `dataset:{path,repo_id,revision}`, `assets`, `output_dir`,
`parameters`, `source`. Selection is optional (all by default). Asset keys must
match the profile. Each asset has `path`, `provider`, `repo_id`, `revision` and
`sha256:{relative_filename:sha256}`. Materialize self-contained asset directories;
symlinks escaping the directory are rejected. Every model/processor file used
must be in the inventory. Base config and weights must additionally match the
profile's published hashes. Credentials do not belong in the Run config.

`source` supplies `cyclo_commit`, `upstream_commit`, and `image_id` in
`sha256:<64 hex digits>` form for the image. Source commits are compared against
the image receipt. The wrapper adds the build receipt hash and environment
evidence to saved metadata. Reload checks Run parameters, dataset/asset/source
identities and the same image receipt, without reopening the original assets.

All of these parameters are explicit, not inherited from pretrained training:

```json
{
  "steps": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "save_interval": 1,
  "save_total_limit": 1,
  "learning_rate": 0.001,
  "optimizer": "sgd",
  "seed": 42,
  "image_size": [224, 224],
  "horizon": 16,
  "repeated_diffusion_steps": 1,
  "gradient_checkpointing": true,
  "max_grad_norm": 1.0
}
```

This is a smoke-test recipe, not a recommended task-performance recipe. `sgd`
uses zero momentum; `adamw` uses betas [0.9,0.95], eps 1e-8, zero weight decay.
Neither is substituted on OOM. No model parameters are silently frozen.
`save_interval` is measured in optimizer updates, `save_total_limit` counts
complete snapshots retained. A final snapshot is always saved. Output must be
empty; optimizer resume is not implemented. Progress is emitted only AFTER
optimizer updates in the agreed `CYCLO_TRAINING_PROGRESS` JSON format.

## Saved Metadata

`train_config.json` includes the resolved Run config, actual model dimensions,
initialization report, probe evidence and the following mapping (names abbreviated
here only; real artifacts always contain the complete ordered lists):

```json
{
  "channel_selection": {"state_names": ["joint_b", "joint_a"], "action_names": ["joint_a"]},
  "training_adapter": {
    "name": "cyclo_channels",
    "resolved": {
      "source": {"version": 1, "state_names": ["joint_a", "joint_b"], "action_names": ["joint_a", "joint_b"], "dataset": {"repo_id": "owner/data", "revision": "pinned-revision"}},
      "selected": {"version": 1, "state_names": ["joint_b", "joint_a"], "action_names": ["joint_a"], "dataset": {"repo_id": "owner/data", "revision": "pinned-revision"}}
    }
  },
  "model_dimensions": {"state_dim": 2, "action_dim": 1}
}
```

`cyclo_io_mapping.json` is exactly the `resolved.selected` object above. Padding
columns are not physical channels and do not appear as invented names.
`initialization.json` and the matching `train_config.initialization` record the
reinitialized heads, shape mismatches, discarded disabled-VGGT keys and vocabulary
restoration. Only dimension-dependent state/action encoders/decoder are reset;
unrelated key/shape mismatches fail. Extra vocabulary rows are restored, not
trimmed; standard processor IDs are not remapped or assigned fabricated tokens.
Qwen's shared embedding/LM-head structure is retained only when both source
tensors are exactly equal; otherwise loading fails instead of overwriting one.

Snapshots contain `checkpoints/steps_N_pytorch_model.pt`, saved `processor/`,
`profile.json`, `model_config.json` (including expanded Qwen config),
`train_config.json`, `dataset_statistics.json`, `cyclo_io_mapping.json`,
`initialization.json`, and `checkpoint.json` inventory/checksums. Writes are
synchronous and atomically exposed. Reload checks the full inventory, metadata
agreement, strict model weights and seeded action predictions in a fresh process.
The training probe's `reload_verified:false` is intentionally immutable; the
separate reload receipt is publication evidence, not a mutation of the snapshot.
These snapshots are not yet the existing Cyclo inference bundle format.

## Bench Changes Required

1. Enable ABot partial selection only after accepting the contract above. Remove
   the current native all-channels gate; retain profile/data validation.
2. Accept `selected` dimensions. Preserve the explicit channel order submitted
   to Cyclo; do not reorder in another resolver. Compare exported mapping.
3. Pin the updated profile hash and Cyclo/upstream source commits together.
4. Materialize/checksum ModelScope base assets and HF processor assets separately.
5. Collect image receipt, final image ID, source/profile/lock hashes, checkpoint
   inventory and fresh-process reload evidence. Reject mismatches before publish.
6. Keep conversion, Hub packaging, and artifact publication in Bench. Do not
   infer safe partial-action robot deployment from training success.

Qualification results and final commit are reported separately after validation.

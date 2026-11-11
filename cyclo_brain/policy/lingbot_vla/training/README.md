# Native training

Status: **single-GPU full-model two-step training/save/reload passed**.
The existing inference Worker and upstream submodule are unchanged.
`Dockerfile.training` prepares an isolated environment, not a qualified training
Run. The wrapper provides explicit validation, training and fresh-process reload
commands. The qualified scope is recorded below; production Runs and the
inference export/publication contract still need integration qualification.

## Execution contract

The image entrypoint is `/opt/lingbot-env/bin/python`. Run JSON contains
`version: 1`, absolute paths `profile`, `dataset`, `assets_manifest`, `recipe`,
`output_dir`, and an optional `settings` object. Use the pinned upstream recipe
`/lingbot-vla-v2/configs/vla/real_robot/real_robot.yaml`. An output directory must
not already exist; failures are not silently resumed or overwritten.

```bash
python -m lingbot_vla_training.runtime validate /run/run.json
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lingbot_vla_training.runtime train /run/run.json
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lingbot_vla_training.runtime reload /output/checkpoints/step-2
```

The initial wrapper restricts execution to one GPU. Native FSDP2, Muon, full
model training and all teacher objectives remain enabled. Compilation is off;
multi-rank masked-loss reduction and compilation are not yet qualified.
Settings default to two optimizer steps, microbatch 1, gradient accumulation 1,
chunk size 50, image size 256, gradient checkpointing on and no image augmentation.
`save_steps` counts optimizer updates between saves; **`save_total_limit`** counts
completed checkpoints retained (default 1). Final-step saving is unconditional.

The asset manifest has `version: 1` and an `assets` object with `pretrained`,
`processor`, `moge`, `morgbd`, `video`, and `video_config`. Each asset records
`repo_id`, immutable `revision`, local `path`, and a `files` list of
`{"path": "relative/name", "sha256": "..."}` entries. All files and the pretrained shard inventory are verified before
allocating model weights. Downloads and dataset format conversion are external.

Checkpoints are staged, checksummed and atomically renamed before retention.
They contain native safetensors weights/config, processor assets,
`train_config.json`, raw-order `norm_stats.json`, explicit `cyclo_io_mapping.json`,
`probe.safetensors`, and `complete.json`. Incomplete checkpoints are not eligible
for retention deletion. Reload verifies file hashes, builds a new model in a
new process, and compares fixed-noise losses and auxiliary predictions with the
saved reference. Frozen teacher targets accompany the probe.
The probe uses the native training forward under `no_grad`, preserving module
modes, router counters and RNG state. The upstream eval-only MoE shortcut uses
atomic accumulation and is not the numerical reference for this check.
Reload tolerances remain `rtol=1e-4`, `atol=1e-5`.

These are **weights-only** checkpoints, not native DCP optimizer checkpoints.
`resume_supported: false` is recorded explicitly. A successful reload is not
training-resume validation, and this layout is not yet the existing inference
Worker's exported bundle layout.

### Run example

Paths below are container paths; Bench supplies and mounts these files. Asset
downloads must finish before launching the offline container.

```json
{
  "version": 1,
  "profile": "/opt/lingbot-build/profiles/task_000759.yaml",
  "dataset": "/run/dataset",
  "assets_manifest": "/run/assets.json",
  "recipe": "/lingbot-vla-v2/configs/vla/real_robot/real_robot.yaml",
  "output_dir": "/run/output",
  "settings": {
    "max_steps": 2,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "save_steps": 1,
    "save_total_limit": 1,
    "num_workers": 0,
    "seed": 42,
    "chunk_size": 50,
    "image_size": 256,
    "image_augment": false,
    "enable_gradient_checkpointing": true,
    "use_compile": false
  }
}
```

For standalone torchrun in an offline container, use `--hostname localhost` so
rendezvous resolves a local address rather than an unreachable Docker hostname:

```bash
docker run --rm --hostname localhost --network none --gpus all --shm-size 4g \
  -v /absolute/run-directory:/run cyclo/lingbot-training:preparation \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lingbot_vla_training.runtime train /run/run.json
docker run --rm --hostname localhost --network none --gpus all --shm-size 4g \
  -v /absolute/run-directory:/run:ro cyclo/lingbot-training:preparation \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lingbot_vla_training.runtime reload /run/output/checkpoints/step-2
```

`train_config.json` records the resolved profile/recipe/settings, asset inventory,
input Run hash, Cyclo build identity, Python path, GPU, world size and explicit
weights-only/no-resume format. `cyclo_io_mapping.json` version 2 records policy
ID, absolute action representation, raw channel names, canonical slots,
transforms, ordered cameras and normalization. It is not the LeRobot fork's
mapping schema. Bench should preserve these native files without replacing them.
`complete.json` version 1 records optimizer step, file SHA-256 inventory and
`resume_supported: false`. Only completed `checkpoints/step-N` directories are
publishable. The `reload_verified` JSON event identifies the exact completion
receipt hash and compared outputs; a successful process exit is also required.

## Locked environment

Build from the Cyclo repository root:

```bash
docker build -f cyclo_brain/policy/lingbot_vla/Dockerfile.training \
  --build-arg FLASH_ATTN_CUDA_ARCHS=120 --build-arg MAX_JOBS=2 \
  -t cyclo/lingbot-training:preparation .
docker run --rm --network none cyclo/lingbot-training:preparation \
  /opt/lingbot-build/environment.py smoke
```

This does not start a Worker, ROS service, GPU job, or training loop.
The Python executable is `/opt/lingbot-env/bin/python`; working directory is
`/lingbot-vla-v2`. For a committed build, pass `--build-arg CYCLO_COMMIT=<SHA>`.
An uncommitted preparation build is explicitly recorded as `uncommitted`.

`environment.toml` pins the CUDA base/uv image digests, upstream source-tree
hash, requirements input/lock hashes, LeRobot dataset source revision/archive
hash, and FlashAttention source/hash. All Python package versions and archive
hashes are in `requirements.lock`. `requirements-build.lock` bootstraps only
the locked build tools before no-build-isolation source builds.

The installed VLA stack keeps Torch 2.8.0/CUDA 12.8, Transformers 4.57.3,
NumPy 1.26.4, and PyArrow 21.0.0. FlashAttention 2.8.3 is source-built with
`FLASH_ATTENTION_FORCE_BUILD=TRUE`, explicit `FLASH_ATTN_CUDA_ARCHS` and
`MAX_JOBS`. Listed build architectures are not GPU-training qualification.

MoGe and MDM code comes from the unchanged pinned LingBot source tree; the
LeRobot dataset source comes from v0.4.2's immutable commit. These source APIs
are added to `PYTHONPATH`, **not installed as full distributions**: their
full training/demo dependency metadata conflicts with the VLA pins. This image
does not promise support for their standalone training or demo CLIs.
Required runtime libraries are resolved together in the Cyclo lock. Source
paths and identities are recorded explicitly rather than patched metadata.
The verified LeRobot archive contributes `src` and `LICENSE`; documentation
links are excluded, and executable source links remain rejected.
The pinned LeRobot dataset reader declares `CODEBASE_VERSION = "v3.0"`.
The wrapper instead reads prepared v2.1 metadata/Parquet through PyArrow and
decodes video using the pinned LingBot decoder. It rejects other layouts;
conversion remains owned by Bench, not this environment helper.

Excluded teacher-training/demo tools include MLflow, Gradio, and PLY export
helpers. `numpydantic==1.6.11` replaces the upstream script's conflicting
`1.9.0 --no-deps` installation; the actual upstream normalization schema is
included in smoke checks. None of these environment decisions disables depth,
future-depth or future-video supervision. Actual teacher weights and forwards
were exercised separately in the full-model qualification below.

The build fails on source/lock drift, failed dependency checks, import failure,
or PyArrow round-trip failure. Its `/opt/lingbot-build/build_receipt.json` records
the contract, installed distribution versions, Python/CUDA versions,
FlashAttention build inputs and smoke results, with `training_qualified: false`.
Bench must additionally record the **final image ID** after building; an image
cannot truthfully record its own final ID inside itself.

The base image and Python artifacts are pinned. Ubuntu apt repositories are
not snapshotted, so this is not a claim of bit-for-bit OS/image reproducibility.
Source-only modules are not checked by `uv pip check`; their imports are tested
separately. Import success is not teacher numerical or full-model validation.
Source imports and Parquet checks run before CUDA compilation. FlashAttention
build caches are separated by GPU architecture and job count.

Regenerate locks only as an explicit reviewed environment update, with uv
0.12.21, Python 3.12 and `--python-platform x86_64-manylinux_2_28`:

```bash
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes cyclo_brain/policy/lingbot_vla/training/requirements-build.in \
  -o cyclo_brain/policy/lingbot_vla/training/requirements-build.lock
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes cyclo_brain/policy/lingbot_vla/training/requirements.in \
  -o cyclo_brain/policy/lingbot_vla/training/requirements.lock
```

Review the resulting changes and update `policy_hashes` in `environment.toml`.
Do not run dependency resolution at image startup.

## Input contract

`lingbot_vla_training.inputs.InputContract.from_profile(profile, info)` validates
the input portion of the Bench version-2 profile, against the dataset's
`meta/info.json`. It does **not** validate asset receipts or certify physical
compatibility with pretrained slots.

- Model identity: `lingbot-vla-v2` / `lingbot_vla:v2`.
- One dataset, all raw state/action channels in metadata order. No prefix-based
  selection, robot-name fallback, or inferred canonical positions.
- `canonical_mapping.state` and `.action` explicitly map each raw name to a
  unique integer slot in `0..54`. State and action layouts can differ.
- `transforms.state` and `.action` cover each raw vector once, in order. Each
  slice supplies `origin_key`, `start`, `end`, `names`, `semantic_role`,
  `canonical_slots`, `subtract_state: false`, and `relative_type: null`.
- Camera feature keys come from `cameras`; their role order comes exclusively
  from `camera_order`, not dictionary order. No global three-camera constant.
- Initial normalization support is **mean_std only**. Bench's broader
  `min_max`/`quantile` enum is not implemented here and is rejected, not silently
  converted.
- Depth, future-depth, and future-video teachers must all be enabled, frozen,
  with `episode_boundary: pad_and_mask`.

Internal statistics passed to `InputTransform` contain separate `state` and
`action` objects, each with `names`, `mean`, and `std` in exact raw channel order.
They are one-dimensional, finite, and have nonnegative standard deviations.
The numerical transform matches upstream `(value - mean) / (std + 1e-6)`.
No clipping, integration, state subtraction, or default identity statistics
are applied. Normalization precedes canonical zero padding; unused slots stay
zero. `restore_actions` gathers the explicit slots and reverses normalization.

This statistics structure is **not** the native inference bundle's grouped
`norm_stats.json` format. The wrapper estimates it from unique frames in the
training split, without validation frames or repeated chunk samples. The
checkpoint stores it together with the profile and canonical mapping. Native
grouped-statistics export remains a separate qualification gate.
Do not reuse relative-action statistics from a previous run for absolute targets.

`prepare` returns canonical state/actions, active channel masks,
`action_is_pad`, upstream-compatible `chunk_joint_mask`, and a combined
`action_loss_mask`. The combined mask excludes both padded steps and unused
channels. The runtime passes the combined mask as native `joint_mask`; upstream
does not consume `action_is_pad` itself. A Cyclo-owned instance adapter excludes
invalid future samples before native future-depth/video loss reduction. Current
targets remain unmasked. Numerical and gradient tests cover mixed-validity and
all-padded batches. The full-model run below exercised both a valid future and
an episode-end padded future.

## Time axis

`plan_frames` takes episode-local frame index/length, chunk size, dataset FPS,
and an explicit warmup flag. It implements the pinned upstream recipe with one
future frame:

- Action samples: offsets `0..chunk_size-1`, clamped within the same episode.
- Current image: offset zero.
- Future image: offset `chunk_size-1`; intended time interval
  `(chunk_size-1)/fps`.
- Effective video FPS: `fps/(chunk_size-1)`.
- Warmup, when enabled, is a duplicate of the current image, **not** a sample
  from the previous frame/episode.
- Clamped future frames have `future_valid: false`. The intended time interval
  is preserved; a repeated last frame must not become a valid future target.
- Chunk sizes below two are rejected for this future-teacher recipe.

The dataset adapter decodes current/future frames with shared image augmentation
parameters. Multi-future-frame sampling is unsupported and rejected.

## Experimental robot profile

The locally available `peanut_10000step` bundle contains
`robot_config.yaml` and `lingbotvla_cli.yaml`. Their stored grouping produces
the following zero-based canonical positions:

| Raw channels | Canonical slots | Previous action transform |
| --- | --- | --- |
| Left arm 7, right arm 7 | 0..13 | Subtract corresponding state |
| Left/right grippers | 28..29 | No subtraction |
| Head 2 | 34..35 | No subtraction |
| Base `linear_x`, `linear_y`, `angular_z` | 36..38 | No subtraction |
| Lift | 51 | No subtraction |

The old config added `lift.position: 4` after the other groups, using the first
of the final four slots, rather than assigning lift to `waist.position`.
The group name `base.position` did not convert velocity values to positions.
This establishes what our earlier run requested, not what those slots meant in
the original pretrained data.

Task_000759's metadata has 22 named channels, four video features, and 15 FPS.
`profiles/task_000759.yaml` selects left head, left wrist and right wrist. The
user approved the previous lift/base slots for software qualification only.
All actions, including arms, are now absolute. Physical equivalence with the
original pretrained slot semantics is not established. Camera selection and
slot mapping belong exclusively to this profile, not shared runtime defaults.

## Upstream audit

Audited gitlink: `ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8`.

1. `data/vla_data/base_dataset.py` samples current and end-of-chunk images.
   `data/vla_data/utils.py` keeps action padding but does not forward a dedicated
   future-image boundary mask to the auxiliary loss paths.
2. `models/vla/vision_models/module_utils.py:get_video_target` repeats current
   for warmup and runs the video teacher under `torch.no_grad()`.
3. `models/vla/lingbot_vla/modeling_lingbot_vla.py:depth_emb_forward`
   computes future-depth embedding loss without an episode-validity mask.
   Future loss masking requires a Cyclo-owned adapter and numerical/gradient
   tests, not just a new dataset flag.
4. Resolving the pinned `requirements.txt` plus `requirements-depth.txt` with
   uv 0.12.21, Python 3.12, Linux AMD64 fails:
   `mlflow==2.22.5` requires `pyarrow>=4,<20`, conflicting with
   `pyarrow==21.0.0`.
5. The vendored `lingbot-depth/pyproject.toml` declares Torch 2.6.0 and
   xformers 0.0.29.post2, whereas the VLA environment uses Torch 2.8.0.
   Upstream's `--no-deps` installation is not evidence of compatibility.
6. `tools/create_train_env.sh` writes a `.pth` referencing
   `vision_models/morgbd_clean/3rd/utils3d`, which is absent in this checkout.
   MoGe instead declares a revision-pinned external utils3d dependency.

The dependency approach is a separately locked teacher-inference runtime,
not the full teacher-training tooling. It retains the source paths for all
three VLA auxiliary losses. Full pretrained/teacher qualification is still required.

## Verification and remaining gates

Run CPU input tests and inference regressions:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  cyclo_brain/policy/lingbot_vla/tests
```

Input tests use synthetic channels and include parity with the actual upstream
normalizer when Torch/torchvision and the submodule are available.
They are not robot-profile approval or full-model training validation.

### Full-model qualification, 2026-10-21

Validated on the user-authorized Verda host with one RTX PRO 6000 Blackwell
Server Edition (96 GiB, SM120). Existing `cyclo_mjlab` was left running.
Training and reload used separate containers with networking disabled, the same
image, and no source bind-mount overrides:

```text
image: sha256:b2707694c03e944f1145f9264ff3290b06a3b285bac193f7284672e752e090d6
upstream: ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8
build identity: uncommitted source snapshot
runtime.py SHA256: f80e200d4490d48e741a3688d0bf86bf2194db2cc1e7525deceed08668307594
Python: /opt/lingbot-env/bin/python (3.12.3)
FlashAttention build: 2.8.3, FLASH_ATTN_CUDA_ARCHS=120, MAX_JOBS=4
checkpoint completion SHA256:
e57620cd684611617285aae48878d60392ae551c22c8cec14291ee65913e2ab0
```

The run used `profiles/task_000759.yaml`, prepared v2.1 data with 98 episodes
and 10,578 frames, and the exact example settings above. The native recipe
supplies learning rate `5e-5`. All **6,375,906,359** model parameters remained
trainable; all three teacher objectives were enabled. Teachers were frozen.

| Optimizer step | Action | Current depth | Future depth | Video | Total |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.53195524 | 0.01564888 | 0 | 0.00131226 | 0.55004126 |
| 2 | 0.58213568 | 0.01470526 | 0.01422984 | 0.00306658 | 0.61528236 |

The total also includes native MoE regularization. Step 1 used an episode-end
padded future, so future depth was correctly masked; current-video supervision
remained active. Step 2 exercised valid future targets. Both backward passes,
finite gradient checks and optimizer updates completed. Peak Torch allocated
GPU memory was **82,609,735,680 bytes (76.94 GiB)**. This is not total driver
memory or evidence that an 80 GiB GPU is sufficient. The qualified target is
96 GiB; other hardware and larger batches need a separate test.

Both steps saved full FP32 sharded safetensors checkpoints. With
`save_steps=1, save_total_limit=1`, step 1 was removed only after step 2 completed.
Fresh-process reload verified the completion inventory and compared six loss
values and four auxiliary predictions at the unchanged tolerances. It emitted
`reload_verified` for step 2 and exited successfully. This checks the native
training forward, not the deployment eval-mode MoE shortcut or optimizer resume.

Build/import checks passed with PyArrow **21.0.0**, including Parquet roundtrip.
The CPU regression suite passed **220 tests, 1 skipped**; its separate CPU
environment uses PyArrow 23.0.1 and does not substitute for the image check.
The skipped test requires the native GPU inference environment.

Evidence on the validation host is under `/opt/cyclo-lingbot-validation/`:
`training-004.log`, `reload-004.log`, `run.json`, `assets/assets.json`, and
`runs/full-model-smoke-004/checkpoints/step-2/complete.json`.
The image receipt deliberately still says `training_qualified: false`: it
describes build-time checks, not a later GPU run. Bench must attach this external
qualification evidence and record the final image ID separately.
All six training-package Python files in that image were SHA-256 compared with
the delivery source. Subsequent documentation changes do not change that code.

### Validated asset identities

Acquire files at these immutable Hugging Face revisions, then hash every local
file into the asset manifest before training. Mount paths are deployment inputs,
not robot/model constants in the wrapper. The complete tested file inventories
are also preserved in the checkpoint's `train_config.json`.

| Role | Repository | Revision | Repository files |
| --- | --- | --- | --- |
| pretrained | `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` | `model.safetensors.index.json`, all six indexed model shards |
| processor | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | config/tokenizer/image-processor files below; no Qwen weight shards |
| moge | `Ruicheng/moge-2-vitb-normal` | `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368` | `model.pt` |
| morgbd | `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` | `depth/model.pt` |
| video | `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` | `dino_video/teacher_step_10000.pth` |
| video_config | `robbyant/lingbot-vla-v2-6b` | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` | `dino_video/config.yaml` |

Processor inventory: `chat_template.json`, `config.json`,
`generation_config.json`, `merges.txt`, `preprocessor_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `video_preprocessor_config.json`,
`vocab.json`. For a teacher role, `path` is its local containing directory and
`files` contains exactly one entry relative to that directory.

Teacher file SHA-256 values:

```text
moge         16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6
morgbd       d70c5191eab853d436763b35d40ff99d13534b4bcd43e4d02823656968159e5b
video        086285efd8d65bc66e96b363807c4010ee5c790b7452b765edaa23837a63705b
video_config 333befe36960b094f2604084802639ab4396c1fa51e51f8d1cb160e29522ac08
```

### Remaining integration gates

- Bench must pin the delivered Cyclo commit and build with that identity. The
  tested snapshot must not be described as a committed build retroactively.
- Export this wrapper's explicit-slot/raw-order statistics to the existing
  inference bundle and verify transform parity plus Worker loading. The current
  training checkpoint must not be presented as a deployment-ready bundle.
- Bench owns Run/VM integration, dataset preparation, packaging, publication and
  checksum-verified download. None of those end-to-end flows were tested here.
- Long-run convergence, real-robot quality, other GPU architectures, multi-GPU,
  larger batches, compilation and optimizer resume remain unqualified.

No Bench-source, existing inference Worker, or upstream submodule changes were
made for this wrapper. The approved lift/base mapping remains an experimental
software-test profile, not a claim of pretrained physical-slot equivalence.

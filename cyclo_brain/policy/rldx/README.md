# RLDX-1 Worker

The pinned `RLDX-1/` upstream is hosted in its own Python 3.10 / Torch 2.8
Worker. The existing Cyclo Runtime owns command publication, Stop, chunk
alignment and Initial Pose. This Worker only receives observations and returns
action chunks through the common Engine protocol. There is no extra ZMQ server.

## Current Scope

- Policy ID: `rldx:rldx1`; checkpoint root: `/workspace/model/rldx`.
- PT-IMG fine-tunes, current RGB images/state, absolute-valued action channels.
- Saved RLDX eval processor handles image geometry, normalization and decoding.
  Cyclo applies configured robot camera rotations, not an additional resize.
- State channels are selected by `JointState.name` from each received message,
  then arranged in training order. Missing, duplicate, invalid or stale required
  joint data blocks inference instead of falling back to positional indexing.
  Actions are checked and reordered to the complete robot action layout; never truncated.
- Only cameras named in exported metadata are subscribed. The peanut dataset
  contains left head, left wrist and right wrist, not right head.
- Memory, motion, physics and RTC checkpoints are rejected at LOAD. Multi-frame video
  is not enabled by silently repeating a current frame. These require separately
  tested observation/execution contracts before enabling them.
- Initial image target is AMD64 NVIDIA systems. ARM64 image support is not yet
  provided. Do not deploy an emulated AMD64 image to a Jetson.

## Build

```bash
git submodule update --init cyclo_brain/policy/rldx/RLDX-1
# RTX 5090 (SM120); use RLDX_FLASH_ATTN_CUDA_ARCHS=100 for B200.
ARCH=amd64 docker compose -f docker/docker-compose.yml build rldx
./docker/container.sh start-rldx
```

`rldx_engine`, `configs`, common runtime/SDK and robot configuration are
source-mounted like the other Workers. Stop inference before edits; restart
the Worker to reload Python. Upstream dependency changes require a rebuild.
The main Cyclo process must reload its catalog to discover the new runtime.
No existing Cyclo or LeRobot service is restarted by the training commands.

## Training

Use the dedicated training image or a separate locked Pixi environment and a
new dataset/output directory. Upstream code and the original dataset are not
patched. Keep Pixi's cache and `TMPDIR` on a sufficiently large filesystem.
Source compilation of FlashAttention needs the locked CUDA 12.8 toolkit.
The prepared LeRobot v2.1 dataset is supplied by Bench; this training path
does not detect or convert dataset versions.
The installer requires Pixi 0.65.0, matching the pinned v6 lockfile.

```bash
python3 scripts/install_training_environment.py /path/to/RLDX-1
export PATH=/path/to/RLDX-1/.pixi/envs/rldx/bin:$PATH
export RLDX_REPO=/path/to/RLDX-1
export RLDX_DATASET_PATH=/path/to/bench-prepared-v21
export RLDX_OUTPUT=/path/to/new-smoke-run
STEPS=10 bash scripts/train.sh
```

The converter rejects missing textual instructions, missing/duplicate channel
names, non-finite values and ambiguous video boundaries. It verifies frame
counts and dimensions after lossless H.264 trimming. It recomputes vector
statistics/quantiles from all episodes. It never resizes or rotates images.
An interrupted export remains marked `INCOMPLETE` and is not used for training.

The upstream launcher creates `OUTPUT/EXPERIMENT/checkpoint-N`. After training
finishes the wrapper copies `cyclo_input_metadata.json` beside each retained
checkpoint. This sidecar records robot type, ordered state/action names, FPS
and cameras. It is required to validate Cyclo's robot mapping at LOAD, rather
than assuming 22 channels always mean the same robot. For an intermediate
checkpoint in an ongoing run, copy this file from the prepared dataset before
exporting the checkpoint. Preserve the entire checkpoint and `processor/`.
The Run config's `save_total_limit` controls checkpoint retention independently
of `save_steps`. The shell wrapper uses `SAVE_TOTAL_LIMIT` for the same value;
both default to 1. Allow space for both the old and new checkpoint
during rotation, plus the launcher's final model export. Filesystem free space
does not account for a separate user/project quota.

The separate training image uses upstream `pixi.toml` and `pixi.lock` with
Cyclo's pinned policy in `training/environment.toml`. It fails on hash or
lock mismatch. Its Python is `/rldx/.pixi/envs/rldx/bin/python`; the image
writes `training_environment.json` there and copies it to the selected
checkpoint. FlashAttention is built from the pinned Git commit for the
selected `FLASH_ATTN_CUDA_ARCHS` (default `120`). The lock supplies
`pyarrow==24.0.0`, including the common Pixi dependencies. Inference keeps
its existing separate installer and environment.

With one GPU the upstream launcher treats `BATCH_SIZE` (default 16) as the
microbatch size. `GRAD_ACCUM=4` gives an effective batch of 64, not 16.

```bash
python scripts/checkpoint_smoke.py /path/to/checkpoint-10 "$RLDX_DATASET_PATH"
# Use a NEW output directory for the full run, not the smoke checkpoint.
RLDX_OUTPUT=/path/to/new-full-run STEPS=10000 SAVE_STEPS=1000 bash scripts/train.sh
```

Ten steps test loading, backpropagation, saving and real-weight inference, not
task performance. Recorded-data inference is open-loop and publishes no robot
commands. Live safety/robot performance validation is a separate user step.

## Tests

```bash
python3 -m pytest cyclo_brain/policy/rldx/tests
```

Mapping/contract tests do not load large models. `checkpoint_smoke.py` uses real
weights and reports input shapes, instruction, action dimensions and latency.

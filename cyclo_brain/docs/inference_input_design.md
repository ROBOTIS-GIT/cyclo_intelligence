# Inference Input Pipeline Implementation and Validation

## Purpose and Responsibilities

User-facing YAML configures only Cyclo's additional preprocessing and registered handler selection.
Python adapters and handlers manage basic input packing, internal graph construction, computation,
and memory. Runtime manages execution facts and request timing.
The common package has no Torch or LeRobot dependency. This migration targets LeRobot;
the separate GR00T Worker retains its existing path.

```text
Robot topics -> Worker RobotClient -> Required latest values / reception history
                                               |
                                     Python-owned input graph
                                               |
                                     before stage -> saved processor
                                               |
                                     after stage -> model inference
                                               |
                                     saved postprocessor -> action chunk
                                               |                |
                                     result stage               v
                                     state proposal         Cyclo Runtime
                                                            alignment / interpolation / publication
                                                                |
Worker state commit <--- Engine context <------------------------+
                         plan adoption / publication / completion / discard
```

Images and features are not relayed through the Cyclo container.
Successful command publication does not mean the physical robot has completed the motion.

## File Layout

| Location | Responsibility |
| --- | --- |
| `policy/common/runtime/inference_inputs/graph.py` | DAG validation and compilation at LOAD; evaluation by stage |
| `inference_inputs/operators.py` | Common selection, slicing, stack/concat, axis, and dtype operations |
| `inference_inputs/memory.py` | State proposals, commits, discards, and generation resets |
| `inference_inputs/providers.py` | Reception, reset, and shutdown contracts for registered external data providers |
| `inference_inputs/resources.py` | Shared retention budget for history, features, and caches |
| `inference_context/` | Existing reception history, freshness, execution facts, and protocol contracts |
| `policy/lerobot/configs/inference_inputs/*.yaml` | Per-policy additional preprocessing and custom handler selection |
| `policy/lerobot/lerobot_engine/input_config.py` | User configuration validation, internal graph construction, and sequential image operations |
| `policy/lerobot/lerobot_engine/input_pipeline.py` | LeRobot bindings, Torch operations, and saved processor integration |
| `policy/lerobot/lerobot_engine/adapters/` | Model public APIs and validation of execution modes, assets, and compatibility |

The former `configs/image_preprocessing/` and `lerobot_engine/input_plan.py` were removed.
Checkpoint weights, `config.json`, and upstream LeRobot code were not changed by this migration.

## User YAML and Internal Graphs

```yaml
preprocessing: identity
```

This single line is sufficient when no additional transforms are needed. Declare image transforms
in order under `preprocessing.images`, and select custom processing through
`preprocessing.custom.handler` and `options`. See the
[user configuration guide](../policy/lerobot/configs/inference_inputs/README.md) for examples.
`sources`, `nodes`, and `outputs` are not allowed in user-facing YAML.

The graph, memory, and execution examples below are **internal representations for Python handler development**.
They are not settings to paste directly into per-model YAML files. Handlers must be explicitly
registered in the model adapter's `input_handlers` and run only when selected in YAML.
The builder that validates module options runs before model loading. Complex model API computations
are registered as operations through the existing `input_extensions` hook. Connections use registered
names only; arbitrary imports are not allowed.

`sources` declares data origins, `nodes` connects operations, and `outputs` defines final inputs by stage.
Every declared node executes once per request, including state-writing nodes not directly connected
to an output. Do not declare unused sources or operations in a Python handler's internal graph.

- `before`: before the saved LeRobot processor.
- `after`: after the processor. `processed` is the actual processor output.
- `result`: after the model and postprocessor. `model_action` and `postprocessed_action` are distinct.
- References to later stages, cycles, unregistered operations, and invalid options are rejected.
- `identity` means Cyclo performs no additional transformation at that node.
- The default Python graph handles rotation, RGB float32 conversion, BCHW batching, and device transfer.
- User image operations execute in list order, preserving the Torch/OpenCV numerical paths and order.
  OpenCV after Torch is rejected because it would require implicit quantization.
- Model-internal transforms and saved processors run unchanged. YAML does not disable or replace them.

Configuration is reloaded only on LOAD after Clear/UNLOAD. File changes during START/RESUME are not applied.
Each model type uses one YAML file; there is no per-checkpoint configuration selector in the UI.

Default behavior:
- Spatial identity for ACT, the Pi family, SmolVLA, FastWAM, VLA-JEPA, and similar policies preserves model-internal behavior.
- Diffusion's OpenCV bilinear/checkpoint-size settings are existing Cyclo compatibility defaults, not confirmation of the training data's resizing method.
- Multi-Task DiT retains the existing 224x224 Torch bilinear/antialias test settings. These are not universal defaults for all training data.
- XVLA does not automatically resolve different camera sizes when its internal padding setting is disabled.
- State/action channels use checkpoint `cyclo_io_mapping.json`. Legacy checkpoints require exact dimensions without padding/truncation; robot command-group definitions remain unchanged.

## Observation History

Cyclo does not automatically construct history by interpreting `n_obs_steps` or
`observation_delta_indices` in the model config. Explicit `size: checkpoint` references and
config reads for model loading and compatibility checks remain supported.

Example internal history graph constructed by a registered Python handler:

```yaml
sources:
  arm_history:
    source: joint:follower_arm_left
    frame_offsets: [-1, 0]
    fps: 15
    max_age_s: 0.04
nodes:
  history:
    op: stack
    inputs: [arm_history]
    options: {axis: 0, sequence: true}
outputs:
  before: {arm_history: history}
  after: {"*": processed}
startup: {missing: wait}
```

This demonstrates internal history assembly, not a complete model configuration or user-facing YAML.
Add `to_tensor`, a batch axis, and normalization operations as needed.
Time offsets in seconds, such as `offsets_s: [-0.1, 0]`, are also supported.
Frame offsets use the explicitly specified FPS; they are not inferred from Control Hz or Dataset FPS.

- History records actual topic callbacks. Reading the same latest value on successive model calls does not create new frames.
- Subscribe only to required topics and retain history only for sources that request past values.
- Timestamps use monotonic reception time on the same host. This is not camera capture-time alignment or cross-device clock synchronization.
- Missing or stale observations, and selecting the same sample for distinct time positions, are rejected.
- `startup.missing` is `wait` or `error`. Waiting is limited to observation readiness; the Worker never waits for future execution feedback while holding its lock.
- Source IDs, reception times, computation times, and axis, unit, coordinate-frame, and normalization semantics remain distinct.
- Diffusion/Multi-Task DiT's public `select_action` uses model-internal history. Default YAML passes one latest observation without constructing duplicate history.

## State and Execution Conditions

State updates are proposed first and committed only when the declared event is confirmed.
Conditions in `memory.slots` are independent of `execution.request_after`.

| Event | Meaning |
| --- | --- |
| `prediction_success` | A valid prediction has completed |
| `plan_accepted` | Runtime has adopted the execution plan |
| `first_publication` | The first command of that plan has been published |
| `published_count` | N distinct commands of that plan have been published |
| `plan_terminal` | All commands of the plan have been published; discard or failure is not success |

Runtime checks prerequisites for the next request against its local execution ledger.
The Worker does not wait for the next command or future feedback while holding the request lock.
Partial failures, plans entirely discarded by alignment, and unreachable command-count conditions
are not treated as success. ZOH repeats are outside the original plan's command range and do not
count as plan progress.

`memory_read` and `memory_write` access only registered state slots.
Initial memory requires an explicit literal or `initial: input` for bootstrapping; missing initialization fails.
Memory derived from model outputs uses public raw/postprocessed results in the `result` stage.
YAML cannot access model private attributes or arbitrary Python paths.

Session, generation, prediction, command, and event IDs distinguish duplicates and delayed responses.
Stop/Clear, a new generation, or an instruction/model change resets session memory and caches.
Initial Pose Sync and normal inference use separate generations.
Failures that cannot roll back model-internal state are not retried in the same generation.

## Extensions and Performance

Implement complex feature computation in registered Python modules.
Register operations through the LeRobot adapter's `input_extensions(registry, bindings, engine)`.
Builders in `input_handlers` translate options into internal operation configurations; YAML does not import modules.

- Operator compilers run at LOAD and return small functions called for each request.
- Inputs are immutable by default. Registered operations that need to modify them declare `mutates_inputs=True` and receive private copies.
- External sources use `Binding(..., provider=...)` and implement `start(queries, budget)`, `resolve_samples`, `reset`, and `close`.
- Providers must return actual sample IDs and original reception timestamps, and register their own retained buffers with the supplied Budget.
- Registered modules are trusted code. The system does not sandbox arbitrary incorrectly implemented Python modules.
- GPU computation runs in registered Torch modules; the common core does not depend on Torch.

Cross-request caching is allowed only for reviewed operations declared `cacheable=True`.
The Python handler's internal graph must declare all input and instruction dependencies and use actual sample IDs.
Computation settings and encoders are fixed for the LOAD lifetime; caches are cleared on generation changes.
Cross-request caching in the `result` stage is prohibited because model outputs may be stochastic.
Memory-read IDs include the committed state version so changes remain distinguishable even for the same image.

The default retention limit is 256 MiB across CPU and GPU, shared by history, pending/committed features,
and caches. This is a budget for retained numerical buffers, not a limit on total process RSS.
Model weights, Python metadata, temporary copies, encoder activations, and allocator reserves are separate.
Encoder computation does not run inside topic callbacks or while holding control locks.

## Protocol and Limitations

The Engine protocol is 3.1. Extended state contracts negotiate `feedback_schema=2`.
Existing schema 1 JSON fields and formats, and ROS service fields, remain unchanged.
Schema 2 distinguishes pre-interpolation prediction IDs from planned command ranges.
Incompatible Worker/Runtime combinations are rejected during LOAD contract negotiation.

This foundation alone does not imply support for every model, including LingBot-VA and RTC/TT-RTC.
Each model still needs an adapter that accounts for its public API, action coordinate frame,
memory commit timing, and bootstrapping. Interactive requests for additional observations, online learning,
per-candidate branching memory, physical execution-completion ACKs, remote clock synchronization,
and arbitrary large-model feature APIs are outside this implementation's scope.
Unsupported options, operations, and execution contracts are rejected at LOAD; model-internal
requirements are not discovered automatically.

## Simplified YAML Migration Validation (2026-09-17)

- 171 LeRobot adapter/input tests and 40 image-operation tests passed.
- 410 common Runtime/Catalog/Dockerfile tests passed.
- All 13 default configurations matched the previous path without numerical differences at rotations 0 and 270.
- Validated YAML handler selection, option rejection, model-specific registration scope, and memory-based feature combination using a small real encoder.
- Three-camera CPU input assembly p95: 2.44 ms previously, 2.52 ms on the new path, within the 2.88 ms limit.
- Both paths made 9 Torch copy calls and had total positive self allocations of 10,948,620 bytes.
- Real model checkpoint, GPU, robot, and ARM64 validation was not rerun for this migration.
- The earlier validation records below do not replace real model validation after this migration.
- Applying only this user-configuration change requires updating the LeRobot Worker code.
  Supplying new YAML to an old Worker causes LOAD to fail. If the earlier common pipeline migration
  has not been deployed, Cyclo must also be updated as described below.

## Earlier Pipeline Validation Records

- Checked image values, shapes, and rotation compatibility for the 13 default profiles.
- Tested past/current feature combination, execution feedback, and resets using a small real Torch encoder. The model body and transport were mocked.
- Real pinned LeRobot Diffusion public API: 11 tests passed, including 1/2/4 observations, repeated action-queue calls, and resets.
- Real ACT 40k checkpoint: 6 inference calls, cached LOAD, and UNLOAD passed in an isolated CPU environment. No robot commands were published.
- Latest-input assembly p95: 2.52 ms previously, 2.59 ms with the new graph, within the `baseline * 1.1 + 0.2 ms` limit.
- Both paths made 9 Torch `copy_` calls on the same inputs, with total positive self allocations of 10,948,620 bytes each. Total allocations are not peak resident memory.
- tracemalloc peak: 758,569 bytes previously, 761,897 bytes with the new graph. These figures exclude Torch tensor storage.
- These p95 results measure three-camera CPU input assembly, not full GPU inference or actual 100 Hz control jitter.
- 190 UI tests, 77 Docker/Supervisor tests, and 47 Initial Pose Sync tests in an isolated ROS environment passed.
- 341 common Runtime tests, 146 LeRobot tests, 40 image-operation tests, 47 action-processing tests, 50 RobotClient unit tests, 4 ROS CDR tests, and 36 Catalog tests passed.
- 6 real pinned WALL-X/LeRobot GR00T import and saved-processor tests passed. One ACT test in the same file was skipped because its checkpoint-path environment variable was not set; ACT was validated separately with a real-checkpoint smoke test.
- ARM64 validation remains incomplete because the robot hostname `ffw-snpr48a1110.local` could not be resolved.
- GPU peak memory, maximum temporary-copy memory, actual control jitter, and real robot/model performance validation remain incomplete.

Running services were not restarted. No version changes, commits, pushes, or image deployments were performed.
AMD64 validation used isolated execution with new source code mounted read-only into an existing image.
Full build and startup validation of the candidate image itself remains a separate task.
Deployment requires rebuilding and recreating both Cyclo and LeRobot images. COPY instructions for the new
common packages were added to both LeRobot Dockerfiles, and the Compose YAML mount path was changed.

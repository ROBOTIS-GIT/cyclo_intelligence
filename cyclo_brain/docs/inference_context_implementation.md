# Inference Context

## Action Steps

`TaskInfo.action_steps` is a Runtime execution setting, not a checkpoint override.
Zero (the UI's empty `All` input) preserves full-chunk execution. Positive values
select at most N original waypoints after alignment and before interpolation and
blending. A shorter chunk is not padded. Source positions retain their original
indices; planning records retain the original source length and the actual
accepted command count. Unselected predictions are not publication receipts.
Interpolation and ZOH repetitions do not count as additional model waypoints.

The same selection applies to Sync and Async. Scheduling and safety holds remain
unchanged. Sync drains the selected plan before requesting again; Async retains
its existing prefetch schedule. Stop or cancellation can prevent the whole plan
from being published. Publication is not proof of physical motion completion.

The backend owns the setting and includes it in central TaskInfo updates. Edit it
only while Ready or Paused. START/RESUME applies it to new plans without reloading
weights; cancelled in-flight results cannot enter the new generation. Slow Start
and saved-pose Return are independent. `InferenceCommand.action_steps=-1` means
preserve on START/RESUME and all on LOAD. Its status includes `loaded_action_steps`.

The ControlLoop also records `observed_chunk_size` from valid current-generation
responses before alignment, waypoint selection, or interpolation. The existing
Runtime STATUS and central InferenceStatus topic carry this read-only measurement;
no extra Worker RPC or per-browser polling is needed. After the first result the
UI shows `Max T` beside the selected limit, independent of the requested value
and without rewriting it. This is not a command receipt or proof that an
edited setting has been applied; edits still take effect on Start. Subsequent raw
chunk lengths may differ, so the measurement is not used as a guaranteed maximum.
Pause/Stop preserve it; deconfiguration resets it. Model-owned step queues report
zero rather than exposing their internal horizon. Unknown, stale, loading, or
different-model status does not supply a displayed length.

Model-owned `step` queues (including the current Diffusion public adapter) reject
numeric limits: one API result is not an exposed prediction chunk. All remains
supported. Supporting limits requires a reviewed public chunk/queue lifecycle
adapter, not private queue access or rewriting `n_action_steps`. Fixed-length
command resampling also cannot be combined with a waypoint limit.

After updating, regenerate `interfaces`, rebuild dependent ROS packages and the
UI, then restart Cyclo processes. Workers need matching rebuilt ROS definitions
if they consume the changed interfaces. Model checkpoints are unchanged. Do not
mix old and new ROS type hashes across running participants.

## Context Scope

The Runtime retains command execution ownership. Workers receive robot topics
directly and assemble only the inputs requested by the loaded adapter. Images
are not relayed through Cyclo. Existing chunk policies remain the compatibility
baseline; history and execution feedback are opt-in.

Multi-Task DiT is no longer supported by Cyclo. LingBot-VA's candidate adapter
remains excluded. RTC/TT-RTC are not implemented merely by adding context.
Model-specific execution semantics still need a reviewed adapter and tests.

Diffusion now uses the reviewed public step path for its model-owned observation
and action queues. This replaces its invalid latest-only chunk call; other chunk
adapters are unchanged. See the [input design decision](inference_input_design.md)
for the implemented YAML pipeline, verification and the boundary of declarative config.

## Ownership

```text
Robot topics -> Worker RobotClient -> ObservationSession -> Input Graph
                                          |                    |
                                 requested history only    model processor
                                                               |
                                                        model prediction
                                                               |
Cyclo ControlLoop <- Zenoh action response <--------------------+
       |
       +-> action processing -> robot command publication
       |
       +-> execution ledger -> requested feedback -> Worker
```

| Location | Responsibility |
| --- | --- |
| `policy/lerobot/lerobot_engine/adapters/` | Registry, model constraints, loaders, predictors and execution adapters |
| `policy/lerobot/configs/inference_inputs/`, `lerobot_engine/input_pipeline.py` | Explicit YAML inputs and framework operators |
| `policy/common/runtime/inference_inputs/` | Common graph, registered operators/providers and transactional memory |
| `policy/common/runtime/inference_context/inputs.py` | Model-independent queries, provider routing and tensor assembly |
| `inference_context/observation.py`, `history.py`, `reception.py` | Opt-in history, freshness, warmup and callback reset barriers |
| `inference_context/execution.py`, `execution_inputs.py`, `contract.py` | Wire records, requested execution history and LOAD contract |
| `sdk/action_chunk_processing/.../tracked_buffer.py` | Pending/in-flight commands, planning and publication receipts |
| `runtime/main_runtime/execution_feedback.py` | Bounded ledger projection and acknowledgement |
| `runtime/main_runtime/step_schedule.py` | Publication-paced step scheduling, separate from chunk alignment |
| `runtime/engine_process/worker.py` | Session identity, generation checks and retry deduplication |

Paths in shortened rows are relative to `policy/common/runtime/`.
See [adapter contracts](../policy/lerobot/lerobot_engine/adapters/README.md) for
hook signatures and how to register an input plan. Predictor construction belongs
to `loading.py`; model-specific implementation belongs to the adapter module.

## Input Semantics

- `SampleQuery` declares a source and optional non-positive time offsets.
  `ExecutionQuery` declares published commands, pending commands or execution facts.
- Only requested temporal sources allocate history. Latest-only policies do not
  acquire a history queue. Histories contain actual callback samples, not repeated
  reads of one cached image/state.
- Timestamps are local monotonic reception times, not synchronized exposure times.
  History cadence comes from the reviewed training contract, not automatically
  from Control Hz or Dataset FPS.
- Input plans compile at LOAD. Readiness is checked before image copies and GPU
  transforms; resolved inputs are reused during assembly.
- RobotClient subscribes only to required observations; shared physical joint
  topics use one subscription. Existing callers retain default subscriptions.
- Context generation changes clear retained inputs; cached LOAD and UNLOAD detach
  old capture sessions. Missing, stale or over-budget inputs fail explicitly.

## Execution Semantics

| Mode | Behavior |
| --- | --- |
| Existing chunk | Public chunk prediction, existing interpolation/alignment and sync/async scheduling |
| Contextual chunk | Same chunk scheduling with explicitly requested observation/execution history |
| Step | One public `select_action` result; next model step waits for publication and a Dataset FPS period |

Step mode bypasses interpolation and prefetch. It repeats the last position target
between steps, but expires nonzero Twist after the step period. Preview-only step
execution is rejected because it cannot produce publication receipts.

Predicted, planned and published commands are distinct records. Publication ACKs
do not confirm physical robot motion. Adapters requiring an exact match between
their cached actions and published commands reject modified publication values.
The adapter owns command-space/model-space conversion and bootstrap semantics.

Initial prediction and history warmup can run asynchronously in preparation state.
Stop performs local hold without waiting for that prediction; late results cannot
enter a replacement generation. Failed hold retains the pending stop state.
Once preparation completes, normal action timeouts apply. History wait time is
excluded from chunk alignment latency only when explicitly negotiated.

## Communication And Lifecycle

Engine protocol 3.1 retains execution context and `UPDATE_CONTEXT`, adding opt-in
feedback schema 2 for state commits and request prerequisites. Runtime and Worker
must be updated together; incompatible protocol majors are rejected before LOAD.
Context retries replay a successful result instead of advancing a stateful model
twice. Failed inference requires a new generation before retry.

Only unacknowledged bounded events and the requested pending-plan prefix travel
with the request. Contexts use a 64 KiB compression threshold and an 8 MiB wire
and decoded limit. Oversize payloads and lost journal continuity fail closed.

Worker status uses a separate metadata service and one shared polling loop per
runtime while readers are interested. Cached status is not authorization to LOAD.
Slow client construction does not hold the registry heartbeat lock.

Policy Runtime is a separate process in the integrated `cyclo_intelligence`
ROS launch, not a separate s6 longrun. A process lock prevents duplicate instances.
Component-only Orchestrator launch intentionally does not start Policy Runtime.
Shutdown rejects new lifecycle operations and removes readiness before cleanup.
Forced termination still requires a robot-controller command watchdog.

## Verification

See [verification summary](policy_extensibility_optimization.md) for tested scope,
performance measurements and remaining limitations. Physical behavior and robot
stop debugging are user-owned; automated tests use fake robot command sinks.

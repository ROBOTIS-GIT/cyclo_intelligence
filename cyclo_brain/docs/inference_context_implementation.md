# Inference Context

## Scope

The Runtime retains command execution ownership. Workers receive robot topics
directly and assemble only the inputs requested by the loaded adapter. Images
are not relayed through Cyclo. Existing chunk policies remain the compatibility
baseline; history and execution feedback are opt-in.

Multi-Task DiT is selectable in the Catalog for robot validation. LingBot-VA's
candidate adapter remains excluded. RTC/TT-RTC are not implemented merely by adding context.
Model-specific execution semantics still need a reviewed adapter and tests.

## Ownership

```text
Robot topics -> Worker RobotClient -> ObservationSession -> InputAssembler
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
| `policy/lerobot/lerobot_engine/input_plan.py` | Default latest-observation input declaration |
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

Engine protocol 3.0 adds execution context and `UPDATE_CONTEXT`. Runtime and Worker
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

# Adapter Contracts

The registry selects model-specific behavior once at LOAD. The common Worker,
control loop and input assembler do not need a new model-name branch.

## Registration

1. Add a model module here exporting an `AdapterDefinition`.
2. Register the checkpoint policy type in `registry.py`.
3. Add the selectable policy to `cyclo_brain/policy/lerobot/manifest.yaml` and its
   `configs/inference_inputs/<policy_type>.yaml` preprocessing settings.
4. Add dependencies only when the existing Worker environment lacks them.
5. Test the adapter contract before exposing a new execution mode to users.

Current hooks:

| Hook | Responsibility |
| --- | --- |
| `contract` | Chunk or publication-paced step execution |
| `step_factory` | Step prediction, processor reset and publication feedback |
| `checkpoint_validator` | Check assets and config before weight allocation |
| `requested_config_validator` | Check that the requested policy accepts the checkpoint format |
| `layout_validator` | Optional model-specific checks after common channel mapping validation, before waiting for topics |
| `batch_validator` | Check processed inputs, after the saved processor |
| `policy_loader` | Optional weight construction/device placement; saved processors still use the common loader |
| `predictor_factory` | Optional batch predictor prepared once per weights load, without replacing policy methods |
| `input_extensions` | Register trusted operators and source bindings for internal graphs |
| `input_handlers` | Map YAML handler names to Python graph builders, scoped to this model |
| `history_max_bytes` | Explicit memory budget for requested observation history |

Existing chunk policies resolve to the default adapter. FastWAM CPU loading,
instruction encoding/offload and MolmoAct2 action-mode defaults are adapter-owned.
`policy_loader(policy_class, config_class, model_path, device)` returns an eval
policy with placement already applied. `predictor_factory(policy, device, request)`
returns a callable accepting one processed batch. The Engine retains it with the
cached weights and drops it on UNLOAD or replacement. It must handle instruction
changes from subsequent batches; a cached LOAD does not reconstruct it.

Diffusion and Multi-Task DiT use the public step adapter: one latest observation
per published model step; temporal and action queues belong to `select_action`.
Do not also attach an observation history plan for those online APIs. Diffusion
validates camera sizes after the saved processor, before advancing its queues,
and rejects enabled relative-action processors that would reanchor cached actions.
The optional `PublicStepAdapter.batch_validator` sees the processed batch; failure
latches the step until reset, just like a processor/model failure.

`n_obs_steps` and `observation_delta_indices` are not universal inference contracts.
The latter is a training dataset sampling property and can contain future frames.
Review the public API before exposing temporal input options through a handler.
Cyclo does not derive sampling or layout from these model properties.
See the [design decision](../../../../docs/inference_input_design.md).

Chunk dispatch falls back to `select_action` only if there is no callable public
chunk method. Exceptions from an existing method, including `NotImplementedError`,
are model failures, not permission to advance the policy a second time. A model
with a placeholder chunk method must explicitly select its reviewed execution
adapter instead. Cached LOAD resets public policy/pre/postprocessor session state
for ordinary chunk policies without reloading weights. Contextual and step models
continue to reset at their initial context and generation boundaries.

The single-robot chunk predictor returns `(1, T, A)` (or `(1, A)` for one
action). Multiple batches/candidates must be explicitly resolved by the adapter;
the engine rejects them instead of silently commanding the first batch. The
postprocessor-to-NumPy boundary also accepts unbatched `(T, A)` and `(A,)` values
for existing step/processor APIs. Invalid batched output is rejected before a CPU
transfer. This does not infer whether a model uses absolute or relative actions.

## Temporal Observations

`input_extensions(registry, bindings, engine)` optionally registers additional
operator compilers or source binding factories. It runs during LOAD. The common
graph compiles Python-owned composition. User YAML only selects extra transforms
and an optional `preprocessing.custom.handler` with its options.
`input_handlers={"name": builder}` explicitly registers a trusted
`builder(base_graph, options) -> graph` function. Validate options strictly in the
builder; it runs before model allocation. Register model-bound calculations with
`input_extensions`, which runs once the model exists. Merely registering a handler
does not activate it; the YAML must select its name. There are no YAML imports.
The builder may extend the default graph, or replace it for a different model
input contract. If replacing it, it must explicitly preserve or consume any image
settings it claims to support; incompatible options must be rejected.
Standard bindings expose robot cameras/state and instruction. Tensor conversion,
temporal offsets and final output keys belong to the builder, not user wiring.
Registered code must use reviewed public model APIs, not private attributes.

Declare `SampleQuery` offsets in seconds, chronological and non-positive. Use an
explicit maximum sample age for history. Derive cadence from reviewed training
metadata; `n_obs_steps` alone does not specify a time interval. Do not assume that
Control Hz or action waypoint Dataset FPS defines camera history spacing.

`ObservationSession` retains only sources with temporal offsets. Other sources
use latest snapshots. Supported robot sources are `camera:<name>`, `joint:<group>`
and numeric sensor fields such as `sensor:odom.linear_velocity`. Current-only
queries may also request a complete sensor value such as `sensor:odom`.

- History consists of actual callbacks, never repeated reads of a cached frame.
- All offsets refer to local monotonic reception time, not sensor exposure time.
- Distinct requested times must resolve to distinct received samples.
- Missing, stale or memory-exhausted history prevents prediction.
- Histories reset at execution generation changes and detach on reload/unload.
- Chunk policies needing history negotiate context without becoming step policies.
- A latest sample in a step query must cross its publication reception barrier;
  negative offsets may legitimately refer to observations before that publication.
- History readiness is checked before copying current camera images. Inputs are
  resolved once before tensor transforms, so a missing later field does not cause
  partial GPU preprocessing.
- LeRobot creates a deferred RobotClient, compiles its input plan, then starts only
  the declared topic subscriptions. A synthetic joint view subscribes to its
  physical parent once. This is LOAD-time setup, not hot subscription replacement.
  Existing RobotClient callers retain their default subscription behavior.
- Selected physical joint groups sharing a topic and message type use one
  subscription and one vector conversion. All physical groups still expose the
  full message vector, while synthetic children remain name-based slices. Public
  snapshots return copies; message replacement cannot mutate retained samples.

### History Warmup

A temporal session derives its initial observation wait budget from its longest
negative offset plus one second for reception/readiness. The negotiated
`observation_warmup_timeout_s` is bounded to 120 seconds; excessive declarations
fail LOAD instead of silently shortening the requested history. After a valid
snapshot, ordinary reads return to the one-second observation wait. Reset makes
the session warm up again from new callbacks.

The Runtime adds this budget to the existing first-model-prediction budget (or
configured `GET_ACTION_TIMEOUT_S` when no special first budget exists). Preparation
is asynchronous, including Initial Pose Sync, and does not authorize robot motion.
Stop performs local hold and discards any late prediction. START cannot reuse the
session while that old prediction is still finishing. Heartbeat safety monitoring
remains active. Once prepared, subsequent action requests keep the normal timeout.

Temporal GET_ACTION responses include `observation_wait_s` in the existing JSON
metadata field. This is measured locally from snapshot-read start to its selected
reception anchor, before tensor transforms or model prediction. The Runtime checks
that it does not exceed the whole request duration, then subtracts it only for
action alignment and refill-latency estimation. Waiting for history must not cause
freshly predicted waypoints to be discarded as if they were already late. Missing
or malformed negotiated timing fails closed, including before pose-sync motion.
Legacy non-temporal models retain their existing metadata and timing behavior.

The end-to-end synthetic integration is in `tests/test_temporal_engine.py` under
the LeRobot policy directory. It uses actual RobotClient callbacks and Worker
requests, but substitutes small fake weights and disables transport/robot commands.

## Execution Inputs

In a Python handler's internal graph, use execution sources, not time-offset
sensor queries, for execution facts (the following is NOT user YAML):

```yaml
sources:
  previous_commands:
    source: execution:published:command
    count: 2
    min_count: 0
  next_commands:
    source: execution:pending:command
    count: 3
    min_count: 0
```

The former returns the most recent published records in order; the latter returns
the upcoming plan prefix, not its tail. A field transform receives a tuple of
records as one value. `values` are actual emitted command values; `planned_values`
retain the prediction before command adjustments. Publication is not robot motion
confirmation. Convert command coordinates to model coordinates explicitly in the
adapter, using the reviewed training contract.

`min_count=0` explicitly permits an empty bootstrap input before the first action.
The adapter decides its meaning. A missing required record fails prediction; the
engine never invents an action or waits for robot topics to fill execution inputs.
Only requested records are retained in bounded buffers. ACK-only updates do not
erase requested history, retries do not duplicate it, and reset boundaries discard
old-generation publications even if they remain unacknowledged on the wire.

`execution:events` exposes terminal action/planning/reset records.
`execution:context` returns the current immutable validated wire snapshot without
an additional history buffer; that snapshot is a delta, not a complete event log.
Optional `max_age_s` uses the same host's monotonic clock for published/events.
Pending commands have no age filter because they are future plans, not observations.

At LOAD, compiled input plans negotiate `pending_command_count`: zero when no
pending query exists, or the largest requested prefix. A raw `execution:context`
query preserves the full plan. The Runtime copies/serializes only that prefix;
the actual execution queue, terminal receipts, reset records and ACK interval are
unchanged. Step callbacks consuming execution context directly must declare any
pending-plan requirement in the input plan, just like batch inputs. Existing
contracts without this field retain full-plan behavior. This additive LOAD field
requires an updated Runtime and Worker together; older strict readers reject it.

Protocol 3.1 retains the 64 KiB compression threshold and the 8 MiB wire and
decoded limits. Large terminal/discard journals use one losslessly compressed
service request; no extra staging calls or partial model-state updates occur.
Compression is only used when it reduces bytes. Oversized contexts and gaps beyond
the bounded journal retention still fail explicitly without truncation. Runtime
rejects protocol 2.x Workers at DESCRIBE, before LOAD. Local two-process TCP/CDR
tests cover large receipts and retry deduplication, not remote-network deadlines.

Handler-defined memory/request prerequisites negotiate `feedback_schema=2`; legacy
contracts retain schema 1 wire fields. Runtime checks publication prerequisites
locally, without blocking Worker inference on future feedback. See the input
pipeline guide for proposal commit/discard conditions, budgets and result stages.

## Validation Boundaries

These providers do not automatically make an arbitrary action-feedback model
compatible. Each model still needs an explicit execution-space/cache/bootstrap
contract and tests for publication changes, lost ACKs, resets and delayed
predictions. For the tested transport/checkpoint scope, see the
[verification summary](../../../../docs/policy_extensibility_optimization.md).
Sensor exposure synchronization and remote clock mapping
are not provided by local reception timestamps.

# Policy Extensibility And Execution Efficiency

## Result

The code-structure audit completed on 2026-09-14. Architecture and contracts are
documented in [Inference Context](inference_context_implementation.md) and the
[adapter guide](../policy/lerobot/lerobot_engine/adapters/README.md).
This is not production deployment approval or physical robot validation.

| Concern | Implemented change |
| --- | --- |
| Model-name branches in common code | Registry-selected adapter hooks for loading, validation, inputs, prediction and reset |
| Unnecessary input work | LOAD-time plans, selected subscriptions, freshness before pixel copies, opt-in bounded history |
| Repeated context work | Single parsing, bounded ACK deltas, requested pending-plan prefix and lossless compression |
| Status contention | Separate metadata service, shared nonblocking status cache and per-runtime client-creation locks |
| Execution ambiguity | Distinct predicted/planned/emitted values, generation ownership and retry deduplication |
| Resource retention | Cached LOAD reuses weights; replacement/UNLOAD release predictors, subscriptions and model resources |
| Startup/shutdown ambiguity | Launch-managed Runtime, process lock, readiness cleanup and process-group shutdown tests |

The final audit also rejected missing input-plan factories at registration and
fixed Worker healthchecks that could mistake `down ..., normally up` for healthy.
Healthchecks compare structured `s6-svstat -o up` output with `true`, plus the
existing ready marker. Process-up alone is not model readiness.

## Regression Evidence

Before the cleanup, the latest disjoint host groups passed 820 tests:

| Group | Passed |
| --- | ---: |
| Common Runtime | 310 |
| LeRobot adapter/engine | 115 |
| Image preprocessing | 37 |
| UI | 184 |
| RobotClient/action processing | 97 |
| Docker/Supervisor | 77 |

After cleanup, the affected groups were rerun: 117 LeRobot tests (including two
new predictor-lifecycle checks), 37 image preprocessing, 310 Runtime and 77
Docker/Supervisor tests passed, totaling 541. UI and SDK counts above are from
the preceding audit, not new runs. Cleanup consolidated the predictor setup into
`LoadingMixin`, removed the redundant optimization mixin and obsolete test mocks,
and shortened documentation; it did not change execution contracts or upstreams.

Generated ROS interfaces and dependency-heavy policy processor tests are not
included in these host counts. Broad collection initially failed on missing ROS
types and a manual subscriber script; those failures are not counted as passes.

Additional isolated verification:

- Native ROS client to SDK server and SDK client to ROS server on AMD64/ARM64,
  preserving sequence IDs and UTF-8 context for 12, 13,324 and 851,980-byte payloads.
- Two-process TCP/CDR context transport, retry deduplication, and sync/async
  ControlLoop fault handling with fake robot command sinks.
- Cyclo/LeRobot candidate images on both architectures and standalone GR00T
  images: startup, packaged imports, DESCRIBE and lifecycle cleanup.
- Actual ACT and Multi-Task DiT checkpoint predictions with replay observations;
  standalone GR00T actual weights with synthetic observations on both architectures.
- ARM64 GR00T repeated reloads: three fresh instances, six predictions each,
  without accumulation at the final CUDA-allocation measurement.
- Final AMD64 GR00T: two cycles of three finite `(16, 16)` predictions including
  cached LOAD and UNLOAD, with zero robot commands. This particular transport was
  in-process; it does not measure robot-network latency.

Final AMD64 GR00T first prediction took 0.990 seconds, subsequent predictions
0.052-0.054 seconds. Peak CUDA allocation was 5.938 GiB; final allocation was
33,554,432 bytes. This bounded check is not a memory-leak proof for all workloads.

## Efficiency Measurements

Local CPU benchmark, with old/new wire content and snapshot equality assertions:

| Case | Before | After |
| --- | ---: | ---: |
| Encode 2,000 receipts | 10.970 ms | 8.425 ms |
| Extract 20 unacknowledged events from 4,096 retained events | 45.561 us | 1.158 us |

Run `python3 cyclo_brain/policy/common/runtime/tests/benchmark_execution_context.py`.
These are local measurements, not network/control jitter claims. Parsing a
2,000-receipt context still took 9.681 ms; optimization does not eliminate all cost.

## Reusable Integration Tools

These are explicit opt-in scripts, not automatically collected unit tests. Use
isolated containers, explicit memory limits and reviewed local assets. Do not run
them against production robot services.

| Script | Additional evidence beyond unit tests |
| --- | --- |
| `policy/common/runtime/tests/context_transport_smoke.py` | Real two-process service transport and ControlLoop fault paths |
| `sdk/robot_client/tests/subscription_transport_smoke.py` | Actual SDK subscriptions and message reception |
| `policy/lerobot/tests/public_step_smoke.py` | Pinned upstream public queue/reset behavior without neural allocation |
| `policy/lerobot/tests/checkpoint_context_smoke.py` | Actual checkpoint/processor predictions, reset and cached LOAD |
| `policy/lerobot/tests/zenoh_checkpoint_transport.py` | Separate-process transport helper for the checkpoint test |
| `policy/groot/tests/checkpoint_context_smoke.py` | Standalone GR00T actual weights and cleanup |
| `policy/groot/tests/resource_lifecycle_smoke.py` | Real CUDA allocation release under injected construction/robot-setup failures |

## Limits And Release Checks

- New model semantics still require a reviewed adapter. Temporal observation,
  publication history and arbitrary future feedback contracts are not equivalent.
- Local reception timestamps do not provide sensor exposure synchronization or
  remote-clock mapping. Step-model refill latency is not hidden by fast queue hits.
- More than 8 MiB of context or a gap beyond retained history is unsupported and
  fails explicitly. Physical command execution cannot be inferred from an ACK.
- Physical robot behavior, Stop/watchdog testing and real-network control jitter
  are user-owned and were not exercised by automatic robot commands.
- The existing AMD64 Blackwell GR00T profile does not pass `pip check`: strict
  upstream package pins differ from the selected Torch/vision/Flash Attention/
  TorchCodec profile, nightly dates differ, and TensorRT/CUDA bindings/decord
  metadata conflicts remain. Successful imports and checkpoints do not establish
  universally compatible dependencies. Release pinning remains separate work.
- Offline GR00T loading is not established. Initial tests failed on missing
  Cosmos weights, a tokenizer metadata lookup and gated HF authentication. The
  successful run used an isolated copy of the verified cache, an existing token
  mounted read-only, and network access for metadata. Production assets were not
  rewritten and upstream code was not patched to hide those failures.

No version bump, commit, push or deployment was part of this audit.

## Local Evidence Archive

### Pre-commit Recheck (2026-09-14)

After removing the temporary observation-dump implementation, separate test
processes passed 937 tests:

| Group | Passed |
| --- | ---: |
| Common Runtime | 312 |
| LeRobot adapter/engine (excluding opt-in real dependency tests) | 115 |
| RobotClient observation/safety and action processing | 89 |
| Container script, Dockerfiles and Supervisor | 88 |
| Catalog and image preprocessing | 77 |
| UI (29 suites, including BT serialization) | 190 |
| Standalone GR00T engine factory | 6 |
| Dynamic service definitions and ROS/CDR interoperability | 7 |
| Orchestrator inference lifecycle and launch | 53 |

Interfaces were rebuilt from scratch in an isolated temporary install. The
Orchestrator tests initially failed collection without generated interfaces and
the complete ROS/library/source paths; they passed with that fresh install.
No robot commands, running-service changes, new image builds or ARM64 reruns
were part of this recheck. The temporary observation recorder and its dedicated
tests/docs are removed; captured data, checkpoints and training scripts remain
ignored local artifacts, not runtime dependencies or commit contents.

### Earlier Evidence

Detailed chronological notes and intermediate failures are retained locally in
`docker/workspace/inference_context_campaign_20260910/cleanup_archive_20260914/`.
Machine-readable reports remain in the campaign's `reports/` directory, including
`optimization_amd64_candidate_groot_checkpoint.json`. These ignored local files
are not required by runtime code or tests and are not a portable release artifact.

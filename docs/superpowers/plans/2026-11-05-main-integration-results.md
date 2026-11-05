# Main Integration Verification

## Scope

- Integration branch: `integration/main-leaderboard-20261105`.
- Feature parent: `5970d90e0b4b63e0eb3826d890859916fc9a648c`.
- Main parent: `17134be33197febc86aafd7de5f9f591ff4bd1a4`.
- Validation worktree: `/tmp/cyclo-main-integration-20261105`.
- The active feature checkout, model data, live services and pinned submodule revisions were not changed.
- No robot commands, deployment, push or model training were performed.

## Resolution Decisions

- Keep centralized Policy Runtime, status-driven Slow Start, action-step contracts and STOP/cleanup safety.
- Keep manifest-based policy discovery and explicit local Worker builds. BT Docker operations use Supervisor reservations and must not build or pull as a side effect.
- Import main's Autonomy Studio, Action/Mission Canvas, navigation storage, BT support routes and converter fixes.
- Initialize Navigation alongside Worker build lifecycle management; close the build manager even if startup fails.
- Share the process-local policy catalog loader through `shared.policy_catalog`, without importing model packages or Orchestrator internals into BT.
- Prepare LeRobot's pinned sources and CPU torchvision in Python CI. Keep architecture-specific Docker behavior and release version changes.
- Reject automatic reload after ambiguous RESUME failure. An intervening Stop/Pause invalidates late RESUME results. Explicit Clear and Start remain the recovery path.
- For BT INFERENCE LOAD, inspect a running Worker's readiness without requesting Docker mutation. Only a known stopped/missing container needs guarded START; checkpoint replacement remains Orchestrator-owned.
- Make both ROS entrypoints tolerate an already-closed context during shutdown, without changing service cleanup order. Full isolated bringup exposed this pre-existing SIGINT error.

## Automated Evidence

Counts are separate suite runs; repeated and overlapping tests are not a unique-test total.

| Area | Result | Environment / qualification |
| --- | --- | --- |
| UI | 81 suites, 941 passed | `npm ci --legacy-peer-deps`; full Jest suite and production build |
| Supervisor and Worker build lifecycle | 201 passed | Mock Docker/runtime; no live mutations |
| Catalog and central Runtime | 455 passed | Host; temporary Unix sockets permitted |
| Orchestrator lifecycle and recording session | 119 passed | Verified both source-mounted and baked into the validation image with generated ROS interfaces |
| Entrypoint shutdown | 80 passed | Open/closed context, SIGINT, external shutdown, native shutdown race, unrelated exceptions, cleanup order |
| Safety repeat | 148 passed on each of 5 runs | Control loop, service handler, saved pose, Worker snapshot, Runtime socket |
| Recovery repeat | 100 passed on each of 3 runs | Final Orchestrator recovery and Slow Start suites in isolated ROS environment |
| BT and shared | 156 passed | Host; mocked ROS; repeated LOAD and cancellation without unwanted Docker STOP |
| RobotClient and generated ROS/CDR | 72 passed | New main image, generated Jazzy interfaces, no network |
| Action chunk processing | 93 passed | Host |
| LeRobot engine/input/extensions | 280 passed, 2 skipped | Existing LeRobot Worker image, merged source read-only |
| GR00T and RLDX adapters | 39 passed | Host |
| LingBot adapter/training helpers | 220 passed, 1 skipped | Existing LingBot image, merged source read-only |
| ABot adapter/training helpers | 87 passed, 48 skipped | Inference image lacks training data dependencies; not full training validation |
| Converter | 258 passed, 1 skipped | MCAP/ROS collection stubs; checks transformation logic, not real bag decoding |
| Real LeRobot dataset reader | 3 passed | Separate Worker run closes converter suite's missing-LeRobot skip; instructions reach DataLoader |
| Recording and reader | 98 passed | Host test doubles |
| Data editing and replay helpers | 36 passed | Host test doubles |
| CLI and dependency profile checks | 43 passed | Includes CI prerequisite regressions |
| Dockerfile contracts | 26 passed | Static checks |
| Compose | Passed for amd64 and arm64 | Configuration validation, not ARM execution |

Representative commands (from the integration worktree):

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=cyclo_brain/policy/common \
  python3 -m pytest -q cyclo_brain/policy/common/catalog/tests cyclo_brain/policy/common/runtime/tests
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q docker/supervisor_api/test_app.py docker/supervisor_api/test_worker_build.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=orchestrator:shared \
  python3 -m pytest -q orchestrator/tests/bt shared/tests
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=cyclo_data:orchestrator:shared \
  python3 -m pytest -q orchestrator/tests/data_processing/test_rosbag_to_lerobot_converter.py cyclo_data/tests/converter
docker build --progress plain -f docker/Dockerfile.amd64 \
  -t cyclo-validation/main-integration:20261105 .
```

Run suites separately: several legacy test modules install global ROS/reader stubs during collection. Do not collect `test_rosbag_subscriber.py` as a unit test: it is a live subscriber script.

The RESUME recovery regression was demonstrated before the fix (15 failing cases). Final cases include lost responses, late successful responses, same-client Stop/Pause, active Slow Start, retained recording, and operator Stop/Clear after failure. No automatic reload or cleanup is permitted on ambiguous RESUME failure.

Real SIGINT testing also exposed a native Jazzy race inside `try_shutdown()` itself. Entrypoints suppress only the known already-shutdown error when the context is confirmed closed; other errors still propagate. Unit mocks alone did not expose this issue.

## Image and Browser Evidence

Final validation tag: `cyclo-validation/main-integration:20261105`.
Image ID: `sha256:f0cf7133f8db4994f71c3ab78d16b9988e12d2cb65677f6552520ecd78bad791`.

The amd64 main image built successfully, including all six colcon packages and UI assets. An isolated s6 startup used `--network none`, no Docker socket, no GPU, no host ports and a dedicated ROS domain. Health, manifest catalog, OpenAPI navigation/BT routes, Nginx UI and API proxy returned successfully. Installed shared catalog and BT imports also passed without source mounts. The temporary container was stopped and removed.

After the shutdown-race fix, full ROS bringup/shutdown passed three consecutive source-bound runs and one final run using only baked image code. Policy Runtime, Orchestrator, cyclo_data, rosbridge, bag recorder and web video server all exited cleanly; launch returned zero without forced termination. The final image also passed all 199 lifecycle/entrypoint cases. Idle shutdown still logs best-effort recording/status requests after ROS transport closes; this is not verification of shutdown during active recording.

Browser evidence is under `/tmp/cyclo-ui-smoke-20261105`: 14 functional check groups, 17 screenshots, zero browser errors. All API and ROS interactions were mocked. Five Workers, build/rebuild/update semantics, Action Steps, BT inspector and Autonomy Studio navigation were exercised at 1440x1000, 768x1024 and 390x844. Browser and temporary server were closed.

## Limits and Remaining Manual Checks

- No real robot STOP, Slow Start, saved-pose motion, camera replay or model inference was executed on this merged image.
- No ARM64/Jetson build or execution was performed. The inherited ARM CUDA library-path change needs hardware validation.
- Existing Worker images were used for adapter tests; their image builds and thirteen checkpoint inference runs from before the merge are baseline evidence only.
- ABot/LingBot full training, teacher weights and checkpoint reload were not rerun. Optional dependency skips are not passes.
- Real Nav2 goal execution, robot mapping, actual camera/URDF rendering, mission persistence across deployments and live recording/conversion need operator validation.
- Inference toolbar clipping is pre-existing in feature-leaderboard; mobile Action/Mission Canvas overflow is inherited from main. Functional browser success is not a responsive-layout pass. No unrelated UI redesign was included.
- UI build reports existing MediaPipe source-map, stale Browserslist and bundle-size warnings. ROS build reports existing deprecated C++ service QoS overload warnings.
- Staged whitespace checks pass except three imported font license texts, which retain main's original CRLF/trailing spaces verbatim.
- GitHub-hosted CI itself has not run for this integration branch; local/static checks do not establish hosted CI success.

Before replacing active containers, Stop/Clear inference, preserve deployment-specific configuration, and test the validation image with existing Worker images in an isolated ROS domain. Keep original images available for rollback.

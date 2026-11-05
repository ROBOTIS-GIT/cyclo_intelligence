# Main Integration Verification Plan

**Goal:** Integrate main 17134be3 into feature-leaderboard 5970d90 without regressing centralized inference safety or local Worker builds.

**Architecture:** Keep Cyclo's central Policy Runtime and manifest-driven Workers. Bring in main's Navigation, Mission/Action Canvas, converter fixes and release metadata; adapt their boundary calls to the current contracts.

**Constraints:** Work only in the isolated integration tree. Preserve model data and submodule pins. Do not restart live services, publish robot commands, push or replace the active branch during validation.

## Tasks
- [x] Establish clean source baseline and create isolated integration worktree.
- [x] Resolve Docker/Compose/CI/docs and converter conflicts; retain external Node builds and no Worker pulls.
- [x] Integrate Supervisor navigation initialization and BT routes with Worker build lifecycle and safety reservations.
- [x] Integrate BT commands with dynamic policy discovery and central Runtime.
- [x] Preserve Runtime-driven Slow Start and stale-client protection in Orchestrator.
- [x] Integrate UI controls and run the complete UI suite and production build.
- [x] Run converter, recording, catalog, Runtime, SDK, model adapter, Docker and Supervisor suites separately where test stubs interfere.
- [x] Review automatic merges, conflict markers, compile/syntax checks, release paths and preserved submodule pins.
- [x] Build an isolated validation image and validate startup without a robot network where practical.
- [x] Obtain independent integration review; record exact evidence and any blocked hardware tests before finalizing.

## Review Focus
- STOP during lifecycle work and failed publishing must retain centralized safety behavior.
- New BT Docker commands must neither bypass mutation reservations nor pull/build implicitly.
- Supervisor startup must initialize Navigation without losing Worker build shutdown cleanup.
- Converter task instructions, frame indices, independent action hold and cache versions must remain compatible.
- UI automatic merges must preserve recording/pose/evaluation behavior and dynamic model selection.

## Baseline
Before integration, Docker/Supervisor tests: 152 passed; Worker UI tests: 8 passed. Five amd64 Blackwell Worker builds and thirteen representative checkpoint inference tests were previously completed on 5970d90's source changes. These are baseline evidence, not evidence for this merged tree.

## Review Follow-ups
- Fixed Python CI prerequisites: initialize the pinned LeRobot submodule and install matching CPU torch/torchvision together.
- Moved process-local catalog loading into shared so BT does not depend on Orchestrator internals.
- Reproduced and fixed an unsafe main recovery interleaving: lost RESUME response followed by successful STOP could cause fresh LOAD/START. Ambiguous failures now retain the session; late replies cannot override newer Stop/Pause intent. Final lifecycle suite: 119 passed.
- BT repeated INFERENCE LOAD must check readiness without mutating an already-loaded Worker. Explicit Docker operations continue to require mutation reservations.
- Browser smoke passed functional checks but identified pre-existing inference-toolbar clipping and narrow-screen Canvas overflow. Record these separately from merge regressions.
- Full isolated ROS bringup exposed a pre-existing SIGINT shutdown race. Both entrypoints now handle only the known already-closed-context error. Three source-bound bringup/shutdown runs and one final baked-image run passed; all six processes exited cleanly.

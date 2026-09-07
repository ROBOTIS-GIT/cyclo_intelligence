# Architecture - cyclo_intelligence

As-built inference topology after `docker/container.sh start`.

- Detailed code map: [`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md)
- Visual map: [`cyclo_brain/docs/architecture.html`](../cyclo_brain/docs/architecture.html)
- Refactoring report: [`cyclo_brain/docs/policy_runtime_refactoring_implementation.html`](../cyclo_brain/docs/policy_runtime_refactoring_implementation.html)

## Container Topology

```text
cyclo_intelligence
├── UI / nginx
├── supervisor_api
├── orchestrator
├── cyclo_data
└── policy-runtime
    ├── lifecycle and global session
    ├── action chunk processing
    ├── safety and initial pose sync
    └── robot command publishers

lerobot_server                   groot_server
└── engine-process               └── engine-process
    ├── observation subscribers      ├── observation subscribers
    ├── LeRobot policy               ├── GR00T policy
    └── action chunk inference       └── action chunk inference
```

The central Policy Runtime owns control behavior once. Worker containers own
only framework-specific model dependencies, observation preprocessing, and
inference.

## Data Flow

```text
UI / BT
  -> Orchestrator
  -> /policy/inference_command
  -> Policy Runtime
  -> /<runtime>/engine_command
  -> selected Engine Worker
  -> action chunk
  -> Policy Runtime buffer/control loop
  -> RobotClient command publishers
  -> Robot
```

Workers subscribe to camera, state, and sensor topics directly over
ROS2/Zenoh. Images are not relayed through the Cyclo container. Policy Runtime
subscribes only to joint state when robot command safety or initial pose sync
requires it.

## Contracts

| Contract | Owner | Purpose |
|---|---|---|
| `/policy/inference_command` | Policy Runtime | Public policy lifecycle and session control |
| `/<runtime>/inference_command` | Policy Runtime | Temporary compatibility aliases |
| `/<runtime>/engine_command` | Engine Worker | Internal model load, action, describe, status, unload |
| `policy/<runtime>/manifest.yaml` | Catalog | Policy IDs, capabilities, UI parameters, checkpoint root |

`EngineCommand` echoes `seq_id` so late responses can be discarded after a
timeout. Its protocol and each Worker descriptor are checked before LOAD.

## Deployment Boundary

UI, Orchestrator, control loop, safety, and action processing changes rebuild
the Cyclo image. A model adapter or framework dependency change rebuilds only
its Worker image. Wire-protocol changes require Cyclo and affected Workers to
be released as one compatible set.

# orchestrator

ROS 2 package with two independent processes:

- `orchestrator_node` — the control plane: session state, UI command
  routing, policy container lifecycle. Pairs with `cyclo_data` (data
  plane) over well-defined srv boundaries.
- `bt_node` — the behaviour-tree engine (`orchestrator/bt/`). It shares
  no Python with the control plane (enforced by
  `tests/bt/test_package_boundary.py`) and talks to it only through
  `interfaces` services; it is started on demand by the supervisor API as
  the `bt_node` s6 service. Mission sequencing (nav2 goals + per-waypoint
  trees) is currently driven by the Autonomy Studio UI, and saved trees are
  owned by the supervisor API (`/api/bt/trees`).

```
orchestrator/
├── orchestrator_node.py       ROS2 Node entry — class OrchestratorNode.
│                              Follows the cyclo_data <pkg>_node.py
│                              convention.
├── launch/                    ros2 launch files.
│   ├── orchestrator.launch.py          OrchestratorNode only.
│   ├── orchestrator_bringup.launch.py  OrchestratorNode + rosbridge
│   │                                   + rosbag_recorder +
│   │                                   web_video_server.
│   └── bt_node.launch.py      BT node bringup.
├── bt/                        Behaviour Tree subsystem (bt_node process).
│     ├── bt_core.py           NodeStatus, BTNode base classes.
│     ├── bt_node.py           BehaviorTreeNode ROS2 Node
│     │                        (orchestrator_bt_node). Provides
│     │                        /bt/nodes/catalog, /bt/load_and_run,
│     │                        /bt/set_running, /bt/status, and
│     │                        /bt/active_nodes.
│     ├── bt_nodes_loader.py   XML → runtime tree assembly via the
│     │                        dynamic node registry.
│     ├── node_registry.py     Scans actions/controls and builds the
│     │                        Behavior Trees catalog from class signatures.
│     ├── blackboard.py        Shared-state blackboard.
│     ├── constants.py         Runtime defaults for BT actions.
│     ├── actions/             Built-in and user-defined action nodes.
│     ├── controls/            loop / sequence / base_control.
│     ├── templates/           Copy-and-edit templates for custom
│     │                        Action / Control BT nodes.
│     └── trees/               Example behavior tree XML, installed under
│                              share/orchestrator/bt/trees/ and seeded
│                              into the user tree directory
│                              (CYCLO_BT_TREES_DIR, default
│                              /workspace/bt/trees) by the supervisor.
│
├── internal/                  Node-local utilities — not part of
│     │                        the inter-package import surface
│     ├── communication/       ROS2 client wrappers.
│     │   ├── communicator.py              Pub/sub for sensor topics.
│     │   ├── container_service_client.py  InferenceCommand.srv
│     │   │                                 dispatcher.
│     │   │                                 + stop_training /
│     │   │                                 get_training_status.
│     │   └── cyclo_data_client.py         cyclo_data srv wrapper.
│     ├── device_manager/      Hardware health / heartbeat monitor.
│     └── file_browser/        BrowseFile.srv implementation.
│
├── training/                  Training container client-side.
│   └── zenoh_training_manager.py
│                              Client for the /<backend>/train srv
│                              on policy containers. Left in the
│                              orchestrator package for now.
│
├── timer/                     Shared TimerManager wrapper.
│
├── ui/                        React UI app. Built by the
│                              Dockerfile.{arm64,amd64} stage-1
│                              node:22 stage and copied into
│                              /usr/share/nginx/html.
│
└── scripts/                   Orchestrator-specific dev helpers.
    └── test_rosbridge_connection.py
                               Manual rosbridge smoke test.
                               Data-side CLIs live in cyclo_data.
```

## Responsibilities — what stays here vs moves to cyclo_data

| Area | Owner | Why |
| --- | --- | --- |
| Session state (`on_recording`, `on_inference`, `operation_mode`, etc.) | orchestrator | central state the UI polls via `/task/status` |
| UI command routing (`/send_command`) | orchestrator | UI-side boundary — orchestrator translates to the appropriate downstream srv |
| Robot control plane publishers | orchestrator | synchronous `JointTrajectory` / `Twist` commands from tree nodes |
| Policy container lifecycle | orchestrator | `InferenceCommand` dispatch, client ownership |
| Behaviour tree catalog + execution | `bt_node` (this package, separate process) | owns every `/bt/*` ROS interface: `/bt/nodes/catalog`, `/bt/load_and_run`, `/bt/set_running`, `/bt/status`, `/bt/active_nodes` |
| Saved trees, `bt_node` lifecycle, robot capability check | supervisor API | `/api/bt/trees`, `/api/bt/support`, `/api/services/bt_node/*`; the supported robot list is `shared.robot_configs.schema.BT_SUPPORTED_ROBOT_TYPES` |
| Recording / conversion / HF / editing | cyclo_data | data-plane workers |
| Dataset visualisation | cyclo_data | `video_file_server`, replay handlers |

## Key srv / topic surface

| Direction | srv / topic | Notes |
| --- | --- | --- |
| UI → orchestrator | `SendCommand.srv` | START_RECORDING / START_INFERENCE / etc. — routed by `user_interaction_callback` to cyclo_data / policy containers |
| orchestrator → policy | `InferenceCommand.srv` | `ContainerServiceClient.inference_command(CMD_*, ...)` |
| orchestrator → cyclo_data | `RecordingCommand` / `StartConversion` / `HfOperation` / `EditDataset` | `CycloDataClient` wraps each |
| cyclo_data → orchestrator | `/data/status` topic | Relayed into `/task/status` for the UI |

## BT node lifecycle

`BehaviorTreeNode` (`bt/bt_node.py`) runs as the `bt_node` executable.
The normal bringup launch no longer starts it automatically:
```
ros2 launch orchestrator orchestrator_bringup.launch.py
```

Autonomy Studio → **Action Canvas** owns the normal process lifecycle through
the supervisor API:

- **Task Engine Turn On** starts the `bt_node` s6 service and refreshes the catalog.
- **Task Engine Turn Off** stops the `bt_node` s6 service, but only after task execution
  has been explicitly stopped.

For isolated debugging, launch only the BT node with:
```
ros2 launch orchestrator bt_node.launch.py robot_type:=ffw_sg2_rev1
```

`bt_node.launch.py` receives `robot_type`, checks it against
`shared.robot_configs.schema.BT_SUPPORTED_ROBOT_TYPES` (the only place the
supported list is defined), derives joint/topic parameters from
`shared/robot_configs/<robot_type>_config.yaml`, and starts with no
preloaded tree. The Action Canvas workspace supplies task XML through
`/bt/load_and_run`.

Action Canvas **Run Task/Stop Task** controls task execution, not the `bt_node`
process:

- **Run Task** cleans up a previous terminal execution when needed, serializes the
  current graph, and calls `/bt/load_and_run`.
- **Stop Task** calls `/bt/set_running` with `false`.
- **Run Task** is disabled while the `bt_node` process is down.
- While a task is running, **Stop Task** is enabled and **Run Task** is disabled.
- When a task completes or fails, **Stop Task** is disabled and **Run Task** becomes
  available again. **Task Engine Turn Off** performs the required runtime cleanup before
  stopping the process from one of these terminal states.

## Custom BT nodes

User-defined nodes are plain Python files under
`orchestrator/orchestrator/bt/actions/` or
`orchestrator/orchestrator/bt/controls/`. The BT registry scans those
folders dynamically, so editing `actions/__init__.py` or `controls/__init__.py`
is not required for Behavior Trees discovery or XML execution. Those files are
only for package-level imports.

Start from the templates in `orchestrator/orchestrator/bt/templates/`
(installed to `share/orchestrator/bt/templates/`):

- `action_template.py` subclasses `BaseAction`, defines constructor kwargs,
  implements `tick()`, and resets local runtime state.
- `control_template.py` subclasses `BaseControl`, defines constructor kwargs,
  ticks children, reports active child IDs, and resets its child index.

Class names become XML tags. Constructor kwargs become Behavior Trees ports; type
hints and defaults become port types and default values. No `META`,
`NODE_TAG`, `PORT_METADATA`, or description block is required.

Simple nodes need only an `__init__()`, `tick()`, and optional `reset()`.
Use `from_xml_params(context, name, params)` only when a node needs runtime
dependencies from the loader, such as the ROS node, topic config, joint names,
or helper methods. Built-in examples include `Rotate`, `JointControl`, and
`SendCommand`.

After adding or deleting a node file, open Autonomy Studio → **Action Canvas**
and click **Refresh Steps**.
If running from an installed package instead of a source-mounted workspace,
rebuild/restart first so the new file exists in the install space.

### SendCommand targets

`SendCommand` owns both inference commands and the lifecycle of their policy
backend without exposing arbitrary Docker names or images:

- `target=INFERENCE` (the default): `LOAD`, `RESUME`, `STOP`, or `CLEAR`.
  `LOAD` first ensures the selected `model` backend is installed, running, and
  internally ready, then continues through the existing inference stages.
- `target=DOCKER`: `START`, `STOP`, or `RESTART` for the allowlisted `groot` or
  `lerobot` backend selected by `model`. `START` is idempotent.

When `START` or `RESTART` finds no local image, the supervisor tries the exact
release image from the registry and falls back to that backend's allowlisted
Compose build. Trees saved before the `target` port existed continue to load as
`target=INFERENCE`.

## Action Canvas XML saving

The Action Canvas workspace lists, opens and saves XML task files through
the supervisor API (`GET/POST /api/bt/trees`, `GET /api/bt/trees/<name>`).
Files live under `CYCLO_BT_TREES_DIR` (default `/workspace/bt/trees`, i.e.
`docker/workspace/bt/trees` on the host) — user data, outside any package.
A duplicate file name is rejected by default to prevent accidental
overwrite; the UI shows an explicit **Overwrite** action only after the
server reports a name conflict.

On first listing the supervisor seeds that directory, without overwriting
anything, from the packaged examples (`share/orchestrator/bt/trees`) and
from the directory releases before 1.4.0 saved into
(`orchestrator/orchestrator/bt/trees` of the checkout), so previously saved
trees keep appearing after an upgrade. Seeding is one-way: once a tree is
in the user directory it is never refreshed from the sources.

## Entry points

After `colcon build`:

- `orchestrator_node` — main orchestrator node.
- `bt_node` — behaviour tree runner.

Both dropped into `install/orchestrator/lib/orchestrator/`.

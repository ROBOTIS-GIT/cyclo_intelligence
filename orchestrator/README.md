# orchestrator

Standalone ROS2 node that owns the control plane — session state, UI
command routing, policy container lifecycle, behaviour-tree driven
execution. Pairs with `cyclo_data` (data plane) over well-defined srv
boundaries.

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
├── config/                    Robot-specific YAML
│                              (ffw_sg2_rev1_config.yaml, etc.).
│                              Top-level key is 'orchestrator'
│                              (Step 2 Import Fixer).
│
├── bt/                        Behaviour Tree subsystem.
│     ├── bt_core.py           NodeStatus, BTNode base classes.
│     ├── bt_node.py           BehaviorTreeNode ROS2 Node
│     │                        (orchestrator_bt_node). Loads tree
│     │                        XML from share/orchestrator/bt/trees/
│     │                        at startup.
│     ├── bt_nodes_loader.py   XML → runtime tree assembly.
│     ├── blackboard.py        Shared-state blackboard.
│     ├── constants.py         Tree-loading magic strings.
│     ├── actions/             8 action nodes (move_arms, move_head,
│     │                        move_lift, rotate, wait,
│     │                        send_command, inference_until_gripper,
│     │                        inference_until_position_with_gripper).
│     ├── controls/            loop / sequence / base_control.
│     ├── trees/               Robot-specific tree XML
│     │                        (ffw_sg2_rev1.xml, korea_mat.xml).
│     │                        Installed under share/orchestrator/
│     │                        bt/trees/.
│     └── bringup/             bt_node_params.yaml installed to
│                              share/orchestrator/bt/bringup/.
│
├── internal/                  Node-local utilities — not part of
│     │                        the inter-package import surface
│     │                        (drift D4, Step 2).
│     ├── communication/       ROS2 client wrappers.
│     │   ├── communicator.py              Pub/sub for sensor topics.
│     │   ├── container_service_client.py  InferenceCommand.srv
│     │   │                                 dispatcher (Step 4-F).
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
                               (Data-side CLIs moved to cyclo_data
                               in Step 7.)
```

## Responsibilities — what stays here vs moves to cyclo_data

| Area | Owner | Why |
| --- | --- | --- |
| Session state (`on_recording`, `on_inference`, `operation_mode`, etc.) | orchestrator | central state the UI polls via `/task/status` |
| UI command routing (`/send_command`) | orchestrator | UI-side boundary — orchestrator translates to the appropriate downstream srv |
| Robot control plane publishers | orchestrator | synchronous `JointTrajectory` / `Twist` commands from tree nodes |
| Policy container lifecycle | orchestrator | `InferenceCommand` dispatch, client ownership |
| Behaviour tree execution | orchestrator | `/bt/load_and_run` is a control-plane trigger |
| Recording / conversion / HF / editing | cyclo_data | data-plane workers (Step 3 atomic swaps) |
| Dataset visualisation | cyclo_data | `video_file_server`, replay handlers |

## Key srv / topic surface

| Direction | srv / topic | Notes |
| --- | --- | --- |
| UI → orchestrator | `SendCommand.srv` | START_RECORDING / START_INFERENCE / etc. — routed by `user_interaction_callback` to cyclo_data / policy containers |
| orchestrator → policy | `InferenceCommand.srv` | `ContainerServiceClient.inference_command(CMD_*, ...)` |
| orchestrator → cyclo_data | `RecordingCommand` / `StartConversion` / `HfOperation` / `EditDataset` | `CycloDataClient` wraps each |
| cyclo_data → orchestrator | `/data/status` topic | Relayed into `/task/status` for the UI |

## BT node lifecycle

`BehaviorTreeNode` (`bt/bt_node.py`) runs as a separate executable
(entry_point: `bt_node`) that plugs into the orchestrator ROS2
graph. Launch it with:
```
ros2 launch orchestrator bt_node.launch.py robot_type:=ffw_sg2_rev1
```

The tree XML is loaded from `share/orchestrator/bt/trees/<tree>.xml`;
params come from `share/orchestrator/bt/bringup/bt_node_params.yaml`.

## Entry points

After `colcon build`:

- `orchestrator_node` — main orchestrator node (Step 5-B rename).
- `bt_node` — behaviour tree runner (Step 5-A).

Both dropped into `install/orchestrator/lib/orchestrator/`.

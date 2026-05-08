# cyclo_data

Standalone ROS2 node that owns the data plane — everything that
produces, stores, transforms, reads, or publishes datasets. The
orchestrator never touches a bag writer or an HF API client
directly; it issues a srv call and `cyclo_data` does the work.

```
cyclo_data/
├── cyclo_data_node.py         ROS2 Node entry — class CycloDataNode.
│                              Wires up the services below on an
│                              rmw_zenoh-backed ROS2 node.
├── services/                  Thin srv callbacks that dispatch to
│   │                          the subsystem workers below.
│   ├── recording_service.py   RecordingCommand.srv
│   ├── conversion_service.py  StartConversion.srv /
│   │                          GetConversionStatus.srv
│   ├── hub_service.py         HfOperation.srv
│   └── edit_service.py        EditDataset.srv
├── recorder/                  Recording subsystem.
│   ├── rosbag_recorder/       C++ recorder node (existing from
│   │                          physical_ai_tools)
│   ├── rosbag_control.py      Python wrapper that starts / stops
│   │                          the C++ node via ROS2 srv.
│   ├── session_manager.py     DataManager — session state +
│   │                          folder naming.
│   └── replay_handler.py      Replay-data API (consumed by the
│                              video_file_server).
├── reader/                    Low-level bag reading + metadata.
├── converter/                 MCAP → MP4 → LeRobot conversion chain.
│   ├── orchestrator.py        pipeline_worker coordinator.
│   ├── pipeline_worker.py     Long-running worker (atomic swap
│   │                          pattern — Step 3 C2e).
│   ├── rosbag2mp4.py          MCAP → MP4 stage.
│   ├── to_lerobot_v21.py      MP4 → LeRobot v2.1 stage.
│   ├── to_lerobot_v30.py      MP4 → LeRobot v3.0 stage.
│   ├── video_encoder/         ffmpeg wrapper.
│   └── scripts/               CLI — convert_rosbag_to_lerobot
│                              (console_script, Step 7).
├── editor/                    Episode edits.
│   ├── episode_editor.py
│   └── scripts/               CLI — remove_head_lift_joints
│                              (console_script, Step 7).
├── quality/                   Timestamp gap / drop analysis.
├── hub/                       HuggingFace Hub upload / download.
│   ├── hf_worker.py
│   ├── endpoint_store.py      User-configurable HF endpoint list.
│   └── templates/             Markdown templates for HF model
│                              cards.
└── visualization/             Data visualization.
    ├── rosbag_visualizer.py
    ├── video_file_server.py   HTTP server on port 8082 — serves
    │                          replay-data / rosbag-list / task-markers.
    │                          nginx /data-api/ proxies here.
    └── scripts/               CLI — visualize_rosbag
                               (console_script, Step 7).
```

## 7-way decomposition

The split above follows PLAN §4.2's "what does this file's subject
do?" rule. The prior `physical_ai_server/data_processing/` directory
had 14 files in one folder, each answering a different question
(recording? conversion? quality? hub upload?). Breaking them out
per-subsystem makes the orchestrator ↔ cyclo_data srv boundary
easy to map to code.

## Atomic swaps (Step 3 pattern)

Conversion, HF upload, and recording each migrated from orchestrator
to cyclo_data via "atomic swap" — one commit adds the srv +
worker pair to cyclo_data, the next flips orchestrator's handler to
forward via the new srv, the next removes the orchestrator-side
remnants. That pattern is traced in REVIEW §9 (recording is the
most detailed write-up: C2d-plan through C2d-5).

## srv boundary with orchestrator

| srv | File |
| --- | --- |
| `RecordingCommand` | [`services/recording_service.py`](services/recording_service.py) |
| `StartConversion` / `GetConversionStatus` | [`services/conversion_service.py`](services/conversion_service.py) |
| `HfOperation` | [`services/hub_service.py`](services/hub_service.py) |
| `EditDataset` | [`services/edit_service.py`](services/edit_service.py) |

Progress on long-running tasks (conversion, HF upload) is published
on `/data/status` as `DataOperationStatus`; orchestrator relays it
into `/task/status` so the UI doesn't see two topics.

## Console scripts (Step 7)

`setup.py` registers three entry points (D9 resolved in Step 7):

```
visualize_rosbag          → cyclo_data.visualization.scripts.visualize_rosbag:main
convert_rosbag_to_lerobot → cyclo_data.converter.scripts.convert_rosbag_to_lerobot:main
remove_head_lift_joints   → cyclo_data.editor.scripts.remove_head_lift_joints:main
```

After `colcon build`, run any of these by name:
```
ros2 run cyclo_data visualize_rosbag /path/to/file.mcap --detailed
```

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
│   ├── rosbag_recorder/       C++ recorder node.
│   ├── rosbag_control.py      Python wrapper that starts / stops
│   │                          the C++ node via ROS2 srv.
│   ├── session_manager.py     DataManager — session state +
│   │                          folder naming.
│   └── replay_handler.py      Replay-data API (consumed by the
│                              video_file_server).
├── reader/                    Low-level bag reading + metadata.
├── converter/                 MCAP → MP4 → LeRobot conversion chain.
│   ├── orchestrator.py        pipeline_worker coordinator.
│   ├── pipeline_worker.py     Long-running worker for conversion tasks.
│   ├── rosbag2mp4.py          MCAP → MP4 stage.
│   ├── to_lerobot_v21.py      MP4 → LeRobot v2.1 stage.
│   ├── to_lerobot_v30.py      MP4 → LeRobot v3.0 stage.
│   ├── video_encoder/         ffmpeg wrapper.
│   └── scripts/               CLI — convert_rosbag_to_lerobot.
├── editor/                    Episode edits.
│   ├── episode_editor.py
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
    └── scripts/               CLI — visualize_rosbag.
```

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

## Recording camera calibration

The recorder keeps the first calibrated `CameraInfo` per camera and releases
that camera's calibration subscription. This does not stop its image stream.
Save copies available calibration (including its original timestamp) to
`camera_info/<camera>.yaml` without waiting for missing cameras. Discard
does not write calibration or clear the cache.

Calibration must remain fixed while using the same robot profile. If replacing
a camera or changing its resolution, ROI, or calibration on the same topic,
finish recording and let any pending archive or transcode complete. Restart
the `cyclo_data` node after the driver publishes the new calibration and
before recording again. Starting another episode does not refresh the cache;
switching to a different robot profile does.

## Console scripts

`setup.py` registers two entry points:

```
visualize_rosbag          → cyclo_data.visualization.scripts.visualize_rosbag:main
convert_rosbag_to_lerobot → cyclo_data.converter.scripts.convert_rosbag_to_lerobot:main
```

After `colcon build`, run any of these by name:
```
ros2 run cyclo_data visualize_rosbag /path/to/file.mcap --detailed
```

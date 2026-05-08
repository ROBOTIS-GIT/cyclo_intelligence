# Architecture — cyclo_intelligence

As-built runtime topology — what the system actually looks like once
`docker/container.sh start` has brought everything up.

## Container topology

```
+--------------------------------------------------------------+
|                    Host (Jetson / workstation)               |
|                                                              |
|  +-------------------+        +-------------------+          |
|  |  cyclo_intelligence|       |  policy container |          |
|  |  (unified image)  |◄──Zenoh───►  (lerobot /    |          |
|  |                   |  rmw   |   groot)          |          |
|  +-------------------+        +-------------------+          |
|                                                              |
+--------------------------------------------------------------+

                              │
                              │ orchestrator/cyclo_data
                              │ publish joint commands
                              ▼

                    +----------------+
                    |  Robot hardware |
                    +----------------+
```

There are only two long-lived containers: `cyclo_intelligence`
(always up) and one policy container per active backend (pulled + up
on demand, stopped when idle). More policy backends can live in
parallel — each gets its own container using the same two-process
template.

## Inside `cyclo_intelligence`

```
cyclo_intelligence container

  /init (s6-overlay)
    │
    ├── s6-agent              cyclo_manager FastAPI on a UDS, for host-side
    │                         integrations
    │
    ├── supervisor_api        FastAPI on 127.0.0.1:8100
    │                         — nginx /api/ proxies here
    │                         — s6-rc for in-container services
    │                         — (planned) Docker SDK for backend
    │                           containers
    │
    ├── nginx                 :80  serves the React UI static build
    │                                + /api/ → supervisor_api
    │                                + /data-api/ → video_file_server
    │                                + /files/    → workspace bind mount
    │
    ├── orchestrator          ROS2 node — OrchestratorNode
    │                                    (orchestrator_node.py)
    │     ├── session state
    │     ├── UI command routing
    │     ├── cyclo_data srv dispatch
    │     ├── policy container lifecycle
    │     │   (InferenceCommand.srv)
    │     └── BT control
    │
    ├── cyclo_data            ROS2 node — CycloDataNode
    │                                    (cyclo_data_node.py)
    │     ├── recorder/       rosbag_recorder (C++) + session_manager
    │     ├── reader/         bag reader + metadata
    │     ├── converter/      MCAP→MP4→LeRobot chain
    │     ├── editor/         episode edits
    │     ├── quality/        timestamp gap analysis
    │     ├── hub/            HuggingFace upload/download
    │     └── visualization/  video_file_server (port 8082)
    │
    └── web_video_server      on demand (image streaming)
```

Every longrun has a matching `<name>-log` consumer that wraps
`logutil-service` into `/var/log/<name>/`.
[`docker/s6-services/`](../docker/s6-services/) holds the service
definitions; [`docker/Dockerfile.arm64`](../docker/Dockerfile.arm64)
ties them into the image.

## Inside a policy container (LeRobot)

```
lerobot container

  /init (s6-overlay)
    │
    ├── inference-server      Process A — runtime/inference_server.py
    │     ├── PyTorch policy (GPU)
    │     ├── ROS2Subscriber   camera + follower joint_states
    │     │                    (direct, not round-tripped via
    │     │                     orchestrator)
    │     ├── ROS2ServiceServer /lerobot/inference_command
    │     ├── Zenoh sub        cyclo/policy/lerobot/run_inference
    │     └── Zenoh pub        cyclo/policy/lerobot/action_chunk_raw
    │
    └── control-publisher     Process B — runtime/control_publisher.py
          ├── Zenoh sub        cyclo/policy/lerobot/action_chunk_raw
          │     └── ActionChunkProcessor
          │           (L2 align → interpolate → blend → buffer)
          ├── 100 Hz loop      time.monotonic + next_t drift resync
          ├── Zenoh pub        cyclo/policy/lerobot/run_inference
          │                    (trigger on low buffer)
          └── ROS2Publisher    per-group JointTrajectory / Twist
                               to /leader/<name>/joint_trajectory /
                                  /cmd_vel
```

GR00T uses the same template — its container will sit next to LeRobot
once the N1.5 / 1.6 / 1.7 pin decision is made (pending).

## Key flows

### 1. Data collection

```
   UI (React)                   orchestrator                   cyclo_data
      │                              │                              │
      │── POST /send_command ───────►│                              │
      │       (START_RECORDING)      │                              │
      │                              │── RecordingCommand.srv ─────►│
      │                              │        (START)               │
      │                              │                              │── rosbag_recorder C++
      │                              │                              │   starts writing MCAP
      │                              │                              │
      │                              │◄── /data/recording/status ───│
      │◄── /task/status (relay) ─────│      (TaskStatus)            │
```

`orchestrator` never touches the bag writer directly — it forwards
the command to `cyclo_data` and relays the status topic back up to
the UI. This keeps the data plane + control plane separation clean.

### 2. Inference with the LeRobot policy

```
   UI                    orchestrator              lerobot container
    │                         │                          │
    │── POST /send_command ───►                          │
    │       (START_INFERENCE) │                          │
    │                         │── InferenceCommand.srv ─►│  Process A
    │                         │        (LOAD)            │  loads policy
    │                         │                          │  subscribes
    │                         │                          │  observations
    │                         │◄── response action_keys ─│
    │                         │                          │
    │                         │── InferenceCommand.srv ─►│  Process A
    │                         │        (START)           │  now accepts
    │                         │                          │  triggers
    │                         │                          │
    │                         │                          │  Process B
    │                         │                          │  (always running
    │                         │                          │   100 Hz since
    │                         │                          │   boot):
    │                         │                          │
    │                         │                          │  pops last_action,
    │                         │                          │  publishes
    │                         │                          │  JointTrajectory
    │                         │                          │
    │                         │                          │  Zenoh trigger →
    │                         │                          │  Zenoh chunk →
    │                         │                          │  ActionChunkProcessor
```

During inference the 100 Hz loop never crosses the container
boundary — that's by design. All the orchestrator has to do is
dispatch LOAD → START, and later PAUSE / RESUME / STOP / UNLOAD
on user intent.

### 3. Conversion (MCAP → LeRobot dataset)

```
   UI                 orchestrator              cyclo_data
    │                      │                        │
    │── CONVERT_MP4 ──────►│                        │
    │                      │── StartConversion.srv ►│  atomic swap:
    │                      │                        │   ├── pipeline_worker
    │                      │                        │   └── video_encoder
    │                      │◄── /data/status        │
    │◄── /task/status ─────│   (OP_CONVERSION       │
    │                      │    progress)           │
```

Long-running task delegated with progress streamed back through
`DataOperationStatus` relayed into `TaskStatus`. Same pattern for HF
upload / download, episode edits.

### 4. UI control plane (`/api/*`)

```
   UI (React)
    │
    ├── GET  /api/services              ─┐
    ├── GET  /api/services/{n}/status    │
    ├── POST /api/services/{n}/start     │── nginx → 127.0.0.1:8100
    ├── POST /api/services/{n}/stop      │   (supervisor_api)
    └── POST /api/backends/{n}/pull      │   — 501 stub for now
                                          ─┘  (backends follow-up)
```

`supervisor_api` wraps `s6-rc -u change` / `s6-rc -d change` for the
three user-controllable in-container services (`orchestrator`,
`cyclo_data`, `web_video_server`), and is the future home of the
Docker-SDK backend lifecycle that will drive
`docker compose pull / up / down` for `lerobot` / `groot`.

## Where the code lives

| Concern | Source |
| --- | --- |
| Two-process policy runtime | [`cyclo_brain/policy/<backend>/runtime/`](../cyclo_brain/policy/) |
| ActionChunkProcessor | [`cyclo_brain/sdk/post_processing/`](../cyclo_brain/sdk/post_processing/) |
| Orchestrator ROS2 node | [`orchestrator/orchestrator_node.py`](../orchestrator/orchestrator_node.py) |
| cyclo_data ROS2 node | [`cyclo_data/cyclo_data_node.py`](../cyclo_data/cyclo_data_node.py) |
| ROS2 srv/msg definitions | [`interfaces/`](../interfaces/) |
| supervisor_api | [`docker/supervisor_api/app.py`](../docker/supervisor_api/app.py) |
| s6 service definitions | [`docker/s6-services/`](../docker/s6-services/) |
| Unified Dockerfile | [`docker/Dockerfile.arm64`](../docker/Dockerfile.arm64) · [`.amd64`](../docker/Dockerfile.amd64) |
| Policy backend Dockerfile | [`cyclo_brain/policy/lerobot/Dockerfile.arm64`](../cyclo_brain/policy/lerobot/Dockerfile.arm64) |
| Compose + launcher | [`docker/docker-compose.yml`](../docker/docker-compose.yml) · [`docker/container.sh`](../docker/container.sh) |

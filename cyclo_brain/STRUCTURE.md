# cyclo_brain — Target Structure

`cyclo_brain`은 **2개 Python process**로 구성한다.
각 process 안의 기능은 여러 class/module로 나눈다.

- Visual map: [`cyclo_brain/docs/architecture.html`](docs/architecture.html)
- Rule: runtime 구조가 바뀌면 이 파일은 설명 기준으로, `cyclo_brain/docs/architecture.html`은 시각화 기준으로 함께 업데이트한다.

- **Main process**: service, session, command publish, control loop를 조율한다.
- **Engine process**: 모델 로딩, inference 실행, inference용 sensor/state 구독을 맡는다.

---

## 1. Big Picture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Host / cyclo_intelligence                                                    │
│                                                                              │
│  UI / Orchestrator or Standalone CLI                                         │
│      │                                                                       │
│      │  same command shape: args or /<backend>/inference_command             │
│      │  LOAD / START / PAUSE / RESUME / STOP / UNLOAD                        │
│      ▼                                                                       │
│  External ROS2 / Zenoh                                                       │
└──────┬───────────────────────────────────────────────────────────────────────┘
       │
       │ service call
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Policy Container: <backend>_server                                           │
│                                                                              │
│  Process 1: Main process                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ main_runtime package                                                   │  │
│  │ one Python process, multiple classes/modules                           │  │
│  │                                                                        │  │
│  │  ┌───────────────────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ ServiceHandler        │       │ SessionState                     │  │  │
│  │  │                       │       │                                  │  │  │
│  │  │ - LOAD                │──────▶│ - unloaded / loaded / running    │  │  │
│  │  │ - START / PAUSE       │       │ - paused / stopped               │  │  │
│  │  │ - RESUME / STOP       │       │ - gate inference + publish       │  │  │
│  │  │ - UNLOAD              │       └──────────────────────────────────┘  │  │
│  │  └───────────────────────┘                                             │  │
│  │                                                                        │  │
│  │  ┌───────────────────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ RobotClient           │       │ InferenceRequester               │  │  │
│  │  │                       │       │                                  │  │  │
│  │  │ - publish robot cmds  │       │ - request model load             │  │  │
│  │  │ - command topic setup │       │ - request one inference step     │  │  │
│  │  │ - Main uses publish   │       │ - receive action list            │  │  │
│  │  └───────────▲───────────┘       └────────────────┬─────────────────┘  │  │
│  │              │                                   │ action_list         │  │
│  │              │ publish_action                     ▼                    │  │
│  │  ┌───────────┴───────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ ControlLoop           │◀──────│ ActionChunkProcessor             │  │  │
│  │  │                       │ pop   │                                  │  │  │
│  │  │ - timer-like loop     │       │ - action list buffer             │  │  │
│  │  │ - one action per tick │       │ - optional post-processing       │  │  │
│  │  │ - cadence follows ACP │       │ - pop one action per tick        │  │  │
│  │  └───────────────────────┘       └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│                                  │ LOAD_POLICY / GET_ACTION / UNLOAD_POLICY  │
│                                  ▼                                           │
│  Process 2: Engine process                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ <backend>Engine                                                        │  │
│  │                                                                        │  │
│  │ ┌─────────────────────┐  ┌───────────────────────────────────────────┐ │  │
│  │ │ PolicyLoader         │  │ Optimizer                                │ │  │
│  │ │ - load policy        │  │ - optional area                          │ │  │
│  │ │ - weights/processors │  │ - TensorRT / GPU / runtime optimization  │ │  │
│  │ └─────────────────────┘  └───────────────────────────────────────────┘ │  │
│  │ ┌─────────────────────┐  ┌───────────────────────────────────────────┐ │  │
│  │ │ Preprocessor         │  │ Predictor                                 │ │  │
│  │ │ - use RobotClient    │  │ - run inference once per request          │ │  │
│  │ │ - build model input  │  │ - return action list (T, D)               │ │  │
│  │ └─────────────────────┘  └───────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │ Engine uses RobotClient for sensor/state topics.                       │  │
│  │ Engine never publishes robot commands.                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       │ RobotClient publish
                                       │ /cmd_vel
                                       │ /leader/*/joint_trajectory
                                       ▼
                                  ┌───────────┐
                                  │   Robot   │
                                  └───────────┘
```

---

## 2. Target Directory Shape

```text
cyclo_brain/
├── sdk/
│   ├── robot_client/                  # robot topic client for command or observation
│   ├── action_chunk_processing/       # action_list post-processing
│   └── zenoh_ros2_sdk/                # ROS2-over-Zenoh transport SDK
│
└── policy/
    ├── common/
    │   ├── runtime/
    │   │   ├── engine.py              # InferenceEngine ABC
    │   │   ├── main_runtime/          # Process 1 package
    │   │   │   ├── main.py            # starts one Main Python process
    │   │   │   ├── service_handler.py # ServiceHandler class
    │   │   │   ├── session_state.py   # SessionState class
    │   │   │   ├── inference_requester.py
    │   │   │   ├── control_loop.py    # ControlLoop class, uses RobotClient
    │   │   │   └── zenoh_client.py    # internal EngineCommand service client
    │   │   ├── engine_process/        # Process 2 package
    │   │   │   ├── worker.py
    │   │   │   └── protocol.py
    │   └── s6-services/
    │       ├── main-runtime/
    │       └── engine-process/
    │
    ├── lerobot/
    │   ├── Dockerfile.{arm64,amd64}
    │   ├── lerobot/
    │   └── lerobot_engine/
    │       ├── engine.py
    │       ├── loading.py                # PolicyLoader convention
    │       ├── optimization.py           # optional TensorRT/GPU/runtime optimization
    │       ├── io_mapping.py
    │       ├── preprocessing.py          # Preprocessor convention
    │       ├── prediction.py             # Predictor convention
    │       └── constants.py
    │
    └── groot/
        ├── Dockerfile.{arm64,amd64}
        ├── Isaac-GR00T/
        └── groot_engine/
            ├── engine.py
            ├── loading.py                # PolicyLoader convention
            ├── optimization.py           # optional TensorRT GPU optimizer
            ├── io_mapping.py
            ├── preprocessing.py
            └── prediction.py
```

---

## 3. Runtime Data Flow

```text
1. LOAD

External or standalone CLI
  └─ InferenceCommand(LOAD, model_path, robot_type, task_instruction)
       └─▶ Main process
             ├─ RobotClient.configure(robot_type) for command publish
             ├─ Engine process.load_policy(model_path, robot_type)
             │    └─ RobotClient.configure(robot_type) for observation
             ├─ ActionChunkProcessor.clear()
             └─ session = loaded


2. START + RUN

External
  └─ InferenceCommand(START)
       └─▶ Main process
             └─ session = running

Main control loop
  ├─ action = ActionChunkProcessor.pop_action()
  ├─ if action exists:
  │    └─ RobotClient.publish_action(action)
  │
  └─ if buffer is low and no request is in flight:
       ├─ Engine process.get_action(task_instruction)
       │    ├─ RobotClient.get_observation()
       │    ├─ run policy inference
       │    └─ return action_list
       └─ ActionChunkProcessor.push_actions(action_list)

Control loop cadence
  ├─ post-processing enabled:
  │    ├─ ActionChunkProcessor converts model action list to control actions
  │    ├─ matching / RTC aligner / future smoothing can run here
  │    ├─ example: 16 model actions → 100 control actions
  │    └─ ControlLoop runs at processed output cadence, normally 100Hz
  │
  └─ post-processing disabled:
       ├─ ActionChunkProcessor buffers raw action list as-is
       └─ ControlLoop runs at model/action-list cadence, not forced to 100Hz


3. PAUSE / RESUME / STOP / UNLOAD

PAUSE
  External ─▶ Main ─▶ session = paused
  control loop keeps running, but does not publish robot commands

RESUME
  External ─▶ Main ─▶ session = running
  control loop resumes publishing from buffer

STOP
  External ─▶ Main ─▶ session = stopped
  ActionChunkProcessor.clear()

UNLOAD
  External ─▶ Main
     ├─ Engine process.cleanup()
     │    └─ RobotClient.close() for observation
     ├─ RobotClient.close() for command publish
     ├─ ActionChunkProcessor.clear()
     └─ session = unloaded
```

---

## 4. Responsibility Boundary

| Area | Owner |
|---|---|
| External command service | Main process |
| Session state | Main process |
| Control loop | Main process |
| Robot sensor/state input | Engine process uses RobotClient |
| Robot command output | Main process uses RobotClient |
| Action list buffer/post-processing | ActionChunkProcessor |
| Model load/inference | Engine process |
| Optional optimization | Engine process optimizer class |
| Backend-specific policy code | `<backend>_engine/` |

---

## 5. Stable Contracts

| Contract | Shape |
|---|---|
| External service | `/<backend>/inference_command` |
| Internal engine service | `/<backend>/engine_command` via `zenoh_ros2_sdk` service |
| Main → Engine | `LOAD_POLICY`, `GET_ACTION`, `UNLOAD_POLICY` |
| Main → RobotClient | `configure`, `publish_action`, `close` |
| Engine → RobotClient | `configure`, `get_observation`, `close` |
| Engine output | `action_list` shaped `(T, D)` |
| Processor output | one action vector per control tick |
| Runtime processes | `main-runtime`, `engine-process` |
| Main internal modules | classes inside one Main process, not extra processes |

### 5.1 Internal engine service

```text
GET_ACTION request:
  seq_id
  task_instruction

GET_ACTION response:
  seq_id
  success
  message
  action_list
  chunk_size
  action_dim

Main rules:
  - one GET_ACTION in-flight by default
  - timeout is configurable per backend/model/deployment
  - timeout means "response not received in time", not "inference failed"
  - late/stale responses are discarded by seq_id
  - only the latest accepted response enters ActionChunkProcessor
```

### 5.2 Timeout policy

```text
LOAD_POLICY timeout:
  backend-specific and can be long
  includes model load, processor load, optimizer build/load

GET_ACTION timeout:
  runtime safety timeout
  configurable; default is a fallback, not a performance guarantee
  should account for model size and user hardware
```

### 5.3 Backend integration contract

```text
policy/<backend>/
├── Dockerfile.{arm64,amd64}        # per-opensource dependency isolation
├── <opensource-submodule>/         # git submodule
└── <backend>_engine/
    ├── engine.py                   # implements InferenceEngine ABC
    ├── loading.py                  # PolicyLoader convention
    ├── optimization.py             # optional TensorRT/GPU/runtime optimization
    ├── io_mapping.py               # robot/model key mapping
    ├── preprocessing.py            # RobotClient observation → model input
    └── prediction.py               # model input → action_list
```

`common/runtime/engine.py`의 `InferenceEngine` ABC가 필수 process-boundary 계약이다.
`loading.py`, `optimization.py`, `preprocessing.py`, `prediction.py`는 backend 내부 표준 레이아웃이며,
별도 abstract를 강제하지 않는다.

`optimization.py`는 선택 영역이다. TensorRT/GPU/runtime 최적화가 필요 없는 backend는
no-op으로 두거나 파일을 생략할 수 있다.

### 5.4 Action list contract

```text
action_list:
  shape: (T, D)
  T: model action steps
  D: flattened robot action dimension
  action_keys: model output modality order

ActionChunkProcessor:
  does not reorder action dimensions
  may match / RTC-align / interpolate / blend / smooth over time

Robot publish path:
  splits final action vector by robot command schema
```

---

## 6. Design Rule

```text
Main owns session flow.
Main owns the control loop.
Main can be split into ServiceHandler, SessionState, InferenceRequester, ControlLoop classes.
RobotClient is the common robot I/O client.
Main process uses RobotClient to publish robot commands.
Engine process uses RobotClient to read sensor/state topics for inference.
Engine implements InferenceEngine ABC.
Engine may split internally into PolicyLoader, Optimizer, Preprocessor, Predictor.
Optimizer is optional.
ActionChunkProcessor owns optional action-list post-processing and buffering.
If post-processing converts 16 actions to 100 actions, the loop can run at 100Hz.
If post-processing is disabled, the loop cadence must follow the raw action list.
```

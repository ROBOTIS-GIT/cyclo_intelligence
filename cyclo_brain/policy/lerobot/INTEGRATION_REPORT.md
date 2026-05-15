# LeRobot Integration Report

## Overview

LeRobot has been integrated into Cyclo Intelligence following the opensource_integration_workflow.

## Integration Details

### Executor Features

The `executor.py` provides the following capabilities:

| Feature | Status | Description |
|---------|--------|-------------|
| Training | ✅ | Start/stop/monitor training with progress tracking |
| Inference | ✅ | Real-time inference with ROS2 sensor data |
| Policy List | ✅ | List available LeRobot policies |
| Checkpoint List | ✅ | List training checkpoints |
| Model List | ✅ | List cached models |

### Supported Policies

| Policy | Category | Description |
|--------|----------|-------------|
| act | Imitation Learning | Action Chunking Transformer |
| diffusion | Imitation Learning | Diffusion Policy |
| vqbet | Imitation Learning | VQ-BeT |
| tdmpc | RL | TD-MPC |
| pi0 | VLA | Physical Intelligence VLA |
| pi0_fast | VLA | Optimized Pi0 |
| pi05 | VLA | Pi0.5 |
| smolvla | VLA | SmolVLA |
| groot | VLA | NVIDIA GR00T |
| sac | RL | Soft Actor-Critic |

### Architecture

```
Cyclo Intelligence Web UI (React UI)
        |
        v (WebSocket 9090)
Cyclo Intelligence Orchestrator (ROS2 + rmw_zenoh_cpp)
        |
        v (Zenoh Protocol 7447)
LeRobot Executor (Docker Container)
        |
        v
LeRobot Training/Inference APIs
```

### ROS2 Services (via Zenoh)

| Service | Type | Description |
|---------|------|-------------|
| /lerobot/train | TrainModel | Start training |
| /lerobot/infer | StartInference | Start inference |
| /lerobot/stop | StopTraining | Stop current task |
| /lerobot/status | TrainingStatus | Get status |
| /lerobot/policy_list | PolicyList | List policies |
| /lerobot/checkpoint_list | CheckpointList | List checkpoints |
| /lerobot/model_list | ModelList | List models |

### ROS2 Topics (via Zenoh)

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| /lerobot/progress | TrainingProgress | Published | Training metrics |
| /lerobot/action | ActionOutput | Published | Inference outputs |

## File Structure

```
cyclo_intelligence/
├── cyclo_brain/policy/
│   ├── lerobot/                     # LeRobot integration folder
│   │   ├── lerobot/                 # LeRobot repository (git submodule)
│   │   ├── executor.py              # Zenoh-based executor
│   │   ├── Dockerfile               # Container definition
│   │   ├── entrypoint.sh            # Container entrypoint
│   │   ├── workspace/               # Dataset/Model storage (gitignore)
│   │   ├── test_executor.py         # Unit tests
│   │   ├── README.md                # Integration guide
│   │   └── INTEGRATION_REPORT.md    # This file
│   └── zenoh_ros2_sdk/              # Zenoh ROS2 SDK (shared)
└── docker/
    └── docker-compose.yml           # Compose configuration
```

## Docker Configuration

### Build

```bash
cd cyclo_intelligence
docker compose -f docker/docker-compose.yml build lerobot
```

### Run

```bash
# Start with Zenoh router and server
docker compose -f docker/docker-compose.yml up zenoh_router lerobot

# Or start all services
docker compose -f docker/docker-compose.yml up
```

### Volume Mappings

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| cyclo_brain/policy/lerobot/workspace | /workspace | Dataset/Model storage |
| cyclo_brain/sdk/zenoh_ros2_sdk | /zenoh_sdk | Zenoh SDK |

## Usage Examples

### Start Training (via ROS2)

```python
# From orchestrator node
from interfaces.srv import TrainModel

client = node.create_client(TrainModel, '/lerobot/train')
request = TrainModel.Request()
request.policy_type = "act"
request.dataset_path = "lerobot/pusht"
request.steps = 100000
request.batch_size = 8

future = client.call_async(request)
```

### Start Inference (via ROS2)

```python
from interfaces.srv import StartInference

client = node.create_client(StartInference, '/lerobot/infer')
request = StartInference.Request()
request.model_path = "/path/to/checkpoint"
request.inference_freq = 30.0  # Hz

future = client.call_async(request)
```

## Testing

### Unit Tests

```bash
cd cyclo_intelligence/cyclo_brain/policy/lerobot
python -m pytest tests/test_executor.py -v
```

### Integration Tests

```bash
# Build and run container
docker compose -f docker/docker-compose.yml build lerobot
docker compose -f docker/docker-compose.yml run lerobot shell

# Inside container
python -c "from executor import LeRobotExecutor; print('OK')"
```

## Known Limitations

1. **GPU Required**: Training and inference require NVIDIA GPU
2. **Zenoh Router**: Must be running before starting executor
3. **ROS2 Interfaces**: Requires interfaces package

## Next Steps

1. [ ] Add WebSocket progress streaming to UI
2. [ ] Implement checkpoint download from HuggingFace Hub
3. [ ] Add distributed training support
4. [ ] Add model fine-tuning interface

## Changelog

### 2026-01-20

- Initial integration following opensource_integration_workflow
- Added executor.py with training/inference support
- Added unit tests
- Created integration report

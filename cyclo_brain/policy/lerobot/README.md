# LeRobot Integration for Cyclo Intelligence

## Overview

LeRobot integration for Cyclo Intelligence. This folder contains the executor and Docker configuration for running LeRobot training and inference via Zenoh communication.

## Folder Structure

```
cyclo_brain/policy/lerobot/
├── lerobot/                 # LeRobot repository (git submodule)
│   └── (HuggingFace LeRobot source code)
├── executor.py              # Zenoh communication + train/infer execution
├── Dockerfile               # Container build file
├── entrypoint.sh            # Container entrypoint
├── workspace/               # Dataset/Model/Results (gitignore)
├── test_executor.py         # Unit tests
├── README.md                # This document
└── INTEGRATION_REPORT.md    # Integration detail report
```

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU (CUDA 12.1+)
- Cyclo Intelligence repository

## Quick Start

### 1. Clone LeRobot (if not already done)

```bash
cd cyclo_intelligence/cyclo_brain/policy/lerobot
git clone https://github.com/huggingface/lerobot.git lerobot
```

### 2. Build Docker Image

```bash
cd cyclo_intelligence
docker compose -f docker/docker-compose.yml build lerobot
```

### 3. Download Test Dataset

```bash
# Create workspace directory
mkdir -p cyclo_brain/policy/lerobot/workspace

# Download lerobot/pusht dataset
huggingface-cli download lerobot/pusht \
  --local-dir cyclo_brain/policy/lerobot/workspace/lerobot/pusht \
  --repo-type dataset
```

### 4. Run Container

```bash
docker compose -f docker/docker-compose.yml up lerobot
```

## Architecture

```
Cyclo Intelligence Web UI (React UI)
        │
        ▼ WebSocket (9090)
Cyclo Intelligence Orchestrator (ROS2 + rmw_zenoh_cpp)
        │
        ▼ Zenoh Protocol (7447)
LeRobot Executor (Docker Container)
        │
        ▼
LeRobot Training/Inference APIs
```

## Supported Policies

| Policy | Category | Description |
|--------|----------|-------------|
| act | Imitation Learning | Action Chunking Transformer |
| diffusion | Imitation Learning | Diffusion Policy |
| vqbet | Imitation Learning | VQ-BeT |
| tdmpc | RL | TD-MPC |
| pi0 | VLA | Physical Intelligence VLA |
| pi0_fast | VLA | Optimized Pi0 |
| smolvla | VLA | SmolVLA |
| sac | RL | Soft Actor-Critic |

## ROS2 Services (via Zenoh)

| Service | Description |
|---------|-------------|
| /lerobot/train | Start training |
| /lerobot/infer | Start inference |
| /lerobot/stop | Stop current task |
| /lerobot/status | Get status |
| /lerobot/policy_list | List available policies |
| /lerobot/checkpoint_list | List checkpoints |
| /lerobot/model_list | List cached models |

## ROS2 Topics (via Zenoh)

| Topic | Direction | Description |
|-------|-----------|-------------|
| /lerobot/progress | Published | Training metrics (step, loss, epoch) |
| /lerobot/action | Published | Inference action outputs |

## Docker Configuration

### Volume Mappings

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| cyclo_brain/policy/lerobot/workspace | /workspace | Dataset/Model storage |
| cyclo_brain/sdk/zenoh_ros2_sdk | /zenoh_sdk | Zenoh SDK |

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| RMW_IMPLEMENTATION | rmw_zenoh_cpp | ROS2 Zenoh middleware |
| ROS_DOMAIN_ID | 30 | ROS2 domain |
| ZENOH_CONFIG_OVERRIDE | (see compose) | Zenoh client configuration |

## Testing

### Unit Tests

```bash
cd cyclo_intelligence/cyclo_brain/policy/lerobot
python -m pytest test_executor.py -v
```

### Integration Test

```bash
# Start container
docker compose -f docker/docker-compose.yml up -d lerobot

# Enter container
docker exec -it lerobot_server bash

# Test import
python -c "from executor import LeRobotExecutor; print('OK')"
```

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Zenoh connection failed

```bash
# Ensure Zenoh daemon is running in orchestrator
docker exec -it orchestrator bash
zenoh  # or: ros2 run rmw_zenoh_cpp rmw_zenohd
```

### LeRobot import error

```bash
# Check LeRobot is properly cloned
ls -la cyclo_brain/policy/lerobot/lerobot/

# Ensure PYTHONPATH includes lerobot
export PYTHONPATH="/app/lerobot:$PYTHONPATH"
```

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/lerobot)
- [Zenoh ROS2 SDK](https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk)
- [Cyclo Intelligence Workflow](../../ai_system_agents/opensource_integration_workflow/)

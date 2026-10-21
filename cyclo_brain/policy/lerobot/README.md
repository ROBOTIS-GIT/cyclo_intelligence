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
        ▼ WebSocket (7090)
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
| ZENOH_CONFIG_OVERRIDE | local shared-memory default or remote router example in `/root/.bashrc` | Zenoh client configuration |

At runtime, `lerobot_server` sources `/root/.bashrc` before starting
`main-runtime` and `engine-process`. Enter the container, edit the Cyclo
ROS/Zenoh block near the top of `/root/.bashrc`, and choose either the local
`ZENOH_CONFIG_OVERRIDE` line or the commented remote router example. Then restart
`lerobot_server`. `docker restart` preserves the edit;
recreating or updating the container resets `/root/.bashrc` to the image
default.

## ACT checkpoints with a subset of robot joints

ACT input state and output actions must match the training layout exactly.
For a subset policy, put `cyclo_inference.json` beside the downloaded
`config.json`. State modalities and action keys are independent, ordered
lists of names from the selected robot configuration. For example, **only
when training used both arms as state and the right arm as action**:

```json
{
  "robot_type": "ffw_sg2_rev1",
  "state_modalities": ["arm_left", "arm_right"],
  "action_keys": ["arm_right"]
}
```

For this robot the example requires 16 state dimensions and 8 action
dimensions. A policy trained with only right-arm state must instead use
`"state_modalities": ["arm_right"]` and have 8 state dimensions. Confirm
the layout from training metadata; dimensions alone cannot identify the arm.
ACT rejects mismatched dimensions instead of padding or truncating state.

Modern LeRobot checkpoints require their stored pre/postprocessor files and
normalization tensors. LeRobot 0.3 ACT checkpoints instead carry these buffers
inside `model.safetensors`; the compatibility loader restores every model
tensor strictly and supplies the embedded statistics to the official ACT
processor factory. Neither format re-estimates statistics from live sensors.

The runtime honors ACT's `n_action_steps`. With temporal ensembling enabled,
it calls the official `select_action` once per fresh observation and returns
one action. Main disables chunk interpolation, alignment and blending for
single-step ACT, limits its buffer to one action, and uses `Inference Hz` as
the loop rate. Start/resume and completion of initial pose synchronization
reset the policy's ensemble history. Actual throughput still depends on
hardware latency and must be measured before robot execution.

For models trained with bilinear antialiased, aspect-preserving resize and
centered zero padding, add `"image_preprocessing": "letterbox"` to the mapping.
This applies the configured camera rotation before resizing/padding, and
applies the saved image normalization once afterward. The default remains
the existing direct resize for other models. An `inference_contract.json`
with `fps` is enforced at LOAD for single-step ACT.
Set the UI's `Inference Hz` to the training dataset FPS; it determines the
time spacing of predicted actions. Cyclo may interpolate them at `Control Hz`.
Validate first with the UI's simulation mode, which still reads live cameras
and joints but publishes only the action preview.

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
# Ensure the externally managed Zenoh router is running on the endpoint in ZENOH_CONFIG_OVERRIDE.
ss -ltnp 'sport = :7447'
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
# ACT TensorRT

Cyclo supports optional `acceleration_mode=tensorrt_act` with a verified,
checkpoint-specific `act_model_fp32.trt`. Native pre/post processing and ACT
temporal ensembling remain unchanged. See
[TensorRT inference setup](../../../docker/TENSORRT_INFERENCE.md) for UI and
offline preparation. Other LeRobot families require their own adapters.

# Cyclo Intelligence

Open-source full-stack Physical AI platform — data recording, conversion,
training, inference, and robot execution in a single repository.

For detailed usage and tutorials, please refer to the documentation below.
  - [Documentation for AI Worker](https://ai.robotis.com/)

## Clone

```bash
git clone --recurse-submodules https://github.com/ROBOTIS-GIT/cyclo_intelligence.git
```

## Folders at a glance

| Folder | Role |
| --- | --- |
| [`shared/`](shared/) | Robot configs, IO helpers, logger |
| [`cyclo_brain/`](cyclo_brain/) | Training + inference (per-backend containers under `policy/`) |
| [`cyclo_data/`](cyclo_data/) | Data recording / conversion / hub upload (ROS2 node) |
| [`orchestrator/`](orchestrator/) | Session state, UI, behaviour-tree control (ships React UI) |
| [`interfaces/`](interfaces/) | ROS2 msg / srv definitions |
| [`docker/`](docker/) | Unified compose, s6-services, Dockerfiles (arm64 / amd64) |
| [`docs/`](docs/) | Architecture |

## Quick start (Jetson / ARM64 — same on AMD64)

```bash
docker/container.sh start          # build + start unified image
docker/container.sh status         # check s6 service state
# UI:          http://localhost/
# control API: http://localhost/api/health
ROBOT_TYPE=ffw_sg2_rev1 docker/container.sh start-lerobot   # policy on demand
```

`docker/container.sh` auto-detects `uname -m`, so the same commands work on
both Jetson and an AMD64 workstation.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for runtime topology
and data flow.

## Submodules (pinned commit)

- `cyclo_brain/sdk/zenoh_ros2_sdk/` ← [ROBOTIS-GIT/zenoh_ros2_sdk](https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk)
- `cyclo_brain/policy/lerobot/lerobot/` ← [huggingface/lerobot](https://github.com/huggingface/lerobot)
- `cyclo_brain/policy/groot/Isaac-GR00T/` ← [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

## Related

  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)
  - [Docker Images](https://hub.docker.com/r/robotis/ros/tags)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

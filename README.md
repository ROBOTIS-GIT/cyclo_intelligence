# Cyclo Intelligence

Cyclo Intelligence is a Physical AI platform for data recording, dataset
conversion, policy training, inference, and robot execution with AI Worker.
For detailed usage instructions, please refer to the documentation below.
  - [Documentation for AI Worker Imitation Learning](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/)

To install Cyclo Intelligence:

```bash
curl -fsSL https://raw.githubusercontent.com/ROBOTIS-GIT/cyclo_intelligence/main/install.sh | bash
```

On ROBOTIS robot PCs whose host name starts with `ffw`, the installer places
the checkout under `/mnt/ssd/cyclo_intelligence` and bind-mounts it at
`~/cyclo_intelligence`. On other PCs, it installs directly under
`~/cyclo_intelligence`.

To learn more about the ROS 2 packages for the AI Worker, visit:
  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)

To explore our open-source platforms in a simulation environment, visit:
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)

For usage instructions and demonstrations of the AI Worker, check out:
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)

To access datasets and pre-trained models for our open-source platforms, see:
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)

To use Docker images for running Cyclo Intelligence, visit:
  - [Docker Images](https://hub.docker.com/u/robotis)

# LeRobot Worker

This directory provides the LeRobot Engine-only Worker used by Cyclo.
Training remains available through the LeRobot CLI, while inference lifecycle,
action processing, safety, and robot command publication are owned by the
central Policy Runtime in `cyclo_intelligence`.

## Layout

```text
policy/lerobot/
├── manifest.yaml              # policies and runtime capabilities
├── lerobot/                   # ROBOTIS LeRobot fork submodule
├── lerobot_engine/            # InferenceEngine adapter
├── Dockerfile.amd64
├── Dockerfile.arm64
└── tests/
```

The container starts only the `engine-process` s6 service. It hosts
`/lerobot/engine_command`, publishes `/lerobot/worker_heartbeat`, subscribes
to the observations required by the selected checkpoint, and returns action
chunks. It does not own `ControlLoop` and does not publish robot commands.

## Build And Start

From the repository root:

```bash
./docker/container.sh start-policy lerobot --build
./docker/container.sh enter-policy lerobot
```

The compatibility commands remain available:

```bash
./docker/container.sh start-lerobot --build
./docker/container.sh enter-lerobot
```

The container executes files baked into the image. Use `--build` after Worker
source or dependency changes.

## Models And Data

The shared host directory `docker/workspace` is mounted at `/workspace`.
LeRobot checkpoints are selected under `/workspace/model/lerobot`. The list of
policies shown by Cyclo comes from `manifest.yaml`; the Worker `DESCRIBE`
response must match that manifest before LOAD is accepted.

Inference is called externally through `/policy/inference_command` with a
namespaced policy ID such as `lerobot:act`. `/lerobot/inference_command` is a
temporary compatibility alias hosted by the central Runtime, not by this
Worker.

## ROS And Zenoh

Edit the Cyclo block near the top of the Worker's `/root/.bashrc`, then restart
the container so s6 starts `engine-process` with the new settings:

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
```

For a remote router, use the commented client endpoint example already present
in the image bashrc. `docker restart` preserves a container-local edit;
recreate/update resets it to the image default.

## Training

Enter the Worker and use the version-pinned LeRobot CLI. For example:

```bash
lerobot-train \
  --dataset.repo_id=<dataset_repo_id> \
  --dataset.root=/workspace/lerobot/<dataset_folder> \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/workspace/model/lerobot/<output_folder>
```

Training output placed below `/workspace/model/lerobot` is immediately visible
to the Cyclo model browser.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  cyclo_brain/policy/lerobot/tests \
  cyclo_brain/policy/lerobot/lerobot_engine/tests
```

Container readiness requires `engine-process` and its ready marker. Cyclo's
Supervisor additionally verifies fresh heartbeat and a compatible `DESCRIBE`
response before reporting `Backend ready`.

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [Cyclo architecture](../../docs/architecture.html)

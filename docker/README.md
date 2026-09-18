# Source-Mounted Containers

The default Compose deployment follows AI Worker's source-workspace pattern.
Cyclo binds this checkout to `/root/ros2_ws/src/cyclo_intelligence` and uses
`colcon build --symlink-install`. Build/install directories stay inside the
container. `/opt/cyclo/policy`, `/opt/cyclo/sdk`, `/orchestrator_config`, and
`/opt/supervisor_api` link to that same workspace instead of separate code copies.

LeRobot and GR00T workers bind Cyclo's engine adapters, common runtime/catalog,
RobotClient, Zenoh SDK, and robot configuration read-only. LeRobot's input YAML
directory remains read-only mounted. Upstream model libraries, virtual
environments, CUDA, and other dependencies remain in the worker images.
The catalog is mounted beside `/policy_runtime` at `/catalog`, matching the
worker's existing sibling-catalog discovery. This avoids nested mounts that
would need to create directories inside the read-only runtime source.

Nginx binds URDF and mesh directories directly; it need not traverse `/root`.
Images still contain source and asset snapshots for standalone use without
these mounts. Compose intentionally runs the host checkout, not that snapshot.

## First Application

Stop/Clear inference and saved-pose return before maintenance. Preserve any
container-local `.bashrc` Zenoh settings: recreating a container discards them.

Build/recreate Cyclo once to install the new source links and mounts:

```bash
./docker/container.sh start --build
```

Existing workers need recreation to install the mounts, not an image rebuild
when their dependencies are already compatible. The Supervisor detects missing
or incorrectly sourced mounts and uses its normal maintenance interlock. For a
manual LeRobot recreation with the already-built local image, after Stop/Clear:

```bash
ARCH=amd64 docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.override.yml \
  up -d --no-build --pull never --force-recreate lerobot
```

Use `ARCH=arm64` on ARM64 and `groot` instead of `lerobot` for that worker.
Direct Docker/Compose commands bypass the Supervisor's safety interlock.
Do not use them while robot commands or a pending hold/return are active.

## After Editing

| Change | Apply it |
| --- | --- |
| Existing Orchestrator, Cyclo Data, Runtime or SDK Python | Stop relevant work, then restart the affected process/launch |
| Worker adapter, engine runtime or SDK Python | Stop/Clear, then restart the affected Worker process |
| Input YAML | Clear, then LOAD again |
| Robot configuration | Restart its consumers and reload the model |
| URDF or mesh asset | Reload its backend consumer and refresh the viewer |
| Supervisor Python | Restart Supervisor API |
| UI JavaScript/CSS | `./docker/container.sh build-ui`, then refresh the browser |
| ROS interfaces, C++, new ROS Python modules, entry points or package metadata | Stop affected processes, run `cb` inside Cyclo, source `install/setup.bash` again and restart |
| Worker manifest | Recreate that Worker and restart Cyclo's catalog consumers |
| Dependency, CUDA, model-library, Dockerfile or image-owned s6 script | Rebuild/recreate the affected image/container |
| Compose mounts/environment | Recreate the affected container |

`cb` is the existing shell alias for the workspace's explicit package list and
`--symlink-install`; it is not a Docker build. Changed ROS wire definitions must
remain compatible with every participating process and SDK cache. Update/rebuild
their generated artifacts together before restarting communication.

Bind mounts change files immediately, but already-imported Python modules do
not reload themselves. A process restart is still required. A container restart
does not apply new mounts. Manifest files are individual file mounts: an editor
or Git can replace their inode, so recreate workers after editing a manifest.

UI bundles remain compiled assets. Source mounting does not compile React;
`build-ui` replaces the served static files without restarting Cyclo. Bare
`container.sh start` may pull a published image; it is not the everyday Python
reload command. Use process restarts to keep the locally built environment.

## Operating Constraints

- Do not edit, switch branches, or pull while inference/return is active. Stop
  first, update the shared sources, then restart all affected consumers together.
- Runtime and workers now read the same checkout. A protocol or shared SDK
  change requires restarting both sides, even without image rebuilding.
- Read-only worker mounts protect source files from worker writes, not from host
  edits. Host code and image dependencies must still be a compatible pair.
- Keep the checkout at its mounted path. Moving it requires container recreation.
- The writable Cyclo workspace supports ROS build tooling; root-owned build
  metadata may be created in the checkout, as with AI Worker.
- This is a development-oriented deployment. Reproducible releases need a
  pinned checkout and matching images; an image tag alone no longer identifies
  the running Python code.

## Verification

```bash
python3 -m pytest -q docker/test_dockerfiles.py docker/test_container_sh.py \
  docker/supervisor_api/test_app.py
```

Inspect mounts without changing running services:

```bash
docker inspect cyclo_intelligence --format '{{json .Mounts}}'
docker exec cyclo_intelligence readlink -f /opt/cyclo/policy/common/runtime
docker inspect lerobot_server --format '{{json .Mounts}}'
```

The Runtime path should resolve inside `/root/ros2_ws/src/cyclo_intelligence`,
and the worker source mounts should refer to this checkout on the Docker host.

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

### Model Worker Images

Cyclo itself still uses a published image. All model Workers use local images:
no Worker image is pulled automatically. Existing compatible images are reused;
new installations build from this checkout and its recorded submodules.

This includes LeRobot and GR00T. Their `robotis/*-zenoh` names in Compose are
local build tags, not an instruction to download or publish Worker images.
Only `robotis/cyclo-intelligence` is published by this repository's release
workflows. CUDA/base images and model weights are still downloaded when needed.
For the 1.4.x to 1.5.0 upgrade, follow [the upgrade steps](#upgrading-from-14x)
and rebuild Workers instead of assuming an old image with the same tag is compatible.

```bash
# Common SDK plus the selected worker only (run from the repository root).
git submodule update --init -- \
  cyclo_brain/sdk/zenoh_ros2_sdk \
  cyclo_brain/policy/lerobot/lerobot
./docker/container.sh build-policy lerobot     # build only
./docker/container.sh start-lerobot            # reuse, or build if missing
./docker/container.sh start-lerobot --build    # rebuild and start
```

The same commands apply to `groot`, `abot`, `lingbot_vla`, and `rldx`, subject
to each Worker's supported architecture. Build dependencies/base images still
require network access. CUDA/FlashAttention settings remain Compose build inputs.

### Selective Source Checkout

A normal clone without `--recurse-submodules` does not download model sources.
The repository's `install.sh` initializes only the common SDK, not model sources.
Initialize the common SDK and only the worker source you need using the command
above, replacing the LeRobot path with one of these paths:

| Worker | Model submodule path |
| --- | --- |
| `lerobot` | `cyclo_brain/policy/lerobot/lerobot` |
| `groot` | `cyclo_brain/policy/groot/Isaac-GR00T` |
| `abot` | `cyclo_brain/policy/abot/ABot-Manipulation` |
| `lingbot_vla` | `cyclo_brain/policy/lingbot_vla/lingbot-vla-v2` |
| `rldx` | `cyclo_brain/policy/rldx/RLDX-1` |

Repeat the same targeted update after updating Cyclo. Do not use `--remote`:
workers must use the commits recorded by the Cyclo checkout, not upstream branch
tips. Building a worker checks its top-level source and the common SDK; other
models and optional nested simulation benchmarks need not be initialized.
These commands do not remove sources you already downloaded or change running
containers. Stop/Clear before updating a checkout used by running services.

Cyclo's core image needs the common SDK but no model submodule. The model source
is needed when building that worker image, not when reusing a compatible image.
The SDK is still required on the host for the source-mounted deployment.

GR00T and RLDX simulation benchmarks are optional and are excluded from their
inference images even if present locally. To work on an upstream benchmark,
initialize only its subtree, for example:

```bash
git -C cyclo_brain/policy/groot/Isaac-GR00T submodule update --init --recursive -- \
  external_dependencies/LIBERO
```

Follow upstream's separate evaluation setup and review the benchmark code and
asset licenses. An inference worker image is not a simulator evaluation image.
Source selection does not change model-weight or third-party license terms.

### Upgrading from 1.4.x

1. Stop/Clear inference and finish or discard recording. Stop pose return and
   navigation before changing a checkout used by running services.
2. Back up deployment-specific Compose overrides, Zenoh settings, model/data
   folders, the checkout commit and image IDs. Container recreation discards
   settings stored only inside the container.
3. Update Cyclo and initialize the common SDK and selected worker sources as
   described above.
4. Recreate core with the matching published image, or use
   `./docker/container.sh start --build` before that image is published. Rebuild
   ROS interfaces and restart their consumers together; old and new service
   definitions are not a supported mixed deployment.
5. Rebuild selected Workers with `./docker/container.sh start-lerobot --build`
   (replace `lerobot` as needed). An existing image tag does not guarantee
   compatibility with the new source.
6. Restore network settings and verify Preview camera/joint mappings before
   enabling physical robot commands.

For rollback, stop work and restore both the old source checkout and matching
images. Restoring only an image is insufficient with source bind mounts. Preserve
data backups when changing formats. Merging into main does not publish images;
publication follows the existing main-to-jazzy release workflow.

### UI Builds

In the UI, use **Build** for a missing image, or the wrench to rebuild an existing
one. Builds run in Supervisor, survive browser navigation, and expose bounded
logs through status polling. ON never silently builds or pulls. After a rebuild,
use **Update Container** if shown, then ON. Builds do not start/recreate workers.
Stop/Clear before Worker maintenance. Only one UI build runs at a time.

UI builds require a full Git checkout mounted in Cyclo, Docker socket access,
Compose V2 and Buildx. The Cyclo Dockerfiles install these tools; older Cyclo
images need an image update before UI builds work. Host CLI builds use the host's
Docker tools. Existing container-local Zenoh settings must be preserved before
any container recreation.

Do not edit the checkout or stop Supervisor during a build. Graceful Supervisor
shutdown cancels its build and releases the maintenance reservation. If Supervisor
is forcibly killed, confirm no build is running before restarting Policy Runtime
to clear an abandoned reservation. Build logs are in memory, not an audit archive.

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

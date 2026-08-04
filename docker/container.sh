#!/bin/bash
#
# cyclo_intelligence container helper. It auto-detects ARCH from uname -m
# and manages the main runtime plus optional policy containers.
#
# Usage:
#   docker/container.sh start              # → cyclo_intelligence
#   docker/container.sh start-lerobot      # → lerobot (idle until LOAD)
#   docker/container.sh start-groot        # → groot (idle until LOAD)
#   docker/container.sh enter              # → shell in cyclo_intelligence
#   docker/container.sh build-ui           # → rebuild React UI only
#   docker/container.sh enter-lerobot      # → shell in lerobot_server
#   docker/container.sh enter-groot        # → shell in groot_server
#   docker/container.sh logs               # → compose logs -f
#   docker/container.sh status             # → s6 svstat on all containers
#   docker/container.sh stop               # → compose down
#   docker/container.sh help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f ${SCRIPT_DIR}/docker-compose.yml"
# Standard docker-compose override convention. Auto-discovery is
# disabled when -f is passed explicitly (above), so we re-enable it
# manually: if a sibling docker-compose.override.yml exists, layer it
# on top. Lets developers shadow image tags / mounts locally without
# editing the canonical compose file.
[ -f "${SCRIPT_DIR}/docker-compose.override.yml" ] \
    && COMPOSE="${COMPOSE} -f ${SCRIPT_DIR}/docker-compose.override.yml"

MAIN_SERVICE="cyclo_intelligence"
MAIN_CONTAINER="cyclo_intelligence"
LEROBOT_SERVICE="lerobot"
LEROBOT_CONTAINER="${LEROBOT_CONTAINER_NAME:-lerobot_server}"
GROOT_SERVICE="groot"
GROOT_CONTAINER="${GROOT_CONTAINER_NAME:-groot_server}"

# Auto-detect host architecture for Dockerfile / image tag selection
MACHINE_ARCH=$(uname -m)
if [ "$MACHINE_ARCH" = "aarch64" ] || [ "$MACHINE_ARCH" = "arm64" ]; then
    export ARCH="arm64"
    echo "[container.sh] Detected ARM64 architecture (Jetson)"
else
    export ARCH="amd64"
    echo "[container.sh] Detected AMD64 architecture (x86_64)"
fi

# Optional opt-in rebuild. Default is to use the pre-built Hub image.
# Pass `--build` (or `-b`) on any start* command to rebuild from source.
BUILD_FLAG=""
NEW_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --build|-b) BUILD_FLAG="--build" ;;
        *)          NEW_ARGS+=("$arg") ;;
    esac
done
set -- "${NEW_ARGS[@]}"

# Pre-create host bind-mount targets so docker doesn't auto-create them
# as root-owned directories (which then can't be written to from the
# host without sudo). Compose always mounts docker/workspace and
# docker/huggingface into the containers. Storage location is decided by
# the repository install path: on robots the installer places the repo on
# SSD, while local installs stay under the user's home directory.
ensure_host_dir() {
    if [ -L "$1" ] && [ ! -e "$1" ]; then
        echo "[container.sh] Error: stale symlink: $1" >&2
        echo "[container.sh] Fix the symlink target or remove it before starting." >&2
        exit 1
    fi
    [ -d "$1" ] || mkdir -p "$1"
}

canonical_path() {
    local resolved
    resolved="$(readlink -f "$1" 2>/dev/null || true)"
    if [ -n "$resolved" ]; then
        printf '%s\n' "$resolved"
    else
        printf '%s\n' "$1"
    fi
}

prepare_host_mounts() {
    local workspace_dir="${SCRIPT_DIR}/workspace"
    local huggingface_dir="${SCRIPT_DIR}/huggingface"
    local workspace_real
    local huggingface_real

    if [ -n "${CYCLO_WORKSPACE_DIR:-}" ] || [ -n "${CYCLO_HUGGINGFACE_DIR:-}" ]; then
        echo "[container.sh] Warning: CYCLO_WORKSPACE_DIR and CYCLO_HUGGINGFACE_DIR are ignored." >&2
        echo "[container.sh] Compose always mounts docker/workspace and docker/huggingface." >&2
    fi

    ensure_host_dir "$workspace_dir"
    ensure_host_dir "${workspace_dir}/dataset"
    ensure_host_dir "${workspace_dir}/rosbag2"
    ensure_host_dir "${workspace_dir}/lerobot"
    ensure_host_dir "${workspace_dir}/model"
    ensure_host_dir "${workspace_dir}/model/lerobot"
    ensure_host_dir "${workspace_dir}/model/groot"
    ensure_host_dir "$huggingface_dir"

    workspace_real="$(canonical_path "$workspace_dir")"
    huggingface_real="$(canonical_path "$huggingface_dir")"

    echo "[container.sh]   workspace:   ${workspace_dir} -> ${workspace_real}"
    echo "[container.sh]   huggingface: ${huggingface_dir} -> ${huggingface_real}"
}

CYCLO_AGENT_SOCKETS_DIR="${CYCLO_AGENT_SOCKETS_DIR:-/var/run/robotis/agent_sockets/cyclo_intelligence}"
export CYCLO_AGENT_SOCKETS_DIR
mkdir -p "$CYCLO_AGENT_SOCKETS_DIR" 2>/dev/null \
    || sudo mkdir -p "$CYCLO_AGENT_SOCKETS_DIR" 2>/dev/null \
    || true

# X11 forwarding for UI windows (rviz, plotjuggler, etc.) when started
# from an interactive shell. Silently skipped if DISPLAY isn't set.
setup_x11() {
    if [ -n "$DISPLAY" ]; then
        xhost +local:docker > /dev/null 2>&1 || true
    fi
}

container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1\$"
}

compose_service_image() {
    local service="$1"
    $COMPOSE config --format json 2>/dev/null \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["services"][sys.argv[1]]["image"])' "$service"
}

container_workspace_source() {
    docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$1" 2>/dev/null || true
}

expected_workspace_source() {
    canonical_path "${SCRIPT_DIR}/workspace"
}

paths_equal() {
    [ "$(canonical_path "$1")" = "$(canonical_path "$2")" ]
}

remove_stale_policy_container() {
    local service="$1"
    local container="$2"
    local expected_image
    local expected_id
    local current_id
    local current_workspace
    local expected_workspace

    expected_image="$(compose_service_image "$service" 2>/dev/null || true)"
    if [ -z "$expected_image" ]; then
        echo "[container.sh] Warning: could not resolve compose image for $service; skipping stale-container check."
        return 0
    fi

    expected_id="$(docker image inspect -f '{{.Id}}' "$expected_image" 2>/dev/null || true)"
    current_id="$(docker inspect -f '{{.Image}}' "$container" 2>/dev/null || true)"
    if [ -n "$expected_id" ] && [ -n "$current_id" ] && [ "$expected_id" != "$current_id" ]; then
        echo "[container.sh] Removing stale $container (expected $expected_image). It will be recreated on next start."
        docker rm -f "$container" >/dev/null || true
        return 0
    fi

    current_workspace="$(container_workspace_source "$container")"
    if [ -n "$current_id" ] && [ -z "$current_workspace" ]; then
        echo "[container.sh] Removing stale $container (/workspace mount missing). It will be recreated on next start."
        docker rm -f "$container" >/dev/null || true
        return 0
    fi
    expected_workspace="$(expected_workspace_source)"
    if [ -n "$current_id" ] && ! paths_equal "$current_workspace" "$expected_workspace"; then
        echo "[container.sh] Removing stale $container (/workspace mounted from $current_workspace, expected $expected_workspace). It will be recreated on next start."
        docker rm -f "$container" >/dev/null || true
    fi
}

remove_stale_policy_containers() {
    remove_stale_policy_container "$LEROBOT_SERVICE" "$LEROBOT_CONTAINER"
    remove_stale_policy_container "$GROOT_SERVICE" "$GROOT_CONTAINER"
}

show_help() {
    cat <<EOF
Usage: $0 <command>

Main image (cyclo_intelligence):
  start            Build (if needed) and start cyclo_intelligence
  enter            Open an interactive bash in cyclo_intelligence
  logs             Tail cyclo_intelligence logs

LeRobot policy container:
  start-lerobot    Build + start lerobot. Container boots idle and
                   only configures itself once orchestrator dispatches
                   InferenceCommand.LOAD with a robot_type.
  enter-lerobot    Open an interactive bash in lerobot_server

GR00T policy container:
  start-groot      Build + start groot (N1.7 baseline). Same boot-idle
                   + LOAD-time configure pattern as lerobot.
  enter-groot      Open an interactive bash in groot_server

Lifecycle:
  status           s6-svstat on all containers (when running)
  stop             compose down (prompts for confirmation)
  help             Show this help

UI development:
  build-ui         Rebuild only orchestrator/ui and copy the static build into
                   the running cyclo_intelligence nginx root. Uses npm inside
                   cyclo_intelligence when available, with a node:22 fallback.
  test-ui [args]   Run React tests. Extra args are passed after npm test,
                   e.g. test-ui -- --watchAll=false

Flags (any start* command):
  --build, -b      Rebuild image from local Dockerfile instead of using
                   the pre-built image pulled from Docker Hub. Default
                   is to use the pulled image (fast, no source build
                   required). Use this only when iterating on Dockerfile.

Environment:
  GPU_ARCH         default | blackwell   (optional, amd64 only)
  FLASH_ATTN_BUILD_JOBS
                   flash-attn source build parallelism for GR00T Blackwell
                   images (default 1)
  FLASH_ATTN_NVCC_THREADS
                   nvcc threads per flash-attn build job (default 1)
  FLASH_ATTN_CUDA_ARCHS
                   CUDA archs for GR00T Blackwell flash-attn builds
                   (default 120)
  VERSION          image tag version (default: 1.2.0 for cyclo)
  ROS/Zenoh        Edit /root/.bashrc inside each container, then restart that
                   container. docker restart preserves edits; recreating the
                   container resets /root/.bashrc to the image default.
  Storage          Containers mount docker/workspace and docker/huggingface
                   from this checkout. Use install.sh on robots so the checkout
                   itself lives on /mnt/ssd.
  CYCLO_UI_NODE_IMAGE
                   Node image for build-ui/test-ui (default node:22).
EOF
}

ui_dir() {
    canonical_path "${SCRIPT_DIR}/../orchestrator/ui"
}

main_ui_dir() {
    printf '%s\n' "/root/ros2_ws/src/cyclo_intelligence/orchestrator/ui"
}

main_container_has_npm() {
    container_running "$MAIN_CONTAINER" \
        && docker exec "$MAIN_CONTAINER" sh -lc 'command -v npm >/dev/null 2>&1'
}

enter_bash() {
    local container="$1"
    docker exec -it "$container" bash
}

run_ui_npm_in_main() {
    docker exec \
        -u "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -w "$(main_ui_dir)" \
        "$MAIN_CONTAINER" \
        npm "$@"
}

run_ui_npm_external() {
    local dir
    dir="$(ui_dir)"
    docker run --rm --network host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "${dir}:/ui" \
        -w /ui \
        "${CYCLO_UI_NODE_IMAGE:-node:22}" \
        npm "$@"
}

run_ui_npm() {
    if main_container_has_npm; then
        run_ui_npm_in_main "$@"
    else
        run_ui_npm_external "$@"
    fi
}

ensure_ui_dependencies() {
    local dir
    if main_container_has_npm; then
        if docker exec "$MAIN_CONTAINER" test -x "$(main_ui_dir)/node_modules/.bin/react-scripts"; then
            return 0
        fi

        echo "[container.sh] Installing UI dependencies inside ${MAIN_CONTAINER}..."
        run_ui_npm_in_main ci --legacy-peer-deps
        return 0
    fi

    dir="$(ui_dir)"
    if [ -x "${dir}/node_modules/.bin/react-scripts" ]; then
        return 0
    fi

    echo "[container.sh] Installing UI dependencies with ${CYCLO_UI_NODE_IMAGE:-node:22}..."
    run_ui_npm ci --legacy-peer-deps
}

clean_ui_build_dir() {
    local dir
    if container_running "$MAIN_CONTAINER"; then
        docker exec "$MAIN_CONTAINER" sh -lc "rm -rf '$(main_ui_dir)/build'"
        return 0
    fi

    dir="$(ui_dir)"
    docker run --rm --network none \
        -v "${dir}:/ui" \
        -w /ui \
        "${CYCLO_UI_NODE_IMAGE:-node:22}" \
        sh -c 'rm -rf build'
}

build_ui() {
    local dir
    dir="$(ui_dir)"
    ensure_ui_dependencies

    echo "[container.sh] Building React UI only..."
    clean_ui_build_dir
    run_ui_npm run build

    if ! container_running "$MAIN_CONTAINER"; then
        echo "[container.sh] UI build complete: ${dir}/build"
        echo "[container.sh] ${MAIN_CONTAINER} is not running, so nginx was not updated."
        return 0
    fi

    echo "[container.sh] Copying UI build into ${MAIN_CONTAINER} nginx root..."
    docker cp "${dir}/build/." "${MAIN_CONTAINER}:/usr/share/nginx/html/"
    docker exec "$MAIN_CONTAINER" sh -c 'nginx -s reload 2>/dev/null || true'
    echo "[container.sh] UI updated. Refresh the browser to load the new bundle."
}

test_ui() {
    ensure_ui_dependencies
    echo "[container.sh] Running React UI tests..."
    if [ "$#" -eq 0 ]; then
        run_ui_npm test -- --watchAll=false
    else
        run_ui_npm test "$@"
    fi
}

start_main() {
    prepare_host_mounts
    setup_x11
    if [ -n "$BUILD_FLAG" ]; then
        echo "[container.sh] Building $MAIN_SERVICE from local Dockerfile; skipping pre-built image pull."
    else
        echo "[container.sh] Pulling pre-built image..."
        echo "[container.sh] Local Dockerfile changes are ignored without --build."
        $COMPOSE pull --ignore-pull-failures "$MAIN_SERVICE" || true
    fi
    echo "[container.sh] Starting $MAIN_SERVICE (ARCH=$ARCH${BUILD_FLAG:+, rebuild on})..."
    $COMPOSE up -d $BUILD_FLAG "$MAIN_SERVICE"
    echo "[container.sh] Done. 'docker/container.sh status' to check s6 services."
}

start_lerobot() {
    prepare_host_mounts
    setup_x11
    if [ -n "$BUILD_FLAG" ]; then
        echo "[container.sh] Building $LEROBOT_SERVICE from local Dockerfile; skipping pre-built image pull."
    else
        echo "[container.sh] Pulling pre-built images..."
        echo "[container.sh] Local Dockerfile/s6 changes are ignored without --build."
        $COMPOSE pull --ignore-pull-failures "$LEROBOT_SERVICE" || true
    fi
    remove_stale_policy_container "$LEROBOT_SERVICE" "$LEROBOT_CONTAINER"
    echo "[container.sh] Starting $LEROBOT_SERVICE (ARCH=$ARCH${BUILD_FLAG:+, rebuild on})..."
    $COMPOSE up -d $BUILD_FLAG "$LEROBOT_SERVICE"
}

start_groot() {
    prepare_host_mounts
    setup_x11
    if [ -n "$BUILD_FLAG" ]; then
        echo "[container.sh] Building $GROOT_SERVICE from local Dockerfile; skipping pre-built image pull."
    else
        echo "[container.sh] Pulling pre-built images..."
        echo "[container.sh] Local Dockerfile/s6 changes are ignored without --build."
        $COMPOSE pull --ignore-pull-failures "$GROOT_SERVICE" || true
    fi
    remove_stale_policy_container "$GROOT_SERVICE" "$GROOT_CONTAINER"
    echo "[container.sh] Starting $GROOT_SERVICE (ARCH=$ARCH${BUILD_FLAG:+, rebuild on})..."
    $COMPOSE up -d $BUILD_FLAG "$GROOT_SERVICE"
}

enter_main() {
    if ! container_running "$MAIN_CONTAINER"; then
        echo "Error: $MAIN_CONTAINER is not running. Run 'start' first." >&2
        exit 1
    fi
    setup_x11
    enter_bash "$MAIN_CONTAINER"
}

enter_lerobot() {
    if ! container_running "$LEROBOT_CONTAINER"; then
        echo "Error: $LEROBOT_CONTAINER is not running. Run 'start-lerobot' first." >&2
        exit 1
    fi
    enter_bash "$LEROBOT_CONTAINER"
}

enter_groot() {
    if ! container_running "$GROOT_CONTAINER"; then
        echo "Error: $GROOT_CONTAINER is not running. Run 'start-groot' first." >&2
        exit 1
    fi
    enter_bash "$GROOT_CONTAINER"
}

show_logs() {
    $COMPOSE logs -f
}

show_status() {
    echo "=== Containers ==="
    docker ps --format '{{.Names}}\t{{.Status}}' \
        | grep -E "^(${MAIN_CONTAINER}|${LEROBOT_CONTAINER}|${GROOT_CONTAINER})\\b" \
        || echo "(none running)"

    # s6-overlay installs s6-svstat under /package/admin/s6-*/command/
    # rather than a stable PATH location, so resolve it dynamically.
    # `sh -c` inside docker exec lets the inner shell glob the version
    # directory without depending on bash being available.
    local svstat_setup='
        S6_SVSTAT=$(ls /package/admin/s6-*/command/s6-svstat 2>/dev/null | head -1)
        [ -z "$S6_SVSTAT" ] && S6_SVSTAT=$(command -v s6-svstat 2>/dev/null)
        [ -z "$S6_SVSTAT" ] && { echo "  (s6-svstat not found)"; exit 0; }
    '

    if container_running "$MAIN_CONTAINER"; then
        echo ""
        echo "=== ${MAIN_CONTAINER} s6 services ==="
        docker exec "$MAIN_CONTAINER" sh -c "
            ${svstat_setup}
            for svc in /run/service/*/; do
                name=\$(basename \"\$svc\")
                printf '  %-30s %s\n' \"\$name\" \"\$(\$S6_SVSTAT \"\$svc\" 2>&1)\"
            done
        " || true
    fi

    for cont in "$LEROBOT_CONTAINER" "$GROOT_CONTAINER"; do
        if container_running "$cont"; then
            echo ""
            # Not every policy container uses s6-overlay (e.g. lerobot
            # has PID 1 running executor.py directly). Detect /run/service
            # first; fall back to top-level process list otherwise.
            if docker exec "$cont" sh -c '[ -d /run/service ]' 2>/dev/null; then
                echo "=== ${cont} s6 services ==="
                docker exec "$cont" sh -c "
                    ${svstat_setup}
                    for svc in /run/service/*/; do
                        name=\$(basename \"\$svc\")
                        printf '  %-30s %s\n' \"\$name\" \"\$(\$S6_SVSTAT \"\$svc\" 2>&1)\"
                    done
                " || true
            else
                echo "=== ${cont} processes (no s6-overlay) ==="
                docker exec "$cont" sh -c 'ps -eo pid,user,comm,args | head -8' || true
            fi
        fi
    done
}

stop_all() {
    echo "Warning: this will stop and remove all compose-managed containers."
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $COMPOSE down
    else
        echo "Cancelled."
    fi
}

case "${1:-help}" in
    start)           start_main ;;
    start-lerobot)   start_lerobot ;;
    start-groot)     start_groot ;;
    enter)           enter_main ;;
    enter-lerobot)   enter_lerobot ;;
    enter-groot)     enter_groot ;;
    build-ui)        build_ui ;;
    test-ui)         shift; test_ui "$@" ;;
    logs)            show_logs ;;
    status)          show_status ;;
    stop)            stop_all ;;
    help|-h|--help)  show_help ;;
    *)
        echo "Error: unknown command '$1'" >&2
        show_help
        exit 1
        ;;
esac

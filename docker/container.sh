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
# docker/huggingface into the containers; when SSD storage is available,
# these repo-local paths are turned into symlinks to CYCLO_SSD_ROOT.
ensure_host_dir() {
    if [ -L "$1" ] && [ ! -e "$1" ]; then
        echo "[container.sh] Error: stale symlink: $1" >&2
        echo "[container.sh] Fix the symlink target or remove it before starting." >&2
        exit 1
    fi
    [ -d "$1" ] || mkdir -p "$1"
}

storage_root_usable() {
    local root="$1"
    local probe="${root}/.cyclo-write-test.$$"

    mkdir -p "$root" 2>/dev/null || return 1
    mkdir "$probe" 2>/dev/null || return 1
    rmdir "$probe" 2>/dev/null || true
}

ssd_mountpoint_for_root() {
    local root="$1"

    if [ -n "${CYCLO_SSD_MOUNTPOINT:-}" ]; then
        canonical_path "$CYCLO_SSD_MOUNTPOINT"
        return 0
    fi

    case "$root" in
        /mnt/ssd|/mnt/ssd/*) printf '%s\n' "/mnt/ssd" ;;
        *)                   printf '%s\n' "" ;;
    esac
}

mountpoint_configured() {
    local mountpoint_path="$1"

    [ -r /etc/fstab ] \
        && awk -v mountpoint_path="$mountpoint_path" \
            '$2 == mountpoint_path { found=1 } END { exit found ? 0 : 1 }' \
            /etc/fstab
}

try_mount_ssd_root() {
    local root="$1"
    local mountpoint_path

    mountpoint_path="$(ssd_mountpoint_for_root "$root")"
    if [ -z "$mountpoint_path" ] || mountpoint -q "$mountpoint_path"; then
        return 0
    fi
    if ! mountpoint_configured "$mountpoint_path"; then
        return 1
    fi

    echo "[container.sh] SSD mountpoint is not mounted: $mountpoint_path"
    echo "[container.sh] Attempting to mount $mountpoint_path"
    if [ "$(id -u)" -eq 0 ]; then
        mount "$mountpoint_path" 2>/dev/null || true
    elif command -v sudo >/dev/null 2>&1; then
        if [ -t 0 ]; then
            sudo mount "$mountpoint_path" 2>/dev/null || true
        else
            sudo -n mount "$mountpoint_path" 2>/dev/null || true
        fi
    fi
    mountpoint -q "$mountpoint_path"
}

ssd_root_usable() {
    local root="$1"
    local mountpoint_path

    mountpoint_path="$(ssd_mountpoint_for_root "$root")"
    if [ -n "$mountpoint_path" ] && ! mountpoint -q "$mountpoint_path"; then
        return 1
    fi
    storage_root_usable "$root"
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

path_within() {
    local path="$1"
    local root="$2"
    case "$path" in
        "$root"|"$root"/*) return 0 ;;
        *)                 return 1 ;;
    esac
}

prepare_required_ssd_root() {
    local root="$1"
    local mountpoint_path

    try_mount_ssd_root "$root" || true
    if ssd_root_usable "$root"; then
        return 0
    fi
    mountpoint_path="$(ssd_mountpoint_for_root "$root")"
    if [ -n "$mountpoint_path" ] && ! mountpoint -q "$mountpoint_path"; then
        return 1
    fi

    if [ "$(id -u)" -eq 0 ]; then
        mkdir -p "$root"
        chown "$(id -u):$(id -g)" "$root" 2>/dev/null || true
    elif command -v sudo >/dev/null 2>&1; then
        sudo mkdir -p "$root"
        sudo chown "$(id -u):$(id -g)" "$root"
    else
        return 1
    fi

    ssd_root_usable "$root"
}

path_is_empty_dir() {
    [ -d "$1" ] && [ -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]
}

backup_path_for() {
    local path="$1"
    local stamp
    local candidate
    local index=0

    stamp="$(date +%Y%m%d-%H%M%S)"
    candidate="${path}.local-before-ssd-${stamp}"
    while [ -e "$candidate" ] || [ -L "$candidate" ]; do
        index=$((index + 1))
        candidate="${path}.local-before-ssd-${stamp}.${index}"
    done
    printf '%s\n' "$candidate"
}

migrate_local_dir_to_ssd() {
    local src_path="$1"
    local target_path="$2"
    local label="$3"

    if [ ! -d "$src_path" ] || [ -L "$src_path" ]; then
        return 0
    fi
    if path_is_empty_dir "$src_path"; then
        return 0
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        echo "[container.sh] Error: rsync is required to migrate existing ${label} data to SSD." >&2
        exit 1
    fi

    echo "[container.sh] Migrating existing ${label} data to ${target_path} without overwriting SSD files."
    rsync -rltHP --omit-dir-times --no-owner --no-group --no-perms \
        --ignore-existing --remove-source-files "$src_path"/ "$target_path"/
    find "$src_path" -depth -type d -empty -delete || true
}

prepare_ssd_link() {
    local link_path="$1"
    local target_path="$2"
    local ssd_root="$3"
    local label="$4"
    local link_real
    local target_real
    local backup_path

    mkdir -p "$target_path"
    target_real="$(canonical_path "$target_path")"

    if [ -L "$link_path" ] && [ -e "$link_path" ]; then
        link_real="$(canonical_path "$link_path")"
        if path_within "$link_real" "$ssd_root"; then
            return 0
        fi
    fi

    if [ -e "$link_path" ] || [ -L "$link_path" ]; then
        if [ -d "$link_path" ] && [ ! -L "$link_path" ]; then
            migrate_local_dir_to_ssd "$link_path" "$target_real" "$label"
        fi
    fi

    if [ -e "$link_path" ] || [ -L "$link_path" ]; then
        if path_is_empty_dir "$link_path" && [ ! -L "$link_path" ]; then
            rmdir "$link_path"
        elif [ ! -L "$link_path" ]; then
            backup_path="$(backup_path_for "$link_path")"
            mv "$link_path" "$backup_path"
            echo "[container.sh] Preserved remaining local ${label} data at ${backup_path}."
        else
            echo "[container.sh] Error: ${link_path} is a symlink outside ${ssd_root}." >&2
            exit 1
        fi
    fi

    ln -s "$target_real" "$link_path"
}

prepare_ssd_links() {
    local ssd_root="$1"
    local workspace_dir="${SCRIPT_DIR}/workspace"
    local huggingface_dir="${SCRIPT_DIR}/huggingface"

    prepare_ssd_link "$workspace_dir" "${ssd_root}/workspace" "$ssd_root" "workspace"
    prepare_ssd_link "$huggingface_dir" "${ssd_root}/huggingface" "$ssd_root" "huggingface"
}

refresh_storage_label() {
    local ssd_root="$1"
    local workspace_real="$2"
    local huggingface_real="$3"

    if path_within "$workspace_real" "$ssd_root" \
        && path_within "$huggingface_real" "$ssd_root"; then
        printf '%s\n' "SSD"
    else
        printf '%s\n' "repo-local"
    fi
}

require_ssd_storage() {
    local ssd_root="$1"
    local workspace_dir="$2"
    local huggingface_dir="$3"
    local workspace_real
    local huggingface_real

    if ! prepare_required_ssd_root "$ssd_root"; then
        echo "[container.sh] Error: SSD storage root is not writable: $ssd_root" >&2
        echo "[container.sh] Mount/create the SSD path or choose CYCLO_STORAGE_MODE=local." >&2
        exit 1
    fi

    prepare_ssd_links "$ssd_root"

    workspace_real="$(canonical_path "$workspace_dir")"
    huggingface_real="$(canonical_path "$huggingface_dir")"
    if ! path_within "$workspace_real" "$ssd_root" \
        || ! path_within "$huggingface_real" "$ssd_root"; then
        echo "[container.sh] Error: repo-local storage paths do not resolve under $ssd_root." >&2
        echo "[container.sh] Current workspace:   $workspace_dir -> $workspace_real" >&2
        echo "[container.sh] Current huggingface: $huggingface_dir -> $huggingface_real" >&2
        exit 1
    fi
}

setup_storage() {
    local storage_mode="${CYCLO_STORAGE_MODE:-auto}"
    local ssd_root="${CYCLO_SSD_ROOT:-/mnt/ssd/cyclo_intelligence}"
    local workspace_dir="${SCRIPT_DIR}/workspace"
    local huggingface_dir="${SCRIPT_DIR}/huggingface"
    local workspace_real
    local huggingface_real
    local storage_label="repo-local"

    case "$storage_mode" in
        auto|ssd|local) ;;
        *)
            echo "[container.sh] Error: unknown CYCLO_STORAGE_MODE='$storage_mode' (expected auto, ssd, or local)" >&2
            exit 1
            ;;
    esac

    if [ -n "${CYCLO_WORKSPACE_DIR:-}" ] || [ -n "${CYCLO_HUGGINGFACE_DIR:-}" ]; then
        echo "[container.sh] Warning: CYCLO_WORKSPACE_DIR and CYCLO_HUGGINGFACE_DIR are ignored." >&2
        echo "[container.sh] Compose always mounts docker/workspace and docker/huggingface." >&2
    fi

    ssd_root="$(canonical_path "$ssd_root")"

    if [ "$storage_mode" != "local" ]; then
        try_mount_ssd_root "$ssd_root" || true
    fi

    if [ "$storage_mode" = "auto" ] && ssd_root_usable "$ssd_root"; then
        prepare_ssd_links "$ssd_root"
    fi

    if [ "$storage_mode" = "ssd" ]; then
        require_ssd_storage "$ssd_root" "$workspace_dir" "$huggingface_dir"
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
    storage_label="$(refresh_storage_label "$ssd_root" "$workspace_real" "$huggingface_real")"

    if [ "$storage_mode" = "local" ] && [ "$storage_label" = "SSD" ]; then
        echo "[container.sh] Warning: CYCLO_STORAGE_MODE=local requested, but repo-local paths resolve under $ssd_root." >&2
    fi

    echo "[container.sh] Using ${storage_label} storage"
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
  VERSION          image tag version (default: 1.0.0 for cyclo)
  ROS_DOMAIN_ID    default 30
  CYCLO_STORAGE_MODE
                   auto | ssd | local (default auto). Containers always
                   mount docker/workspace and docker/huggingface. Auto uses
                   CYCLO_SSD_ROOT when writable and migrates local-only files
                   without overwriting SSD files.
  CYCLO_SSD_ROOT   SSD relocation root (default /mnt/ssd/cyclo_intelligence)
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
    setup_storage
    setup_x11
    echo "[container.sh] Pulling pre-built image..."
    $COMPOSE pull --ignore-pull-failures "$MAIN_SERVICE" || true
    echo "[container.sh] Starting $MAIN_SERVICE (ARCH=$ARCH${BUILD_FLAG:+, rebuild on})..."
    $COMPOSE up -d $BUILD_FLAG "$MAIN_SERVICE"
    echo "[container.sh] Done. 'docker/container.sh status' to check s6 services."
}

start_lerobot() {
    setup_storage
    setup_x11
    echo "[container.sh] Pulling pre-built images..."
    $COMPOSE pull --ignore-pull-failures "$LEROBOT_SERVICE" || true
    remove_stale_policy_container "$LEROBOT_SERVICE" "$LEROBOT_CONTAINER"
    echo "[container.sh] Starting $LEROBOT_SERVICE (ARCH=$ARCH${BUILD_FLAG:+, rebuild on})..."
    $COMPOSE up -d $BUILD_FLAG "$LEROBOT_SERVICE"
}

start_groot() {
    setup_storage
    setup_x11
    echo "[container.sh] Pulling pre-built images..."
    $COMPOSE pull --ignore-pull-failures "$GROOT_SERVICE" || true
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
    docker exec -it "$MAIN_CONTAINER" bash
}

enter_lerobot() {
    if ! container_running "$LEROBOT_CONTAINER"; then
        echo "Error: $LEROBOT_CONTAINER is not running. Run 'start-lerobot' first." >&2
        exit 1
    fi
    docker exec -it "$LEROBOT_CONTAINER" bash
}

enter_groot() {
    if ! container_running "$GROOT_CONTAINER"; then
        echo "Error: $GROOT_CONTAINER is not running. Run 'start-groot' first." >&2
        exit 1
    fi
    docker exec -it "$GROOT_CONTAINER" bash
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

#!/bin/bash
#
# cyclo_intelligence container helper — mirrors the physical_ai_tools
# docker/container.sh pattern (auto-detects ARCH from uname -m). Step 6
# expands this from lerobot-only to also managing the main
# cyclo_intelligence service.
#
# Usage:
#   docker/container.sh start              # → cyclo_intelligence
#   docker/container.sh start-lerobot      # → lerobot (idle until LOAD)
#   docker/container.sh start-groot        # → groot (idle until LOAD)
#   docker/container.sh enter              # → shell in cyclo_intelligence
#   docker/container.sh enter-lerobot      # → shell in lerobot_server
#   docker/container.sh enter-groot        # → shell in groot_server
#   docker/container.sh logs               # → compose logs -f
#   docker/container.sh status             # → s6 svstat on all containers
#   docker/container.sh stop               # → compose down
#   docker/container.sh help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f ${SCRIPT_DIR}/docker-compose.yml"

MAIN_SERVICE="cyclo_intelligence"
MAIN_CONTAINER="cyclo_intelligence"
LEROBOT_SERVICE="lerobot"
LEROBOT_CONTAINER="lerobot_server"
GROOT_SERVICE="groot"
GROOT_CONTAINER="groot_server"

# Auto-detect host architecture for Dockerfile / image tag selection
MACHINE_ARCH=$(uname -m)
if [ "$MACHINE_ARCH" = "aarch64" ] || [ "$MACHINE_ARCH" = "arm64" ]; then
    export ARCH="arm64"
    echo "[container.sh] Detected ARM64 architecture (Jetson)"
else
    export ARCH="amd64"
    echo "[container.sh] Detected AMD64 architecture (x86_64)"
fi

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
  start-groot      Build + start groot (N1.6 baseline). Same boot-idle
                   + LOAD-time configure pattern as lerobot.
  enter-groot      Open an interactive bash in groot_server

Lifecycle:
  status           s6-svstat on all containers (when running)
  stop             compose down (prompts for confirmation)
  help             Show this help

Environment:
  GPU_ARCH         default | blackwell   (optional, amd64 only)
  VERSION          image tag version (default: 1.0.0)
  ROS_DOMAIN_ID    default 30
EOF
}

start_main() {
    setup_x11
    echo "[container.sh] Pulling pre-built images (ignoring local-only failures)..."
    $COMPOSE pull --ignore-pull-failures "$MAIN_SERVICE" || true
    echo "[container.sh] Starting $MAIN_SERVICE (ARCH=$ARCH)..."
    $COMPOSE up -d --build "$MAIN_SERVICE"
    echo "[container.sh] Done. 'docker/container.sh status' to check s6 services."
}

start_lerobot() {
    setup_x11
    echo "[container.sh] Pulling pre-built images..."
    $COMPOSE pull --ignore-pull-failures "$LEROBOT_SERVICE" || true
    echo "[container.sh] Starting $LEROBOT_SERVICE (ARCH=$ARCH)..."
    $COMPOSE up -d --build "$LEROBOT_SERVICE"
}

start_groot() {
    setup_x11
    echo "[container.sh] Pulling pre-built images..."
    $COMPOSE pull --ignore-pull-failures "$GROOT_SERVICE" || true
    echo "[container.sh] Starting $GROOT_SERVICE (ARCH=$ARCH)..."
    $COMPOSE up -d --build "$GROOT_SERVICE"
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

    if container_running "$MAIN_CONTAINER"; then
        echo ""
        echo "=== ${MAIN_CONTAINER} s6 services ==="
        docker exec "$MAIN_CONTAINER" /bin/sh -c \
            'for svc in /run/service/*/; do
               name=$(basename "$svc")
               s6-svstat "$svc" 2>/dev/null | sed "s/^/  ${name}: /"
             done' || true
    fi

    if container_running "$LEROBOT_CONTAINER"; then
        echo ""
        echo "=== ${LEROBOT_CONTAINER} s6 services ==="
        docker exec "$LEROBOT_CONTAINER" s6-svstat /run/service/inference-server 2>/dev/null || true
        docker exec "$LEROBOT_CONTAINER" s6-svstat /run/service/control-publisher 2>/dev/null || true
    fi

    if container_running "$GROOT_CONTAINER"; then
        echo ""
        echo "=== ${GROOT_CONTAINER} s6 services ==="
        docker exec "$GROOT_CONTAINER" s6-svstat /run/service/inference-server 2>/dev/null || true
        docker exec "$GROOT_CONTAINER" s6-svstat /run/service/control-publisher 2>/dev/null || true
    fi
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

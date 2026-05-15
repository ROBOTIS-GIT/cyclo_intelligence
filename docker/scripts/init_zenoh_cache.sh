#!/usr/bin/env bash
#
# Pre-populate the zenoh_ros2_sdk message-definition cache on the host
# so the lerobot / groot containers can mount it as
# /root/.cache/zenoh_ros2_sdk and skip the runtime git fetch (which
# trips over the upstream "master" branch rename and the docker-cp
# ownership-mismatch path).
#
# One-shot setup. Idempotent — re-runs git fetch on existing clones
# and exits clean.
#
# Pinned commits mirror cyclo_brain/sdk/zenoh_ros2_sdk/zenoh_ros2_sdk/
# _repositories.py. Bump there and re-run this script if upstream pins
# move.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cache_dir="${repo_root}/docker/zenoh_cache"

declare -A REPOS=(
    [common_interfaces]="https://github.com/ros2/common_interfaces.git"
    [rcl_interfaces]="https://github.com/ros2/rcl_interfaces.git"
)
branch="jazzy"

mkdir -p "${cache_dir}"

# Mark the cache dir as colcon-invisible. zenoh_cache holds upstream
# ROS message packages (common_interfaces, rcl_interfaces) that have
# their own package.xml; without this sentinel, anyone running a plain
# `colcon build` from the workspace root would discover them and try
# to (re)build std_msgs / test_msgs / etc. See Dockerfile.arm64 `cb`
# alias for the build path that's already path-pinned and safe.
touch "${cache_dir}/COLCON_IGNORE"

# Clone via a throw-away docker container so the resulting tree is
# owned by uid 0 — matches the in-container root that lerobot/groot
# run as, sidestepping GitPython's "dubious ownership" check.
git_image="alpine/git:latest"

for name in "${!REPOS[@]}"; do
    url="${REPOS[$name]}"
    target="${cache_dir}/${name}"
    if [ -d "${target}/.git" ]; then
        echo "[zenoh_cache] ${name}: refreshing"
        docker run --rm --network host \
            -v "${cache_dir}:/cache" \
            "${git_image}" \
            -C "/cache/${name}" fetch --depth 1 origin "${branch}"
        docker run --rm --network host \
            -v "${cache_dir}:/cache" \
            "${git_image}" \
            -C "/cache/${name}" reset --hard "origin/${branch}"
    else
        echo "[zenoh_cache] ${name}: cloning ${url} (${branch})"
        docker run --rm --network host \
            -v "${cache_dir}:/cache" \
            "${git_image}" \
            clone --depth 1 --branch "${branch}" "${url}" "/cache/${name}"
    fi
done

echo
echo "[zenoh_cache] done. Next:"
echo "  docker compose up -d --force-recreate lerobot groot"

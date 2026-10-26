#!/usr/bin/env bash
# Build-time ROS Jazzy definitions; no robot driver checkout is required.
set -euo pipefail

cache_dir="${1:?usage: prepare_message_cache.sh CACHE_DIR}"
mkdir -p "${cache_dir}"

fetch_messages() {
    local name="$1" revision="$2"
    git init "${cache_dir}/${name}"
    git -C "${cache_dir}/${name}" remote add origin "https://github.com/ros2/${name}.git"
    git -C "${cache_dir}/${name}" fetch --depth 1 origin "${revision}"
    # The SDK requests the local jazzy ref. Keep it pinned to this revision
    # so message loading never needs a network fetch at runtime.
    git -C "${cache_dir}/${name}" checkout -B jazzy FETCH_HEAD
    test "$(git -C "${cache_dir}/${name}" rev-parse HEAD)" = "${revision}"
}

fetch_messages common_interfaces a941f14bb318d8d904505ed935ccbb97f24a70a4
fetch_messages rcl_interfaces 7aa3caf43377ea6ad615bc1040832e2c7566bfbe
touch "${cache_dir}/COLCON_IGNORE"

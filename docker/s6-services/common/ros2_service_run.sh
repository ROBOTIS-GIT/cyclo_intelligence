#!/bin/bash
# Reusable ROS2 service run script template.
# Launches a ROS2 command for any s6 longrun service defined under
# /etc/s6-overlay/s6-rc.d/<name>/run.
#
# Usage (from the service's run script):
#     exec /command/with-contenv env \
#         SERVICE_NAME=<name> \
#         ROS2_COMMAND="ros2 launch <pkg> <file>.launch.py" \
#         bash /usr/local/lib/s6-services/ros2_service_run.sh
#
# If ROS2_COMMAND is not set, defaults to:
#     ros2 launch orchestrator ${SERVICE_NAME}.launch.py

set -e

SERVICE_NAME="${SERVICE_NAME}"
if [ -z "${SERVICE_NAME}" ]; then
    echo "Error: SERVICE_NAME environment variable must be set" >&2
    exit 1
fi

# Default ROS env (override via container environment).
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-30}
export ROS_DISTRO=${ROS_DISTRO:-jazzy}
export COLCON_WS=${COLCON_WS:-/root/ros2_ws}
export PATH=/opt/venv/bin:${PATH}
export PYTHONPATH=/opt/venv/lib/python3.12/site-packages:${PYTHONPATH}

# s6-overlay debug logging
export S6_VERBOSITY=${S6_VERBOSITY:-1}

echo "[${SERVICE_NAME}] Starting service..."
echo "[${SERVICE_NAME}] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[${SERVICE_NAME}] ROS_DISTRO=${ROS_DISTRO}"
echo "[${SERVICE_NAME}] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[${SERVICE_NAME}] COLCON_WS=${COLCON_WS}"
echo "[${SERVICE_NAME}] PID: $$"

# Record process group id so the finish script can target the whole group.
PGID=$(ps -o pgid= -p $$ | tr -d ' ')
echo "[${SERVICE_NAME}] Process group: ${PGID}"
echo "${PGID}" > /run/${SERVICE_NAME}.pgid || true

# Source ROS2 environment
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${COLCON_WS}/install/setup.bash

# Determine the command to execute
if [ -n "${ROS2_COMMAND}" ]; then
    ROS2_CMD="${ROS2_COMMAND}"
    echo "[${SERVICE_NAME}] Executing custom command: ${ROS2_CMD}"
else
    ROS2_CMD="ros2 launch orchestrator ${SERVICE_NAME}.launch.py"
    echo "[${SERVICE_NAME}] Executing default command: ${ROS2_CMD}"
fi

# Execute the ROS2 command. 'exec' ensures the command becomes PID 1 of
# this service so s6 can signal it and its children. stdout/stderr are
# piped to the matching <name>-log consumer via producer-for/consumer-for.
exec bash -c "${ROS2_CMD}"

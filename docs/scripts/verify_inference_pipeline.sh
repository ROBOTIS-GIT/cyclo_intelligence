#!/bin/bash
set +u
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
set -u

# ensure clean — UNLOAD first
ros2 service call /lerobot/inference_command interfaces/srv/InferenceCommand "{command: 5}" 2>&1 | tail -3
sleep 2

echo "=== LOAD ==="
ros2 service call /lerobot/inference_command interfaces/srv/InferenceCommand \
    "{command: 0, model_path: '/workspace/training_outputs/act_task0013_5000/checkpoints/last', robot_type: 'ffw_sg2_rev1'}" \
    2>&1 | tail -3

sleep 2
echo "=== START ==="
ros2 service call /lerobot/inference_command interfaces/srv/InferenceCommand "{command: 1}" 2>&1 | tail -3

sleep 4
echo "=== service states ==="
ros2 service list 2>&1 | grep -E "/lerobot/(inference_command|engine_command)"
echo ""
echo "=== /cmd_vel pub rate (5s, if mobile action exists) ==="
timeout 5 ros2 topic hz /cmd_vel 2>&1 | grep -E "average rate|does not"

# Cleanup
ros2 service call /lerobot/inference_command interfaces/srv/InferenceCommand "{command: 4}" 2>&1 | tail -1
ros2 service call /lerobot/inference_command interfaces/srv/InferenceCommand "{command: 5}" 2>&1 | tail -1

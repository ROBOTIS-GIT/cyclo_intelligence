"""Strict SH5 deployment contract for the ViTacFormer backend."""

IMAGE_KEY = "observation.images.rgb.cam_left_head"
STATE_KEY = "observation.state"
TACTILE_BATCH_KEY = "observation.tactile.vitacformer"

CAMERA_NAME = "cam_left_head"
ACTION_KEYS = ("arm_left", "arm_right", "hand_left", "hand_right")

STATE_HISTORY_SIZE = 6
STATE_HISTORY_HZ = 10.0
TACTILE_HISTORY_SIZE = 18
TACTILE_HISTORY_HZ = 30.0
TACTILE_BASELINE_SAMPLES = 20

JOINT_NAMES = (
    *(f"arm_l_joint{i}" for i in range(1, 8)),
    *(f"arm_r_joint{i}" for i in range(1, 8)),
    *(f"finger_l_joint{i}" for i in range(1, 21)),
    *(f"finger_r_joint{i}" for i in range(1, 21)),
)

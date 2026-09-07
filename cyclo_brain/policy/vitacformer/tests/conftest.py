"""Exercise the real upstream RobotClient with transport-only test doubles."""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[4]
for path in (
    ROOT / "cyclo_brain/policy/vitacformer",
    ROOT / "cyclo_brain/policy/common/runtime",
    ROOT / "cyclo_brain/sdk/robot_client",
    ROOT / "shared/shared/robot_configs",
):
    sys.path.insert(0, str(path))

transport = types.ModuleType("zenoh_ros2_sdk")


class Subscriber:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


transport.ROS2Subscriber = Subscriber
transport.ROS2Publisher = Subscriber
transport.ROS2ServiceServer = Subscriber
transport.get_message_class = lambda _name: object
sys.modules["zenoh_ros2_sdk"] = transport

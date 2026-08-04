# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        module = sys.modules[name]
        for key, value in attrs.items():
            setattr(module, key, value)
        return

    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        sys.modules.setdefault(parent, ModuleType(parent))

    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


_stub_module("geometry_msgs.msg", Twist=_Dummy)
_stub_module("nav_msgs.msg", Odometry=_Dummy)
_stub_module("sensor_msgs.msg", JointState=_Dummy)
_stub_module("trajectory_msgs.msg", JointTrajectory=_Dummy)
_stub_module("rclpy.serialization", deserialize_message=lambda *args, **kwargs: None)
_stub_module(
    "rosbag2_py",
    SequentialReader=_Dummy,
    StorageOptions=_Dummy,
    ConverterOptions=_Dummy,
    StorageFilter=_Dummy,
)

from cyclo_data.recorder.replay_handler import ReplayDataHandler  # noqa: E402


def test_missing_bag_replay_response_keeps_3d_viewer_fields(tmp_path):
    result = ReplayDataHandler().get_replay_data(str(tmp_path / "missing"))

    assert result["success"] is False
    assert result["urdf_path"] == ""
    assert result["end_effector_links"] == []


def test_robot_semantic_layout_resolves_replay_urdf_path(tmp_path, monkeypatch):
    config_dir = tmp_path / "robot_configs"
    urdf_dir = config_dir / "urdf"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "test_bot.urdf").write_text("<robot name='test_bot' />\n")
    (config_dir / "test_bot_config.yaml").write_text(
        """
orchestrator:
  ros__parameters:
    test_bot:
      urdf_path: urdf/test_bot.urdf
      visualization:
        end_effector_links:
          - tool0
      observation:
        state:
          arm:
            topic: /state
            joint_names: [joint_1, joint_2]
      action:
        arm:
          topic: /action
          joint_names: [joint_1, joint_2]
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ORCHESTRATOR_CONFIG_PATH", str(tmp_path / "unused"))
    monkeypatch.setenv("ROBOT_CLIENT_CONFIG_DIR", str(config_dir))

    layout = ReplayDataHandler()._load_robot_semantic_layout("test_bot")

    assert layout["urdf_path"] == str((urdf_dir / "test_bot.urdf").resolve())
    assert layout["end_effector_links"] == ["tool0"]
    assert layout["state"]["arm"]["topic"] == "/state"
    assert layout["action"]["arm"]["topic"] == "/action"

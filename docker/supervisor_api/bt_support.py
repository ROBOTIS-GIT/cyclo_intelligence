# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Seongwoo Kim

"""Robot capability lookup for the behavior-tree engine.

``shared/robot_configs/schema.py`` owns the list of robot types the BT
engine supports. The supervisor does not run inside the ROS workspace, so
the schema module is loaded by path from the robot-config bundle (the same
``/orchestrator_config`` mount the policy containers use), with the source
checkout as a fallback for editable deployments.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import List, Optional

ROBOT_CONFIGS_DIR_ENV = "CYCLO_ROBOT_CONFIGS_DIR"
_SCHEMA_FILENAME = "schema.py"

_schema_module = None


def robot_configs_dir_candidates() -> List[Path]:
    """Return the directories searched for ``schema.py``, in order."""
    candidates: List[Path] = []
    configured = os.environ.get(ROBOT_CONFIGS_DIR_ENV, "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("/orchestrator_config"))
    colcon_ws = os.environ.get("COLCON_WS", "/root/ros2_ws")
    candidates.append(
        Path(colcon_ws) / "src" / "cyclo_intelligence" / "shared" / "shared" / "robot_configs"
    )
    return candidates


def _load_schema():
    global _schema_module
    if _schema_module is not None:
        return _schema_module
    for directory in robot_configs_dir_candidates():
        path = directory / _SCHEMA_FILENAME
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("cyclo_robot_config_schema", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _schema_module = module
        return module
    raise RuntimeError(
        "robot config schema not found; set "
        f"{ROBOT_CONFIGS_DIR_ENV} to the shared/robot_configs directory"
    )


def bt_supported_robot_types() -> List[str]:
    """Return the robot types the behavior-tree engine supports."""
    schema = _load_schema()
    return [str(name) for name in schema.bt_supported_robot_types()]


def default_bt_robot_type() -> str:
    return bt_supported_robot_types()[0]


def is_bt_supported(robot_type: Optional[str]) -> bool:
    return str(robot_type or "").strip() in bt_supported_robot_types()

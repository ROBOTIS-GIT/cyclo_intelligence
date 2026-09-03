#!/usr/bin/env python3
#
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

"""Cyclo Intelligence Behavior Tree actions package."""

from orchestrator.bt.actions.base_action import BaseAction

__all__ = [
    'ArmStateGate',
    'BaseAction',
    'JointControl',
    'Rotate',
    'SendCommand',
    'SendCommandAction',
    'Wait',
]


def __getattr__(name):
    """Lazily expose built-in actions without importing ROS-heavy modules."""
    if name == 'JointControl':
        from orchestrator.bt.actions.joint_control import JointControl
        return JointControl
    if name == 'Rotate':
        from orchestrator.bt.actions.rotate import Rotate
        return Rotate
    if name in ('SendCommand', 'SendCommandAction'):
        from orchestrator.bt.actions.send_command import SendCommand
        return SendCommand
    if name == 'Wait':
        from orchestrator.bt.actions.wait import Wait
        return Wait
    if name == 'ArmStateGate':
        from orchestrator.bt.actions.arm_state_gate import (
            ArmStateGate,
        )
        return ArmStateGate
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

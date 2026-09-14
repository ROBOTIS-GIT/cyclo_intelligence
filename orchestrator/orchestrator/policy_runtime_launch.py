"""Launch the shared runtime before its ROS clients, without respawning a session."""

import os
from pathlib import Path
import time

from launch.actions import (
    EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
    SetEnvironmentVariable, TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable


def managed_runtime_actions(dependents, *, startup_timeout_s=30.0):
    root = Path(os.environ.get('CYCLO_POLICY_ROOT', '/opt/cyclo/policy'))
    python_paths = [
        root / 'common/runtime', root / 'common',
        root.parent / 'sdk/zenoh_ros2_sdk', root.parent / 'sdk/robot_client',
        root.parent / 'sdk/action_chunk_processing',
    ]
    pythonpath = os.pathsep.join(map(str, python_paths))
    if os.environ.get('PYTHONPATH'):
        pythonpath += os.pathsep + os.environ['PYTHONPATH']
    runtime = ExecuteProcess(
        cmd=['python3', '-m', 'main_runtime'],
        name='policy_runtime', output='screen', respawn=False,
        additional_env={'PYTHONPATH': pythonpath, 'PYTHONUNBUFFERED': '1'},
        sigterm_timeout='30', sigkill_timeout='5',
    )
    marker = Path(os.environ.get(
        'POLICY_RUNTIME_READY_MARKER', '/run/cyclo/policy-runtime.ready'))

    def wait_until_ready(event, _context):
        deadline = time.monotonic() + startup_timeout_s

        def poll(context):
            if context.is_shutdown:
                return []
            try:
                ready = marker.read_text().strip() == str(event.pid)
            except OSError:
                ready = False
            if ready:
                return list(dependents)
            if time.monotonic() >= deadline:
                return [EmitEvent(event=Shutdown(reason='Policy Runtime startup timed out'))]
            return [TimerAction(period=0.1, actions=[OpaqueFunction(function=poll)])]

        return [OpaqueFunction(function=poll)]

    defaults = {
        'ROS_DOMAIN_ID': '30',
        'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp',
        'ZENOH_CONFIG_OVERRIDE': 'transport/shared_memory/enabled=true',
        'ZENOH_SHM_ENABLED': 'true',
        'ZENOH_TRANSPORT_SHM_ENABLED': 'true',
    }
    return [
        *[SetEnvironmentVariable(name, EnvironmentVariable(name, default_value=default))
          for name, default in defaults.items()],
        RegisterEventHandler(OnProcessStart(target_action=runtime, on_start=wait_until_ready)),
        RegisterEventHandler(OnProcessExit(
            target_action=runtime,
            on_exit=[EmitEvent(event=Shutdown(reason='Policy Runtime exited; inference is stopped'))],
        )),
        runtime,
    ]

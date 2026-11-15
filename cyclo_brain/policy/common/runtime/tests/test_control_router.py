"""Session routing must preserve ordinary/RAIL and support TT-RTC defaults."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

runtime_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(runtime_root))
robot_stub = ModuleType("robot_client")
robot_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_stub)

from main_runtime.control_router import ControlRouter, ControlLoop, PlaygroundControlLoop


def test_tt_rtc_environment_default_constructs_playground():
    router = ControlRouter(SimpleNamespace(), action_request_mode="tt_rtc", trace_path=None, rail_interpolation_hz=100)
    assert type(router._loop) is PlaygroundControlLoop
    ordinary = router._create_loop(ControlLoop)
    assert ordinary._action_request_mode == "async"


def test_ordinary_default_preserves_target_controller():
    router = ControlRouter(SimpleNamespace(), action_request_mode="async", rail_interpolation_hz=200)
    assert type(router._loop) is ControlLoop


def test_rail_and_playground_are_rejected_before_mutation():
    router = ControlRouter(SimpleNamespace())
    original = router._loop
    import pytest
    with pytest.raises(ValueError, match="cannot be enabled together"):
        router.configure(action_request_mode="tt_rtc", vla_rail_enabled=True)
    assert router._loop is original

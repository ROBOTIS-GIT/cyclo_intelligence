"""Exercise real ROS launch with harmless child processes, no robot/network I/O."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("launch")
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, TimerAction
from launch.events import Shutdown


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "policy_runtime_launch_test", ROOT / "orchestrator/orchestrator/policy_runtime_launch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RUNTIME = r'''
import importlib.util, os, pathlib, signal, threading, time
spec = importlib.util.spec_from_file_location("lock", os.environ["TEST_LOCK_MODULE"])
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)
with lock.runtime_process_lock():
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    root = pathlib.Path(os.environ["TEST_OUTPUT"])
    (root / "pid").write_text(str(os.getpid()))
    mode = os.environ["TEST_MODE"]
    if mode == "startup_failure":
        raise SystemExit(7)
    time.sleep(0.15)
    marker = pathlib.Path(os.environ["POLICY_RUNTIME_READY_MARKER"])
    if mode != "no_readiness":
        marker.write_text(str(os.getpid()))
    if mode == "crash":
        time.sleep(0.5)
        os._exit(9)
    done.wait(10)
    marker.unlink(missing_ok=True)
    (root / "stopped").touch()
'''

DEPENDENT = r'''
import json, os, pathlib, signal, threading
done = threading.Event()
signal.signal(signal.SIGINT, lambda *_: done.set())
signal.signal(signal.SIGTERM, lambda *_: done.set())
root = pathlib.Path(os.environ["TEST_OUTPUT"])
(root / "dependent.json").write_text(json.dumps({
    "ready_pid": pathlib.Path(os.environ["POLICY_RUNTIME_READY_MARKER"]).read_text(),
    "zenoh": os.environ["ZENOH_CONFIG_OVERRIDE"],
}))
if os.environ["TEST_MODE"] == "orchestrator_exit":
    raise SystemExit(12)
done.wait(10)
(root / "dependent_stopped").touch()
'''


@pytest.mark.parametrize("mode", ["normal", "startup_failure", "crash", "no_readiness", "duplicate", "orchestrator_exit"])
def test_launch_readiness_exit_and_singleton(mode, tmp_path, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setenv("TEST_OUTPUT", str(tmp_path))
    monkeypatch.setenv("TEST_MODE", mode)
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ZENOH_CONFIG_OVERRIDE", 'mode="client";connect/endpoints=["tcp/example:7447"]')
    monkeypatch.setenv("POLICY_RUNTIME_CONTROL_SOCKET", str(tmp_path / "runtime.sock"))
    monkeypatch.setenv("POLICY_RUNTIME_READY_MARKER", str(tmp_path / "ready"))
    lock_path = ROOT / "cyclo_brain/policy/common/runtime/main_runtime/process_lock.py"
    monkeypatch.setenv("TEST_LOCK_MODULE", str(lock_path))
    (tmp_path / "ready").write_text("stale-pid")
    lock_spec = importlib.util.spec_from_file_location("test_launch_lock", lock_path)
    lock_module = importlib.util.module_from_spec(lock_spec)
    lock_spec.loader.exec_module(lock_module)

    def harmless_runtime(**kwargs):
        assert kwargs["cmd"] == ["python3", "-m", "main_runtime"]
        assert kwargs["respawn"] is False
        assert "ZENOH_CONFIG_OVERRIDE" not in kwargs["additional_env"]
        kwargs["cmd"] = [sys.executable, "-c", RUNTIME]
        return ExecuteProcess(**kwargs)

    monkeypatch.setattr(module, "ExecuteProcess", harmless_runtime)
    dependent = ExecuteProcess(cmd=[sys.executable, "-c", DEPENDENT])
    if mode == "orchestrator_exit":
        orch_spec = importlib.util.spec_from_file_location(
            "test_orchestrator_launch", ROOT / "orchestrator/launch/orchestrator.launch.py")
        orch = importlib.util.module_from_spec(orch_spec)
        orch_spec.loader.exec_module(orch)
        monkeypatch.setattr(orch, "get_package_share_directory", lambda _: str(tmp_path))
        monkeypatch.setattr(orch, "Node", lambda **kwargs: ExecuteProcess(
            cmd=[sys.executable, "-c", DEPENDENT], on_exit=kwargs["on_exit"]))
        dependent = orch.generate_launch_description().entities[0]
    actions = module.managed_runtime_actions([dependent], startup_timeout_s=0.4 if mode == "no_readiness" else 3.)
    actions.append(TimerAction(period=1.5, actions=[EmitEvent(event=Shutdown(reason="test complete"))]))
    service = LaunchService(noninteractive=True)
    service.include_launch_description(LaunchDescription(actions))
    # SetEnvironmentVariable launch actions also modify the test process. Keep
    # ROS/RMW defaults from leaking into unrelated serialization tests.
    with patch.dict(os.environ):
        with lock_module.runtime_process_lock() if mode == "duplicate" else nullcontext():
            service.run()

    if mode in ("normal", "crash", "orchestrator_exit"):
        data = json.loads((tmp_path / "dependent.json").read_text())
        assert data["ready_pid"] == (tmp_path / "pid").read_text()
        assert data["zenoh"] == os.environ["ZENOH_CONFIG_OVERRIDE"]
        if mode != "orchestrator_exit":
            assert (tmp_path / "dependent_stopped").exists()
    else:
        assert not (tmp_path / "dependent.json").exists()
    if mode in ("normal", "no_readiness", "orchestrator_exit"):
        assert (tmp_path / "stopped").exists()
    if (tmp_path / "pid").exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)

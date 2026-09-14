import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "runtime_process_lock_test", Path(__file__).resolve().parents[1] / "main_runtime/process_lock.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_singleton_rejects_second_owner_and_allows_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_RUNTIME_CONTROL_SOCKET", str(tmp_path / "runtime.sock"))
    with module.runtime_process_lock():
        with pytest.raises(RuntimeError, match="already running"):
            with module.runtime_process_lock():
                pytest.fail("duplicate runtime acquired the socket lock")
    with module.runtime_process_lock():
        assert (tmp_path / "runtime.sock.lock").exists()

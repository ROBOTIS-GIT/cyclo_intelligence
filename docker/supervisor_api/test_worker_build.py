import os
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervisor_api.worker_build import WorkerBuilds


def wait_finished(builds, name):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = builds.status(name)
        if result["state"] != "building":
            return result
        time.sleep(0.01)
    pytest.fail("build did not finish")


def command(code):
    return [sys.executable, "-u", "-c", code]


def test_build_runs_in_background_and_keeps_bounded_logs():
    builds = WorkerBuilds()
    released = threading.Event()
    verified = threading.Event()
    try:
        result = builds.start("lerobot", command("print('step\\n' * 300); print('x' * 10000)"),
                              os.environ.copy(), verified.set, released.set)
        assert result["state"] == "building"
        result = wait_finished(builds, "lerobot")
        assert result["state"] == "succeeded"
        assert result["returncode"] == 0
        assert verified.is_set() and released.is_set()
        assert len(result["log_tail"]) <= 200
        assert max(map(len, result["log_tail"])) <= 2048
    finally:
        builds.close()


@pytest.mark.parametrize("code", ["raise SystemExit(7)", "raise RuntimeError('bad build')"])
def test_failed_build_never_verifies_old_image_and_allows_retry(code):
    builds = WorkerBuilds()
    released = threading.Event()
    verified = threading.Event()
    try:
        builds.start("groot", command(code), os.environ.copy(), verified.set, released.set)
        assert wait_finished(builds, "groot")["state"] == "failed"
        assert released.is_set() and not verified.is_set()
        builds.start("groot", command("print('retry')"), os.environ.copy(), verified.set, released.set)
        assert wait_finished(builds, "groot")["state"] == "succeeded"
    finally:
        builds.close()


def test_missing_result_image_marks_zero_exit_build_failed():
    builds = WorkerBuilds()
    released = threading.Event()
    def missing():
        raise RuntimeError("expected image is missing")
    try:
        builds.start("abot", command("pass"), os.environ.copy(), missing, released.set)
        result = wait_finished(builds, "abot")
        assert result["state"] == "failed"
        assert "expected image" in result["message"]
        assert released.is_set()
    finally:
        builds.close()


def test_only_one_build_and_shutdown_releases_reservation():
    builds = WorkerBuilds()
    released = threading.Event()
    builds.start("lerobot", command("import time; print('started'); time.sleep(60)"),
                 os.environ.copy(), lambda: None, released.set)
    with pytest.raises(RuntimeError, match="already"):
        builds.start("groot", command("pass"), os.environ.copy(), lambda: None, lambda: None)
    builds.close()
    assert builds.status("lerobot")["state"] == "failed"
    assert released.is_set()


def test_missing_executable_releases_reservation():
    builds = WorkerBuilds()
    released = threading.Event()
    try:
        builds.start("rldx", ["/nonexistent/cyclo-builder"], os.environ.copy(), lambda: None, released.set)
        assert wait_finished(builds, "rldx")["state"] == "failed"
        assert released.is_set()
    finally:
        builds.close()


def test_thread_start_failure_does_not_leave_building_status(monkeypatch):
    builds = WorkerBuilds()
    def fail(_thread):
        raise RuntimeError("thread unavailable")
    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        builds.start("abot", command("pass"), os.environ.copy(), lambda: None, lambda: None)
    assert builds.status("abot")["state"] == "failed"
    builds.close()


def test_shutdown_kills_child_holding_pipe_after_parent_exits():
    builds = WorkerBuilds()
    released = threading.Event()
    code = """
import subprocess, sys, time
subprocess.Popen([sys.executable, '-u', '-c',
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('child ready'); time.sleep(60)"])
time.sleep(60)
"""
    try:
        builds.start("abot", command(code), os.environ.copy(), lambda: None, released.set)
        deadline = time.monotonic() + 5
        while "child ready" not in builds.status("abot")["log_tail"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        builds.close()
        assert released.is_set()
        assert builds.status("abot")["state"] == "failed"
    finally:
        builds.close()

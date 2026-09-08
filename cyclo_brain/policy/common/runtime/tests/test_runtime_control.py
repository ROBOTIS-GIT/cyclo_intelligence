#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from main_runtime.runtime_control import RuntimeControlServer  # noqa: E402
from main_runtime.service_handler import ServiceHandler  # noqa: E402
from main_runtime.session_state import SessionState  # noqa: E402


class RuntimeControlServerTests(unittest.TestCase):
    @staticmethod
    def _request(path, data=b'{}'):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(str(path))
            client.sendall(data)
            client.shutdown(socket.SHUT_WR)
            return json.loads(client.recv(65536))

    def test_idle_malformed_and_fragmented_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.sock"
            server = RuntimeControlServer(
                lambda request: {"ok": True, "echo": request}, str(path),
                io_timeout_s=0.1,
            )
            server.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as idle:
                    idle.connect(str(path))
                    self.assertTrue(self._request(path)["ok"])
                for data in (b'{bad', b'[]', b'x' * 65537):
                    self.assertFalse(self._request(path, data)["ok"])
                    self.assertTrue(self._request(path)["ok"])
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(2)
                    client.connect(str(path))
                    client.sendall(b'{"value":')
                    client.sendall(b'"fragmented"}')
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(json.loads(client.recv(4096))["echo"],
                                     {"value": "fragmented"})
            finally:
                server.close()

    def test_undelivered_reservation_is_released(self):
        service = ServiceHandler(
            SessionState(), None,
            SimpleNamespace(initial_pose_sync_hold_required=lambda: False), None,
        )
        entered, release, cleaned = (threading.Event() for _ in range(3))

        def handler(request):
            if request.get("operation") != "begin_worker_mutation":
                return {"ok": True}
            allowed, reason, token = service.begin_worker_mutation("lerobot")
            entered.set()
            release.wait(2)
            return {"ok": True, "allowed": allowed, "reason": reason, "token": token}

        def cleanup(request, response):
            service.release_undelivered_worker_mutation(request, response)
            cleaned.set()

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "control.sock")
            server = RuntimeControlServer(handler, path, on_response_failure=cleanup)
            server.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(path)
                    client.sendall(b'{"operation":"begin_worker_mutation","runtime_id":"lerobot"}')
                    client.shutdown(socket.SHUT_WR)
                    self.assertTrue(entered.wait(1))
                release.set()
                self.assertTrue(cleaned.wait(1))
                self.assertTrue(self._request(path)["ok"])
                allowed, _, token = service.begin_worker_mutation("lerobot")
                self.assertTrue(allowed)
                self.assertTrue(service.end_worker_mutation("lerobot", token))
            finally:
                release.set()
                server.close()

    def test_disconnected_client_does_not_kill_server(self):
        entered = threading.Event()
        release = threading.Event()

        def handler(request):
            if request.get("slow"):
                entered.set()
                release.wait(2)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "control.sock")
            server = RuntimeControlServer(handler, path)
            server.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(path)
                    client.sendall(b'{"slow":true}')
                    client.shutdown(socket.SHUT_WR)
                    self.assertTrue(entered.wait(1))
                release.set()
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(1)
                    client.connect(path)
                    client.sendall(b'{}')
                    client.shutdown(socket.SHUT_WR)
                    self.assertTrue(json.loads(client.recv(4096))["ok"])
            finally:
                release.set()
                server.close()

    def test_round_trip_and_socket_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.sock"
            server = RuntimeControlServer(
                lambda request: {"ok": True, "echo": request.get("value")},
                str(path),
            )
            server.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(path))
                    client.sendall(b'{"value":"ready"}')
                    client.shutdown(socket.SHUT_WR)
                    response = json.loads(client.recv(4096).decode("utf-8"))
                self.assertEqual(response, {"echo": "ready", "ok": True})
            finally:
                server.close()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

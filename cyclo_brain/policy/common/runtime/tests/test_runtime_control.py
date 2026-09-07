#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from main_runtime.runtime_control import RuntimeControlServer  # noqa: E402


class RuntimeControlServerTests(unittest.TestCase):
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

#!/usr/bin/env python3

"""Small local Unix-socket API used by Supervisor for lifecycle interlocks."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Callable


class RuntimeControlServer:
    def __init__(
        self,
        handler: Callable[[dict], dict],
        path: str = "/run/cyclo/policy-runtime.sock",
    ) -> None:
        self._handler = handler
        self._path = Path(path)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self._path))
        os.chmod(self._path, 0o660)
        server.listen(8)
        server.settimeout(0.5)
        self._socket = server
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name="policy-runtime-control",
        )
        self._thread.start()

    def close(self) -> None:
        self._shutdown.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    raw = connection.recv(65536)
                    request = json.loads(raw.decode("utf-8")) if raw else {}
                    response = self._handler(request)
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                connection.sendall(
                    (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                )

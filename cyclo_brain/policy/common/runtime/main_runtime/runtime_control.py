#!/usr/bin/env python3

"""Small local Unix-socket API used by Supervisor for lifecycle interlocks."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class RuntimeControlServer:
    def __init__(
        self,
        handler: Callable[[dict], dict],
        path: str = "/run/cyclo/policy-runtime.sock",
        *,
        on_response_failure: Callable[[dict, dict], None] | None = None,
        io_timeout_s: float = 1.0,
    ) -> None:
        self._handler = handler
        self._on_response_failure = on_response_failure
        self._io_timeout_s = io_timeout_s
        self._path = Path(path)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        self._shutdown.clear()
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
            args=(server,),
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

    def _serve(self, server: socket.socket) -> None:
        while not self._shutdown.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(self._io_timeout_s)
                request = {}
                try:
                    # Supervisor half-closes the write side to delimit a request.
                    raw = bytearray()
                    deadline = time.monotonic() + self._io_timeout_s
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("control request read timed out")
                        connection.settimeout(remaining)
                        chunk = connection.recv(65537 - len(raw))
                        if not chunk:
                            break
                        raw.extend(chunk)
                        if len(raw) > 65536:
                            raise ValueError("control request exceeds 64 KiB")
                    if not raw:
                        continue
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("control request must be a JSON object")
                    response = self._handler(request)
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                try:
                    connection.settimeout(self._io_timeout_s)
                    connection.sendall(
                        (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                    )
                except (OSError, TypeError, ValueError):
                    logger.warning("Policy Runtime control response failed", exc_info=True)
                    if self._on_response_failure is not None:
                        try:
                            self._on_response_failure(request, response)
                        except Exception:
                            logger.exception("Failed to clean up undelivered control response")

"""Synchronous Trigger client for AI Worker's command-source arbiter."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICE_PREFIX = "/command_source_arbiter"


@dataclass
class ArbiterResult:
    success: bool
    message: str


class CommandArbiterClient:
    COMMANDS = ("select_none", "select_policy")

    def __init__(self, node: Node, callback_group: CallbackGroup | None = None):
        group = callback_group or ReentrantCallbackGroup()
        self._clients = {
            command: node.create_client(
                Trigger,
                f"{SERVICE_PREFIX}/{command}",
                callback_group=group,
            )
            for command in self.COMMANDS
        }

    def call(self, command: str, timeout_s: float = 2.0) -> ArbiterResult:
        client = self._clients.get(command)
        if client is None:
            return ArbiterResult(False, f"unknown arbiter command: {command}")
        timeout = max(0.01, float(timeout_s))
        deadline = time.monotonic() + timeout
        if not client.wait_for_service(timeout_sec=min(0.5, timeout)):
            return ArbiterResult(
                False,
                f"command-source arbiter unavailable: {command}",
            )
        future = client.call_async(Trigger.Request())
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        remaining_s = max(0.0, deadline - time.monotonic())
        if not done.wait(timeout=remaining_s):
            future.cancel()
            return ArbiterResult(False, f"arbiter {command} timed out")
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            return ArbiterResult(False, f"arbiter {command} raised: {exc}")
        if response is None:
            return ArbiterResult(False, f"arbiter {command} returned no response")
        return ArbiterResult(bool(response.success), str(response.message or ""))

    def wait_select_policy(self, timeout_s: float = 5.0) -> ArbiterResult:
        deadline = time.monotonic() + max(0.01, float(timeout_s))
        last = ArbiterResult(False, "policy source not ready")
        while time.monotonic() < deadline:
            remaining_s = max(0.01, deadline - time.monotonic())
            last = self.call(
                "select_policy",
                timeout_s=min(1.0, remaining_s),
            )
            if last.success:
                return last
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return last

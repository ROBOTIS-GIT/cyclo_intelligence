#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Process A — generic policy inference server.

Policy-agnostic Zenoh frontend. Implements the cyclo_intelligence
two-process contract (D16):

- Always-on InferenceCommand srv at ``/<backend>/inference_command``.
- Always-on configure publisher on ``cyclo/policy/<backend>/configure``;
  Process B (``control_publisher.py``) listens and lazily configures
  per-robot publishers when LOAD lands.
- Always-on lifecycle publisher on ``cyclo/policy/<backend>/lifecycle``
  (``loaded`` / ``running`` / ``paused`` / ``stopped`` / ``unloaded``)
  so Process B can gate triggers without waiting for in-flight requests
  to time out.
- Zenoh trigger sub on ``cyclo/policy/<backend>/run_inference`` (only
  while loaded), chunk pub on ``cyclo/policy/<backend>/action_chunk_raw``.

All policy-specific work — model loading, observation construction,
inference call — is delegated to the ``InferenceEngine`` instance
returned by the per-policy module's ``create_engine()`` factory.

Configuration via environment variables:

- ``POLICY_BACKEND`` — backend name used for topic/service prefixes.
  Required (no default — refusing to boot is safer than silently
  publishing on the wrong topic when two backends share a router).
- ``POLICY_ENGINE_MODULE`` — Python module path for the concrete engine
  (e.g. ``lerobot_engine``). Defaults to ``"<POLICY_BACKEND>_engine"``.
- ``POLICY_ENGINE_FACTORY`` — factory callable name in that module
  (default ``"create_engine"``).
- ``ZENOH_ROUTER_IP`` / ``ZENOH_ROUTER_PORT`` / ``ROS_DOMAIN_ID`` —
  Zenoh wiring (defaults match docker-compose).

See ``docs/specs/policy-runtime-contracts.md`` for the full topic /
srv / mount contract reference.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional


# -- zenoh_ros2_sdk import shim ------------------------------------------------
_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(_ZENOH_SDK_PATH):
    sys.path.insert(0, _ZENOH_SDK_PATH)

from zenoh_ros2_sdk import (  # noqa: E402
    ROS2Publisher,
    ROS2ServiceServer,
    ROS2Subscriber,
    get_logger,
)


# -- robot_client msg defs import shim -----------------------------------------
# /robot_client_sdk is the bind-mount root in the container; for source-tree
# dev runs we fall back to <repo_root>/cyclo_brain/sdk/robot_client. _parents[3]
# resolves to cyclo_brain/ for both the legacy
# policy/<backend>/runtime/inference_server.py layout and the new
# policy/common/runtime/inference_server.py layout (same depth).
_parents = Path(__file__).resolve().parents
_default_rc = (
    str(_parents[3] / "sdk" / "robot_client") if len(_parents) > 3 else ""
)
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", _default_rc)
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from robot_client.messages import (  # noqa: E402
    ACTION_CHUNK_DEF,
    INFERENCE_COMMAND_REQUEST_DEF,
    INFERENCE_COMMAND_RESPONSE_DEF,
)


# -- Engine ABC + concrete engine module --------------------------------------
# Common ABC lives next to this file; the concrete engine module is
# bind-mounted into /app and imported by name (POLICY_ENGINE_MODULE).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import InferenceEngine  # noqa: E402


def _resolve_engine() -> InferenceEngine:
    backend = os.environ.get("POLICY_BACKEND", "").strip()
    if not backend:
        raise RuntimeError(
            "POLICY_BACKEND env var is required (e.g. 'lerobot', 'groot')."
        )
    module_name = os.environ.get("POLICY_ENGINE_MODULE", f"{backend}_engine")
    factory_name = os.environ.get("POLICY_ENGINE_FACTORY", "create_engine")
    # /app is added to sys.path by the s6 run script (PYTHONPATH); also
    # add it here defensively so direct invocation works in tests.
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    engine = factory()
    if not isinstance(engine, InferenceEngine):
        raise TypeError(
            f"{module_name}.{factory_name}() returned {type(engine).__name__}, "
            f"expected an InferenceEngine subclass"
        )
    return engine


logger = get_logger("inference_server")


# -- Constants -----------------------------------------------------------------

BACKEND = os.environ.get("POLICY_BACKEND", "").strip() or "policy"
SERVICE_NAME = f"/{BACKEND}/inference_command"
TRIGGER_TOPIC = f"cyclo/policy/{BACKEND}/run_inference"
CHUNK_TOPIC = f"cyclo/policy/{BACKEND}/action_chunk_raw"
CONFIGURE_TOPIC = f"cyclo/policy/{BACKEND}/configure"
LIFECYCLE_TOPIC = f"cyclo/policy/{BACKEND}/lifecycle"

# InferenceCommand enum — must match interfaces/srv/InferenceCommand.srv.
CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6


# -- InferenceServer -----------------------------------------------------------


class InferenceServer:
    """Process A — Zenoh frontend; defers model work to ``InferenceEngine``."""

    def __init__(
        self,
        engine: InferenceEngine,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: Optional[str] = None,
        namespace: str = "/",
    ):
        self._engine = engine
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._node_name = node_name or f"{BACKEND}_inference_server"
        self._namespace = namespace

        # Lifecycle flags
        self._loaded = False
        self._running = False
        self._paused = False

        self._action_keys: List[str] = []
        self._task_instruction: str = ""

        # Zenoh handles (created on LOAD, torn down on UNLOAD)
        self._trigger_sub: Optional[ROS2Subscriber] = None
        self._chunk_pub: Optional[ROS2Publisher] = None

        # Always-on (process lifetime).
        self._command_srv: Optional[ROS2ServiceServer] = None
        self._configure_pub: Optional[ROS2Publisher] = None
        self._lifecycle_pub: Optional[ROS2Publisher] = None

        self._shutdown = threading.Event()

    # -- Common kwargs --------------------------------------------------------

    def _common_kwargs(self) -> dict:
        return {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
            "domain_id": self._domain_id,
            "node_name": self._node_name,
            "namespace": self._namespace,
        }

    # -- Main lifecycle -------------------------------------------------------

    def start_service(self) -> None:
        """Bring up command srv + configure/lifecycle pubs. Block until shutdown."""
        common = self._common_kwargs()
        self._command_srv = ROS2ServiceServer(
            service_name=SERVICE_NAME,
            srv_type="interfaces/srv/InferenceCommand",
            callback=self._handle_command,
            request_definition=INFERENCE_COMMAND_REQUEST_DEF,
            response_definition=INFERENCE_COMMAND_RESPONSE_DEF,
            **common,
        )
        self._configure_pub = ROS2Publisher(
            topic=CONFIGURE_TOPIC,
            msg_type="std_msgs/msg/String",
            **common,
        )
        self._lifecycle_pub = ROS2Publisher(
            topic=LIFECYCLE_TOPIC,
            msg_type="std_msgs/msg/String",
            **common,
        )
        logger.info(f"InferenceCommand service up at {SERVICE_NAME}")
        logger.info(f"configure pub: {CONFIGURE_TOPIC}")
        logger.info(f"lifecycle pub: {LIFECYCLE_TOPIC}")
        logger.info("ZENOH_SUB_READY")  # s6 readiness marker

        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)

    def shutdown(self) -> None:
        self._shutdown.set()
        self._teardown_runtime()
        for attr in ("_command_srv", "_configure_pub", "_lifecycle_pub"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _publish_configure(self, robot_type: str) -> None:
        """Tell Process B which robot to publish for. '' = deconfigure."""
        if self._configure_pub is None:
            return
        try:
            self._configure_pub.publish(data=robot_type)
            logger.info(f"configure broadcast: robot_type='{robot_type}'")
        except Exception as e:
            logger.error(f"configure publish failed: {e}", exc_info=True)

    def _publish_lifecycle(self, state: str) -> None:
        logger.info(f"lifecycle: {state}")
        if self._lifecycle_pub is None:
            return
        try:
            self._lifecycle_pub.publish(data=state)
        except Exception as e:
            logger.error(f"lifecycle publish failed: {e}", exc_info=True)

    # -- Command handler ------------------------------------------------------

    def _handle_command(self, request):
        cmd = int(request.command)
        try:
            if cmd == CMD_LOAD:
                return self._cmd_load(request)
            if cmd == CMD_START:
                return self._cmd_start()
            if cmd == CMD_PAUSE:
                return self._cmd_pause()
            if cmd == CMD_RESUME:
                return self._cmd_resume(request)
            if cmd == CMD_STOP:
                return self._cmd_stop()
            if cmd == CMD_UNLOAD:
                return self._cmd_unload()
            if cmd == CMD_UPDATE_INSTRUCTION:
                return self._cmd_update_instruction(request)
            return self._make_response(
                success=False, message=f"Unknown command: {cmd}"
            )
        except Exception as e:
            logger.error(f"Command {cmd} failed: {e}", exc_info=True)
            return self._make_response(success=False, message=str(e))

    def _cmd_load(self, request):
        if self._loaded:
            return self._make_response(
                success=False, message="policy already loaded — UNLOAD first"
            )
        if not request.model_path:
            return self._make_response(success=False, message="model_path is required")
        if not request.robot_type:
            return self._make_response(success=False, message="robot_type is required")

        # The engine handles the heavy lifting: weights, processors,
        # RobotClient sensor wiring, optional accelerators (e.g. TRT).
        result = self._engine.load_policy(request)

        if not result.get("success"):
            return self._make_response(
                success=False,
                message=result.get("message", f"{BACKEND} load_policy failed"),
            )

        self._setup_zenoh_io()

        self._action_keys = list(result.get("action_keys", []))
        self._task_instruction = request.task_instruction or ""
        self._loaded = True
        self._publish_configure(request.robot_type)
        self._publish_lifecycle("loaded")

        logger.info(f"LOAD ok — action_keys={self._action_keys}")
        return self._make_response(
            success=True,
            message=result.get("message", f"loaded {request.model_path}"),
            action_keys=self._action_keys,
        )

    def _cmd_start(self):
        if not self._loaded:
            return self._make_response(success=False, message="LOAD first")
        self._paused = False
        self._running = True
        logger.info("START")
        self._publish_lifecycle("running")
        return self._make_response(success=True, message="running")

    def _cmd_pause(self):
        if not self._running:
            return self._make_response(success=False, message="not running")
        self._paused = True
        logger.info("PAUSE — ignoring triggers; Process B holds last action")
        self._publish_lifecycle("paused")
        return self._make_response(success=True, message="paused")

    def _cmd_resume(self, request):
        if not self._running:
            return self._make_response(success=False, message="not running")
        if request.task_instruction:
            self._task_instruction = request.task_instruction
        self._paused = False
        logger.info("RESUME")
        self._publish_lifecycle("running")
        return self._make_response(success=True, message="resumed")

    def _cmd_stop(self):
        self._running = False
        self._paused = False
        logger.info("STOP")
        self._publish_lifecycle("stopped")
        return self._make_response(success=True, message="stopped")

    def _cmd_unload(self):
        self._teardown_runtime()
        self._publish_configure("")
        self._publish_lifecycle("unloaded")
        logger.info("UNLOAD")
        return self._make_response(success=True, message="unloaded")

    def _cmd_update_instruction(self, request):
        """Update the current task_instruction without reloading weights.

        InferenceCommand variant ``CMD_UPDATE_INSTRUCTION=6``. Engines
        with language conditioning (smolvla, pi0, eo1, xvla, wallx)
        read the new instruction on the next ``get_action_chunk``
        call. Engines without language conditioning (act, diffusion,
        multi_task_dit) ignore it. The response carries no action_keys
        and no policy reload occurs.
        """
        if not self._loaded:
            return self._make_response(success=False, message="LOAD first")
        if not self._running:
            return self._make_response(
                success=False, message="not running — START first"
            )
        new_instruction = (request.task_instruction or "").strip()
        if not new_instruction:
            return self._make_response(
                success=False, message="task_instruction must be non-empty"
            )
        self._task_instruction = new_instruction
        logger.info(f'instruction updated: "{new_instruction}"')
        return self._make_response(
            success=True, message=f'instruction updated: "{new_instruction}"'
        )

    # -- Zenoh trigger / chunk pub --------------------------------------------

    def _setup_zenoh_io(self) -> None:
        common = self._common_kwargs()
        self._trigger_sub = ROS2Subscriber(
            topic=TRIGGER_TOPIC,
            msg_type="std_msgs/msg/UInt64",
            callback=self._on_trigger,
            **common,
        )
        self._chunk_pub = ROS2Publisher(
            topic=CHUNK_TOPIC,
            msg_type="interfaces/msg/ActionChunk",
            msg_definition=ACTION_CHUNK_DEF,
            **common,
        )
        logger.info(f"zenoh trigger sub: {TRIGGER_TOPIC}")
        logger.info(f"zenoh chunk pub:   {CHUNK_TOPIC}")

    def _on_trigger(self, msg) -> None:
        if not self._running or self._paused:
            return
        if not self._engine.is_ready:
            logger.warning("trigger received but engine not ready")
            return

        seq_id = int(msg.data)

        # Engines only need the current task_instruction (model_path /
        # robot_type were already consumed by load_policy).
        req = SimpleNamespace(task_instruction=self._task_instruction)
        result = self._engine.get_action_chunk(req)

        if not result.get("success"):
            logger.warning(
                f"trigger seq={seq_id} — inference failed: "
                f"{result.get('message')}"
            )
            return

        self._publish_chunk(seq_id, result)

    def _publish_chunk(self, seq_id: int, result: dict) -> None:
        try:
            # zenoh_ros2_sdk's publisher.publish() calls .view() on the data
            # array (treats it as a numpy buffer for fast CDR encoding), so
            # the engine MUST return a numpy ndarray — wrapping in list()
            # crashes with AttributeError: 'list' object has no attribute 'view'.
            self._chunk_pub.publish(
                seq_id=seq_id,
                chunk_size=int(result["chunk_size"]),
                action_dim=int(result["action_dim"]),
                data=result["action_chunk"],
            )
            logger.info(
                f"chunk pub seq={seq_id} T={result['chunk_size']} "
                f"D={result['action_dim']}"
            )
        except Exception as e:
            logger.error(f"chunk publish failed: {e}", exc_info=True)

    # -- Teardown -------------------------------------------------------------

    def _teardown_runtime(self) -> None:
        """Close Zenoh trigger/pub + cleanup engine. Keep srv +
        configure_pub + lifecycle_pub alive — those are process-lifetime."""
        self._running = False
        self._paused = False

        for attr in ("_trigger_sub", "_chunk_pub"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

        try:
            self._engine.cleanup()
        except Exception as e:
            logger.warning(f"engine cleanup raised: {e}", exc_info=True)

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._action_keys = []
        self._task_instruction = ""
        self._loaded = False

    # -- Response builder -----------------------------------------------------

    def _make_response(
        self,
        success: bool,
        message: str = "",
        action_keys: Optional[List[str]] = None,
    ):
        ResponseClass = self._command_srv.response_msg_class
        return ResponseClass(
            success=bool(success),
            message=str(message),
            action_keys=list(action_keys) if action_keys else [],
        )


# -- Main ----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    engine = _resolve_engine()
    server = InferenceServer(
        engine=engine,
        router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
        router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
        domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
    )
    try:
        server.start_service()
    except KeyboardInterrupt:
        logger.info("shutdown via SIGINT")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process A — GR00T policy inference server.

Mirrors cyclo_brain/policy/lerobot/runtime/inference_server.py with two
GR00T-specific adaptations:

1. Model lifecycle delegated to GR00TInference (runtime/inference_engine.py)
   which already encapsulates Gr00tPolicy + RobotClient + TensorRT
   acceleration + the GR00T-flavored preprocess/postprocess.
2. Observation collection runs inside RobotClient (Process A doesn't
   own raw ROS2Subscribers). LeRobot's flat ROS2Subscriber array doesn't
   fit GR00T's per-modality dict-of-cameras + sensor-backed odom →
   mobile bridge.

The cyclo_intelligence two-process contract (D16) is identical to
lerobot's:

- Always-on InferenceCommand srv at /groot/inference_command.
- Always-on configure publisher on cyclo/policy/groot/configure;
  Process B (control_publisher.py) listens and lazily configures
  per-robot publishers when LOAD lands.
- Zenoh trigger sub on cyclo/policy/groot/run_inference (only while
  loaded), chunk pub on cyclo/policy/groot/action_chunk_raw.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
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
# Container deployments set ROBOT_CLIENT_SDK_PATH; for source-tree dev we
# fall back to <repo_root>/cyclo_brain/sdk/robot_client. The conditional
# avoids IndexError when the file lives in a shallow path (e.g. baked into
# /app/runtime/ inside the container with the env unset).
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


# -- GR00T inference engine import (sibling file) ------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_engine import GR00TInference  # noqa: E402


logger = get_logger("inference_server")


# -- Constants -----------------------------------------------------------------

BACKEND = "groot"
SERVICE_NAME = f"/{BACKEND}/inference_command"
TRIGGER_TOPIC = f"cyclo/policy/{BACKEND}/run_inference"
CHUNK_TOPIC = f"cyclo/policy/{BACKEND}/action_chunk_raw"
CONFIGURE_TOPIC = f"cyclo/policy/{BACKEND}/configure"
# Lifecycle broadcasts ("loaded" / "running" / "paused" / "stopped" /
# "unloaded"). Process B uses these to know whether Process A is honoring
# triggers — without this signal, after PAUSE→RESUME Process B waits the
# full REQUEST_TIMEOUT_S for a stale in-flight trigger to time out before
# it tries again, producing a multi-second resume latency.
LIFECYCLE_TOPIC = f"cyclo/policy/{BACKEND}/lifecycle"

# InferenceCommand enum — must match interfaces/srv/InferenceCommand.srv.
CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6


# -- InferenceServer -----------------------------------------------------------


class InferenceServer:
    """Process A — GR00T policy lifecycle + Zenoh trigger/chunk plumbing."""

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: str = "groot_inference_server",
        namespace: str = "/",
    ):
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._node_name = node_name
        self._namespace = namespace

        # Lifecycle flags
        self._loaded = False
        self._running = False
        self._paused = False

        # Inference engine wraps Gr00tPolicy + RobotClient + TRT setup.
        self._inference: Optional[GR00TInference] = None
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

    # -- Main lifecycle -------------------------------------------------------

    def start_service(self) -> None:
        """Bring up the InferenceCommand service + configure publisher.

        Both stay up for the whole process lifetime so orchestrator can
        dispatch LOAD even before the first model is picked, and Process
        B always has someone to send configure messages to.
        """
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
            "domain_id": self._domain_id,
            "node_name": self._node_name,
            "namespace": self._namespace,
        }
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
        if self._command_srv is not None:
            try:
                self._command_srv.close()
            except Exception:
                pass
            self._command_srv = None
        if self._configure_pub is not None:
            try:
                self._configure_pub.close()
            except Exception:
                pass
            self._configure_pub = None
        if self._lifecycle_pub is not None:
            try:
                self._lifecycle_pub.close()
            except Exception:
                pass
            self._lifecycle_pub = None

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
        """Broadcast a state transition to Process B.

        ``state`` is one of: 'loaded', 'running', 'paused', 'stopped',
        'unloaded'. Process B uses 'running' to mean 'triggers will be
        honored' — anything else means 'don't bother triggering, hold
        last action'. Sending after every transition keeps the receiver
        idempotent (e.g. a duplicated 'running' just re-confirms).
        """
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

        # GR00TInference handles the heavy lifting: Gr00tPolicy load,
        # RobotClient (which owns its own ROS2Subscribers for cameras /
        # joint groups / odom), and the optional TensorRT engine build
        # for the DiT action head (~3-5 min first time on Orin, cached
        # to disk under <model_path>/dit_model_bf16.trt).
        self._inference = GR00TInference()
        result = self._inference.load_policy(request)

        if not result.get("success"):
            self._inference = None
            return self._make_response(
                success=False,
                message=result.get("message", "GR00T load_policy failed"),
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
        """Re-condition the running policy on a new language instruction
        without touching lifecycle state. Multi-task language-conditioned
        policies (GR00T N1.6) read self._task_instruction at every trigger
        in _on_trigger, so the next inference picks up the new value."""
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
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
            "domain_id": self._domain_id,
            "node_name": self._node_name,
            "namespace": self._namespace,
        }
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
        if self._inference is None:
            logger.warning("trigger received but inference engine missing")
            return

        seq_id = int(msg.data)

        # GR00TInference.get_action_chunk takes a request-shaped object
        # with task_instruction. We don't pass model_path / robot_type
        # because they're already set on the loaded engine.
        req = SimpleNamespace(task_instruction=self._task_instruction)
        result = self._inference.get_action_chunk(req)

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
            # we must pass the raw numpy array. Wrapping it in list() turns
            # it into a Python list and crashes with
            # AttributeError: 'list' object has no attribute 'view'.
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
        """Close Zenoh trigger/pub + release inference engine. Keep srv +
        configure_pub alive — those are process-lifetime."""
        self._running = False
        self._paused = False

        for handle_attr in ("_trigger_sub", "_chunk_pub"):
            handle = getattr(self, handle_attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_attr, None)

        if self._inference is not None:
            try:
                # Releases RobotClient (closes obs subscribers) + clears
                # the cached policy_info / robot_info dicts. The Gr00tPolicy
                # itself drops when we null the reference below.
                self._inference.cleanup()
            except Exception:
                pass
            self._inference = None

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

    server = InferenceServer(
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

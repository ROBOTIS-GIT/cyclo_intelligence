#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Process B — generic 100 Hz control publisher.

Policy-agnostic. Listens for ``cyclo/policy/<backend>/configure`` from
Process A, builds per-robot publishers + ActionChunkProcessor, then runs
the 100 Hz tick loop:

- Subscribe ``cyclo/policy/<backend>/action_chunk_raw`` (Process A).
- Push raw chunks through ``ActionChunkProcessor`` (L2 align →
  interpolate → blend → buffer) imported from
  ``cyclo_brain/sdk/post_processing/``.
- Publish ``cyclo/policy/<backend>/run_inference`` triggers when the
  buffer falls below the refill threshold.
- Pop one action per tick, fan out via ``split_action`` to per-group
  ROS2 publishers (``JointTrajectory`` for joint groups, ``Twist`` for
  mobile).

Robot-type binding is dynamic (D16): the container starts in an
unconfigured state and stays idle until ``LOAD`` lands at A and
broadcasts ``robot_type`` on the configure topic. This lets one
container support many robots — UNLOAD/LOAD with a different
``robot_type`` reconfigures everything in place.

Configuration via environment variables:

- ``POLICY_BACKEND`` — required, must match Process A's value.
- ``REQUEST_TIMEOUT_S`` — chunk-arrival deadline before we drop the
  in-flight ``_requesting`` flag (default 5 s; bump for slow VLAs like
  GR00T post-TRT-load by setting e.g. ``REQUEST_TIMEOUT_S=8.0``).

See ``docs/specs/policy-runtime-contracts.md`` for the full topic /
srv / mount contract reference.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# -- robot config schema helper -----------------------------------------------
# shared/robot_configs/ is bind-mounted into the container at
# /orchestrator_config/, so schema.py lands beside the per-robot yamls.
_SCHEMA_DIR = os.environ.get("ORCHESTRATOR_CONFIG_PATH", "/orchestrator_config")
if os.path.isdir(_SCHEMA_DIR) and _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)
try:
    import schema as robot_schema  # type: ignore[import-not-found]
except ImportError:
    # Source-tree fallback for unit tests on the host. _parents[3]
    # resolves to cyclo_brain/ for both the legacy
    # policy/<backend>/runtime/ layout and the new policy/common/runtime/
    # layout (same depth from cyclo_brain).
    _src_schema_dir = (
        Path(__file__).resolve().parents[3].parent
        / "shared" / "shared" / "robot_configs"
    )
    if _src_schema_dir.is_dir():
        sys.path.insert(0, str(_src_schema_dir))
    import schema as robot_schema  # type: ignore[import-not-found]


# -- zenoh_ros2_sdk import shim ------------------------------------------------
_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(_ZENOH_SDK_PATH):
    sys.path.insert(0, _ZENOH_SDK_PATH)

from zenoh_ros2_sdk import (  # noqa: E402
    ROS2Publisher,
    ROS2Subscriber,
    get_logger,
    get_message_class,
)


# -- post_processing SDK import shim -------------------------------------------
_parents = Path(__file__).resolve().parents
_default_pp = (
    str(_parents[3] / "sdk" / "post_processing") if len(_parents) > 3 else ""
)
_POST_PROCESSING_PATH = os.environ.get("POST_PROCESSING_SDK_PATH", _default_pp)
if os.path.exists(_POST_PROCESSING_PATH) and _POST_PROCESSING_PATH not in sys.path:
    sys.path.insert(0, _POST_PROCESSING_PATH)

from post_processing import (  # noqa: E402
    ActionChunkProcessor,
    build_action_joint_map,
    split_action,
)
from post_processing.ros_msg_helpers import make_joint_trajectory  # noqa: E402
from post_processing.runtime_paths import dev_sdk_path  # noqa: E402


# -- robot_client msg defs import shim -----------------------------------------
_ROBOT_CLIENT_PATH = os.environ.get(
    "ROBOT_CLIENT_SDK_PATH",
    dev_sdk_path(__file__, 3, "sdk", "robot_client"),
)
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from robot_client.messages import ACTION_CHUNK_DEF  # noqa: E402


logger = get_logger("control_publisher")


# -- Mixins --------------------------------------------------------------------
# Placed after the sys.path / SDK imports above so the mixin modules find
# zenoh_ros2_sdk / post_processing / robot_client.messages / schema on
# their own imports. Bare imports — runtime is bind-mounted at
# /policy_runtime so sibling lookup just works. See S5 design note in
# docs/plans/2026-05-10-cyclo_brain-refactor.md.
from _cp_lifecycle import LifecycleMixin  # noqa: E402
from _cp_pipeline import PipelineMixin  # noqa: E402
from _cp_publishers import PublishersMixin  # noqa: E402


# -- Constants -----------------------------------------------------------------

BACKEND = os.environ.get("POLICY_BACKEND", "").strip()
if not BACKEND:
    raise RuntimeError(
        "POLICY_BACKEND env var is required (e.g. 'lerobot', 'groot')."
    )
TRIGGER_TOPIC = f"cyclo/policy/{BACKEND}/run_inference"
CHUNK_TOPIC = f"cyclo/policy/{BACKEND}/action_chunk_raw"
CONFIGURE_TOPIC = f"cyclo/policy/{BACKEND}/configure"
LIFECYCLE_TOPIC = f"cyclo/policy/{BACKEND}/lifecycle"

CONTROL_HZ = 100.0
INFERENCE_HZ = 15.0
CHUNK_ALIGN_WINDOW_S = 0.3

# Refill when buffer falls below this many waypoints (200 ms of slack).
REFILL_MARGIN_S = 0.2
# Give up on a trigger if no chunk arrives within this window. Slow VLAs
# (GR00T post-TRT-load, large Pi0 etc.) need more headroom — override via
# REQUEST_TIMEOUT_S env var.
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "5.0"))
# Best-effort real-time priority — requires CAP_SYS_NICE + rtprio ulimit.
RT_PRIO = 80


# -- Helpers -------------------------------------------------------------------


def _try_rt_priority(prio: int = RT_PRIO) -> None:
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        logger.info(f"acquired SCHED_FIFO prio {prio}")
    except (PermissionError, OSError, AttributeError) as e:
        logger.warning(
            f"could not set SCHED_FIFO prio {prio} ({e}); continuing with "
            f"default scheduler — check container has rtprio ulimit + "
            f"CAP_SYS_NICE"
        )


# -- ControlPublisher ----------------------------------------------------------


class ControlPublisher(LifecycleMixin, PipelineMixin, PublishersMixin):

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: Optional[str] = None,
        namespace: str = "/",
    ):
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._node_name = node_name or f"{BACKEND}_control_publisher"
        self._namespace = namespace

        # Configuration state — populated in configure(robot_type), cleared in
        # deconfigure(). Guarded by _config_lock for safe transitions while the
        # 100 Hz tick and Zenoh callbacks may be racing.
        self._config_lock = threading.Lock()
        self._configured = False
        self._robot_type: Optional[str] = None
        self._command_topics: Dict[str, str] = {}
        self._command_msg_types: Dict[str, str] = {}
        self._joint_order: Dict[str, list] = {}
        self._action_keys: list = []
        self._action_joint_map: Optional[Dict[str, Any]] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._preview_joint_names: list = []

        self._refill_threshold = max(1, int(REFILL_MARGIN_S * CONTROL_HZ))

        # Trigger state — reset on every configure/deconfigure.
        self._requesting = False
        self._request_sent_at: float = 0.0
        self._seq_id = 0

        # Robot-specific Zenoh/ROS2 handles (created in configure(), torn down
        # in deconfigure()).
        self._command_pubs: Dict[str, ROS2Publisher] = {}
        self._trigger_pub: Optional[ROS2Publisher] = None
        self._chunk_sub: Optional[ROS2Subscriber] = None
        self._trajectory_preview_pub: Optional[ROS2Publisher] = None

        # Always-on configure + lifecycle subscribers.
        self._configure_sub: Optional[ROS2Subscriber] = None
        self._lifecycle_sub: Optional[ROS2Subscriber] = None
        # Whether Process A is in a state that honors triggers (running).
        self._a_honoring = False

        # Cache generated message classes — _publish_twist /
        # _publish_joint_trajectory run in the 100 Hz tick, so a per-call
        # get_message_class lookup was previously the hot path's biggest
        # fixed cost. Bind once at init.
        self._Vector3 = get_message_class("geometry_msgs/msg/Vector3")
        self._msg_classes = {
            "JointTrajectoryPoint": get_message_class(
                "trajectory_msgs/msg/JointTrajectoryPoint"
            ),
            "Header": get_message_class("std_msgs/msg/Header"),
            "Time": get_message_class("builtin_interfaces/msg/Time"),
            "Duration": get_message_class("builtin_interfaces/msg/Duration"),
        }

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

    # -- Lifecycle ------------------------------------------------------------

    def setup(self) -> None:
        """Bring up the always-on configure + lifecycle subscribers.

        Robot-specific handles are created lazily inside configure() the
        first time a configure msg lands.
        """
        self._configure_sub = ROS2Subscriber(
            topic=CONFIGURE_TOPIC,
            msg_type="std_msgs/msg/String",
            callback=self._on_configure,
            **self._common_kwargs(),
        )
        self._lifecycle_sub = ROS2Subscriber(
            topic=LIFECYCLE_TOPIC,
            msg_type="std_msgs/msg/String",
            callback=self._on_lifecycle,
            **self._common_kwargs(),
        )
        logger.info(f"configure sub: {CONFIGURE_TOPIC}")
        logger.info(f"lifecycle sub: {LIFECYCLE_TOPIC}")

    def shutdown(self) -> None:
        self._shutdown.set()
        self.deconfigure()
        for attr in ("_configure_sub", "_lifecycle_sub"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # configure / deconfigure / _setup_robot_specific_locked /
    # _teardown_robot_specific_locked / _on_configure / _on_lifecycle
    # live in _cp_lifecycle.LifecycleMixin (S5 split).

    def run(self) -> None:
        _try_rt_priority()

        period = 1.0 / CONTROL_HZ
        next_t = time.monotonic()
        logger.info(f"control loop start @ {CONTROL_HZ} Hz (idle until configured)")

        while not self._shutdown.is_set():
            self._tick()

            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Fell behind — resync to now so one long sleep doesn't
                # cascade into a burst of back-to-back ticks.
                next_t = time.monotonic()

    # -- Per-tick work --------------------------------------------------------

    def _tick(self) -> None:
        with self._config_lock:
            if not self._configured or self._processor is None:
                return

            # Recover a stuck trigger: drop the in-flight flag and try
            # again on the next buffer-low check.
            if self._requesting and (time.time() - self._request_sent_at) > REQUEST_TIMEOUT_S:
                logger.warning(
                    f"trigger seq={self._seq_id} timed out after "
                    f"{REQUEST_TIMEOUT_S:.1f}s, resetting"
                )
                self._requesting = False

            # Process A is paused / stopped / loaded / unloaded — don't
            # publish anything. ActionChunkProcessor.pop_action() falls
            # back to last_action when the buffer drains, so without this
            # gate we'd keep emitting the same JointTrajectory at 100 Hz
            # long after PAUSE.
            if not self._a_honoring:
                return

            action = self._processor.pop_action()
            if action is not None:
                self._publish_action_locked(action)

            if (
                self._processor.buffer_size < self._refill_threshold
                and not self._requesting
            ):
                self._send_trigger_locked()

    # _on_chunk / _send_trigger_locked live in _cp_pipeline.PipelineMixin (S5 split).

    # _publish_action_locked / _publish_twist / _publish_joint_trajectory /
    # _publish_trajectory_preview_locked live in _cp_publishers.PublishersMixin
    # (S5 split).


# -- Main ----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    publisher = ControlPublisher(
        router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
        router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
        domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
    )
    try:
        publisher.setup()
        publisher.run()
    except KeyboardInterrupt:
        logger.info("shutdown via SIGINT")
    finally:
        publisher.shutdown()


if __name__ == "__main__":
    main()

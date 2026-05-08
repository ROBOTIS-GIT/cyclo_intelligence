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


class ControlPublisher:

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

    def configure(self, robot_type: str) -> None:
        """Build per-robot publishers + ActionChunkProcessor for ``robot_type``.

        Idempotent for the same robot_type. Switching robot types tears down
        the previous setup before rebuilding.
        """
        with self._config_lock:
            if self._configured and self._robot_type == robot_type:
                logger.info(f"already configured for {robot_type}, skipping")
                return
            if self._configured:
                logger.info(
                    f"reconfiguring from {self._robot_type} → {robot_type}"
                )
                self._teardown_robot_specific_locked()

            section = robot_schema.load_robot_section(robot_type)
            action_groups = robot_schema.get_action_groups(section)

            self._command_topics = {
                f"leader_{m}": cfg["topic"] for m, cfg in action_groups.items()
            }
            self._command_msg_types = {
                f"leader_{m}": cfg["msg_type"] for m, cfg in action_groups.items()
            }
            self._joint_order = {
                f"joint_order.leader_{m}": list(cfg["joint_names"])
                for m, cfg in action_groups.items()
            }
            self._action_keys = sorted(action_groups.keys())
            self._action_joint_map = build_action_joint_map(
                self._action_keys, self._joint_order
            )
            self._preview_joint_names = []
            for m in self._action_keys:
                self._preview_joint_names.extend(
                    self._joint_order.get(f"joint_order.leader_{m}", [])
                )
            logger.info(f"action_keys={self._action_keys}")
            logger.info(f"action_joint_map={self._action_joint_map}")

            self._processor = ActionChunkProcessor(
                inference_hz=INFERENCE_HZ,
                control_hz=CONTROL_HZ,
                chunk_align_window_s=CHUNK_ALIGN_WINDOW_S,
            )
            self._setup_robot_specific_locked()

            self._robot_type = robot_type
            self._configured = True
            logger.info(f"configured for {robot_type}")

    def deconfigure(self) -> None:
        """Tear down per-robot publishers + processor. Safe to call when
        already deconfigured (no-op)."""
        with self._config_lock:
            if not self._configured:
                return
            self._teardown_robot_specific_locked()
            self._processor = None
            self._action_joint_map = None
            self._joint_order = {}
            self._action_keys = []
            self._command_topics = {}
            self._command_msg_types = {}
            self._preview_joint_names = []
            prev = self._robot_type
            self._robot_type = None
            self._configured = False
            self._a_honoring = False
            logger.info(f"deconfigured (was {prev})")

    def _setup_robot_specific_locked(self) -> None:
        """Create command publishers + chunk subscriber + trigger publisher.

        Caller must hold _config_lock.
        """
        common = self._common_kwargs()

        for name, topic in self._command_topics.items():
            msg_type = self._command_msg_types[name]
            self._command_pubs[name] = ROS2Publisher(
                topic=topic, msg_type=msg_type, **common
            )
            logger.info(f"command pub: {name} → {topic} ({msg_type})")

        self._chunk_sub = ROS2Subscriber(
            topic=CHUNK_TOPIC,
            msg_type="interfaces/msg/ActionChunk",
            msg_definition=ACTION_CHUNK_DEF,
            callback=self._on_chunk,
            **common,
        )
        self._trigger_pub = ROS2Publisher(
            topic=TRIGGER_TOPIC,
            msg_type="std_msgs/msg/UInt64",
            **common,
        )
        self._trajectory_preview_pub = ROS2Publisher(
            topic="/inference/trajectory_preview",
            msg_type="trajectory_msgs/msg/JointTrajectory",
            **common,
        )
        logger.info(f"chunk sub:   {CHUNK_TOPIC}")
        logger.info(f"trigger pub: {TRIGGER_TOPIC}")
        logger.info(
            "trajectory preview pub: /inference/trajectory_preview "
            f"({len(self._preview_joint_names)} joints)"
        )

        self._requesting = False
        self._request_sent_at = 0.0
        self._seq_id = 0

    def _teardown_robot_specific_locked(self) -> None:
        """Close command publishers + chunk sub + trigger pub.

        Caller must hold _config_lock.
        """
        for pub in self._command_pubs.values():
            try:
                pub.close()
            except Exception:
                pass
        self._command_pubs.clear()
        for attr in ("_chunk_sub", "_trigger_pub", "_trajectory_preview_pub"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # -- Configure handler ----------------------------------------------------

    def _on_configure(self, msg) -> None:
        """Process A → B configure broadcast. Empty robot_type = deconfigure."""
        try:
            robot_type = (getattr(msg, "data", "") or "").strip()
            if robot_type:
                logger.info(f"configure msg received: robot_type={robot_type}")
                self.configure(robot_type)
            else:
                logger.info("deconfigure msg received")
                self.deconfigure()
        except Exception as e:
            logger.error(f"configure handler failed: {e}", exc_info=True)

    def _on_lifecycle(self, msg) -> None:
        """Process A → B lifecycle broadcast.

        Tracks whether Process A is honoring triggers. When entering the
        honoring state ("running"), drop any in-flight trigger we may
        have sent during a non-honoring state — otherwise the next tick
        wouldn't refire until the stale trigger times out
        (REQUEST_TIMEOUT_S), producing a multi-second resume latency.
        """
        try:
            state = (getattr(msg, "data", "") or "").strip()
            new_honoring = (state == "running")
            with self._config_lock:
                was_honoring = self._a_honoring
                self._a_honoring = new_honoring
                if new_honoring and not was_honoring and self._requesting:
                    logger.info(
                        f"lifecycle: {state} — clearing stale in-flight "
                        f"trigger seq={self._seq_id}"
                    )
                    self._requesting = False
                else:
                    logger.info(f"lifecycle: {state}")
        except Exception as e:
            logger.error(f"lifecycle handler failed: {e}", exc_info=True)

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

    def _on_chunk(self, msg) -> None:
        with self._config_lock:
            if not self._configured or self._processor is None:
                return
            try:
                seq_id = int(getattr(msg, "seq_id", 0))
                chunk_size = int(msg.chunk_size)
                action_dim = int(msg.action_dim)
                data = np.asarray(msg.data, dtype=np.float64)
                if data.size != chunk_size * action_dim:
                    logger.warning(
                        f"chunk seq={seq_id} size mismatch: data.size={data.size} "
                        f"!= {chunk_size} * {action_dim}"
                    )
                    return
                chunk = data.reshape(chunk_size, action_dim)
                n_pushed = self._processor.push_chunk(chunk)
                logger.info(
                    f"chunk rx seq={seq_id} T={chunk_size} D={action_dim} → "
                    f"pushed={n_pushed} buffer={self._processor.buffer_size}"
                )
                self._publish_trajectory_preview_locked(chunk)
            except Exception as e:
                logger.error(f"chunk decode failed: {e}", exc_info=True)
            finally:
                self._requesting = False

    def _send_trigger_locked(self) -> None:
        if self._trigger_pub is None:
            return
        self._seq_id += 1
        try:
            self._trigger_pub.publish(data=self._seq_id)
            self._requesting = True
            self._request_sent_at = time.time()
            logger.debug(f"trigger pub seq={self._seq_id}")
        except Exception as e:
            logger.error(f"trigger publish failed: {e}", exc_info=True)

    def _publish_trajectory_preview_locked(self, chunk: np.ndarray) -> None:
        """Emit the full predicted action chunk as a JointTrajectory for the
        UI's 3D viz. ``chunk`` is shape (T, D) where D matches
        ``len(self._preview_joint_names)``. Caller holds _config_lock.
        """
        if self._trajectory_preview_pub is None:
            return
        joint_names = self._preview_joint_names
        if not joint_names:
            return
        n_names = len(joint_names)
        JointTrajectoryPoint = self._msg_classes["JointTrajectoryPoint"]
        Header = self._msg_classes["Header"]
        Time = self._msg_classes["Time"]
        Duration = self._msg_classes["Duration"]
        try:
            empty = np.zeros(0, dtype=np.float64)
            zero_duration = Duration(sec=0, nanosec=0)
            points = []
            for row in chunk:
                # Defensive trim — D should already equal n_names but the
                # model is the source of truth and we'd rather emit
                # something than crash on a one-off mismatch.
                positions = np.asarray(row[:n_names], dtype=np.float64)
                points.append(
                    JointTrajectoryPoint(
                        positions=positions,
                        velocities=empty,
                        accelerations=empty,
                        effort=empty,
                        time_from_start=zero_duration,
                    )
                )
            self._trajectory_preview_pub.publish(
                header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
                joint_names=list(joint_names),
                points=points,
            )
        except Exception as e:
            logger.error(f"trajectory preview publish failed: {e}", exc_info=True)

    def _publish_action_locked(self, action: np.ndarray) -> None:
        try:
            segments = split_action(
                action, self._action_joint_map, self._joint_order
            )
        except Exception as e:
            logger.error(f"split_action failed: {e}", exc_info=True)
            return

        for publisher_key, values in segments.items():
            pub = self._command_pubs.get(publisher_key)
            if pub is None:
                continue

            try:
                msg_type = self._command_msg_types.get(publisher_key, "")
                if msg_type == "geometry_msgs/msg/Twist":
                    self._publish_twist(pub, values)
                else:
                    joint_names = self._joint_order.get(
                        f"joint_order.{publisher_key}", []
                    )
                    self._publish_joint_trajectory(pub, joint_names, values)
            except Exception as e:
                logger.error(
                    f"publish {publisher_key} failed: {e}", exc_info=True
                )

    def _publish_twist(self, pub: ROS2Publisher, values: np.ndarray) -> None:
        Vector3 = self._Vector3
        linear = Vector3(
            x=float(values[0]) if len(values) > 0 else 0.0,
            y=float(values[1]) if len(values) > 1 else 0.0,
            z=0.0,
        )
        angular = Vector3(
            x=0.0,
            y=0.0,
            z=float(values[2]) if len(values) > 2 else 0.0,
        )
        pub.publish(linear=linear, angular=angular)

    def _publish_joint_trajectory(
        self,
        pub: ROS2Publisher,
        joint_names,
        values: np.ndarray,
    ) -> None:
        header, points = make_joint_trajectory(
            self._msg_classes, joint_names, values
        )
        pub.publish(header=header, joint_names=list(joint_names), points=points)


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

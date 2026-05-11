#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LifecycleMixin — configure / deconfigure / configure-handler / lifecycle-handler.

Extracted from ``control_publisher.py`` so the core class file shows only
the ``__init__`` / ``setup`` / ``shutdown`` / ``run`` / ``_tick``
skeleton + the ``main()`` entrypoint. See the S5 design note in
``docs/plans/2026-05-10-cyclo_brain-refactor.md``.

All methods here are called by code in ``control_publisher.py`` and rely
on attributes set in ``ControlPublisher.__init__`` (``self._config_lock``,
``self._configured``, ``self._command_pubs`` etc.). The ``*_locked``
helpers must be called with ``self._config_lock`` held — the bodies do
not acquire it themselves.
"""

from __future__ import annotations

import os

from zenoh_ros2_sdk import (
    ROS2Publisher,
    ROS2Subscriber,
    get_logger,
)

from post_processing import (
    ActionChunkProcessor,
    build_action_joint_map,
)

from robot_client.messages import ACTION_CHUNK_DEF

import schema as robot_schema  # type: ignore[import-not-found]


logger = get_logger("control_publisher")


# Mirror of constants in control_publisher.py — see S5 design note in plan.
# Topic names are derived from POLICY_BACKEND (same env var that
# control_publisher.py reads at import time) so the mixin module-load
# does not depend on control_publisher having finished importing.
_BACKEND = os.environ.get("POLICY_BACKEND", "").strip()
CHUNK_TOPIC = f"cyclo/policy/{_BACKEND}/action_chunk_raw"
TRIGGER_TOPIC = f"cyclo/policy/{_BACKEND}/run_inference"

INFERENCE_HZ = 15.0
CONTROL_HZ = 100.0
CHUNK_ALIGN_WINDOW_S = 0.3


class LifecycleMixin:
    """Configure / deconfigure / lifecycle handling for ControlPublisher."""

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

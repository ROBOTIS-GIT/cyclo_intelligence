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

"""Process A — LeRobot policy inference server.

Responsibilities (REVIEW §10.1–§10.5):

- Host InferenceCommand.srv at ``/lerobot/inference_command``.
- Subscribe to camera + follower joint-state topics directly via ROS2Subscriber
  (zenoh_ros2_sdk). Obs never round-trips through orchestrator (§5.8).
- Subscribe Zenoh trigger ``cyclo/policy/lerobot/run_inference`` from Process B.
- Publish ``interfaces/msg/ActionChunk`` on Zenoh topic
  ``cyclo/policy/lerobot/action_chunk_raw``.

Not this file's concern:

- 100 Hz control loop, L2 alignment, interpolation — lives in Process B
  (runtime/control_publisher.py) with ActionChunkProcessor (Step 4-A).
- InferenceCommand dispatch from the UI — orchestrator translates
  ``/send_command`` into this service (Step 4-G).

Structure mirrors GR00T ``policy/server_client.py``: single-threaded main,
subscriber/service callbacks run on library-owned threads, the policy call
is synchronous in the callback. No multiprocessing / threading inside this
process.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# -- zenoh_ros2_sdk import shim ------------------------------------------------
# The SDK is mounted into the container at $ZENOH_SDK_PATH; when running unit
# tests on the host it's a sibling folder in the repo.
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
from robot_client.robot_client import derive_robot_config  # noqa: E402


logger = get_logger("inference_server")

# -- Constants -----------------------------------------------------------------

BACKEND = "lerobot"
SERVICE_NAME = f"/{BACKEND}/inference_command"
TRIGGER_TOPIC = f"cyclo/policy/{BACKEND}/run_inference"
CHUNK_TOPIC = f"cyclo/policy/{BACKEND}/action_chunk_raw"
CONFIGURE_TOPIC = f"cyclo/policy/{BACKEND}/configure"
# Lifecycle broadcasts ("loaded" / "running" / "paused" / "stopped" /
# "unloaded"). Process B uses "running" to know whether triggers will
# be honored — without it, PAUSE→RESUME makes Process B wait the full
# REQUEST_TIMEOUT_S for a stale in-flight trigger to time out.
LIFECYCLE_TOPIC = f"cyclo/policy/{BACKEND}/lifecycle"

# InferenceCommand enum — must match interfaces/srv/InferenceCommand.srv.
CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5

# Observation freshness — reject triggers when the latest obs is older than this.
OBS_STALE_S = 2.0


# -- Helpers -------------------------------------------------------------------


def _find_robot_config(robot_type: str) -> Path:
    """Locate the unified <robot_type>_config.yaml.

    Phase 3: SDK yaml retired — Process A reads the same orchestrator
    yaml as control_publisher.py and RobotClient. Bind-mounted at
    /orchestrator_config/ inside policy containers.
    """
    candidates = [
        Path("/orchestrator_config") / f"{robot_type}_config.yaml",
    ]
    # Source-tree fallback for dev runs outside the container.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(
            parent / "shared" / "robot_configs" / f"{robot_type}_config.yaml"
        )

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"robot_config for '{robot_type}' not found in: {[str(c) for c in candidates]}"
    )


def _extract_robot_section(raw: dict, robot_type: str) -> dict:
    """Drill into orchestrator.ros__parameters.<robot> in the unified yaml."""
    try:
        return raw["orchestrator"]["ros__parameters"][robot_type]
    except KeyError as e:
        raise KeyError(
            f"orchestrator.ros__parameters.{robot_type} missing in yaml: {e}"
        ) from e


def _preprocess_image(image: np.ndarray) -> "Any":
    """Resize, BGR→RGB, HWC→CHW, [0,255]→[0,1], add batch dim."""
    import cv2
    import torch

    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def _load_policy(model_path: str):
    """Load a LeRobot checkpoint. Returns a policy or raises.

    Accepts either a flat checkpoint directory (config.json + safetensors at
    the root) or a LeRobot training-output root that nests the actual model
    under ``pretrained_model/`` alongside ``training_state/``. Users
    typically paste the latter path in the UI; auto-descend so we don't
    push that detail onto every operator.
    """
    import json
    import torch
    from lerobot.policies.factory import get_policy_class

    root = Path(model_path)
    nested = root / "pretrained_model"
    if not (root / "config.json").exists() and (nested / "config.json").exists():
        logger.info(f"Descending into pretrained_model: {nested}")
        root = nested
        model_path = str(nested)

    config_path = root / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            policy_type = json.load(f).get("type", "act")
    else:
        policy_type = "act"

    logger.info(f"Loading {policy_type} policy from {model_path}")
    PolicyClass = get_policy_class(policy_type)
    policy = PolicyClass.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device).eval()
    logger.info(f"Model loaded on {device}")
    return policy


# -- InferenceServer -----------------------------------------------------------


class InferenceServer:
    """Process A — policy lifecycle + obs subscription + chunk publishing."""

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: str = "lerobot_inference_server",
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

        # Policy state
        self._policy: Optional[Any] = None
        self._action_keys: List[str] = []
        self._task_instruction: str = ""

        # Obs accumulator
        self._obs_lock = threading.Lock()
        self._latest_obs: Dict[str, Any] = {
            "images": {},
            "joint_states": {},
            "timestamp": None,
        }

        # Zenoh handles (created on LOAD, torn down on UNLOAD)
        self._obs_subscribers: List[ROS2Subscriber] = []
        self._trigger_sub: Optional[ROS2Subscriber] = None
        self._chunk_pub: Optional[ROS2Publisher] = None

        # Command srv is live for the whole process lifetime so orchestrator
        # can hit LOAD even before the first model is picked.
        self._command_srv: Optional[ROS2ServiceServer] = None

        # Configure publisher — broadcasts robot_type to Process B on LOAD/UNLOAD
        # so B can build (or tear down) per-group publishers. Lifetime = process
        # so a B that comes up after A still receives the latest state on
        # subscription (Zenoh latched-1 semantics).
        self._configure_pub: Optional[ROS2Publisher] = None
        self._lifecycle_pub: Optional[ROS2Publisher] = None

        self._shutdown = threading.Event()

    # -- Main lifecycle -------------------------------------------------------

    def start_service(self) -> None:
        """Bring up the InferenceCommand service. Blocks until shutdown."""
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
        # Always-on configure broadcaster — see _publish_configure().
        self._configure_pub = ROS2Publisher(
            topic=CONFIGURE_TOPIC,
            msg_type="std_msgs/msg/String",
            **common,
        )
        # Always-on lifecycle broadcaster — see _publish_lifecycle().
        self._lifecycle_pub = ROS2Publisher(
            topic=LIFECYCLE_TOPIC,
            msg_type="std_msgs/msg/String",
            **common,
        )
        logger.info(f"InferenceCommand service up at {SERVICE_NAME}")
        logger.info(f"configure pub: {CONFIGURE_TOPIC}")
        logger.info(f"lifecycle pub: {LIFECYCLE_TOPIC}")
        logger.info("ZENOH_SUB_READY")  # s6 readiness marker for Process B

        # Main thread idles; all real work happens in sdk callback threads.
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
        """Tell Process B which robot to publish for.

        Empty ``robot_type`` means deconfigure — B closes its per-group
        publishers and goes idle until the next LOAD.
        """
        if self._configure_pub is None:
            return
        try:
            self._configure_pub.publish(data=robot_type)
            logger.info(f"configure broadcast: robot_type='{robot_type}'")
        except Exception as e:
            logger.error(f"configure publish failed: {e}", exc_info=True)

    def _publish_lifecycle(self, state: str) -> None:
        """Broadcast a lifecycle state transition to Process B.

        ``state`` is one of: 'loaded', 'running', 'paused', 'stopped',
        'unloaded'. Process B uses 'running' as the cue that triggers
        will be honored — anything else means hold last action.
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

        model_path = request.model_path
        robot_type = request.robot_type
        self._task_instruction = request.task_instruction or ""

        if not model_path:
            return self._make_response(success=False, message="model_path is required")
        if not robot_type:
            return self._make_response(success=False, message="robot_type is required")

        robot_config_path = _find_robot_config(robot_type)
        with open(robot_config_path) as f:
            raw = yaml.safe_load(f)
        robot_config = derive_robot_config(_extract_robot_section(raw, robot_type))

        # Action keys = follower joint modalities, stripped to bare names
        # so downstream (ActionChunkProcessor.split_action) can map them
        # to "joint_order.leader_<name>" when emitting JointTrajectory.
        # Phase 2: synthetic per-modality views (parent: <physical-group>)
        # ARE the modality source. Physical follower leaves with no
        # children stay supported as a fallback for older robot configs.
        groups = robot_config.get("joint_groups", {})
        parents_seen = {
            cfg.get("parent") for cfg in groups.values() if cfg.get("parent")
        }
        modality_groups: List[str] = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)              # synthetic view
            elif name not in parents_seen:
                modality_groups.append(name)              # leaf physical follower

        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        # Odometry-backed mobile state lives in sensors["odom"] (the
        # robot_client schema funnels nav_msgs/Odometry there instead of
        # joint_groups). Tack 'mobile' onto the modality list so
        # _run_inference splices its [linear_x, linear_y, angular_z] into
        # observation.state — matches the layout used during training in
        # the legacy orchestrator pipeline.
        if "odom" in robot_config.get("sensors", {}):
            modalities = sorted(set(modalities) | {"mobile"})
        if not modalities:
            return self._make_response(
                success=False,
                message=f"no follower joint groups in {robot_config_path}",
            )

        self._action_keys = modalities
        self._policy = _load_policy(model_path)
        self._setup_obs_subscribers(robot_config)
        self._setup_zenoh_io()

        self._loaded = True
        # Tell Process B to wire up per-group publishers for this robot.
        # Has to happen after our own zenoh trigger sub is up so B's first
        # trigger after configure has someone to receive it.
        self._publish_configure(robot_type)
        self._publish_lifecycle("loaded")
        logger.info(f"LOAD ok — action_keys={self._action_keys}")
        return self._make_response(
            success=True,
            message=f"loaded {model_path}",
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
        # Empty robot_type tells Process B to close per-group publishers and
        # go idle until the next LOAD.
        self._publish_configure("")
        self._publish_lifecycle("unloaded")
        logger.info("UNLOAD")
        return self._make_response(success=True, message="unloaded")

    # -- Obs plumbing ---------------------------------------------------------

    def _setup_obs_subscribers(self, robot_config: dict) -> None:
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
            "domain_id": self._domain_id,
            "node_name": self._node_name,
            "namespace": self._namespace,
        }

        for cam_name, cfg in robot_config.get("cameras", {}).items():
            topic = cfg["topic"]
            msg_type = cfg.get("msg_type", "sensor_msgs/msg/CompressedImage")
            sub = ROS2Subscriber(
                topic=topic,
                msg_type=msg_type,
                callback=self._make_image_callback(cam_name),
                **common,
            )
            self._obs_subscribers.append(sub)
            logger.info(f"camera sub: {cam_name} → {topic}")

        # Build {topic → [(modality_key, joint_names), ...]} from follower
        # joint groups. Synthetic per-modality views (those carrying a
        # ``parent``) inherit their parent group's physical topic and are
        # populated by name-based slicing in the callback. Leaf physical
        # follower groups (no parent, no children) get their own entry.
        groups = robot_config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        topic_splits: Dict[str, List[Any]] = {}
        msg_type_by_topic: Dict[str, str] = {}

        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            modality = name[len("follower_"):]
            wanted = list(cfg.get("joint_names", []))
            parent = cfg.get("parent")
            if parent:
                parent_cfg = groups.get(parent, {})
                topic = parent_cfg.get("topic")
                if not topic:
                    continue
                msg_type_by_topic.setdefault(
                    topic,
                    parent_cfg.get("msg_type", "sensor_msgs/msg/JointState"),
                )
                topic_splits.setdefault(topic, []).append((modality, wanted))
            elif name not in parents:
                topic = cfg.get("topic")
                if not topic:
                    continue
                msg_type_by_topic.setdefault(
                    topic, cfg.get("msg_type", "sensor_msgs/msg/JointState")
                )
                topic_splits.setdefault(topic, []).append((modality, wanted))

        for topic, splits in topic_splits.items():
            sub = ROS2Subscriber(
                topic=topic,
                msg_type=msg_type_by_topic[topic],
                callback=self._make_joint_split_callback(splits),
                **common,
            )
            self._obs_subscribers.append(sub)
            logger.info(
                f"joint sub: {topic} → modalities {[m for m, _ in splits]}"
            )

        # Odometry → mobile state. nav_msgs/Odometry's twist.linear.{x,y}
        # + twist.angular.z become a 3-vector that joins observation.state
        # alongside the joint-state modalities.
        odom_cfg = robot_config.get("sensors", {}).get("odom")
        if odom_cfg:
            sub = ROS2Subscriber(
                topic=odom_cfg["topic"],
                msg_type=odom_cfg.get("msg_type", "nav_msgs/msg/Odometry"),
                callback=self._make_odom_callback(),
                **common,
            )
            self._obs_subscribers.append(sub)
            logger.info(f"odom sub: {odom_cfg['topic']} → mobile")

    def _make_image_callback(self, cam_name: str):
        def _cb(msg):
            try:
                import cv2

                data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    return
                with self._obs_lock:
                    self._latest_obs["images"][cam_name] = image
                    self._latest_obs["timestamp"] = time.time()
            except Exception as e:
                logger.debug(f"image decode ({cam_name}) failed: {e}")

        return _cb

    def _make_joint_split_callback(self, splits):
        """JointState callback that fills one or more per-modality buckets.

        ``splits`` is a list of ``(modality_key, expected_joint_names)``
        tuples. When ``expected_joint_names`` is non-empty we slice
        positions out of the message by name (so a single /joint_states
        feed can serve multiple modalities); when empty we just take the
        message's positions verbatim (legacy per-modality follower topic).
        """
        def _cb(msg):
            try:
                names = list(msg.name) if hasattr(msg, "name") else []
                positions = list(msg.position) if hasattr(msg, "position") else []
                if not positions:
                    return
                name_to_pos = dict(zip(names, positions)) if names else {}
                ts = time.time()
                with self._obs_lock:
                    for modality, wanted in splits:
                        if wanted and name_to_pos:
                            try:
                                sliced = [name_to_pos[n] for n in wanted]
                            except KeyError as missing:
                                logger.debug(
                                    f"joint state ({modality}) missing {missing} "
                                    f"in {names}"
                                )
                                continue
                            self._latest_obs["joint_states"][modality] = {
                                "names": list(wanted),
                                "positions": sliced,
                            }
                        else:
                            self._latest_obs["joint_states"][modality] = {
                                "names": list(names),
                                "positions": list(positions),
                            }
                    self._latest_obs["timestamp"] = ts
            except Exception as e:
                logger.debug(f"joint split callback failed: {e}")

        return _cb

    def _make_odom_callback(self):
        """nav_msgs/Odometry → mobile state slot.

        ACT (and other LeRobot policies) trained on the legacy
        orchestrator pipeline expect the 3-vector
        [linear_x, linear_y, angular_z] to participate in observation.state
        alongside the joint-state modalities. Read it off the standard
        Odometry fields and write it into the same _latest_obs structure
        the joint_split callback uses.
        """
        names = ["linear_x", "linear_y", "angular_z"]

        def _cb(msg):
            try:
                t = msg.twist.twist
                positions = [
                    float(t.linear.x),
                    float(t.linear.y),
                    float(t.angular.z),
                ]
                ts = time.time()
                with self._obs_lock:
                    self._latest_obs["joint_states"]["mobile"] = {
                        "names": list(names),
                        "positions": positions,
                    }
                    self._latest_obs["timestamp"] = ts
            except Exception as e:
                logger.debug(f"odom callback failed: {e}")

        return _cb

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
        seq_id = int(msg.data)

        snapshot = self._snapshot_obs()
        if snapshot is None:
            logger.warning(f"trigger seq={seq_id} — no recent obs, skipping")
            return

        chunk = self._run_inference(snapshot)
        if chunk is None:
            return

        self._publish_chunk(seq_id, chunk)

    def _snapshot_obs(self) -> Optional[Dict[str, Any]]:
        with self._obs_lock:
            ts = self._latest_obs["timestamp"]
            if ts is None or (time.time() - ts) > OBS_STALE_S:
                return None
            return {
                "images": dict(self._latest_obs["images"]),
                "joint_states": dict(self._latest_obs["joint_states"]),
                "timestamp": ts,
            }

    def _run_inference(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        import torch

        try:
            batch: Dict[str, Any] = {}
            for cam_name, image in obs["images"].items():
                batch[f"observation.images.{cam_name}"] = _preprocess_image(image)

            state_parts: List[List[float]] = []
            for modality_key in self._action_keys:
                joint_data = obs["joint_states"].get(modality_key)
                if joint_data is None:
                    logger.warning(f"missing joint state for {modality_key}")
                    return None
                state_parts.append(joint_data["positions"])

            flat_state: List[float] = []
            for part in state_parts:
                flat_state.extend(part)
            batch["observation.state"] = torch.tensor(
                flat_state, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                # nn.Module has no `.device` attribute by default — derive
                # from the first parameter (cheap, all params share device).
                device = next(self._policy.parameters()).device
                on_device = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                action = self._policy.select_action(on_device)

            action_array = (
                action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
            )
            if action_array.ndim == 1:
                action_array = action_array[np.newaxis, :]
            return action_array.astype(np.float64)
        except Exception as e:
            logger.error(f"inference failed: {e}", exc_info=True)
            return None

    def _publish_chunk(self, seq_id: int, chunk: np.ndarray) -> None:
        T, D = chunk.shape
        try:
            # zenoh_ros2_sdk's publisher.publish() calls .view() on the
            # data array (treats it as a numpy buffer for fast CDR
            # encoding), so we must pass the raw numpy array. Wrapping in
            # list() crashes with "'list' object has no attribute 'view'".
            self._chunk_pub.publish(
                seq_id=seq_id,
                chunk_size=T,
                action_dim=D,
                data=np.ascontiguousarray(chunk.reshape(-1), dtype=np.float64),
            )
            logger.info(f"chunk pub seq={seq_id} T={T} D={D}")
        except Exception as e:
            logger.error(f"chunk publish failed: {e}", exc_info=True)

    # -- Teardown -------------------------------------------------------------

    def _teardown_runtime(self) -> None:
        """Close obs subs + Zenoh trigger/pub + release model. Keep srv alive."""
        self._running = False
        self._paused = False

        for sub in self._obs_subscribers:
            try:
                sub.close()
            except Exception:
                pass
        self._obs_subscribers.clear()

        for handle_attr in ("_trigger_sub", "_chunk_pub"):
            handle = getattr(self, handle_attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_attr, None)

        if self._policy is not None:
            del self._policy
            self._policy = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        with self._obs_lock:
            self._latest_obs = {"images": {}, "joint_states": {}, "timestamp": None}
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

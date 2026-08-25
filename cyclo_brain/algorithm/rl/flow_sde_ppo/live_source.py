"""Live Cyclo SG2 episode collection for MultiTaskDiT Flow-SDE PPO.

This module is deliberately an adapter boundary.  The PPO implementation does
not know about Zenoh or RobotClient, while this file does not reimplement the
policy or optimizer.  One policy decision samples a complete action chunk and
every primitive action is admitted by the simulator's atomic ActionStep/ACK
transport before the next policy decision is made.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from cyclo_brain.model.multi_task_dit.flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS

from .on_policy import FlowSDEEpisode


ACTION_STEP_TOPIC = "/inference/action_step"
ACTION_STEP_CANCEL_TOPIC = "/inference/action_step_cancel"
ACTION_STEP_ACK_TOPIC = "/inference/action_step_ack"
SIMULATION_RESET_TOPIC = "/simulation/reset"
ACTION_STEP_MSG_TYPE = "interfaces/msg/ActionStep"
ACTION_STEP_ACK_MSG_TYPE = "interfaces/msg/ActionStepAck"

# Keep these definitions next to the client because the restored
# cyclo_intelligence branch intentionally does not install the experimental
# interfaces package in the LeRobot container.  They must remain byte-for-byte
# field compatible with cyclo_lab's opt-in atomic bridge.
ACTION_STEP_DEF = """\
uint64 session_id
uint64 step_id
uint64 source_seq_id
int32 source_action_index
int32 source_chunk_size
string[] action_keys
int32 action_dim
float64[] action
float64 timestamp
"""

ACTION_STEP_ACK_DEF = """\
uint8 STATUS_EXECUTED=0
uint8 STATUS_CANCELLED=1
uint64 session_id
uint64 step_id
uint64 environment_step
uint8 status
float64[] executed_action
float64 timestamp
float64 duration
"""

ACTION_KEYS = ("arm_left", "arm_right", "head", "lift", "mobile")
ACTION_WIDTHS = (8, 8, 2, 1, 3)
ACTION_DIM = 22
EXECUTION_HORIZON = 16
if sum(ACTION_WIDTHS) != ACTION_DIM:
    raise RuntimeError("Cyclo ActionStep modality widths do not form the 22D contract")
CAMERA_TO_POLICY_KEY = {
    "cam_left_wrist": "observation.images.rgb.cam_left_wrist",
    "cam_left_head": "observation.images.rgb.cam_left_head",
    "cam_right_wrist": "observation.images.rgb.cam_right_wrist",
}
STATE_GROUPS = (
    ("follower_arm_left", 8),
    ("follower_arm_right", 8),
    ("follower_head", 2),
    ("follower_lift", 1),
)


class FlowSDELiveError(RuntimeError):
    """Base class for explicit live-collection failures."""


class FlowSDECollectionCancelled(FlowSDELiveError):
    """External operator cancelled collection without producing an update."""


class ActionStepTimeout(FlowSDELiveError):
    """The simulator did not acknowledge an atomic primitive action."""


class SensorBarrierTimeout(FlowSDELiveError):
    """A post-action observation never became newer than its source chunk."""


@dataclass(frozen=True)
class SensorMarker:
    """Reception timestamps for the exact three-camera + 22D observation."""

    timestamps: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.timestamps:
            raise ValueError("Sensor marker cannot be empty")

    def as_dict(self) -> dict[str, float]:
        return dict(self.timestamps)


@dataclass(frozen=True)
class ActionStepReceipt:
    session_id: int
    step_id: int
    environment_step: int
    executed_action: tuple[float, ...]
    simulator_timestamp: float
    duration: float
    received_at: float
    command_max_abs_error: float = 0.0


@dataclass(frozen=True)
class EpisodeOutcome:
    job_id: str
    outcome: str
    sequence: int
    timestamp: Any

    @property
    def reward(self) -> float:
        if self.outcome == "success":
            return 1.0
        if self.outcome == "fail":
            return 0.0
        raise FlowSDECollectionCancelled("Flow-SDE episode was cancelled")


class AtomicOutcomeFile:
    """Read fresh success/fail/cancel labels from an atomically replaced JSON file."""

    VALID_OUTCOMES = frozenset({"success", "fail", "cancel"})

    def __init__(self, path: str | Path, *, job_id: str) -> None:
        self.path = Path(path).expanduser()
        if not job_id or not job_id.strip():
            raise ValueError("Flow-SDE outcome job_id must be non-empty")
        self.job_id = job_id.strip()
        initial = self._read_optional()
        self._last_sequence = initial.sequence if initial is not None else -1

    def _read_optional(self) -> EpisodeOutcome | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FlowSDELiveError(
                f"Outcome control file is not valid atomic JSON: {self.path}"
            ) from exc
        if not isinstance(payload, dict):
            raise FlowSDELiveError("Outcome control payload must be a JSON object")
        job_id = payload.get("job_id")
        outcome = payload.get("outcome")
        sequence = payload.get("sequence")
        if job_id != self.job_id:
            return None
        if outcome not in self.VALID_OUTCOMES:
            raise FlowSDELiveError(
                "Outcome must be one of success, fail, or cancel"
            )
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise FlowSDELiveError("Outcome sequence must be a non-negative integer")
        return EpisodeOutcome(
            job_id=job_id,
            outcome=outcome,
            sequence=sequence,
            timestamp=payload.get("timestamp"),
        )

    def poll(self) -> EpisodeOutcome | None:
        outcome = self._read_optional()
        if outcome is None or outcome.sequence <= self._last_sequence:
            return None
        self._last_sequence = outcome.sequence
        return outcome


@runtime_checkable
class ObservationBatchSource(Protocol):
    def snapshot(self, *, timeout: float | None = None) -> SensorMarker: ...

    def observe(
        self,
        *,
        newer_than: SensorMarker | None = None,
        timeout: float | None = None,
    ) -> tuple[Mapping[str, Tensor], SensorMarker]: ...

    def close(self) -> None: ...


@runtime_checkable
class AtomicActionTransport(Protocol):
    def reset_environment(self) -> None: ...

    def begin_episode(self) -> int: ...

    def execute_chunk(
        self,
        actions: np.ndarray,
        *,
        source_seq_id: int,
    ) -> tuple[ActionStepReceipt, ...]: ...

    def cancel_current(self) -> None: ...

    def close(self) -> None: ...


def _close_endpoint(endpoint: Any) -> None:
    close = getattr(endpoint, "close", None) or getattr(endpoint, "Close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


class ZenohAtomicActionStepTransport:
    """ROS2-compatible Zenoh client for cyclo_lab's atomic SG2 bridge."""

    STATUS_EXECUTED = 0
    STATUS_CANCELLED = 1

    def __init__(
        self,
        *,
        ack_timeout: float = 5.0,
        router_ip: str | None = None,
        router_port: int | None = None,
        domain_id: int | None = None,
    ) -> None:
        if ack_timeout <= 0.0:
            raise ValueError("ActionStep ACK timeout must be positive")
        sdk_path = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            from zenoh_ros2_sdk import ROS2Publisher, ROS2Subscriber
        except Exception as exc:
            raise FlowSDELiveError(
                "zenoh_ros2_sdk is unavailable; run this job in the LeRobot container"
            ) from exc

        common: dict[str, Any] = {
            "router_ip": router_ip or os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
            "router_port": int(
                router_port
                if router_port is not None
                else os.environ.get("ZENOH_ROUTER_PORT", "7447")
            ),
            "domain_id": int(
                domain_id
                if domain_id is not None
                else os.environ.get("ROS_DOMAIN_ID", "0")
            ),
        }
        self._condition = threading.Condition()
        self._receipts: dict[tuple[int, int], ActionStepReceipt | Exception] = {}
        self._ack_timeout = float(ack_timeout)
        self._session_id: int | None = None
        self._step_id = 0
        self._closed = False
        try:
            self._publisher = ROS2Publisher(
                topic=ACTION_STEP_TOPIC,
                msg_type=ACTION_STEP_MSG_TYPE,
                msg_definition=ACTION_STEP_DEF,
                **common,
            )
            self._cancel_publisher = ROS2Publisher(
                topic=ACTION_STEP_CANCEL_TOPIC,
                msg_type="std_msgs/msg/UInt64",
                **common,
            )
            self._reset_publisher = ROS2Publisher(
                topic=SIMULATION_RESET_TOPIC,
                msg_type="std_msgs/msg/Empty",
                **common,
            )
            self._subscriber = ROS2Subscriber(
                topic=ACTION_STEP_ACK_TOPIC,
                msg_type=ACTION_STEP_ACK_MSG_TYPE,
                msg_definition=ACTION_STEP_ACK_DEF,
                callback=self._on_ack,
                **common,
            )
        except Exception as exc:
            for endpoint in (
                getattr(self, "_publisher", None),
                getattr(self, "_cancel_publisher", None),
                getattr(self, "_reset_publisher", None),
                getattr(self, "_subscriber", None),
            ):
                if endpoint is not None:
                    _close_endpoint(endpoint)
            raise FlowSDELiveError(
                "Could not create ActionStep Zenoh endpoints; verify the router and ROS_DOMAIN_ID"
            ) from exc

    def reset_environment(self) -> None:
        if self._closed:
            raise FlowSDELiveError("ActionStep transport is closed")
        try:
            self._reset_publisher.publish()
        except Exception as exc:
            raise FlowSDELiveError(
                "Failed to publish /simulation/reset; live multi-episode PPO cannot continue"
            ) from exc

    def _on_ack(self, message: Any) -> None:
        try:
            session_id = int(message.session_id)
            step_id = int(message.step_id)
            status = int(message.status)
            if status == self.STATUS_CANCELLED:
                value: ActionStepReceipt | Exception = FlowSDECollectionCancelled(
                    f"ActionStep session={session_id} step={step_id} was cancelled by simulator"
                )
            elif status != self.STATUS_EXECUTED:
                value = FlowSDELiveError(f"Unknown ActionStep ACK status={status}")
            else:
                executed = tuple(float(item) for item in message.executed_action)
                if len(executed) != ACTION_DIM or not all(np.isfinite(executed)):
                    value = FlowSDELiveError(
                        "ActionStep ACK must contain exactly 22 finite executed-action values"
                    )
                else:
                    value = ActionStepReceipt(
                        session_id=session_id,
                        step_id=step_id,
                        environment_step=int(message.environment_step),
                        executed_action=executed,
                        simulator_timestamp=float(message.timestamp),
                        duration=float(message.duration),
                        received_at=time.time(),
                    )
            with self._condition:
                self._receipts[(session_id, step_id)] = value
                self._condition.notify_all()
        except Exception:
            # Malformed unrelated traffic must not kill the Zenoh callback
            # thread; the matching sender will time out with a useful error.
            return

    def begin_episode(self) -> int:
        if self._closed:
            raise FlowSDELiveError("ActionStep transport is closed")
        if self._session_id is not None:
            self.cancel_current()
        # time_ns is monotonic enough for one process and naturally fits uint64.
        self._session_id = time.time_ns() & ((1 << 64) - 1)
        self._step_id = 0
        return self._session_id

    def _wait_for_ack(self, session_id: int, step_id: int) -> ActionStepReceipt:
        deadline = time.monotonic() + self._ack_timeout
        with self._condition:
            while (session_id, step_id) not in self._receipts:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ActionStepTimeout(
                        "No /inference/action_step_ack received for "
                        f"session={session_id} step={step_id} within {self._ack_timeout:.1f}s. "
                        "Start cyclo_lab with --action_step_transport and press B."
                    )
                self._condition.wait(remaining)
            result = self._receipts.pop((session_id, step_id))
        if isinstance(result, Exception):
            raise result
        return result

    def execute_chunk(
        self,
        actions: np.ndarray,
        *,
        source_seq_id: int,
    ) -> tuple[ActionStepReceipt, ...]:
        if self._session_id is None:
            self.begin_episode()
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < 1:
            raise ValueError(f"Atomic action chunk must have shape (T, {ACTION_DIM})")
        if not np.isfinite(chunk).all():
            raise ValueError("Atomic action chunk contains non-finite values")
        session_id = int(self._session_id)
        receipts: list[ActionStepReceipt] = []
        for index, action in enumerate(chunk):
            self._step_id += 1
            step_id = self._step_id
            self._publisher.publish(
                session_id=session_id,
                step_id=step_id,
                source_seq_id=int(source_seq_id),
                source_action_index=index,
                source_chunk_size=int(chunk.shape[0]),
                action_keys=list(ACTION_KEYS),
                action_dim=ACTION_DIM,
                action=np.ascontiguousarray(action, dtype=np.float64),
                timestamp=time.time(),
            )
            receipt = self._wait_for_ack(session_id, step_id)
            # Joint-limit clamping is a deterministic environment transform,
            # so it does not invalidate latent-space PPO. Keep the difference
            # visible for diagnostics without rejecting a valid receipt.
            max_abs_error = float(
                np.max(np.abs(np.asarray(receipt.executed_action) - action))
            )
            receipts.append(
                replace(receipt, command_max_abs_error=max_abs_error)
            )
        return tuple(receipts)

    def cancel_current(self) -> None:
        if self._session_id is None or self._closed:
            return
        try:
            self._cancel_publisher.publish(data=int(self._session_id))
        finally:
            self._session_id = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.cancel_current()
        finally:
            self._closed = True
            for endpoint in (
                self._subscriber,
                self._publisher,
                self._cancel_publisher,
                self._reset_publisher,
            ):
                _close_endpoint(endpoint)


class CycloLeRobotObservationSource:
    """Build the exact current 3-camera + 22D preprocessed policy input."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        preprocessor: Any,
        robot_type: str,
        task_instruction: str,
        sensor_timeout: float = 10.0,
        robot: Any | None = None,
    ) -> None:
        if sensor_timeout <= 0.0:
            raise ValueError("Sensor timeout must be positive")
        if not task_instruction or not task_instruction.strip():
            raise ValueError("Task instruction must be non-empty")
        if robot is None:
            client_path = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
            if client_path and client_path not in sys.path:
                sys.path.insert(0, client_path)
            try:
                from robot_client import RobotClient
            except Exception as exc:
                raise FlowSDELiveError(
                    "RobotClient is unavailable; run this job in the LeRobot container"
                ) from exc
            robot = RobotClient(robot_type)
        self.robot = robot
        self.policy = policy
        self.preprocessor = preprocessor
        self.task_instruction = task_instruction.strip()
        self.sensor_timeout = float(sensor_timeout)
        self._last_marker: SensorMarker | None = None

        image_features = getattr(policy.config, "image_features", None)
        if tuple(image_features or ()) != CYCLO_SG2_CAMERA_KEYS:
            raise ValueError("Live Flow-SDE checkpoint camera order is not the Cyclo 3-camera contract")
        self._target_sizes: dict[str, tuple[int, int]] = {}
        for key, feature in image_features.items():
            shape = tuple(int(value) for value in feature.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"Invalid image feature shape for {key}: {shape}")
            self._target_sizes[key] = (shape[2], shape[1])

    @staticmethod
    def _marker_from_status(status: Mapping[str, Any]) -> SensorMarker | None:
        timestamps: list[tuple[str, float]] = []
        cameras = status.get("cameras", {})
        joints = status.get("joint_groups", {})
        sensors = status.get("sensors", {})
        for name in CAMERA_TO_POLICY_KEY:
            timestamp = cameras.get(name, {}).get("timestamp")
            if timestamp is None:
                return None
            timestamps.append((f"camera:{name}", float(timestamp)))
        for name, _width in STATE_GROUPS:
            timestamp = joints.get(name, {}).get("timestamp")
            if timestamp is None:
                return None
            timestamps.append((f"joint:{name}", float(timestamp)))
        odom_timestamp = sensors.get("odom", {}).get("timestamp")
        if odom_timestamp is None:
            return None
        timestamps.append(("sensor:odom", float(odom_timestamp)))
        return SensorMarker(tuple(timestamps))

    @staticmethod
    def _is_newer(candidate: SensorMarker, baseline: SensorMarker) -> bool:
        current = candidate.as_dict()
        return all(current.get(name, float("-inf")) > timestamp for name, timestamp in baseline.timestamps)

    def _wait_for_marker(
        self,
        *,
        newer_than: SensorMarker | None,
        timeout: float,
    ) -> SensorMarker:
        deadline = time.monotonic() + timeout
        last_marker: SensorMarker | None = None
        while time.monotonic() < deadline:
            marker = self._marker_from_status(self.robot.get_status())
            if marker is not None:
                last_marker = marker
                if newer_than is None or self._is_newer(marker, newer_than):
                    return marker
            time.sleep(0.01)
        if newer_than is None:
            raise SensorBarrierTimeout(
                "Required live observation is unavailable: cam_left_head, cam_left_wrist, "
                "cam_right_wrist, four joint groups, and odom are required"
            )
        stale: list[str] = []
        if last_marker is not None:
            current = last_marker.as_dict()
            stale = [
                name for name, timestamp in newer_than.timestamps
                if current.get(name, float("-inf")) <= timestamp
            ]
        raise SensorBarrierTimeout(
            "Post-ActionStep sensor barrier timed out; stale signals=" + repr(stale)
        )

    @staticmethod
    def _prepare_image(
        image: np.ndarray,
        *,
        rotation_deg: int,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        import cv2

        rotation = int(rotation_deg) % 360
        if rotation == 90:
            image = np.rot90(image, k=3)
        elif rotation == 180:
            image = np.rot90(image, k=2)
        elif rotation == 270:
            image = np.rot90(image, k=1)
        elif rotation != 0:
            raise ValueError(f"Unsupported camera rotation: {rotation_deg}")
        width, height = target_size
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        return np.ascontiguousarray(image)

    def _raw_batch(self) -> dict[str, Any]:
        images = self.robot.get_images(format="rgb")
        joint_dict = self.robot.get_joint_positions()
        raw: dict[str, Any] = {}
        for camera_name, policy_key in CAMERA_TO_POLICY_KEY.items():
            image = images.get(camera_name)
            if image is None:
                raise FlowSDELiveError(f"Missing camera frame: {camera_name}")
            config = self.robot._config.get("cameras", {}).get(camera_name, {})
            image = self._prepare_image(
                image,
                rotation_deg=int(config.get("rotation_deg", 0)),
                target_size=self._target_sizes[policy_key],
            )
            tensor = torch.from_numpy(image).to(torch.float32).div_(255.0)
            raw[policy_key] = tensor.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(1)

        state_parts: list[np.ndarray] = []
        for group_name, width in STATE_GROUPS:
            positions = joint_dict.get(group_name)
            if positions is None or np.asarray(positions).size != width:
                raise FlowSDELiveError(
                    f"State group {group_name} must contain {width} values"
                )
            state_parts.append(np.asarray(positions, dtype=np.float32))
        odom = self.robot.get_odom()
        if odom is None:
            raise FlowSDELiveError("Missing odom for 22D state")
        state_parts.append(
            np.asarray(
                [
                    odom["linear_velocity"][0],
                    odom["linear_velocity"][1],
                    odom["angular_velocity"][2],
                ],
                dtype=np.float32,
            )
        )
        state = np.concatenate(state_parts)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise FlowSDELiveError(f"Live SG2 state must be finite {ACTION_DIM}D")
        raw["observation.state"] = torch.from_numpy(state).unsqueeze(0).unsqueeze(1)
        raw["task"] = [self.task_instruction]
        return raw

    def observe(
        self,
        *,
        newer_than: SensorMarker | None = None,
        timeout: float | None = None,
    ) -> tuple[Mapping[str, Tensor], SensorMarker]:
        resolved_timeout = self.sensor_timeout if timeout is None else float(timeout)
        marker = self._wait_for_marker(newer_than=newer_than, timeout=resolved_timeout)
        batch = self.preprocessor(self._raw_batch())
        if not isinstance(batch, Mapping):
            raise FlowSDELiveError("LeRobot preprocessor did not return a mapping")
        # Anchor the next post-action barrier after the reads above. Without
        # this second snapshot a callback racing with _raw_batch() could make
        # the next observe() accept a frame that actually preceded the action.
        post_read_marker = self._marker_from_status(self.robot.get_status())
        if post_read_marker is not None:
            marker = post_read_marker
        self._last_marker = marker
        return batch, marker

    def snapshot(self, *, timeout: float | None = None) -> SensorMarker:
        """Capture the pre-reset generation without invoking the policy processor."""

        resolved_timeout = self.sensor_timeout if timeout is None else float(timeout)
        return self._wait_for_marker(newer_than=None, timeout=resolved_timeout)

    def close(self) -> None:
        close = getattr(self.robot, "close", None)
        if callable(close):
            close()


class CycloFlowSDEEpisodeSource:
    """Collect one labeled on-policy episode from the current Cyclo simulator."""

    def __init__(
        self,
        *,
        observations: ObservationBatchSource,
        actions: AtomicActionTransport,
        outcomes: AtomicOutcomeFile,
        postprocessor: Any,
        max_chunk_decisions: int = 120,
        sensor_timeout: float = 10.0,
    ) -> None:
        if max_chunk_decisions < 1:
            raise ValueError("max_chunk_decisions must be positive")
        if sensor_timeout <= 0.0:
            raise ValueError("sensor_timeout must be positive")
        self.observations = observations
        self.actions = actions
        self.outcomes = outcomes
        self.postprocessor = postprocessor
        self.max_chunk_decisions = int(max_chunk_decisions)
        self.sensor_timeout = float(sensor_timeout)
        self.last_episode_diagnostics: dict[str, float | int] = {}

    def _postprocess(self, normalized: Tensor) -> np.ndarray:
        with torch.no_grad():
            result = self.postprocessor(normalized)
        if not isinstance(result, Tensor):
            result = torch.as_tensor(result)
        chunk = result.detach().float().cpu().numpy()
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.shape != (EXECUTION_HORIZON, ACTION_DIM):
            raise FlowSDELiveError(
                "Postprocessed action chunk must match the deployed Cyclo contract "
                f"({EXECUTION_HORIZON}, {ACTION_DIM}), got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise FlowSDELiveError("Postprocessed action chunk contains non-finite values")
        return np.ascontiguousarray(chunk, dtype=np.float64)

    def collect_episode(self, runner: Any) -> FlowSDEEpisode:
        pending_outcome = self.outcomes.poll()
        if pending_outcome is not None and pending_outcome.outcome == "cancel":
            self.actions.cancel_current()
            raise FlowSDECollectionCancelled("Flow-SDE collection cancelled before rollout")

        # Every episode has an explicit environment lifecycle. Waiting for all
        # required signals to advance makes a silent/missing reset fail closed
        # instead of training on the terminal frame from the prior episode.
        before_reset = self.observations.snapshot(timeout=self.sensor_timeout)
        self.actions.cancel_current()
        self.actions.reset_environment()
        batch, marker = self.observations.observe(
            newer_than=before_reset,
            timeout=self.sensor_timeout,
        )
        self.actions.begin_episode()
        primitive_steps = 0
        command_max_abs_error = 0.0
        transitions = []
        for decision_index in range(self.max_chunk_decisions):
            decision = runner.sample_preprocessed_batch(batch)
            action_chunk = self._postprocess(decision.executed_actions)
            receipts = self.actions.execute_chunk(action_chunk, source_seq_id=decision_index)
            primitive_steps += len(receipts)
            if receipts:
                command_max_abs_error = max(
                    command_max_abs_error,
                    max(receipt.command_max_abs_error for receipt in receipts),
                )

            outcome = pending_outcome or self.outcomes.poll()
            pending_outcome = None
            if outcome is not None:
                if outcome.outcome == "cancel":
                    self.actions.cancel_current()
                    raise FlowSDECollectionCancelled("Flow-SDE collection cancelled by operator")
                transitions.append(
                    decision.as_transition(
                        reward=outcome.reward,
                        terminated=True,
                        truncated=False,
                    )
                )
                self.last_episode_diagnostics = {
                    "chunk_decisions": decision_index + 1,
                    "primitive_steps": primitive_steps,
                    "command_max_abs_error": command_max_abs_error,
                }
                return FlowSDEEpisode(tuple(transitions), bootstrap_value=0.0)

            is_last = decision_index + 1 == self.max_chunk_decisions
            # The barrier is anchored to the pre-chunk observation. cyclo_lab
            # publishes all observations after env.step and only then emits the
            # ACK; cross-topic delivery order may differ, so comparing against
            # ACK wall-clock receipt would incorrectly reject a valid frame.
            next_batch, next_marker = self.observations.observe(
                newer_than=marker,
                timeout=self.sensor_timeout,
            )
            # A human label can land while the post-action camera barrier is
            # resolving. Consume it before sampling or executing another chunk.
            outcome = self.outcomes.poll()
            if outcome is not None:
                if outcome.outcome == "cancel":
                    self.actions.cancel_current()
                    raise FlowSDECollectionCancelled("Flow-SDE collection cancelled by operator")
                transitions.append(
                    decision.as_transition(
                        reward=outcome.reward,
                        terminated=True,
                        truncated=False,
                    )
                )
                self.last_episode_diagnostics = {
                    "chunk_decisions": decision_index + 1,
                    "primitive_steps": primitive_steps,
                    "command_max_abs_error": command_max_abs_error,
                }
                return FlowSDEEpisode(tuple(transitions), bootstrap_value=0.0)
            if is_last:
                transitions.append(
                    decision.as_transition(
                        reward=0.0,
                        terminated=False,
                        truncated=True,
                    )
                )
                with torch.no_grad():
                    conditioning = runner.adapter.encode_conditioning(next_batch)
                    bootstrap_value = float(runner.value(conditioning)[0].detach().cpu())
                self.last_episode_diagnostics = {
                    "chunk_decisions": decision_index + 1,
                    "primitive_steps": primitive_steps,
                    "command_max_abs_error": command_max_abs_error,
                }
                return FlowSDEEpisode(
                    tuple(transitions),
                    bootstrap_value=bootstrap_value,
                )

            transitions.append(
                decision.as_transition(
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                )
            )
            batch, marker = next_batch, next_marker

        raise AssertionError("unreachable Flow-SDE episode loop")

    def close(self) -> None:
        try:
            self.actions.close()
        finally:
            self.observations.close()

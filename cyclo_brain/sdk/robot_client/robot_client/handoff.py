"""Receive-side authority and latched leases for physical SDK output."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import math
import threading
import time


@dataclass(frozen=True)
class HandoffStatus:
    enabled: bool
    state: str
    reason: str
    revision: int
    fault_revision: int
    ready_since: float | None


DISABLED_STATUS = HandoffStatus(False, "ready", "disabled", 0, 0, None)


class OutputAuthorityError(RuntimeError):
    """No send/whole-vector receipt is permitted under the current lease."""

    def __init__(self, status: HandoffStatus, reason: str | None = None):
        self.status = status
        self.reason = reason or status.reason
        super().__init__(f"Output authority {status.state}: {self.reason}")


def physical_output(*, allow_unarmed=False):
    """Keep one ownership revision across nested and multi-topic commands."""
    def decorate(method):
        @wraps(method)
        def guarded(self, *args, **kwargs):
            guard = getattr(self, "_handoff", None)
            if guard is None:
                return method(self, *args, **kwargs)
            with guard.output(allow_unarmed=allow_unarmed):
                return method(self, *args, **kwargs)
        return guarded
    return decorate


class HandoffGuard:
    """Callbacks share only this short-held lock, never an observation/I/O lock.

    Faults and revisions are cumulative, not consumed by reads. Expiry is checked
    both on reads and *before* heartbeat refresh so an unpolled outage is retained.
    No leader heartbeat is monitored: source fallback belongs to AIWorker.
    """

    def __init__(self, config, joint_groups):
        if not isinstance(config, Mapping):
            raise ValueError("inference_handoff must be a mapping")
        self.source_topic = config.get("source_topic", "/source")
        self.control_status_topic = config.get("control_status_topic", "/control/status")
        for topic in (self.source_topic, self.control_status_topic):
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError("inference_handoff topics must be nonempty strings")
        self._timeouts = {}
        for stream in ("source", "status_confirmation", "joint_state"):
            value = config.get(f"{stream}_timeout_s", .5)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{stream}_timeout_s must be positive and finite")
            self._timeouts[stream] = float(value)
        initial_timeout = config.get("initial_confirmation_timeout_s", 2.0)
        if (isinstance(initial_timeout, bool) or not isinstance(initial_timeout, (int, float))
                or not math.isfinite(initial_timeout) or initial_timeout <= 0):
            raise ValueError("initial_confirmation_timeout_s must be positive and finite")
        self._initial_confirmation_timeout = float(initial_timeout)
        interval = config.get("status_confirmation_interval_s", .1)
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or not 0 < interval < self._timeouts["status_confirmation"] / 3):
            raise ValueError("status_confirmation_interval_s must be positive and below timeout / 3")
        self.confirmation_interval = float(interval)
        modes = config.get("allowed_model_modes", [1])
        if not isinstance(modes, (list, tuple)) or not modes or any(
            type(mode) is not int or not 1 <= mode <= 65535 for mode in modes
        ):
            raise ValueError("allowed_model_modes must contain positive uint16 mode IDs")
        self._allowed_modes = frozenset(modes)
        self._required_joints = {name: frozenset(names) for name, names in joint_groups.items()}
        if not self._required_joints or any(not names for names in self._required_joints.values()):
            raise ValueError("inference_handoff requires named JointState feedback")
        self._lock = threading.Lock()
        self._initial_confirmation = threading.Event()
        self._output = threading.local()
        self._armed_revision = None
        self._source = None
        self._local_model_requested = False
        self._external_observed = False
        self._source_seen = None
        self._source_live_confirmed = False
        self._status = None
        self._confirmed_seen = None
        self._confirmation_required_since = time.monotonic()
        self._confirmation_fault = None
        self._epoch = 0
        self._status_version = 0
        self._feedback_fresh = False
        self._joint_seen = {name: None for name in self._required_joints}
        self._joint_valid = {name: False for name in self._required_joints}
        self._snapshot = HandoffStatus(True, "blocked", "source_missing", 0, 0, None)
        self._signature = None
        self._evaluate(time.monotonic())

    def _state(self, now):
        if self._local_model_requested and not self._external_observed:
            joint_fault = self._joint_fault(now)
            if joint_fault:
                return "blocked", joint_fault
            if self._confirmation_fault not in (None, "publisher_unavailable"):
                return "blocked", self._confirmation_fault
            return "ready", "local_model"
        for stream, seen in (("source", self._source_seen),):
            if seen is None:
                return "blocked", f"{stream}_missing"
            if now - seen >= self._timeouts[stream]:
                return "blocked", f"{stream}_stale"
        joint_fault = self._joint_fault(now)
        if joint_fault:
            return "blocked", joint_fault
        if self._source not in ("model_action", "teleop", "switching"):
            return "blocked", "source_invalid"
        requested, active, controller_state, _transition = self._status or (0, 0, 1, 0)
        if controller_state == 5:
            return "blocked", "controller_error"
        if controller_state not in (1, 2, 3, 4):
            return "blocked", "controller_state_invalid"
        if self._confirmation_fault:
            return "blocked", self._confirmation_fault
        seen = self._confirmed_seen if self._confirmed_seen is not None else self._confirmation_required_since
        if now - seen >= self._timeouts["status_confirmation"]:
            return "blocked", "status_confirmation_stale"
        if self._source in ("teleop", "switching"):
            return "yielded", self._source
        if not self._source_live_confirmed:
            return "yielded", "awaiting_live_source"
        if controller_state in (2, 3) or active == 0 or requested != active:
            if self._status is None:
                return "yielded", "awaiting_status_confirmation"
            return "yielded", "controller_transition"
        if active not in self._allowed_modes:
            return "blocked", "model_mode_unsupported"
        if self._confirmed_seen is None:
            return "yielded", "awaiting_status_confirmation"
        # Direct joint model mode intentionally stays HOLDING even while moving.
        return "ready", "model_action"

    def _joint_fault(self, now):
        for group, seen in self._joint_seen.items():
            if seen is None:
                return f"joint_state_missing:{group}"
            if now - seen >= self._timeouts["joint_state"]:
                return f"joint_state_stale:{group}"
            if not self._joint_valid[group]:
                return f"joint_state_invalid:{group}"
        return None

    def request_model_start(self):
        """Explicit local START, never a synthetic controller/source heartbeat.

        Once any external control is observed, this client lifetime cannot fall
        back to local ownership, even after publisher loss or another START.
        """
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            if not self._external_observed:
                self._local_model_requested = True
                if self._confirmation_fault == "feedback_gap":
                    self._confirmation_fault = None
            return self._evaluate(now)

    def observe_external_control(self):
        """Discovery alone revokes local ownership before the first state event."""
        with self._lock:
            self._external_observed = True
            self._evaluate(time.monotonic())

    def _evaluate(self, now):
        source_fresh = (self._local_model_requested and not self._external_observed) or (
            self._source_seen is not None and now - self._source_seen < self._timeouts["source"])
        fresh = (source_fresh
                 and all(seen is not None and now - seen < self._timeouts["joint_state"]
                         for seen in self._joint_seen.values()))
        if self._feedback_fresh and not fresh:
            self._epoch += 1
            self._confirmed_seen = None
            self._confirmation_fault = "feedback_gap"
        self._feedback_fresh = fresh
        state, reason = self._state(now)
        signature = (self._source, self._status, state, reason)
        old = self._snapshot
        if signature != self._signature:
            fault = state == "blocked" and (self._signature is None or (old.state, old.reason) != (state, reason))
            self._snapshot = HandoffStatus(True, state, reason, old.revision + 1,
                                           old.fault_revision + int(fault), now if state == "ready" else None)
            self._signature = signature
        return self._snapshot

    def snapshot(self):
        with self._lock:
            return self._evaluate(time.monotonic())

    def wait_for_initial_confirmation(self):
        """Bound LOAD's discovery warmup; this signal never grants output authority.

        START must still inspect and arm the current snapshot. A timeout leaves
        the model loaded but does not bypass missing, teleop, or faulty control.
        """
        return self._initial_confirmation.wait(self._initial_confirmation_timeout)

    def update_source(self, msg):
        source = getattr(msg, "data", None)
        if not isinstance(source, str):
            source = None
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            self._external_observed = True
            if source != self._source:
                self._epoch += 1
                self._confirmed_seen = None
                self._confirmation_required_since = now
            self._source, self._source_seen = source, now
            self._source_live_confirmed = True
            self._evaluate(now)

    def observe_cached_source(self, msg):
        """A newer cached transition revokes; it never supplies heartbeat credit."""
        source = getattr(msg, "data", None)
        if not isinstance(source, str):
            source = None
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            self._external_observed = True
            if source != self._source:
                self._source = source
                self._source_live_confirmed = False
                self._epoch += 1
                self._confirmed_seen = None
                self._confirmation_required_since = now
                self._evaluate(now)

    @staticmethod
    def _status_fields(msg):
        fields = tuple(getattr(msg, field, None) for field in (
            "requested_control_mode", "active_control_mode", "state", "transition_id"))
        if any(type(value) is not int for value in fields):
            fields = (0, 0, -1, 0)
        return fields

    def update_status(self, msg):
        """Live events revoke immediately but never renew cache confirmation."""
        fields = self._status_fields(msg)
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            self._external_observed = True
            self._status_version += 1
            if fields != self._status:
                if self._confirmed_seen is not None:
                    self._confirmation_required_since = self._confirmed_seen
                self._confirmed_seen = None
            self._status = fields
            self._evaluate(now)

    def confirmation_token(self):
        with self._lock:
            self._evaluate(time.monotonic())
            return self._epoch, self._status_version

    def confirm_status(self, token, msg):
        """Only the adapter's successful, ordered current-publisher query calls this.

        An unchanged cached event renews confirmation without an ownership change.
        Tokens cannot survive a source epoch, feedback gap, or intervening event.
        """
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            if (token != (self._epoch, self._status_version)
                    or not self._feedback_fresh or not self._source_live_confirmed):
                return False
            self._status = self._status_fields(msg)
            self._confirmed_seen = now
            self._confirmation_fault = None
            self._evaluate(now)
            self._initial_confirmation.set()
            return True

    def invalidate_confirmation(self, reason, *, reset_source=False):
        with self._lock:
            if reset_source:
                self._source_seen = None
                self._source_live_confirmed = False
            self._epoch += 1
            self._confirmed_seen = None
            self._confirmation_fault = reason
            self._evaluate(time.monotonic())

    def update_joints(self, group, msg):
        try:
            names, values = list(msg.name), list(msg.position)
            valid = (len(names) == len(values) and len(set(names)) == len(names)
                     and self._required_joints[group].issubset(names)
                     and all(math.isfinite(float(value)) for value in values))
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        with self._lock:
            now = time.monotonic()
            self._evaluate(now)
            self._joint_seen[group], self._joint_valid[group] = now, valid
            self._evaluate(now)

    def _check(self, revision):
        status = self._evaluate(time.monotonic())
        if status.state != "ready":
            raise OutputAuthorityError(status)
        if revision is None:
            raise OutputAuthorityError(status, "unarmed")
        if revision != status.revision:
            raise OutputAuthorityError(status, "ownership_changed")
        return status

    def arm(self, revision):
        with self._lock:
            status = self._check(revision)
            self._armed_revision = revision
            return status

    def check(self):
        with self._lock:
            return self._check(getattr(self._output, "revision", self._armed_revision))

    @contextmanager
    def output(self, *, allow_unarmed):
        previous = getattr(self._output, "revision", None)
        with self._lock:
            revision = previous if previous is not None else self._armed_revision
            if revision is None and allow_unarmed:
                revision = self._evaluate(time.monotonic()).revision
            self._check(revision)
            self._output.revision = revision
        try:
            yield
            self.check()
        finally:
            if previous is None:
                del self._output.revision
            else:
                self._output.revision = previous

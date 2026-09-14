"""Stage complete MLP snapshots; apply only at an inference request boundary."""

from copy import deepcopy
from hashlib import sha256
import threading

import torch

from .stage2 import stage2_spec_fingerprint


class RLTPolicyUpdate:
    def __init__(self, actor, spec):
        self._spec = stage2_spec_fingerprint(spec)
        self._dtype = next(actor.parameters()).dtype
        self._shapes = {name: value.shape for name, value in actor.state_dict().items()}
        self._lock = threading.Lock()
        self._candidate = None
        self._requested = None
        self._auto_apply = False
        self._training_version = 0
        self._inference_version = 0

    def publish(self, learner):
        """Called by the sole learner owner after a complete actor update.

        This copies weights, never exposes the mutable training actor, and
        deliberately does not change the inference actor or write checkpoints.
        """
        if learner.update_pending:
            raise RuntimeError("Finish the RLT update before publishing a policy")
        if stage2_spec_fingerprint(learner.spec) != self._spec:
            raise ValueError("RLT policy update spec differs from the loaded bundle")
        actor = deepcopy(learner.actor).to(device="cpu", dtype=self._dtype).eval().requires_grad_(False)
        state = actor.state_dict()
        if {name: value.shape for name, value in state.items()} != self._shapes:
            raise ValueError("RLT policy update architecture differs from the loaded actor")
        digest = sha256()
        for name, value in state.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError("RLT policy update contains non-finite weights")
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        with self._lock:
            self._training_version += 1
            self._candidate = (self._training_version, actor, digest.hexdigest())

    def set_auto_apply(self, enabled):
        if not isinstance(enabled, bool):
            raise TypeError("Auto Apply must be a boolean")
        with self._lock:
            self._auto_apply = enabled

    def request_apply(self):
        """Pin the candidate selected by the user, even if training continues."""
        with self._lock:
            if self._candidate is None or self._candidate[0] <= self._inference_version:
                raise RuntimeError("No new RLT policy is ready to apply")
            self._requested = self._candidate

    def apply_pending(self, current_actor):
        """The caller serializes this with the complete RLT inference request."""
        with self._lock:
            selected = self._requested or (self._candidate if self._auto_apply else None)
            if selected is None or selected[0] <= self._inference_version:
                return current_actor, None
            version, actor, fingerprint = selected
            target = next(current_actor.parameters())
            # Prepare a separate module before replacing any live reference.
            replacement = deepcopy(actor).to(device=target.device, dtype=target.dtype)
            replacement.eval().requires_grad_(False)
            self._inference_version = version
            self._requested = None
            return replacement, {
                "async_policy_version": version,
                "actor_state_sha256": fingerprint,
            }

    def status(self):
        with self._lock:
            return {
                "auto_apply": self._auto_apply,
                "training_version": self._training_version,
                "inference_version": self._inference_version,
                "pending_version": self._requested[0] if self._requested else None,
                "can_apply": self._training_version > self._inference_version,
            }

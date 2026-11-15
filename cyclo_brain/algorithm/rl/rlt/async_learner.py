"""Opt-in RLT updates between inference requests, without changing the losses.

The owner supplies a training-only learner and a batch sampler over completed
replay. Every inference request must enter ``inference()`` before GPU work.
This is cooperative scheduling, not preemption of an executing CUDA kernel.
Inference must use a separate actor snapshot, not this mutable learner.
Dataset preparation, checkpoints and snapshot deployment belong to the owner.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from copy import deepcopy
import threading

from .stage2 import RLTStage2Batch, RLTStage2Learner


class RLTAsyncLearner:
    def __init__(
        self,
        learner: RLTStage2Learner,
        sample_batch: Callable[[], RLTStage2Batch] | None,
        *,
        after_update: Callable | None = None,
    ) -> None:
        self.learner = learner
        self._sample_batch = sample_batch
        self._after_update = after_update
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._closed = False
        self._busy = False
        self._waiting_inference = 0
        self._waiting_background = 0
        self._training_phase: str | None = None
        self._error: str | None = None
        self._last_update: dict | None = None
        self._next_replay = None
        self._replay_info = {}
        self._replay_context = None
        self._trained_context = None
        self.replay_history = []
        self._max_updates = None
        self._starting_updates = 0
        self._completed_updates = 0
        self._limit_reached = False

    def replace_replay(self, sampler, info, *, checkpoint_context=None):
        """Queue a sampler swap after the complete current critic/actor update."""
        with self._condition:
            if self._closed:
                raise RuntimeError('Async RL learner is closed')
            self._next_replay = (sampler, dict(info), checkpoint_context)
            self._condition.notify_all()

    def checkpoint_snapshot(self):
        """Capture CPU state at a complete update boundary; disk I/O is external.

        Finishes an already-started update even when OFF, but never samples a
        new batch. The inference actor and the user's ON/OFF choice are intact.
        """
        with self.inference(background=True):
            if self._error:
                raise RuntimeError(f'Cannot save a failed learner: {self._error}')
            if self.learner.update_pending:
                try:
                    self._complete_update()
                except Exception as error:
                    with self._condition:
                        self._error = f'{type(error).__name__}: {error}'
                        self._enabled = False
                    raise
            if self._trained_context is None:
                raise RuntimeError('No Async RL updates to save yet')
            context = self._trained_context
            return {
                'learner': self.learner.state_dict(),
                'replay': context['replay'],  # immutable CPU features; no GPU copy
                'sampling_generator': context['generator'].get_state().clone(),
                'history': deepcopy(self.replay_history),
            }

    def _complete_update(self):
        update = self.learner.finish_update()
        if self.replay_history and update is not None:
            self.replay_history[-1]['ending_critic_updates'] = update.completed_critic_updates
        if update is not None:
            with self._condition:
                self._completed_updates = update.completed_critic_updates
                if self._max_updates is not None and (
                    self._completed_updates - self._starting_updates >= self._max_updates
                ):
                    self._limit_reached = True
                    self._enabled = False
        if self._after_update is not None:
            self._after_update(update)
        if update is not None:
            self._last_update = asdict(update)
        return update

    def set_enabled(self, enabled: bool, *, max_updates: int | None = None) -> None:
        """OFF pauses after the current phase; pending actor work is retained."""
        if not isinstance(enabled, bool):
            raise TypeError("Async RL enabled must be a boolean")
        if max_updates is not None and (
            isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1
        ):
            raise ValueError('Async RL max_updates must be a positive integer')
        with self._condition:
            if self._closed:
                raise RuntimeError("Async RL learner is closed")
            if enabled and self._error is not None:
                raise RuntimeError(f"Async RL learner failed: {self._error}")
            if enabled and not self._enabled:
                # A paused critic step is counted when its actor/target phase
                # finishes, not when the request happens between phases.
                self._starting_updates = (int(self.learner.completed_critic_updates)
                    - int(self.learner.update_pending)) if max_updates is not None else 0
                self._completed_updates = self._starting_updates
                self._max_updates = max_updates
                self._limit_reached = False
            self._enabled = enabled
            if enabled and self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="rlt-async-learner", daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    @contextmanager
    def inference(self, *, background=False):
        """Reserve a slot: inference, then background work, then training."""
        with self._condition:
            if background:
                self._waiting_background += 1
            else:
                self._waiting_inference += 1
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: not self._busy and (
                    not background or not self._waiting_inference))
                self._busy = True
            finally:
                if background:
                    self._waiting_background -= 1
                else:
                    self._waiting_inference -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()

    def status(self) -> dict:
        with self._condition:
            return {
                "enabled": self._enabled,
                "closed": self._closed,
                "training_phase": self._training_phase,
                "waiting_inference": self._waiting_inference,
                "error": self._error,
                "last_update": dict(self._last_update) if self._last_update else None,
                "replay": dict(self._replay_info),
                "waiting_data": self._sample_batch is None,
                "max_updates": self._max_updates,
                "updates_this_run": self._completed_updates - self._starting_updates,
                "limit_reached": self._limit_reached,
            }

    def close(self) -> None:
        """Join the current phase, without discarding the caller-owned learner.

        A paused partial update remains on that learner; its owner must finish
        it before checkpointing. Inference itself remains available after close.
        """
        with self._condition:
            self._enabled = False
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    not self._busy and not self._waiting_inference
                    and not self._waiting_background and (
                        (self._next_replay is not None and not self.learner.update_pending)
                        or (self._enabled and (self.learner.update_pending or self._sample_batch is not None))
                    )
                ))
                if self._closed:
                    return
                if self._next_replay is not None and not self.learner.update_pending:
                    self._sample_batch, self._replay_info, self._replay_context = self._next_replay
                    self._next_replay = None
                    self._condition.notify_all()
                    continue
                self._busy = True
                self._training_phase = (
                    "actor_target" if self.learner.update_pending else "critic"
                )
            update = None
            error = None
            try:
                if self.learner.update_pending:
                    update = self._complete_update()
                else:
                    if self._replay_context is not None and self._replay_context is not self._trained_context:
                        self._trained_context = self._replay_context
                        self.replay_history.append({
                            **self._replay_info,
                            'sampling_seed': self._trained_context['sampling_seed'],
                            'dataset_snapshot_fingerprint': self._trained_context['replay'].metadata['dataset_snapshot_fingerprint'],
                            'starting_critic_updates': self.learner.completed_critic_updates,
                            'ending_critic_updates': self.learner.completed_critic_updates,
                        })
                    self.learner.update_critic(self._sample_batch())
                # The split learner reads its metrics back to the CPU before
                # returning, so preceding device work has completed here.
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
            finally:
                with self._condition:
                    if update is not None:
                        self._last_update = asdict(update)
                    if error is not None:
                        self._error = error
                        self._enabled = False
                    self._training_phase = None
                    self._busy = False
                    self._condition.notify_all()

"""15 Hz TT-RTC execution with committed prefixes and finite response deadlines.

The queue is sampled before publishing the current policy tick. Its remaining
actions are the exact prefix supplied to the model. They are never replayed or
replaced by the response; only 24 new suffix actions are appended. A response
that misses the committed prefix boundary is discarded, then an empty-prefix
request recovers from the current observation while the robot holds its tail.
"""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np


class TtrtcExecution:
    HORIZON = 32
    ACTION_DIM = 16
    POSTFIX_STEPS = 24

    def __init__(self, requester, lock, trace, policy_hz=15., prefix_steps=7,
                 clock=time.monotonic):
        if not math.isfinite(policy_hz) or policy_hz != 15.:
            raise ValueError('TT-RTC requires the trained 15 Hz action rate')
        if (isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int)
                or not 0 <= prefix_steps <= 8):
            raise ValueError('TT-RTC prefix steps must be an integer from 0 to 8')
        self.requester, self.lock, self.trace, self.clock = requester, lock, trace, clock
        self.prefix_steps = prefix_steps
        self.output_hz = policy_hz
        self.output_dt = 1 / policy_hz
        self.thread = None
        self.epoch = 0
        try:
            self.clock_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            self.clock_id = ''
        self.reset()

    @property
    def buffer_size(self):
        return len(self.actions)

    def reset(self):
        # Keep the live thread reference: a canceled engine call must finish
        # before another request can enter the engine's single-flight gate.
        self.epoch += 1
        self.actions = deque()
        self.next_request = -math.inf
        self.last_tick = None

    def tick(self, now, task, keys, generation):
        """Called under the control-loop lock, once per 15 Hz policy tick."""
        if self.last_tick is not None and now-self.last_tick > 2*self.output_dt:
            # A delayed control loop cannot catch up by replaying a past prefix
            # rapidly. Drop the old timeline and observe again from a hold.
            self.trace.emit('ttrtc_timeline_reset', generation=generation,
                            reason='control_tick_gap', gap_s=now-self.last_tick)
            self.reset()
        self.last_tick = now
        if (len(self.actions) <= self.prefix_steps and now >= self.next_request
                and (self.thread is None or not self.thread.is_alive())):
            prefix = np.asarray(list(self.actions), dtype=np.float64).reshape(-1, self.ACTION_DIM)
            self.thread = threading.Thread(
                target=self._request,
                args=(task, prefix, now, self.epoch, generation), daemon=True,
            )
            self.thread.start()
        return self.actions.popleft() if self.actions else None

    def _request(self, task, prefix, dispatched, epoch, generation):
        request_id = uuid.uuid4().hex
        d = len(prefix)
        deadline = dispatched + d * self.output_dt if d else None
        self.trace.emit('request_started', request_id=request_id, generation=generation,
                        mode='ttrtc', prefix_steps=d, postfix_steps=self.POSTFIX_STEPS,
                        prefix_deadline_monotonic_s=deadline)
        success, reason = False, None
        try:
            with self.lock:
                if epoch != self.epoch:
                    return
            response = self.requester.get_action(task, ttrtc_prefix_actions=prefix)
            if not response.success:
                raise ValueError(response.message or 'TT-RTC inference failed')
            if response.chunk_size != self.HORIZON or response.action_dim != self.ACTION_DIM:
                raise ValueError('TT-RTC response must have logical shape [32, 16]')
            chunk = np.asarray(response.action_list, dtype=np.float64).reshape(self.HORIZON, self.ACTION_DIM)
            if not np.isfinite(chunk).all():
                raise ValueError('TT-RTC response contains non-finite actions')
            if not np.array_equal(chunk[:d], prefix):
                raise ValueError('TT-RTC response changed the committed action prefix')
            timing = json.loads(getattr(response, 'timing_json', '') or '{}')
            if not isinstance(timing, dict):
                raise ValueError('Invalid TT-RTC inference timing metadata')
            snapshot = timing.get('sensor_snapshot_monotonic_s')
            # Receive timestamps are not hardware acquisition time. Only
            # compare local monotonic clocks on the same Linux boot; ordinary
            # sensor-age validation remains owned by the Engine sensor reader.
            if (d and self.clock_id and timing.get('clock_id') == self.clock_id
                    and isinstance(snapshot, (int, float)) and not isinstance(snapshot, bool)
                    and math.isfinite(snapshot) and snapshot > dispatched+self.output_dt):
                raise ValueError('TT-RTC observation was captured after the prefix start tick')
            with self.lock:
                if epoch != self.epoch:
                    reason = 'stale_generation'
                    return
                # Evaluate ready time inside the publishing lock. A response
                # waiting for this lock has not yet met the handoff deadline.
                ready = self.clock()
                if deadline is not None and ready >= deadline:
                    reason = 'prefix_deadline'
                    self.trace.emit('ttrtc_chunk_rejected', request_id=request_id,
                                    generation=generation, reason=reason,
                                    prefix_steps=d, lateness_s=ready-deadline)
                    self.next_request = ready + self.output_dt
                    return
                self.actions.extend(action.copy() for action in chunk[d:d+self.POSTFIX_STEPS])
                success = True
                self.trace.emit('ttrtc_chunk_installed', request_id=request_id,
                                generation=generation, prefix_steps=d,
                                postfix_steps=self.POSTFIX_STEPS, queued_steps=len(self.actions),
                                request_round_trip_s=ready-dispatched, timing=timing)
        except Exception as exc:
            reason = str(exc)
            with self.lock:
                if epoch == self.epoch:
                    self.next_request = self.clock() + .1
                    self.trace.emit('ttrtc_chunk_rejected', request_id=request_id,
                                    generation=generation, reason=reason)
        finally:
            self.trace.emit('request_completed', request_id=request_id, generation=generation,
                            mode='ttrtc', prefix_steps=d, success=success, reason=reason,
                            latency_s=self.clock()-dispatched)

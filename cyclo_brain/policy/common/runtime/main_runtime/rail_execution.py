"""Opt-in ordinary-Async VLA-RAIL executor."""
import json
import math
import os
import threading
import time
import uuid
import numpy as np
from .rail_trajectory import CubicChunk, RailTrajectory, alignment_offset


class RailExecution:
    def __init__(self, requester, lock, trace, policy_hz=15., clock=time.monotonic,
                 blend_s=.2, alignment_window_s=.1):
        if not all(math.isfinite(v) and v > 0 for v in (policy_hz, blend_s, alignment_window_s)):
            raise ValueError('VLA-RAIL timing must be positive and finite')
        self.requester, self.lock, self.trace, self.clock = requester, lock, trace, clock
        self.dt, self.blend, self.alignment_window = 1/policy_hz, blend_s, alignment_window_s
        self.thread = None
        self.epoch = 0
        self.capture_actions = os.getenv('VLA_RAIL_CAPTURE_ACTIONS', '0') == '1'
        self.reset()

    def reset(self, policy_hz=None):
        self.epoch += 1
        if policy_hz is not None:
            self.dt = 1/policy_hz
        self.trajectory = None
        self.pending = None
        self.next_request = -math.inf
        self.latency = .1
        self.last_request_id = None

    def tick(self, now, task, keys, generation):
        if self.pending is not None:
            chunk, snapshot, request_id, timing = self.pending
            self.pending = None
            prior = self.trajectory.state(now) if self.trajectory else None
            offset = (alignment_offset(chunk, now-snapshot, prior, self.alignment_window, self.blend)
                      if prior is not None else 0.)
            if offset is None:
                self.trace.emit('rail_chunk_rejected', reason='expired horizon', request_id=request_id,
                                generation=generation, age_s=now-snapshot)
                self.next_request = now + .05
            else:
                self.trajectory = RailTrajectory(chunk, now, offset, prior, self.blend)
                self.last_request_id = request_id
                horizon = self.trajectory.duration
                self.next_request = now + max(.02, min(horizon*.5, horizon-self.latency-self.blend))
                self.trace.emit('rail_chunk_installed', request_id=request_id, generation=generation,
                                mode='vla_rail', offset_s=offset, blend_s=self.trajectory.blend,
                                policy_dt_s=self.dt, horizon_s=horizon, timing=timing)
        if (self.thread is None or not self.thread.is_alive()) and now >= self.next_request:
            self.next_request = now+.1
            epoch, dt = self.epoch, self.dt
            self.thread = threading.Thread(target=self._request,
                args=(task, list(keys), generation, epoch, dt), daemon=True)
            self.thread.start()
        if self.trajectory is None:
            return None
        return self.trajectory.action(now)


    def _request(self, task, keys, generation, epoch, dt):
        started = self.clock()
        request_id = uuid.uuid4().hex
        self.trace.emit('request_started', request_id=request_id, generation=generation, mode='vla_rail')
        success, timing, error = False, {}, None
        try:
            response = self.requester.get_action(task)  # ordinary inference only
            arrival = self.clock()
            timing = json.loads(getattr(response, 'timing_json', '') or '{}')
            if not isinstance(timing, dict):
                timing = {}
            if not response.success:
                raise ValueError(response.message or 'VLA-RAIL inference failed')
            actions = np.asarray(response.action_list, dtype=float).reshape(response.chunk_size, response.action_dim)
            chunk = CubicChunk(actions, dt, keys)  # Fit outside the publishing lock.
            snapshot = timing.get('sensor_snapshot_monotonic_s')
            if (isinstance(snapshot, bool) or not isinstance(snapshot, (int, float))
                    or not math.isfinite(snapshot) or not started <= snapshot <= arrival):
                snapshot = started
                timing['rail_alignment_reference'] = 'request_dispatch_fallback'
            else:
                timing['rail_alignment_reference'] = 'cache_snapshot_not_sensor_acquisition'
            with self.lock:
                if epoch != self.epoch:
                    return
                self.latency = .8*self.latency+.2*(arrival-started)
                self.pending = (chunk, snapshot, request_id, timing)
            success = True
        except Exception as exc:
            error = str(exc)
            with self.lock:
                if epoch == self.epoch:
                    self.next_request = self.clock()+.1
        finally:
            self.trace.emit('request_completed', request_id=request_id, generation=generation,
                            mode='vla_rail', success=success, latency_s=self.clock()-started,
                            timing=timing, error=error, status='ok' if success else 'rejected_or_error')

    def record_command(self, now, generation, *, published_at=None, action=None):
        if self.capture_actions and action is not None:
            self.trace.emit('command_published', mode='vla_rail', generation=generation,
                            request_id=self.last_request_id, action=np.asarray(action).tolist(),
                            action_sample_monotonic_s=now, published_monotonic_s=published_at)

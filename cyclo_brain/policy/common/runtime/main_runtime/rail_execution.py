"""Opt-in ordinary-Async VLA-RAIL executor."""
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from .rail_trajectory import CubicChunk
from .rail_handoff import select_handoff
from .rail_output_guard import RailOutputGuard


class RailExecution:
    def __init__(self, requester, lock, trace, policy_hz=15., clock=time.monotonic,
                 blend_s=.2, interpolation_hz=None, tcp_positions=None):
        if not all(math.isfinite(v) and v > 0 for v in
                   (policy_hz, blend_s, policy_hz if interpolation_hz is None else interpolation_hz)):
            raise ValueError('VLA-RAIL timing must be positive and finite')
        self.requester, self.lock, self.trace, self.clock = requester, lock, trace, clock
        self.blend = blend_s
        self.tcp_positions = tcp_positions
        self._configured_interpolation_hz = interpolation_hz
        self.thread = None
        self.epoch = 0
        self.capture_actions = os.getenv('VLA_RAIL_CAPTURE_ACTIONS', '0') == '1'
        try:
            self.clock_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            self.clock_id = ''
        self.reset(policy_hz)

    def reset(self, policy_hz=None):
        self.epoch += 1
        if policy_hz is not None:
            if not math.isfinite(policy_hz) or policy_hz <= 0:
                raise ValueError('VLA-RAIL policy frequency must be positive and finite')
            self.dt = 1/policy_hz
            # RAIL publishes at the original policy rate, like ordinary Async.
            # Changing generic control_hz must not change its sampling or speed.
            self.output_hz = policy_hz
            self.output_dt = self.dt
            self.interpolation_hz = (policy_hz if self._configured_interpolation_hz is None
                                     else self._configured_interpolation_hz)
            self.playback_rate = self.output_hz/self.interpolation_hz
            self.output_guard = RailOutputGuard(max_interval_s=max(.1, 3*self.output_dt))
        self.output_guard.reset()
        self.trajectory = None
        self.pending = None
        self.next_request = -math.inf
        self.last_publish_time = None
        self.last_request_id = None
        self.consecutive_rejections = 0
        self.response_budget_s = 0.

    def tick(self, now, task, keys, generation, *, enforce_envelope=True, initial_positions=None):
        # Evaluate an available response on the next policy tick. Only accepted
        # bridges replace the reference; rejection keeps its finite tail/hold.
        if self.pending is not None:
            chunk, snapshot, request_id, timing = self.pending
            self.pending = None
            prior = self.trajectory.state(now) if self.trajectory else None
            boundary_source = 'reference'
            if self.output_guard.positions is not None and self.trajectory is not None:
                # Arm commands follow this reference without a second
                # regulator. Evaluate the very same execution clock at handoff.
                boundary_source = 'executed_reference'
            elif initial_positions is not None:
                initial = np.asarray(initial_positions, dtype=float)
                if initial.shape != (chunk.action_dim,) or not np.isfinite(initial).all():
                    raise ValueError('Fresh current joint positions required for RAIL transition')
                q = initial[chunk.arm_columns]
                prior = (q, np.zeros_like(q), np.zeros_like(q))
                boundary_source = 'joint_feedback'
            has_previous_chunk = self.trajectory is not None
            observation_age = max(0., now-snapshot)
            # Observation age determines freshness. TCP alignment separately
            # locates current progress on the incoming path; an earlier policy
            # phase can become a new future command after a continuous bridge.
            nominal_offset = observation_age*self.playback_rate
            # Reserve execution coverage for the next response and its finite
            # stop segment. A late response must not require two fixed blends
            # inside a very short wall-clock remainder. Re-time that remainder
            # without extrapolating the prediction beyond its end.
            # Expired warmup responses do not inflate the running estimate.
            round_trip = timing.get('request_round_trip_s', observation_age)
            if (not isinstance(round_trip, (int, float)) or isinstance(round_trip, bool)
                    or not math.isfinite(round_trip) or round_trip < 0):
                round_trip = observation_age
            if nominal_offset < chunk.duration:
                self.response_budget_s = max(float(round_trip), .9*self.response_budget_s)
            minimum_horizon_s = max(2*self.blend,
                                    self.response_budget_s+self.blend+2*self.output_dt)
            if has_previous_chunk and initial_positions is not None:
                feedback = np.asarray(initial_positions, dtype=float)
                if feedback.shape == (chunk.action_dim,) and np.isfinite(feedback).all():
                    error = prior[0]-feedback[chunk.arm_columns]
                    speed_squared = float(np.dot(prior[1], prior[1]))
                    self.trace.emit('rail_tracking', request_id=request_id, generation=generation,
                                    sample_monotonic_s=now,
                                    reference_error_max_rad=float(np.max(np.abs(error))),
                                    reference_error_rms_rad=float(np.sqrt(np.mean(error**2))),
                                    # A local projection, not a measured transport delay.
                                    projected_reference_lead_s=(float(np.dot(error, prior[1])/speed_squared)
                                                                if speed_squared > 1e-12 else None),
                                    positions=feedback[chunk.arm_columns].tolist()
                                              if self.capture_actions else None)
            planning_started = time.perf_counter()
            decision = select_handoff(chunk, now, observation_age, prior,
                                      previous_trajectory=self.trajectory, blend=self.blend,
                                      playback_rate=self.playback_rate,
                                      tcp_positions=self.tcp_positions,
                                      minimum_horizon_s=minimum_horizon_s)
            planning_s = time.perf_counter()-planning_started
            self.trace.emit('rail_handoff_decision', request_id=request_id, generation=generation,
                            accepted=decision.trajectory is not None, reason=decision.reason,
                            planning_s=planning_s, response_budget_s=self.response_budget_s,
                            minimum_horizon_s=minimum_horizon_s, diagnostics=decision.diagnostics)
            if decision.trajectory is None:
                self.consecutive_rejections += 1
                self.trace.emit('rail_chunk_rejected', reason=decision.reason, request_id=request_id,
                                generation=generation, age_s=observation_age,
                                consecutive_rejections=self.consecutive_rejections,
                                fallback='previous_reference_then_terminal_hold',
                                remaining_reference_s=(max(0., self.trajectory.end-now)
                                                       if self.trajectory is not None else 0.))
                self.next_request = now
            else:
                self.consecutive_rejections = 0
                self.trajectory = decision.trajectory
                offset = self.trajectory.offset
                self.last_request_id = request_id
                if self.capture_actions:
                    self.trace.emit('rail_transition', request_id=request_id, generation=generation,
                                    start_monotonic_s=now, offset_s=offset,
                                    boundary_source=boundary_source,
                                    previous_state=([np.asarray(x).tolist() for x in prior]
                                                    if prior is not None else None),
                                    blend_s=self.trajectory.blend,
                                    tail_blend_s=self.trajectory.tail_blend,
                                    duration_s=self.trajectory.duration,
                                    policy_remaining_s=self.trajectory.policy_remaining_s,
                                    time_scale=self.trajectory.time_scale,
                                    gripper_offset_s=self.trajectory.gripper_offset,
                                    gripper_time_scale=self.trajectory.gripper_time_scale,
                                    head_mode=getattr(self.trajectory, 'head_mode', 'dual'),
                                    handoff_diagnostics=decision.diagnostics,
                                    lower=self.trajectory.lower.tolist(), upper=self.trajectory.upper.tolist())
                horizon = self.trajectory.duration
                self.next_request = now
                self.trace.emit('rail_chunk_installed', request_id=request_id, generation=generation,
                                mode='vla_rail', offset_s=offset, blend_s=self.trajectory.blend,
                                tail_blend_s=self.trajectory.tail_blend,
                                policy_dt_s=self.dt, horizon_s=horizon, timing=timing,
                                policy_remaining_s=self.trajectory.policy_remaining_s,
                                time_scale=self.trajectory.time_scale,
                                gripper_offset_s=self.trajectory.gripper_offset,
                                gripper_time_scale=self.trajectory.gripper_time_scale,
                                gripper_alignment=decision.diagnostics.get('gripper_alignment'),
                                observation_age_s=observation_age,
                                nominal_policy_offset_s=nominal_offset,
                                alignment_shift_s=offset-nominal_offset,
                                interpolation_hz=self.interpolation_hz,
                                output_hz=self.output_hz,
                                configured_playback_rate=self.playback_rate,
                                response_budget_s=self.response_budget_s,
                                minimum_horizon_s=minimum_horizon_s,
                                algorithm=('nearest_tcp_handoff_v3'
                                           if decision.diagnostics.get('planner') == 'tcp_nearest_handoff_v3'
                                           else 'cartesian_bridge_admission_v2'),
                                alignment_method=decision.diagnostics.get('alignment_method'),
                                nearest_policy_phase_s=decision.diagnostics.get('nearest_policy_phase_s'),
                                selected_alignment_rank=decision.diagnostics.get('selected_alignment_rank'),
                                alignment_fallback_used=decision.diagnostics.get('alignment_fallback_used', False),
                                alignment_fallback_reason=decision.diagnostics.get('alignment_fallback_reason'),
                                head_mode=getattr(self.trajectory, 'head_mode', 'dual'),
                                planning_s=planning_s,
                                next_request_delay_s=self.next_request-now,
                                position_mismatch_rad=(float(np.max(np.abs(chunk.state(offset)[0]-prior[0])))
                                                       if prior is not None else 0.))
        if (self.pending is None and (self.thread is None or not self.thread.is_alive())
                and now >= self.next_request):
            # One request at a time, without a scheduled pause after success.
            # Failures below retain a small retry backoff.
            self.next_request = now
            epoch, dt = self.epoch, self.dt
            self.thread = threading.Thread(target=self._request,
                args=(task, list(keys), generation, epoch, dt), daemon=True)
            self.thread.start()
        if self.trajectory is None:
            return None
        if self.capture_actions:
            q, v, a = self.trajectory.state(now)
            self.trace.emit('rail_sample', request_id=self.last_request_id, generation=generation,
                            sample_monotonic_s=now, positions=q.tolist(),
                            velocities=v.tolist(), accelerations=a.tolist())
        return self.trajectory.action(now, enforce_envelope=enforce_envelope)


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
            if self.capture_actions:
                self.trace.emit('rail_raw_chunk', request_id=request_id, generation=generation,
                                action_keys=list(keys), actions=actions.tolist(), policy_dt_s=dt,
                                request_monotonic_s=started, arrival_monotonic_s=arrival,
                                timing=dict(timing))
            snapshot = timing.get('sensor_snapshot_monotonic_s')
            received_source = timing.get('timestamp_source') == 'local_monotonic_receive'
            same_clock = bool(self.clock_id) and timing.get('clock_id') == self.clock_id
            received_snapshot = received_source and same_clock
            earliest = 0. if received_snapshot else started
            if (isinstance(snapshot, bool) or not isinstance(snapshot, (int, float))
                    or not math.isfinite(snapshot) or not earliest <= snapshot <= arrival
                    or (received_source and not same_clock)):
                snapshot = started
                timing['rail_alignment_reference'] = ('clock_id_mismatch_dispatch_fallback'
                                                      if received_source and not same_clock
                                                      else 'request_dispatch_fallback')
            else:
                timing['rail_alignment_reference'] = ('sensor_receive_not_sensor_acquisition'
                                                      if received_snapshot else 'cache_snapshot_not_sensor_acquisition')
            with self.lock:
                if epoch != self.epoch:
                    return
                measured_latency = self.clock()-started
                timing['request_round_trip_s'] = arrival-started
                timing['request_ready_latency_s'] = measured_latency
                timing['chunk_fit_duration_s'] = self.clock()-arrival
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

    def prepare_action(self, action, now, generation, initial_positions=None):
        output = self.output_guard.update(action, now, initial_positions)
        if self.trajectory is not None:
            # The arm reference starts at fresh feedback and stays C2 across
            # handoffs. Preserve the already sampled command without resampling.
            columns = self.trajectory.chunk.arm_columns
            output[columns] = np.asarray(action)[columns]
            self.output_guard.positions[columns] = output[columns]
        if self.capture_actions:
            self.trace.emit('rail_output_sample', request_id=self.last_request_id,
                            generation=generation, sample_monotonic_s=now,
                            target=np.asarray(action).tolist(), action=output.tolist())
        return output

    def command_window(self, now, action):
        """Sample a short q/v/a trajectory, including polynomial boundaries.

        Grippers keep the current discrete command throughout this window so
        controller interpolation cannot anticipate the next grasp event.
        """
        trajectory = self.trajectory
        horizon = 3*self.output_dt
        times = list(np.arange(4)*self.output_dt)
        # Global cubic has no interior knots. Each dual-quintic has a midpoint
        # where the jerk may change; include it to reproduce both polynomials.
        boundaries = [trajectory.start_time+trajectory.blend/2,
                      trajectory.start_time+trajectory.blend,
                      trajectory.start_time+trajectory.tail_start,
                      trajectory.start_time+trajectory.tail_start+trajectory.tail_blend/2,
                      trajectory.end]
        times.extend(t-now for t in boundaries if now < t < now+horizon)
        # Avoid near-zero Hermite intervals: rounding position samples taken
        # from a large monotonic timestamp can amplify acceleration enormously
        # when a knot is only nanoseconds away. C2 permits merging those knots.
        offsets_ns = np.unique(np.round(np.asarray(times)*1e9).astype(np.int64))
        min_segment_ns = max(1, round(min(1e-4, self.output_dt*.01)*1e9))
        merged = [offsets_ns[0]]
        for offset in offsets_ns[1:-1]:
            if offset <= merged[-1]:
                continue
            if offset-merged[-1] >= min_segment_ns and offsets_ns[-1]-offset >= min_segment_ns:
                merged.append(offset)
            elif offsets_ns[-1]-merged[-1] >= 2*min_segment_ns:
                # Confine a merged polynomial boundary to a short interval.
                # At policy-rate output the next regular point can be tens of
                # milliseconds away; spanning it would magnify the merge error.
                merged.append(merged[-1]+min_segment_ns
                              if offset-merged[-1] < min_segment_ns
                              else offsets_ns[-1]-min_segment_ns)
        merged.append(offsets_ns[-1])
        offsets = np.asarray(merged)/1e9
        positions = np.tile(action, (len(offsets), 1))
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)
        columns = trajectory.chunk.arm_columns
        for index, offset in enumerate(offsets):
            q, v, a = trajectory.state(now+offset)
            positions[index, columns] = q
            velocities[index, columns] = v
            accelerations[index, columns] = a
        return positions, velocities, accelerations, offsets

    def record_command(self, now, generation, *, published_at=None, action=None):
        interval = None if self.last_publish_time is None else published_at-self.last_publish_time
        self.last_publish_time = published_at
        if self.capture_actions and action is not None:
            self.trace.emit('command_published', mode='vla_rail', generation=generation,
                            request_id=self.last_request_id, action=np.asarray(action).tolist(),
                            action_sample_monotonic_s=now, published_monotonic_s=published_at,
                            sample_to_publish_s=published_at-now, publish_interval_s=interval,
                            trajectory_window_s=3*self.output_dt, output_hz=self.output_hz)

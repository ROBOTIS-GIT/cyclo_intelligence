"""Keep VLA execution unchanged; retime only queued MLP destinations.

Route ownership travels with source targets, not the latest inference response.
At VLA->MLP, finish bounded braking before beginning a rest-to-rest common path.
"""
from collections import deque
import numpy as np
from .common_time_timeline import CommonTimeTimeline
from .tt_rtc_timeline import TTActionTimeline
from .arm_command_smoothing import ArmCommandSmoother, ArmSmoothingConfig


class PolicyTimeTimeline(CommonTimeTimeline):
    def __init__(self, base_config=None, **kwargs):
        super().__init__(**kwargs)
        self._policies=deque()
        self.executing_policy=None
        self._brake_target=None
        self._base_smoother=ArmCommandSmoother(
            self.indices,config=base_config or ArmSmoothingConfig(
                max_velocity_rad_s=self.limits[0],max_acceleration_rad_s2=self.limits[1],
                max_jerk_rad_s3=self.limits[2]),lower=self.lower,upper=self.upper)

    def push_actions(self, actions, scheduled_start_delay_s=None, align=True, *, action_policy_mode='base'):
        if action_policy_mode not in {'base','rlt'}:
            raise ValueError('unknown queued action policy')
        with self._lock:
            count=super().push_actions(actions,scheduled_start_delay_s,align)
            self._policies.extend([action_policy_mode]*count)
            return count

    def replace_queued_actions(self, expected_actions, replacement_actions):
        # Only a prefix-preserving tail trim is supported in the no-overlap mode.
        expected=np.asarray(expected_actions);replacement=np.asarray(replacement_actions)
        if len(replacement)>len(expected) or not np.array_equal(replacement,expected[:len(replacement)]):
            raise ValueError('policy timeline only permits prefix-preserving tail trims')
        with self._lock:
            count=super().replace_queued_actions(expected,replacement)
            self._policies=deque(list(self._policies)[:count])
            return count

    def _base_output(self, target, dt):
        if not self._base_smoother.initialized:
            self._base_smoother.reset(self._last_output_action)
        output=self._base_smoother.step(target,dt)
        self._last_output_action=output.copy()
        return output

    def pop_action(self, elapsed_s=None):
        dt=1./self._control_hz if elapsed_s is None else float(elapsed_s)
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('control interval must be finite and positive')
        dt=min(dt,1./self._control_hz)
        with self._lock:
            if self._seed_pending:
                return super().pop_action(elapsed_s=dt)
            if not self._buffer:
                if self.executing_policy=='base' and self._base_smoother.initialized:
                    if (np.max(abs(self._base_smoother.velocity))>1e-8
                            or np.max(abs(self._base_smoother.acceleration))>1e-8
                            or np.max(abs(self._last_output_action[self.indices]-self._anchor[self.indices]))>1e-8):
                        return self._base_output(self._anchor,dt)
                return None
            policy=self._policies[0]
            if policy=='rlt' and self.executing_policy in {'base','braking'}:
                if self._brake_target is None:
                    self._brake_target=self._last_output_action.copy()
                self.executing_policy='braking'
                output=self._base_output(self._brake_target,dt)
                if (np.max(abs(self._base_smoother.velocity))<1e-8
                        and np.max(abs(self._base_smoother.acceleration))<1e-8
                        and np.max(abs(output[self.indices]-self._brake_target[self.indices]))<1e-8):
                    self._anchor=output.copy()
                    self._elapsed=self._duration=0.
                    self._phase=0
                    self._brake_target=None
                    self.executing_policy='rlt'
                return output
            if policy=='base':
                if self.executing_policy!='base':
                    self._base_smoother.clear()
                    self._phase=0
                    if self._last_output_action is not None:
                        self._anchor=self._last_output_action.copy()
                    self._elapsed=self._duration=0.
                self.executing_policy='base'
                # Hide later MLP targets from the original VLA resampler so it
                # cannot start interpolating toward an MLP target prematurely.
                leading=0
                for owner in self._policies:
                    if owner!='base':break
                    leading+=1
                tail=list(self._buffer)[leading:]
                self._buffer=deque(list(self._buffer)[:leading])
                last_command=None if self._last_output_action is None else self._last_output_action.copy()
                try:
                    raw=TTActionTimeline.pop_action(self)
                    consumed=leading-len(self._buffer)
                finally:
                    self._buffer.extend(tail)
                for _ in range(consumed):self._policies.popleft()
                if last_command is not None:self._last_output_action=last_command
                if raw is None:return None
                return self._base_output(raw,dt)
            self.executing_policy='rlt'
            before=len(self._buffer)
            output=super().pop_action(elapsed_s=dt)
            for _ in range(before-len(self._buffer)):self._policies.popleft()
            return output

    def clear(self):
        with self._lock:
            super().clear()
            self._policies.clear()
            self._base_smoother.clear()
            self._brake_target=None
            self.executing_policy=None

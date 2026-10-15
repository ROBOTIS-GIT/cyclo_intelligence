"""End one child mission without stopping the autonomy session or its reasoner."""


class ChildMissionCancelled(RuntimeError):
    """A validated mission-supervision decision, not a global Stop or a fault."""


def cancel_child_mission(controller, selected, reason):
    c = controller
    # SessionActions has drained its worker before we can replace the runtime.
    # cancel() latches a runtime permanently, so use a fresh ROS executor for return.
    with c.control_lock:
        c._check(c.state['map_name'])
        c.transition(phase='cancelling', control_status='cancelling', runner={**c.state.get('runner', {}), 'phase': 'cancelling'})
        c.event('Cancelling mission', f"{selected['mission_name']} · {reason}")
        c.runtime.cancel()  # Require actuator acknowledgement, including admission-only cancellation.
        if c.action_thread and c.action_thread.is_alive():
            raise RuntimeError('Mission worker has not stopped; patrol return blocked')
        c.runtime.close()
        c._check(c.state['map_name'])
        c.runtime = c.runtime_factory()
        c.runtime.open()
        c._check(c.state['map_name'])
        with c.lock:
            causes = {k: v for k, v in c.state['pause_causes'].items() if k != 'ai'}
        if causes:
            c.runtime.pause()  # A model cancellation must never release an operator hold.
        with c.lock:
            c.transition(mode='observation', phase='returning', mission=None, mission_route=[],
                         runner={}, result=None, mission_dispatched=False,
                         paused=bool(causes), control_status='paused' if causes else 'running', pause_causes=causes,
                         pause_reason='; '.join(causes.values()) or None,
                         observation_context={'phase': 'returning', 'at_spot_id': None,
                             'from_spot_id': None, 'target_spot_id': c.state['resume_checkpoint']['return_spot_id']},
                         recent_missions=(c.state['recent_missions'] + [
                             {'mission_id': selected['mission_id'], 'status': 'cancelled'}])[-20:])
        c.event('Mission cancelled', f"{selected['mission_name']} · {reason}")

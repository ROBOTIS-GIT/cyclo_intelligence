"""Episode roles and capture eligibility shared by recording and training."""
if __package__:
    from .mission_decisions import is_returning, recorded_catalog, task_for, validate_target
else:
    from mission_decisions import is_returning, recorded_catalog, task_for, validate_target

EPISODE_TASKS = {
    'observation': 'mission_selection',
    'patrol': 'patrol_supervision',
    'mission': 'mission_supervision',
}
CAPTURE_CONTRACT = 'task_aligned_v1'


def episode_type_for(episode):
    kind = episode.get('episode_type')
    if kind in EPISODE_TASKS:
        return kind
    # Only pre-contract episodes use this compatibility rule. Never infer a frame task.
    return 'mission' if episode.get('recording_mission_name') not in (None, '', 'ObservationMode') else 'observation'


def task_for_episode(episode):
    return EPISODE_TASKS[episode_type_for(episode)]


def validate_episode_frame(episode, frame):
    """Raise on mixed roles or missing execution evidence; do not relabel the frame."""
    context = frame.get('context') or {}
    kind = episode_type_for(episode)
    if context.get('runtime_source') == 'unverified':
        raise ValueError('Capture-time execution context could not be matched. Record a new frame after the current state is available.')
    if task_for(context) != task_for_episode(episode):
        raise ValueError(f'This {kind.title()} episode cannot capture the current reasoning task. Wait for its execution state or start a matching episode.')
    active = context.get('active_mission')
    observation = context.get('observation') or {}
    if kind == 'mission':
        if is_returning(context):
            raise ValueError('Return movement belongs to Patrol supervision, not the completed mission.')
        if (not isinstance(active, dict) or not active.get('mission_id')
                or active.get('status') not in ('running', 'paused')):
            raise ValueError('Wait for the selected mission to start; running or paused mission context is required.')
        if episode.get('recording_mission_id') and active['mission_id'] != episode['recording_mission_id']:
            raise ValueError('A different mission is running. Select the recorded mission or start a new episode.')
    elif active:
        raise ValueError(f'{kind.title()} frames must not contain an active mission.')
    if episode.get('capture_contract') != CAPTURE_CONTRACT:
        return
    if context.get('reasoning_task') != task_for_episode(episode):
        raise ValueError('Capture is missing its explicit reasoning task. Restart recording with the current UI.')
    if context.get('map_name') != episode.get('map_name'):
        raise ValueError('Capture map does not match this episode. Load the recorded map first.')
    if context.get('source') not in ('server_at_capture', 'record_runner_reported') or context.get('runtime_source') in ('unavailable', 'unverified'):
        raise ValueError('Capture-time execution context is unavailable. Wait for navigation or localization to report the current state and retry.')
    if kind == 'observation':
        selected = episode.get('observation_spot_id')
        arrived = observation.get('at_spot_id') or observation.get('spot_id')
        if (not selected or arrived != selected or observation.get('phase') not in ('observing', 'arrived')
                or context.get('localized') is not True):
            raise ValueError('Arrive and finish pose alignment at the selected observation spot before capturing.')
    elif kind == 'patrol':
        if observation.get('phase') not in ('moving', 'returning', 'approaching', 'settling'):
            raise ValueError('Patrol capture requires an active or paused observation route.')
        if episode.get('capture_segment') == 'mission_return' and observation.get('phase') not in ('returning', 'settling'):
            raise ValueError('This return episode only records navigation back to its observation spot.')
        route = episode.get('patrol_spot_ids') or []
        target = observation.get('target_spot_id') or observation.get('at_spot_id')
        source = observation.get('from_spot_id')
        if (not target or target not in route or source and source not in route):
            raise ValueError('The current patrol segment does not belong to this episode route.')


def frame_matches_episode(episode, frame):
    try:
        validate_episode_frame(episode, frame)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def reasoning_counts(episode, samples):
    """Count trainable labels independently of image/mask collection totals."""
    eligible = [sample for sample in samples if frame_matches_episode(episode, sample['observation'])]
    approved = []
    for sample in eligible:
        labels = sample.get('annotations') or []
        if not labels or labels[-1].get('review_status') != 'approved':
            continue
        label, frame = labels[-1], sample['observation']
        try:
            validate_target(label.get('target') or {}, frame.get('context') or {}, recorded_catalog(frame, episode, label))
        except (ValueError, TypeError, KeyError):
            continue
        approved.append(frame['frame_id'])
    sample = episode.get('qwen_sample') or {}
    sampled = set(sample.get('frame_ids') or []) if sample.get('schema_version') == 'user_count_v1' else set()
    return {
        'reasoning_frame_count': len(eligible),
        'reasoning_approved_count': len(approved),
        'qwen_sample_eligible_count': sum(sample['observation']['frame_id'] in sampled for sample in eligible),
        'qwen_sample_approved_count': sum(fid in sampled for fid in approved),
    }

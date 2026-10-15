"""Split a recording on execution changes without owning robot navigation.

Each episode contains one execution role. Consecutive segments share the source
session and robot metadata; retries reuse the same published child.
"""
from fastapi import HTTPException

if __package__:
    from .collection import digest, read_json, write_json
    from .collection_episode_contract import CAPTURE_CONTRACT
else:
    from collection import digest, read_json, write_json
    from collection_episode_contract import CAPTURE_CONTRACT


def _child(store, source):
    linked = source.get('next_episode_id') or source.get('return_episode_id')
    if linked:
        try:
            child = store.episode(linked)
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(410, 'The next recording segment was deleted. Start a new episode.') from exc
            raise
        if child.get('previous_episode_id', child.get('source_episode_id')) != source['id']:
            raise HTTPException(409, 'The recording segment link is inconsistent. Reload the episode.')
        return child
    # A child is published before closing its source. Recover publication if the
    # subsequent source write failed, including a legacy /return publication.
    children = []
    for path in store.episode_paths(source['map_name']):
        candidate = read_json(store.path(*path.relative_to(store.root).parts))
        previous = candidate.get('previous_episode_id', candidate.get('source_episode_id'))
        if previous == source['id']:
            children.append(candidate)
    if len(children) > 1:
        raise HTTPException(409, 'Multiple recording segments reference this episode. Reload the dataset.')
    return children[0] if children else None


def _finish(store, source, child):
    source['ended_at'] = source.get('ended_at') or child['started_at']
    source['next_episode_id'] = child['id']
    write_json(store.path('raw', source['id'], 'episode.json'), source)


def open_segment(store, eid, request):
    """Called under the collection lock; bad requests leave the source open."""
    source = store.episode(eid)
    if source.get('capture_contract') != CAPTURE_CONTRACT or request.capture_contract != CAPTURE_CONTRACT:
        raise HTTPException(422, 'Recording segments require the typed capture contract.')
    if request.map_name != source['map_name']:
        raise HTTPException(409, 'Recording segments must remain on the same map.')
    if request.group_id is not None and request.group_id != source['group_id']:
        raise HTTPException(409, 'Recording segments must remain in the same collection session.')
    inherited = {key: source[key] for key in ('group_id', 'robot_type', 'environment')}
    inherited.update({key: source[key] for key in ('capture_mode', 'capture_interval_ms')
                      if getattr(request, key) is None and key in source})
    request = request.model_copy(update=inherited)
    fingerprint = digest(request.model_dump(mode='json', exclude_none=True))
    existing = _child(store, source)
    if existing:
        if existing.get('segment_request_digest') != fingerprint:
            raise HTTPException(409, 'This episode already continues in a different recording segment. Reload the episode.')
        if not source.get('ended_at') or source.get('next_episode_id') != existing['id']:
            _finish(store, source, existing)
        return store.get(existing['id'])
    if source.get('ended_at'):
        raise HTTPException(409, 'This episode was saved. Start a new episode before recording another segment.')
    # _start performs map/plan/mission validation before writing any data. Links
    # are published with the child so errors can be retried without duplication.
    child = store._start(request, segment={
        'previous_episode_id': source['id'],
        'capture_segment': 'execution_change',
        'segment_request_digest': fingerprint,
    })
    _finish(store, source, child)
    return child

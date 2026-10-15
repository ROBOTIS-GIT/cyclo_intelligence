"""One-click labeler training policy and model lifecycle (no robot commands)."""
import fcntl
import json
import math
import re
import shutil
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel, Field

class AutomaticTrainingRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)


def automatic_steps(kind, count):
    # Current workers use one instance (SAM) / one frame (Qwen) per update.
    passes, minimum = (10, 100) if kind == 'sam' else (5, 100)
    return min(2000, max(minimum, math.ceil(count * passes)))


def allocate_name(root, kind, map_name, models, base_model=''):
    """Reserve a durable monotonic index, including failed/deleted runs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    size = re.search(r'(\d+(?:\.\d+)?)B(?:$|-)', base_model, re.I)
    prefix = kind + (size.group(1).lower() + 'b' if kind == 'qwen' and size else '')
    slug = re.sub(r'[^\w.-]+', '_', map_name).strip('._-')[:48] or 'map'
    prefix = f'{prefix}_{slug}'
    with (root/'.labeler_sequence.json').open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        text = stream.read()
        state = json.loads(text) if text else {}
        previous = [int(m['model_name'][len(prefix)+1:]) for m in models
                    if m.get('model_name', '').startswith(prefix+'_')
                    and m['model_name'][len(prefix)+1:].isdigit()]
        index = max([state.get(prefix, 0)] + previous) + 1
        state[prefix] = index
        stream.seek(0); stream.truncate(); json.dump(state, stream); stream.flush()
        import os
        os.fsync(stream.fileno())
    return f'{prefix}_{index}'


def manual_sample_ids(store, ep, kind):
    from .collection import rows
    sample = ep.get(kind+'_sample', {})
    selected = set(sample.get('frame_ids', [])) if sample.get('schema_version') == 'user_count_v1' else set()
    result = set()
    for row in rows(store.path('raw', ep['id'], 'samples.jsonl')):
        fid = row['observation']['frame_id']
        if fid not in selected: continue
        history = row.get('regions' if kind == 'sam' else 'annotations', [])
        # Approval of an auto draft must not turn it into a manual seed.
        automated = any(r.get('annotator', '').startswith(('sam-auto', 'qwen:')) for r in history)
        if history and not automated: result.add(fid)
    return result


def saved_sample_episodes(store, map_name, kind):
    return [ep['id'] for ep in store.list() if ep.get('ended_at')
            and ep.get('map_name') == map_name and ep.get(kind+'_sample', {}).get('frame_ids')]


def remove_model(root, model_id, depth):
    """Resolve only metadata-owned directories; never accept a client path."""
    root = Path(root).resolve()
    for metadata in root.glob('/'.join(['*'] * depth + ['model.json'])):
        if metadata.is_symlink() or not metadata.resolve().is_relative_to(root): continue
        model = json.loads(metadata.read_text())
        if model.get('id') != model_id: continue
        directory = metadata.parent
        if directory.is_symlink(): raise HTTPException(409, 'Model directory must not be a symlink')
        shutil.rmtree(directory)
        return {'deleted': model_id}
    raise HTTPException(404, 'Model not found')

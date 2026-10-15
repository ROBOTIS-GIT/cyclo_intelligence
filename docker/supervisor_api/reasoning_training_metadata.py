"""Provenance from actual accepted training rows, not intended model capabilities."""
from collections import Counter
import hashlib
import json
if __package__:
    from .mission_decisions import PROMPT_VERSION, SELECTION, SUPERVISION, PATROL, catalog_missions, task_for, validate_target
else:
    from mission_decisions import PROMPT_VERSION, SELECTION, SUPERVISION, PATROL, catalog_missions, task_for, validate_target


def training_metadata(rows):
    counts = Counter({SELECTION: 0, PATROL: 0, SUPERVISION: 0})
    supervision = Counter({'pause': 0, 'continue': 0, 'cancel': 0})
    snapshots = set()
    decisions = {task: Counter() for task in (SELECTION, PATROL, SUPERVISION)}
    for row in rows:
        context = row['frame'].get('context') or {}
        task = task_for(context)
        value = json.loads(row['sample']['messages'][-1]['content'][0]['text'])
        validate_target(value, context, row['catalog'])
        counts[task] += 1
        decisions[task][value["decision"]] += 1
        if task == SUPERVISION:
            supervision[value['decision']] += 1
        missions = catalog_missions(row['catalog'])
        snapshots.add(hashlib.sha256(json.dumps(missions, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    return {'input_policy': PROMPT_VERSION,
            'reasoning_tasks': sorted(task for task, count in counts.items() if count),
            'task_sample_counts': dict(counts),
            'task_decision_counts': {task: dict(values) for task, values in decisions.items()},
            'supervision_decision_counts': dict(supervision),
            'training_catalog_snapshot_ids': sorted(snapshots)}

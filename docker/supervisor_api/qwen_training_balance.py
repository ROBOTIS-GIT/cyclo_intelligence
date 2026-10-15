"""Deterministic training-only downsampling; source annotations remain untouched."""
from collections import Counter, defaultdict
import json
import random


def target(row):
    message = row['sample']['messages'][-1]['content']
    return json.loads(message if isinstance(message, str) else message[0]['text'])


def row_task(row):
    context = row.get('frame', {}).get('context')
    if context is None:
        for message in row['sample']['messages']:
            if message.get('role') != 'user':
                continue
            content = message.get('content', [])
            for part in content if isinstance(content, list) else []:
                if part.get('type') != 'text':
                    continue
                try:
                    context = json.loads(part['text']).get('context', {})
                except (ValueError, AttributeError):
                    continue
    context = context or {}
    return context.get('reasoning_task') or 'mission_selection'


def counts(rows):
    decisions, missions = Counter(), Counter()
    tasks = {}
    for row in rows:
        value = target(row)
        decisions[value['decision']] += 1
        task = tasks.setdefault(row_task(row), {'total': 0, 'decisions': {}, 'missions': {}})
        task['total'] += 1
        task['decisions'][value['decision']] = task['decisions'].get(value['decision'], 0) + 1
        if value['decision'] == 'start':
            missions[value['mission_id']] += 1
            task['missions'][value['mission_id']] = task['missions'].get(value['mission_id'], 0) + 1
    return {'total': len(rows), 'decisions': dict(decisions), 'missions': dict(missions), 'tasks': tasks}


def stratified(rows, size, rng):
    if not size:
        return []
    groups = defaultdict(list)
    for row in rows:
        groups[target(row).get('mission_id') or 'wait'].append(row)
    quotas = {key: size * len(group) // len(rows) for key, group in groups.items()}
    remaining = size - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(size * len(groups[key]) % len(rows)), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return [row for key in sorted(groups) for row in rng.sample(groups[key], quotas[key])]


def balance_frames(frames, enabled=False):
    report = {'enabled': enabled, 'seed': 42, 'policy': 'per_task_majority_cap_v1',
              'before': counts(frames), 'warnings': []}
    selected = list(frames)
    if enabled:
        tasks = defaultdict(lambda: defaultdict(list))
        for row in frames:
            tasks[row_task(row)][target(row)['decision']].append(row)
        rng = random.Random(42)
        chosen = []
        for task, decisions in sorted(tasks.items()):
            sizes = sorted((len(rows) for rows in decisions.values()), reverse=True)
            if len(sizes) == 1:
                report['warnings'].append(f'{task}: only one decision is labeled; all samples retained.')
            cap = sizes[1] if len(sizes) > 1 else sizes[0]
            for decision, rows in sorted(decisions.items()):
                chosen.extend(stratified(rows, min(len(rows), cap), rng))
        ids = {id(row) for row in chosen}
        selected = [row for row in frames if id(row) in ids]
    report['supervision_retained'] = sum(target(r)['decision'] in ('pause', 'continue', 'cancel') for r in selected)
    report['after'] = counts(selected)
    report['excluded'] = len(frames) - len(selected)
    return selected, report

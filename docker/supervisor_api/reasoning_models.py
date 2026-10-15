"""Resolve completed mission-decision adapters without accepting client file paths."""
import hashlib
import json
from pathlib import Path

from fastapi import HTTPException

if __package__:
    from .qwen_annotation_contract import SCHEMA, parse_target
    from .qwen_base_models import BASE_MODELS
else:
    from qwen_annotation_contract import SCHEMA, parse_target
    from qwen_base_models import BASE_MODELS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def catalog_digest(missions):
    return hashlib.sha256(json.dumps(missions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ReasoningModels:
    def __init__(self, catalog_provider, root='/workspace/model/qwen'):
        self.root = Path(root).resolve()
        self.catalog_provider = catalog_provider

    def validate(self, model):
        if not all(isinstance(model.get(k), str) and model[k].strip() for k in ('id', 'model_name', 'map_name', 'base_model', 'revision', 'adapter_path')):
            raise ValueError('Incomplete model metadata')
        adapter = Path(model['adapter_path']).resolve()
        if not adapter.is_relative_to(self.root):
            raise ValueError('Adapter must be inside the Qwen model directory')
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = (adapter / name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or not path.stat().st_size:
                raise ValueError('Adapter files are missing or empty')
        config = read_json(adapter / 'adapter_config.json')
        if config.get('peft_type') != 'LORA' or config.get('base_model_name_or_path') != model['base_model']:
            raise ValueError('Adapter base model does not match training metadata')
        if not model['base_model'].startswith('Qwen/Qwen3.5-') or not model.get('revision'):
            raise ValueError('This Qwen engine requires a Qwen3.5 model and its training revision')
        return {**model, 'adapter_path': str(adapter)}

    def list(self, map_name):
        if not map_name:
            return []
        models = []
        current_missions = self.catalog_provider(map_name)
        current_snapshot_id = catalog_digest(current_missions)
        def compatibility(model):
            trained = model.get('catalog_snapshot_id')
            source_snapshots = model.get('training_catalog_snapshot_ids')
            compatible = None if not trained else trained == current_snapshot_id
            if source_snapshots:
                compatible = all(value == current_snapshot_id for value in source_snapshots)
            task_counts = model.get('task_sample_counts') or {}
            supervision_counts = model.get('supervision_decision_counts') or {}
            count = task_counts.get('mission_supervision', 0)
            supervised = isinstance(count, int) and count > 0 and any(
                isinstance(supervision_counts.get(action), int) and supervision_counts[action] > 0
                for action in ('pause', 'continue', 'cancel'))
            return {**model, 'current_catalog_snapshot_id': current_snapshot_id,
                    'catalog_compatible': compatible, 'supervision_trained': supervised}
        # Final reasoning training is isolated from the sampled auto-label adapters.
        for path in self.root.glob('reasoning/*/*/model.json'):
            try:
                model = read_json(path)
                if (model.get('schema') != SCHEMA or model.get('map_name') != map_name or
                        model.get('purpose') != 'reasoning'):
                    continue
                model = self.validate(model)
                models.append(compatibility({**model, 'source': 'fine_tuning'}))
            except (ValueError, KeyError, TypeError, OSError):
                continue
        # Keep pre-separation annotation adapters usable. New auto-label-only
        # models declare their purpose and are not offered to Operate.
        for path in self.root.glob('annotation/*/*/model.json'):
            try:
                model = read_json(path)
                if (model.get('schema') != SCHEMA or model.get('map_name') != map_name or
                        model.get('purpose') == 'auto_labeling'):
                    continue
                model = self.validate(model)
                models.append(compatibility({**model, 'source': 'annotation_legacy'}))
            except (ValueError, KeyError, TypeError, OSError):
                continue
        # Earlier command-line training runs have reports instead of model.json.
        # Offer them only if every target mission belongs to the selected map.
        mission_ids = {m['mission_id'] for m in current_missions}
        for path in self.root.glob('*/training_report.json'):
            try:
                report = read_json(path)
                targets = [parse_target(json.dumps(t)) for t in report.get('targets', [])]
                target_ids = {t['mission_id'] for t in targets if t['mission_id']}
                if not target_ids or not target_ids.issubset(mission_ids):
                    continue
                model = self.validate({
                    'id': 'legacy:' + path.parent.name, 'model_name': path.parent.name,
                    'map_name': map_name, 'base_model': report['model'], 'revision': report['revision'],
                    'adapter_path': str(path.parent), 'sample_count': report.get('samples'),
                    'source': 'legacy',
                })
                models.append(compatibility(model))
            except (ValueError, KeyError, TypeError, OSError):
                continue
        for base_model, revision in BASE_MODELS.items():
            models.append({
                'id': 'base:' + base_model.split('/')[-1],
                'model_name': base_model.split('/')[-1],
                'map_name': map_name,
                'base_model': base_model,
                'revision': revision,
                'adapter_path': '',
                'sample_count': 0,
                'source': 'base',
                'catalog_compatible': None,
                'current_catalog_snapshot_id': current_snapshot_id,
            })
        return sorted(models, key=lambda m: (m['source'] != 'fine_tuning', m['source'] == 'base', m['model_name'].casefold()))

    def resolve(self, model_id, map_name):
        model = next((m for m in self.list(map_name) if m['id'] == model_id), None)
        if model is None:
            raise HTTPException(422, 'Selected model is unavailable or does not belong to this map. Refresh the model list.')
        return model


def load_options(model):
    return {'model_id': model['base_model'], 'revision': model['revision'], 'adapter_path': model['adapter_path']}


def matches_loaded(health, model):
    return (health.get('model') == model['base_model'] and
            health.get('revision') == model['revision'] and
            (health.get('adapter') or '') == model['adapter_path'])

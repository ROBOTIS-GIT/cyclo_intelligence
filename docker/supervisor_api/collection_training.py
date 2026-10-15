"""Prepare episode-local typed training messages from immutable recorded context."""
import hashlib
from fastapi import HTTPException
if __package__:
    from .collection import digest
    from .collection_episode_contract import frame_matches_episode
    from .reasoning_training_metadata import training_metadata
    from .mission_decisions import sample_for, recorded_catalog, system_for
else:
    from collection import digest
    from collection_episode_contract import frame_matches_episode
    from reasoning_training_metadata import training_metadata
    from mission_decisions import sample_for, recorded_catalog, system_for


def training_files(store, ep):
    episodes = [ep]
    counts = {'train': 0, 'val': 0, 'test': 0}
    training_rows = []
    groups, sources = {}, []
    region_count = 0
    output = {name: [] for name in ('train.jsonl', 'val.jsonl', 'test.jsonl', 'training_regions.jsonl')}
    for ep in episodes:
        current_catalog = store.current_catalog(ep.get('map_name'))
        bucket = int(hashlib.sha256(ep['group_id'].encode()).hexdigest()[:8], 16) % 100
        split = 'train' if bucket < 80 else 'val' if bucket < 90 else 'test'
        groups[ep['group_id']] = split
        directory = store.path('raw', ep['id'])
        sources.append({'episode_id': ep['id'], 'files': {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('episode.json', 'samples.jsonl')
            if (directory / name).is_file()}})
        if digest(ep['catalog']['missions']) != ep['catalog_snapshot_id']:
            raise HTTPException(422, 'Mission catalog snapshot checksum mismatch')
        by_id = {r['frame_id']: r for r in ep['observations']}
        for label in ep['annotations']:
            if label['review_status'] != 'approved':
                continue
            if len(label['frame_ids']) != 1 or label['frame_ids'][0] not in by_id:
                raise HTTPException(422, 'Invalid annotation frame reference')
            frame = by_id[label['frame_ids'][0]]
            if not frame_matches_episode(ep, frame):
                continue
            captured_catalog = frame.get('catalog') or ep['catalog']
            if frame['catalog_snapshot_id'] != digest(captured_catalog['missions']):
                raise HTTPException(422, 'Observation catalog snapshot checksum mismatch')
            try:
                store.validate_target(label['target'], frame, current_catalog)
            except HTTPException as exc:
                if label['target'].get('mission_id'):
                    raise HTTPException(422, 'A saved label references a deleted Design mission. Relabel the frame before training.') from exc
                raise
            image = store.image(ep['id'], frame['frame_id'])
            relative = str(image.relative_to(directory))
            catalog = recorded_catalog(frame, ep, label)
            store.validate_target(label['target'], frame, catalog)
            sample = sample_for(frame.get('context') or {}, catalog, label['target'], relative)
            training_rows.append({'frame': frame, 'catalog': catalog, 'sample': sample})
            output[split + '.jsonl'].append(sample)
            counts[split] += 1
        # Region supervision remains separate metadata. It is not silently
        # injected as an oracle ROI into Qwen's decision-training prompt.
        regions = {r['object_id']: r for r in store.records(ep['id'], 'regions')}
        for region in regions.values():
            if region['review_status'] != 'approved':
                continue
            frame = by_id.get(region['frame_id'])
            if not frame or region['image_sha256'] != frame['image']['sha256']:
                raise HTTPException(422, 'Region references a missing or changed source image')
            mask_path = store.path('raw', ep['id'], *region['mask']['path'].split('/'))
            if hashlib.sha256(mask_path.read_bytes()).hexdigest() != region['mask']['sha256']:
                raise HTTPException(422, 'Region mask checksum mismatch')
            image_rel = f"images/{region['frame_id']}.jpg"
            store.image(ep['id'], region['frame_id'])
            mask_rel = str(mask_path.relative_to(directory))
            output['training_regions.jsonl'].append( {**region, 'split': split, 'image_path': image_rel,
                   'mask': {**region['mask'], 'path': mask_rel}})
            region_count += 1
    if not sum(counts.values()) and not region_count:
        store.invalidate_training(ep['id'])
        raise HTTPException(422, 'No approved samples or masks to prepare')
    provenance = training_metadata(training_rows)
    manifest = {'schema_version': 'reasoning_episode_training_v1', 'format': 'transformers_multimodal_messages',
                'episode_id': ep['id'], 'map_name': ep.get('map_name'), 'counts': counts,
                'approved_regions': region_count, 'groups': groups, 'sources': sources,
                **provenance,
                'system_prompts': {task: system_for({'reasoning_task': task}) for task in provenance['reasoning_tasks']},
                'catalog': store.catalog_identity(current_catalog),
                'storage_path': str(store.path('raw', ep['id'])),
                'warnings': [f'{s} split is empty; collect more independent scene groups.' for s, n in counts.items() if not n]}
    return output, manifest

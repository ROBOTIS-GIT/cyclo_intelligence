"""Server-owned reasoning datasets. Labels never execute robot commands."""
import base64
from copy import deepcopy
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv('CYCLO_REASONING_DATA_DIR', '/workspace/dataset/reasoning'))
if __package__:
    from .collection_review_counts import generated_review_counts
else:
    from collection_review_counts import generated_review_counts

if __package__:
    from .mission_decisions import decisions_for, recorded_catalog, task_for, validate_target as validate_decision_target
    from .collection_episode_contract import CAPTURE_CONTRACT, EPISODE_TASKS, episode_type_for, frame_matches_episode, reasoning_counts, validate_episode_frame
else:
    from mission_decisions import decisions_for, recorded_catalog, task_for, validate_target as validate_decision_target
    from collection_episode_contract import CAPTURE_CONTRACT, EPISODE_TASKS, episode_type_for, frame_matches_episode, reasoning_counts, validate_episode_frame


def now():
    return int(time.time() * 1000)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def serialized(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            try:
                return fn(self, *args, **kwargs)
            except OSError as exc:
                raise HTTPException(507, 'Dataset storage failed. Check disk space and permissions; retry before continuing.') from exc
    return wrapped


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, data):
    atomic(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, value):
    # Roll back partial writes (e.g. ENOSPC) so retries cannot leave malformed JSONL.
    with path.open('a+b') as stream:
        size = stream.tell()
        try:
            stream.write((json.dumps(value, ensure_ascii=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        except OSError:
            stream.seek(size)
            stream.truncate()
            raise


def jpeg_size(data):
    if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise HTTPException(422, 'A complete JPEG image is required')
    i = 2
    while i < len(data):
        if data[i] != 255:
            break
        while i < len(data) and data[i] == 255:
            i += 1
        marker = data[i]; i += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            break
        length = int.from_bytes(data[i:i+2], 'big')
        if length < 2 or i + length > len(data):
            break
        if marker in (0xC0, 0xC1, 0xC2):
            if length < 8:
                break
            height, width = struct.unpack('>HH', data[i+3:i+7])
            if width and height:
                return width, height
        i += length
    raise HTTPException(422, 'JPEG dimensions are unavailable')


class EpisodeRequest(BaseModel):
    capture_mode: Literal['single', 'continuous'] | None = None
    capture_interval_ms: int | None = Field(default=None, ge=100, le=60000)
    plan_revision: int | None = Field(default=None, ge=0)
    recording_mission_id: str | None = Field(default=None, min_length=1, max_length=128)
    episode_type: Literal['observation', 'patrol', 'mission'] | None = None
    capture_contract: Literal['task_aligned_v1'] | None = None
    patrol_spot_ids: list[str] = Field(default_factory=list, max_length=100)
    observation_spot_id: str | None = Field(default=None, max_length=128)
    return_observation_spot_id: str | None = Field(default=None, min_length=1, max_length=128)
    recording_mission_name: str | None = Field(default=None, max_length=128)
    scenario: str | None = Field(default=None, min_length=1, max_length=300)
    group_id: str | None = Field(default=None, min_length=1, max_length=120)
    robot_type: str = Field(default='unknown', min_length=1, max_length=80)
    environment: Literal['simulation', 'real', 'unknown'] = Field(default_factory=lambda: os.getenv('CYCLO_COLLECTION_ENVIRONMENT', 'unknown'), validate_default=True)
    map_name: str = Field(min_length=1, max_length=128)


class MapCopyRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)


class FrameMetadataRequest(BaseModel):
    scenario: str | None = Field(default=None, min_length=1, max_length=300)
    group_id: str = Field(min_length=1, max_length=120)
    robot_type: str = Field(min_length=1, max_length=80)
    environment: Literal['simulation', 'real', 'unknown'] = 'unknown'


class AnnotationRequest(BaseModel):
    metadata: FrameMetadataRequest | None = None
    frame_id: str = Field(min_length=1, max_length=36)
    decision: Literal['wait', 'start', 'next_observation', 'unknown', 'pause', 'continue', 'cancel']
    mission_id: str | None = Field(default=None, max_length=128)
    reason: str = Field(min_length=1, max_length=1000)
    annotator: str = Field(min_length=1, max_length=120)
    review_status: Literal['draft', 'approved', 'rejected'] = 'draft'
    expected_revision: int = Field(default=0, ge=0)


class InstructionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ExportRequest(BaseModel):
    episode_ids: list[str] = Field(min_length=1, max_length=500)


class CollectionStore:
    def __init__(self, root=ROOT, catalog_provider=None):
        self.root = Path(root)
        self.catalog_provider = catalog_provider or (lambda _: [])
        self.lock = threading.RLock()

    def path(self, *parts):
        for part in parts:
            if not part or part in ('.', '..') or '/' in part or '\\' in part:
                raise HTTPException(400, 'Invalid dataset identifier')
        # Keep the episode-ID API stable while resolving storage under its map.
        if len(parts) >= 2 and parts[0] == 'raw':
            matches = []
            for record in self.episode_paths():
                directory = self.path(*record.parent.relative_to(self.root).parts)
                if read_json(directory / 'episode.json').get('id') == parts[1]:
                    matches.append(record)
            if len(matches) > 1:
                raise HTTPException(409, 'Duplicate episode identifier across maps')
            if matches:
                parts = (*matches[0].parent.relative_to(self.root).parts, *parts[2:])
        candidate = self.root.joinpath(*parts)
        for parent in [candidate, *candidate.parents]:
            if parent == self.root.parent:
                break
            if parent.is_symlink():
                raise HTTPException(400, 'Dataset paths must not contain symlinks')
        return candidate

    def episode_paths(self, map_name=None):
        maps = [self.path(map_name)] if map_name else [p for p in self.root.iterdir() if p.is_dir()] if self.root.exists() else []
        found = []
        for directory in maps:
            if directory.name in ('exports', 'training_inputs', '_staging', '.deleted'):
                continue
            found.extend(directory.glob('raw_episode_*/episode.json'))  # legacy layout
            for kind in EPISODE_TASKS:
                found.extend((directory / kind).glob('raw_episode_*/episode.json'))
        return found

    def next_episode_directory(self, map_name, episode_type='observation'):
        directory = self.path(map_name)
        # Fill deleted slots instead of using the legacy high-water counter.
        # Reserve any matching filesystem entry, including incomplete episodes.
        occupied = set()
        for parent in (directory, *(directory / kind for kind in EPISODE_TASKS)):
            if parent.exists():
                for child in parent.iterdir():
                    suffix = child.name.removeprefix('raw_episode_')
                    if child.name.startswith('raw_episode_') and suffix.isdigit():
                        occupied.add(int(suffix))
        number = 1
        while number in occupied:
            number += 1
        return self.path(map_name, episode_type, f'raw_episode_{number:04d}')

    def records(self, eid, kind):
        samples = rows(self.path('raw', eid, 'samples.jsonl'))
        if kind == 'observations':
            return [sample['observation'] for sample in samples]
        return [record for sample in samples for record in sample.get(kind, [])]

    def reason_presets(self, map_name):
        """Aggregate reusable approved reasons across saved episodes on one map."""
        directory = self.path(map_name)
        if not directory.is_dir():
            return []
        presets = {}
        for episode_path in self.episode_paths(map_name):
            episode = read_json(episode_path)
            if episode.get('map_name') != map_name or not episode.get('ended_at'):
                continue
            for sample in rows(episode_path.parent / 'samples.jsonl'):
                if not frame_matches_episode(episode, sample['observation']):
                    continue
                annotations = sample.get('annotations', [])
                if not annotations:
                    continue
                annotation = annotations[-1]
                target = annotation.get('target', {})
                decision = target.get('decision')
                reason = target.get('reason', '').strip()
                context = sample.get('observation', {}).get('context') or {}
                try:
                    task = task_for(context)
                except ValueError:
                    continue
                decisions = decisions_for(context)
                if annotation.get('review_status') != 'approved' or decision not in decisions or not reason:
                    continue
                mission_id = target.get('mission_id') if decision in ('start', 'pause', 'continue', 'cancel') else None
                key = (task, decision, mission_id, reason)
                if key not in presets:
                    presets[key] = {'reasoning_task': task, 'decision': decision, 'mission_id': mission_id,
                                    'reason': reason, 'count': 0}
                presets[key]['count'] += 1
        return sorted(presets.values(), key=lambda preset: (-preset['count'], preset['reason']))

    def save_samples(self, eid, samples):
        atomic(self.path('raw', eid, 'samples.jsonl'),
               ''.join(json.dumps(sample, ensure_ascii=False) + '\n' for sample in samples).encode())

    def current_catalog(self, map_name):
        """Return a detached snapshot of the current Design mission catalog."""
        missions = deepcopy(self.catalog_provider(map_name)) if map_name else []
        return {'missions': missions, 'snapshot_id': digest(missions)}

    @staticmethod
    def catalog_identity(catalog):
        missions = catalog.get('missions', [])
        return {
            'snapshot_id': catalog.get('snapshot_id') or digest(missions),
            'mission_ids': sorted(m['mission_id'] for m in missions),
            'mission_revisions': {m['mission_id']: m.get('revision', 0) for m in missions},
        }

    @serialized
    def add_record(self, eid, kind, record):
        if kind not in ('annotations', 'predictions', 'regions'):
            raise ValueError('Unknown sample record kind')
        samples = rows(self.path('raw', eid, 'samples.jsonl'))
        fid = record['frame_ids'][0] if kind == 'annotations' else record['frame_id']
        sample = next((s for s in samples if s['observation']['frame_id'] == fid), None)
        if sample is None:
            raise HTTPException(422, 'Sample frame not found')
        if kind == "regions" and record.get("review_status") != "rejected":
            sample["observation"].pop("mask_review", None)
        sample.setdefault(kind, []).append(record)
        self.save_samples(eid, samples)

    def sync_mission_names(self, ep, path, missions=None):
        """Repair mutable episode names by ID; keep the captured catalog immutable."""
        missions = self.catalog_provider(ep['map_name']) if missions is None and ep.get('map_name') else missions or []
        snapshot = ep.get('catalog', {}).get('missions', [])
        mid = ep.get('recording_mission_id')
        old_name = ep.get('recording_mission_name')
        if not mid and old_name:
            matches = [m for m in snapshot if m.get('mission_name') == old_name]
            if len(matches) == 1:
                mid = matches[0]['mission_id']
        current = {m['mission_id']: m['mission_name'] for m in missions}
        changed = False
        return_mid = ep.get('return_mission_id')
        if return_mid in current and ep.get('return_mission_name') != current[return_mid]:
            ep['return_mission_name'] = current[return_mid]
            changed = True
        if mid and ep.get('recording_mission_id') != mid:
            ep['recording_mission_id'] = mid
            changed = True
        if mid in current and old_name != current[mid]:
            # Only the former automatic "map / mission" title is renameable.
            # Never replace names inside operator-authored scene descriptions.
            scenario = ep.get('scenario')
            if isinstance(scenario, str) and [v.strip() for v in scenario.split('/')] == [ep['map_name'], old_name]:
                ep['scenario'] = f"{ep['map_name']} / {current[mid]}"
            ep['recording_mission_name'] = current[mid]
            changed = True
        if changed:
            write_json(path, ep)
        return [{**m, 'mission_name': current.get(m['mission_id'], m['mission_name']),
                 'available': m['mission_id'] in current} for m in snapshot]

    def attach_current_catalog(self, ep, missions=None):
        catalog = self.current_catalog(ep.get('map_name')) if missions is None else {
            'missions': deepcopy(missions), 'snapshot_id': digest(missions)}
        ep['current_catalog'] = catalog
        current_ids = {m['mission_id'] for m in catalog['missions']}
        ep['catalog_status'] = {
            'changed': catalog['snapshot_id'] != ep.get('catalog_snapshot_id'),
            'deleted_mission_ids': sorted(
                m['mission_id'] for m in ep.get('catalog', {}).get('missions', [])
                if m['mission_id'] not in current_ids),
        }
        return catalog

    @serialized
    def episode(self, episode_id):
        path = self.path('raw', episode_id, 'episode.json')
        if not path.is_file():
            raise HTTPException(404, 'Episode not found')
        ep = read_json(path)
        self.sync_mission_names(ep, path)
        return ep

    @staticmethod
    def observation_view(map_name):
        if __package__:
            from .autonomy_config import AutonomyRepository
        else:
            from supervisor_api.autonomy_config import AutonomyRepository
        return AutonomyRepository().view(map_name)

    @serialized
    def start(self, request):
        return self._start(request)

    def _start(self, request, segment=None):
        # Callers hold the collection lock. Segment links are server-owned and
        # included before publishing, so a failed source update is recoverable.
        if not request.map_name.strip() or not request.robot_type.strip():
            raise HTTPException(422, 'Robot and map are required')
        if any(v is not None and not v.strip() for v in (request.scenario, request.group_id)):
            raise HTTPException(422, 'Legacy scenario/group values must not be blank')
        missions = self.current_catalog(request.map_name)['missions']
        if request.recording_mission_name and not any(m.get("mission_name") == request.recording_mission_name for m in missions):
            raise HTTPException(422, "Recording mission does not belong to this map")
        observation = None
        return_observation = None
        view = None
        if request.plan_revision is not None or request.observation_spot_id or request.patrol_spot_ids or request.return_observation_spot_id:
            view = self.observation_view(request.map_name)
            ids = {p['spot_id'] for p in view['plan']['observations']}
            spots = {s['id']: s for s in view['spots']
                     if s.get('metadata', {}).get('role') == 'observation' or s['id'] in ids}
            observation = spots.get(request.observation_spot_id)
            return_observation = spots.get(request.return_observation_spot_id)
            if request.return_observation_spot_id and return_observation is None:
                raise HTTPException(422, 'Return observation spot does not belong to this map')
            if request.observation_spot_id and observation is None:
                raise HTTPException(422, 'Observation spot does not belong to this map')
            if any(s not in spots for s in request.patrol_spot_ids):
                raise HTTPException(422, 'Every patrol stop must be an observation spot on this map')
        catalog = {'missions': missions, 'snapshot_id': digest(missions)}
        eid = str(uuid4())
        if request.map_name.startswith('.') or request.map_name in ('raw', 'exports', 'training_inputs', '_staging', '_unassigned'):
            raise HTTPException(422, 'Reserved map name')
        episode_type = request.episode_type or ('mission' if request.recording_mission_name else 'observation')
        if episode_type != 'mission' and request.recording_mission_name:
            raise HTTPException(422, 'Observation and Patrol episodes cannot record a task mission')
        if request.return_observation_spot_id and episode_type != 'mission':
            raise HTTPException(422, 'Only Mission episodes can specify a return observation spot')
        if request.patrol_spot_ids and episode_type != 'patrol':
            raise HTTPException(422, 'Only Patrol episodes can specify a patrol route')
        if request.capture_contract == CAPTURE_CONTRACT:
            if request.episode_type is None:
                raise HTTPException(422, 'Select Observation, Patrol, or Mission before recording')
            if episode_type == 'observation' and not request.observation_spot_id:
                raise HTTPException(422, 'Select an observation spot before recording')
            if episode_type == 'patrol' and (not request.patrol_spot_ids or len(set(request.patrol_spot_ids)) != len(request.patrol_spot_ids)):
                raise HTTPException(422, 'Select one or more distinct observation spots for the patrol route')
            if episode_type == 'mission' and not request.recording_mission_name:
                raise HTTPException(422, 'Select a mission before recording')
        if __package__:
            from .collection_observation_plan import episode_binding
        else:
            from collection_observation_plan import episode_binding
        binding = episode_binding(request, episode_type, view, missions)
        if binding.get('return_observation_spot_id'):
            return_observation = observation
        directory = self.next_episode_directory(request.map_name, episode_type)
        ep = {'schema_version': 'reasoning_episode_v4' if request.capture_contract else 'reasoning_episode_v3', 'id': eid, **request.model_dump(exclude_none=True),
              'episode_type': episode_type, **binding,
              'group_id': request.group_id or f'episode-{eid}',
              'started_at': now(), 'ended_at': None, 'catalog_snapshot_id': catalog['snapshot_id'],
              'storage_path': str(directory), 'catalog': catalog, **(segment or {})}
        if observation is not None:
            ep['observation_spot'] = observation
        if return_observation is not None:
            point = next((p for p in view['plan']['observations'] if p['spot_id'] == ep.get('return_observation_spot_id')), {})
            ep['return_observation_spot'] = {**deepcopy(return_observation), 'arrival_bt': point.get('arrival_bt', '')}
        if view is not None:
            ep['recording_plan'] = deepcopy(view['plan'])
        if request.recording_mission_name:
            ep['recording_mission_id'] = next(m['mission_id'] for m in missions if m['mission_name'] == request.recording_mission_name)
        directory.mkdir(parents=True)
        try:
            atomic(directory / 'samples.jsonl', b'')
            write_json(directory / 'episode.json', ep)  # publish last
        except Exception:
            shutil.rmtree(directory)
            raise
        return self.get(eid)

    @serialized
    def start_segment(self, eid, request):
        if __package__:
            from .collection_segments import open_segment
        else:
            from collection_segments import open_segment
        return open_segment(self, eid, request)

    @serialized
    def start_return(self, eid):
        if __package__:
            from .collection_returns import open_return_episode
        else:
            from collection_returns import open_return_episode
        return open_return_episode(self, eid)

    @serialized
    def copy_to_map(self, eid, request):
        source = self.get(eid)
        if source.get('map_name') or not source['ended_at']:
            raise HTTPException(409, 'Only ended episodes without a map can be copied to a map.')
        if not request.map_name.strip():
            raise HTTPException(422, 'Select a map first.')
        missions = self.current_catalog(request.map_name)['missions']
        catalog = {'missions': missions, 'snapshot_id': digest(missions)}
        source_dir = self.path('raw', eid)
        # Copy only a regular dataset tree; never follow links out of its root.
        for parent, dirs, files in os.walk(source_dir):
            if any((Path(parent) / name).is_symlink() for name in dirs + files):
                raise HTTPException(400, 'Dataset contains a symlink.')
        for frame in source['observations']:
            self.image(eid, frame['frame_id'])
        new_id = str(uuid4())
        if request.map_name.startswith('.') or request.map_name in ('raw', 'exports', 'training_inputs', '_destination', '_unassigned'):
            raise HTTPException(422, 'Reserved map name')
        episode_type = episode_type_for(source)
        destination = self.next_episode_directory(request.map_name, episode_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(source_dir, destination, ignore=shutil.ignore_patterns(
                'episode.json', 'train.jsonl', 'val.jsonl', 'test.jsonl', 'training_regions.jsonl', 'training_manifest.json'))
            original = self.episode(eid)
            samples = rows(destination / 'samples.jsonl')
            sample_ids = {}
            for sample in samples:
                row = sample['observation']
                row.update(episode_id=new_id, source_catalog_snapshot_id=row['catalog_snapshot_id'],
                           catalog_snapshot_id=catalog['snapshot_id'], catalog=deepcopy(catalog))
                for label in sample.get('annotations', []):
                    old_id = label['sample_id']
                    label.update(episode_id=new_id, source_sample_id=old_id,
                                 sample_id=sample_ids.setdefault(old_id, str(uuid4())))
                    if label['review_status'] != 'rejected':
                        label['review_status'] = 'draft'
                for region in sample.get('regions', []):
                    region['episode_id'] = new_id
                    mask = self.path('raw', eid, *region['mask']['path'].split('/'))
                    if hashlib.sha256(mask.read_bytes()).hexdigest() != region['mask']['sha256']:
                        raise HTTPException(422, 'Region mask checksum mismatch')
            atomic(destination / 'samples.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in samples).encode())
            provenance = {'source_episode_id': eid, 'assigned_at_ms': now(), 'assigned_map_name': request.map_name}
            original.pop('training', None)
            write_json(destination / 'episode.json', {**original, 'id': new_id, 'map_name': request.map_name,
                'episode_type': episode_type,
                'catalog_snapshot_id': catalog['snapshot_id'], 'catalog': catalog, 'source_catalog': source['catalog'], 'map_assignment': provenance,
                'storage_path': str(destination)})
        except Exception:
            if destination.exists():
                shutil.rmtree(destination)
            raise
        return self.get(new_id)

    @serialized
    def delete(self, eid):
        directory = self.path('raw', eid)
        ep = self.episode(eid)
        if not ep['ended_at']:
            raise HTTPException(409, 'End the episode before deleting it.')
        # Delete in place. Keep episode.json until last so failures remain retryable.
        for child in directory.iterdir():
            if child.name == 'episode.json':
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        (directory / 'episode.json').unlink()
        directory.rmdir()
        return {'deleted': True, 'episode_id': eid, 'cleanup_pending': False}

    @serialized
    def list(self):
        if not self.root.exists():
            return []
        found = self.episode_paths()
        episodes = []
        catalogs = {}
        for path in found:
            directory = self.path(*path.parent.relative_to(self.root).parts)
            ep = read_json(directory / 'episode.json')
            map_name = ep.get('map_name')
            if map_name not in catalogs:
                catalogs[map_name] = self.catalog_provider(map_name) if map_name else []
            ep['display_missions'] = self.sync_mission_names(ep, directory / 'episode.json', catalogs[map_name])
            self.attach_current_catalog(ep, catalogs[map_name])
            samples = rows(directory / 'samples.jsonl')
            ep['frame_count'] = len(samples)
            masks = [list({r['object_id']: r for r in sample.get('regions', [])}.values()) for sample in samples]
            ep['mask_labeled_count'] = sum(any(r['review_status'] == 'approved' for r in regions) for regions in masks)
            ep['mask_reviewed_count'] = sum((bool(regions) and all(r['review_status'] in ('approved', 'rejected') for r in regions)) or (not regions and sample['observation'].get('mask_review', {}).get('status') == 'empty') for sample, regions in zip(samples, masks))
            ep['mask_pending_count'] = sum(any(r['review_status'] == 'draft' for r in regions) for regions in masks)
            ep['unlabeled_count'] = sum(1 for sample in samples if not sample.get('annotations') or sample['annotations'][-1].get('review_status') != 'approved')
            ep.update(reasoning_counts(ep, samples))
            ep.update(generated_review_counts(ep, samples))
            sam_sample = ep.get('sam_sample', {})
            mask_sampled_ids = set(sam_sample.get('frame_ids', [])) if sam_sample.get('schema_version') == 'user_count_v1' else set()
            sampled_masks = [regions for sample,regions in zip(samples,masks)
                             if sample['observation']['frame_id'] in mask_sampled_ids]
            approved_masks = [region for regions in sampled_masks for region in regions
                              if region['review_status'] == 'approved']
            ep['sam_sample_approved_frame_count'] = sum(
                any(region['review_status'] == 'approved' for region in regions)
                for regions in sampled_masks
            )
            ep['sam_sample_pending_frame_count'] = sum(
                any(region['review_status'] == 'draft' for region in regions)
                for regions in sampled_masks
            )
            ep['sam_sample_approved_instance_count'] = len(approved_masks)
            ep['sam_sample_label_counts'] = {
                label: {
                    'instance_count': sum(region['object_label'] == label for region in approved_masks),
                    'frame_count': len({region['frame_id'] for region in approved_masks if region['object_label'] == label}),
                }
                for label in {region['object_label'] for region in approved_masks}
            }
            trainable_masks = [region for regions in sampled_masks
                               if not any(item['review_status'] == 'draft' for item in regions)
                               for region in regions if region['review_status'] == 'approved']
            ep['sam_sample_trainable_frame_count'] = len({region['frame_id'] for region in trainable_masks})
            ep['sam_sample_trainable_instance_count'] = len(trainable_masks)
            ep['sam_sample_trainable_label_counts'] = {
                label: {
                    'instance_count': sum(region['object_label'] == label for region in trainable_masks),
                    'frame_count': len({region['frame_id'] for region in trainable_masks if region['object_label'] == label}),
                }
                for label in {region['object_label'] for region in trainable_masks}
            }
            trainable_masks = [region for regions in masks
                               if not any(item['review_status'] == 'draft' for item in regions)
                               for region in regions if region['review_status'] == 'approved']
            ep['sam_trainable_frame_count'] = len({region['frame_id'] for region in trainable_masks})
            ep['sam_trainable_instance_count'] = len(trainable_masks)
            ep['sam_trainable_label_counts'] = {
                label: {
                    'instance_count': sum(region['object_label'] == label for region in trainable_masks),
                    'frame_count': len({region['frame_id'] for region in trainable_masks if region['object_label'] == label}),
                }
                for label in {region['object_label'] for region in trainable_masks}
            }
            episodes.append(ep)
        return sorted(episodes, key=lambda ep: ep['started_at'], reverse=True)

    @serialized
    def get(self, eid):
        ep = self.episode(eid)
        directory = self.path('raw', eid)
        current = self.current_catalog(ep.get('map_name'))
        ep['display_missions'] = self.sync_mission_names(ep, directory / 'episode.json', current['missions'])
        self.attach_current_catalog(ep, current['missions'])
        ep['regions'] = list({r['object_id']: r for r in self.records(eid, 'regions')}.values())
        ep['observations'] = self.records(eid, 'observations')
        ep['predictions'] = [{'frame_id': p['frame_id'], 'prediction': p['prediction']}
                             for p in self.records(eid, 'predictions')]
        ep['annotations'] = list({r['sample_id']: r for r in self.records(eid, 'annotations')}.values())
        samples = rows(directory / 'samples.jsonl')
        by_id = {f['frame_id']: f for f in ep['observations']}
        for sample in samples:
            frame = by_id.get(sample['observation']['frame_id'])
            if frame is None: continue
            for kind, key, prefix in [('sam', 'regions', 'sam-auto'), ('qwen', 'annotations', 'qwen:')]:
                if any(r.get('annotator', '').startswith(prefix) for r in sample.get(key, [])):
                    frame.setdefault('auto_labeling', {}).setdefault(kind, {'status':'generated'})
        ep.update(reasoning_counts(ep, samples))
        ep.update(generated_review_counts(ep, samples))
        return ep

    @serialized
    def end(self, eid):
        ep = self.episode(eid)
        if not ep['ended_at']:
            ep['ended_at'] = now()
            write_json(self.path('raw', eid, 'episode.json'), ep)
        return self.get(eid)

    @serialized
    def capture(self, eid, frame, context, *, capture_info=None, prediction=None, analysis_id=None, metadata=None):
        ep = self.episode(eid)
        if ep['ended_at']:
            raise HTTPException(410, 'Episode has ended. Start a new episode to collect frames.')
        directory = self.path('raw', eid)
        existing = self.records(eid, 'observations')
        if analysis_id:
            saved = next((r for r in existing if r.get('analysis_id') == analysis_id), None)
            if saved:
                return saved
        if len(existing) >= 2000:
            self.end(eid)
            raise HTTPException(410, 'Episode reached 2000 frames. Start a new episode.')
        data = base64.b64decode(frame['image_base64'], validate=True)
        width, height = jpeg_size(data)
        fid = str(uuid4())
        instruction = next((e for e in reversed(ep.get('instructions', [])) if e['kind'] == 'operator_instruction' and e['received_at_ms'] <= int(frame['captured_at'] * 1000)), None)
        context = {**deepcopy(context), 'operator_instruction': context.get('operator_instruction', instruction)}
        if ep.get('capture_contract') == CAPTURE_CONTRACT:
            try:
                validate_episode_frame(ep, {'context': context})
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        else:
            context.setdefault('reasoning_task', 'mission_selection')
        if (ep.get('capture_contract') == CAPTURE_CONTRACT and episode_type_for(ep) == 'observation'
                and context.get('source') == 'record_runner_reported' and not ep.get('observation_binding')):
            # The browser reports arrival; candidate permissions come from the saved map plan.
            view = self.observation_view(ep['map_name'])
            plan = view['plan']
            point = next((p for p in plan['observations'] if p['spot_id'] == ep['observation_spot_id']), None)
            if not any(s['id'] == ep['observation_spot_id'] for s in view['spots']):
                raise HTTPException(422, 'The selected observation spot was deleted. Start a new episode.')
            excluded = set(plan.get('excluded_mission_ids', []))
            context['candidate_mission_ids'] = [mid for mid in (point or {}).get('mission_ids', []) if mid not in excluded]
            context.pop('allowed_mission_ids', None)
            context['plan_snapshot'] = deepcopy(plan)
            context['plan_revision'] = plan.get('revision', 0)
        if __package__:
            from .collection_observation_plan import capture_binding
        else:
            from collection_observation_plan import capture_binding
        context = capture_binding(ep, context)
        catalog = context.pop('mission_catalog_snapshot', None) or self.current_catalog(ep['map_name'])
        if isinstance(catalog, list):
            catalog = {'missions': catalog, 'snapshot_id': digest(catalog)}
        if not isinstance(catalog, dict) or not isinstance(catalog.get('missions'), list):
            raise HTTPException(422, 'Invalid captured mission catalog')
        catalog = {**catalog, 'snapshot_id': digest(catalog['missions'])}
        plan = context.pop('plan_snapshot', None)
        if plan is not None:
            # A saved episode may span plan edits. Keep each capture-time version.
            plan_id = digest(plan)
            ep.setdefault('observation_plans', {})[plan_id] = deepcopy(plan)
            context['plan_snapshot_id'] = plan_id
            write_json(directory / 'episode.json', ep)
        row = {'schema_version': 'reasoning_raw_v2', 'episode_id': eid, 'frame_id': fid,
               'image': {'path': f'images/{fid}.jpg', 'sha256': hashlib.sha256(data).hexdigest(),
                         'width': width, 'height': height},
               'capture': {'received_at_ms': int(frame['captured_at'] * 1000),
                           'source_stamp': frame['frame_stamp'],
                           **(capture_info or {'source': 'camera_capture'})},
               'saved_at_ms': now(), 'analysis_id': analysis_id, 'episode_type': episode_type_for(ep),
               'camera_topic': frame['camera_topic'], 'catalog_snapshot_id': catalog['snapshot_id'],
               'catalog': deepcopy(catalog),
               'metadata': metadata or {key: ep[key] for key in ('scenario', 'group_id', 'robot_type', 'environment') if key in ep},
               'context': context}
        image_path = self.path('raw', eid, 'images', fid + '.jpg')
        atomic(image_path, data)
        try:
            samples = rows(directory / 'samples.jsonl')
            sample = {'observation': row}
            if prediction is not None:
                sample['predictions'] = [{'frame_id': fid, 'recorded_at_ms': now(), **prediction}]
            samples.append(sample)
            self.save_samples(eid, samples)
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
        return row

    @serialized
    def instruction(self, eid, text):
        ep = self.episode(eid)
        if ep['ended_at'] or not text.strip():
            raise HTTPException(422, 'An active episode and instruction text are required')
        event = {'kind': 'operator_instruction', 'text': text.strip(), 'received_at_ms': now(),
                 'source': 'collect_label_context', 'executed': False}
        ep.setdefault('instructions', []).append(event)
        write_json(self.path('raw', eid, 'episode.json'), ep)
        return event

    @serialized
    def prediction(self, eid, fid, result, config):
        self.episode(eid)
        self.add_record(eid, 'predictions', {
            'frame_id': fid, 'recorded_at_ms': now(), 'config': config, 'prediction': result})

    @serialized
    def annotate(self, eid, request, *, source_catalog=None):
        ep = self.get(eid)
        frame = next((r for r in ep['observations'] if r['frame_id'] == request.frame_id), None)
        if not frame:
            raise HTTPException(422, 'Select a saved frame from this episode')
        self.image(eid, request.frame_id)
        if ep.get('capture_contract') == CAPTURE_CONTRACT:
            try:
                validate_episode_frame(ep, frame)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        current_catalog = ep['current_catalog']
        if not ep.get('observation_binding'):
            self.validate_target(request.model_dump(), frame, current_catalog)
        captured_scope = any(key in frame.get('context', {}) for key in ('candidate_mission_ids', 'allowed_mission_ids'))
        label_catalog = (recorded_catalog(frame, ep) if ep.get('observation_binding') else
                         source_catalog or (recorded_catalog(frame, ep) if captured_scope else current_catalog))
        self.validate_target(request.model_dump(), frame, label_catalog)
        if not request.annotator.strip() or not request.reason.strip():
            raise HTTPException(422, 'Annotator and visible evidence are required')
        old = next((r for r in ep['annotations'] if r['frame_ids'] == [request.frame_id]), None)
        revision = old['revision'] if old else 0
        if request.expected_revision != revision:
            raise HTTPException(409, 'Annotation changed. Reload the episode before editing.')
        label = {'schema_version': 'reasoning_label_v1', 'episode_id': eid,
                 'sample_id': old['sample_id'] if old else str(uuid4()), 'frame_ids': [request.frame_id],
                 'target': {k: getattr(request, k) for k in ('decision', 'mission_id', 'reason')},
                 'annotator': request.annotator.strip(), 'review_status': request.review_status,
                 'revision': revision + 1, 'recorded_at_ms': now(),
                 'reasoning_task': task_for(frame.get('context', {})),
                 'episode_type': episode_type_for(ep),
                 'catalog_snapshot_id': label_catalog['snapshot_id'],
                 'catalog': deepcopy(label_catalog)}
        self.invalidate_training(eid)
        if request.metadata is None:
            self.add_record(eid, 'annotations', label)
        else:
            samples = rows(self.path('raw', eid, 'samples.jsonl'))
            sample = next(s for s in samples if s['observation']['frame_id'] == request.frame_id)
            sample['observation']['metadata'] = request.metadata.model_dump(exclude_none=True)
            sample.setdefault('annotations', []).append(label)
            self.save_samples(eid, samples)
        return self.get(eid)

    @staticmethod
    def validate_target(target, frame, catalog):
        try:
            validate_decision_target(target, frame.get('context') or {}, catalog)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @serialized
    def image(self, eid, fid):
        ep = self.get(eid)
        row = next((r for r in ep['observations'] if r['frame_id'] == fid), None)
        if not row:
            raise HTTPException(404, 'Frame not found')
        path = self.path('raw', eid, 'images', fid + '.jpg')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['image']['sha256']:
            raise HTTPException(422, 'Saved image is missing or its checksum changed')
        return path

    def invalidate_training(self, eid):
        ep = self.episode(eid)
        if 'training' in ep:
            ep.pop('training')
            write_json(self.path('raw', eid, 'episode.json'), ep)
        for name in ('training_manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl', 'training_regions.jsonl'):
            self.path('raw', eid, name).unlink(missing_ok=True)

    @serialized
    def export(self, episode_ids):
        episodes = [self.get(eid) for eid in sorted(set(episode_ids))]
        if not episodes or any(not ep['ended_at'] for ep in episodes):
            raise HTTPException(409, 'End selected episodes before preparing training data')
        if len({ep.get('map_name') for ep in episodes}) != 1:
            raise HTTPException(422, 'Select episodes from one map')
        prepared = [self._training_files(ep) for ep in episodes]
        results = []
        for ep, (output, manifest) in zip(episodes, prepared):
            directory = self.path('raw', ep['id'])
            self.invalidate_training(ep['id'])
            try:
                for name, records in output.items():
                    if name == 'training_regions.jsonl' or not records:
                        continue
                    atomic(directory / name, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode())
                metadata = self.episode(ep['id'])
                metadata['training'] = {**manifest, 'regions': output['training_regions.jsonl']}
                write_json(directory / 'episode.json', metadata)
            except Exception:
                self.invalidate_training(ep['id'])
                raise
            results.append(manifest)
        if len(results) == 1:
            return results[0]
        return {'datasets': results, 'storage_path': str(self.path(episodes[0]['map_name'] or '_unassigned')),
                'counts': {k: sum(m['counts'][k] for m in results) for k in ('train', 'val', 'test')},
                'approved_regions': sum(m['approved_regions'] for m in results), 'warnings': []}

    def _training_files(self, ep):
        if __package__:
            from .collection_training import training_files
        else:
            from collection_training import training_files
        return training_files(self, ep)


def create_router(store):
    router = APIRouter(prefix='/collection', tags=['collection'])

    @router.get('/episodes')
    def list_episodes():
        return {'episodes': store.list(), 'storage_root': str(store.root)}

    @router.get('/reason-presets')
    def reason_presets(map_name: str):
        return {'map_name': map_name, 'presets': store.reason_presets(map_name)}

    @router.post('/episodes')
    def start(request: EpisodeRequest):
        return store.start(request)

    @router.get('/episodes/{eid}')
    def get(eid: str):
        return store.get(eid)

    @router.delete('/episodes/{eid}')
    def delete(eid: str):
        return store.delete(eid)

    @router.post('/episodes/{eid}/copy-to-map')
    def copy_to_map(eid: str, request: MapCopyRequest):
        return store.copy_to_map(eid, request)

    @router.post('/episodes/{eid}/end')
    def end(eid: str):
        return store.end(eid)

    @router.post('/episodes/{eid}/segment')
    def segment(eid: str, request: EpisodeRequest):
        return store.start_segment(eid, request)

    @router.post('/episodes/{eid}/return')
    def start_return(eid: str):
        return store.start_return(eid)

    @router.post('/episodes/{eid}/annotations')
    def annotate(eid: str, request: AnnotationRequest):
        return store.annotate(eid, request)

    @router.get('/episodes/{eid}/images/{fid}')
    def image(eid: str, fid: str):
        return FileResponse(store.image(eid, fid), media_type='image/jpeg')

    @router.post('/episodes/{eid}/instructions')
    def instruction(eid: str, request: InstructionRequest):
        return store.instruction(eid, request.text)

    @router.post('/prepare-training')
    def export(request: ExportRequest):
        return store.export(request.episode_ids)

    return router

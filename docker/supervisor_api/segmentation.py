"""SAM2 preview bridge and reviewed region persistence for saved Collect frames."""
import base64
import hashlib
import json
import os
import struct
import threading
import time
import zlib
from collections import OrderedDict
from typing import Literal
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .collection import append, atomic, rows, now

SAM2_URL = f"http://127.0.0.1:{int(os.getenv('SAM2_PORT', '7111'))}"


class Point(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    label: Literal[0, 1]


class PreviewRequest(BaseModel):
    frame_id: str = Field(min_length=1, max_length=36)
    points: list[Point] = Field(min_length=1, max_length=64)


class SaveRequest(BaseModel):
    preview_id: str = Field(min_length=1, max_length=36)
    frame_id: str = Field(min_length=1, max_length=36)
    object_id: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    object_label: str = Field(min_length=1, max_length=120)
    relevance: Literal['target', 'stop_evidence', 'context'] = 'target'
    mission_id: str | None = Field(default=None, max_length=128)
    annotator: str = Field(min_length=1, max_length=120)
    expected_revision: int = Field(default=0, ge=0)
    reviewed: Literal[True]


class BatchItem(SaveRequest):
    preview_id: str | None = Field(default=None, max_length=36)

class RemovedItem(BaseModel):
    object_id: str = Field(min_length=1, max_length=64)
    expected_revision: int = Field(ge=1)

class BatchSaveRequest(BaseModel):
    review_empty: bool = False
    frame_id: str = Field(min_length=1, max_length=36)
    items: list[BatchItem] = Field(default_factory=list, max_length=64)
    removed: list[RemovedItem] = Field(default_factory=list, max_length=64)


class RejectRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    annotator: str = Field(min_length=1, max_length=120)


def sam_request(path, payload=None):
    req = Request(SAM2_URL + path, headers={'Content-Type': 'application/json'},
                  data=None if payload is None else json.dumps(payload).encode())
    try:
        with urlopen(req, timeout=120 if payload else 3) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get('detail', 'SAM2 request failed')
        except ValueError:
            detail = 'SAM2 request failed'
        raise HTTPException(exc.code, detail) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(503, 'SAM2 is unavailable or still loading. Retry the mask.') from exc


def png(width, height, rgba):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b''.join(b'\x00' + rgba[y*width*4:(y+1)*width*4] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def mask_artifacts(rle, width, height):
    """Validate column-major COCO RLE, derive authoritative overlay/mask/bbox."""
    counts = rle.get('counts')
    if rle.get('size') != [height, width] or not isinstance(counts, list) or len(counts) > width * height + 1:
        raise HTTPException(502, 'SAM2 mask dimensions are invalid')
    if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != width * height:
        raise HTTPException(502, 'SAM2 mask RLE is invalid')
    rgba = bytearray(width * height * 4)
    binary = bytearray(width * height * 4)
    # Transparent background in overlay; opaque black/white in binary mask PNG.
    binary[3::4] = b'\xff' * (width * height)
    offset = 0; area = 0; xmin = width; ymin = height; xmax = -1; ymax = -1
    for index, count in enumerate(counts):
        if index % 2:
            area += count
            for pixel in range(offset, offset + count):
                x, y = divmod(pixel, height)
                target = (y * width + x) * 4
                rgba[target:target+4] = bytes((34, 197, 94, 115))
                binary[target:target+4] = b'\xff\xff\xff\xff'
                xmin = min(xmin, x); xmax = max(xmax, x); ymin = min(ymin, y); ymax = max(ymax, y)
        offset += count
    if area == 0:
        raise HTTPException(422, 'SAM2 returned an empty mask; add or move foreground points')
    return {'overlay': png(width, height, rgba), 'mask': png(width, height, binary),
            'area': area, 'bbox_xywh': [xmin, ymin, xmax-xmin+1, ymax-ymin+1]}


class SegmentationService:
    def __init__(self, store, start_container, request=sam_request):
        self.store = store
        self.start_container = start_container
        self.request = request
        self.inference_lock = threading.Lock()
        self.cache_lock = threading.RLock()
        self.previews = OrderedDict()

    def list_regions(self, eid):
        self.store.episode(eid)
        return list({r['object_id']: r for r in self.store.records(eid, 'regions')}.values())

    def preview(self, eid, req):
        if not self.inference_lock.acquire(blocking=False):
            raise HTTPException(409, 'A segmentation preview is already running')
        try:
            ep = self.store.get(eid)
            frame = next((r for r in ep['observations'] if r['frame_id'] == req.frame_id), None)
            if frame is None:
                raise HTTPException(404, 'Select a saved frame from this episode')
            width, height = frame['image']['width'], frame['image']['height']
            if width * height > 4_000_000 or any(p.x >= width or p.y >= height for p in req.points) or not any(p.label for p in req.points):
                raise HTTPException(422, 'Use in-image clicks with at least one foreground point (maximum 4 MP)')
            data = self.store.image(eid, req.frame_id).read_bytes()
            self.start_container()
            deadline = time.monotonic() + 15
            while True:
                try:
                    self.request('/health'); break
                except HTTPException:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.5)
            result = self.request('/segment', {'image_base64': base64.b64encode(data).decode(),
                'image_sha256': frame['image']['sha256'], 'points': [p.model_dump() for p in req.points]})
            artifacts = mask_artifacts(result['rle'], width, height)
            token = str(uuid4())
            preview = {'episode_id': eid, 'frame_id': req.frame_id, 'image_sha256': frame['image']['sha256'],
                       'points': [p.model_dump() for p in req.points], 'prediction': result,
                       'width': width, 'height': height, 'expires': time.monotonic() + 1800, **artifacts}
            with self.cache_lock:
                self.previews[token] = preview
                while len(self.previews) > 128:
                    self.previews.popitem(last=False)
            return {'preview_id': token, 'frame_id': req.frame_id,
                    'overlay_base64': base64.b64encode(artifacts['overlay']).decode(),
                    'area': artifacts['area'], 'bbox_xywh': artifacts['bbox_xywh'],
                    'score': result['score'], 'model': result['model'], 'elapsed_seconds': result['elapsed_seconds']}
        finally:
            self.inference_lock.release()

    def prepare(self, eid, req, review_status, current):
        old = next((r for r in current if r['object_id'] == req.object_id), None)
        preview = self.previews.get(req.preview_id)
        if req.preview_id is None and old and old['frame_id'] == req.frame_id:
            h,w = old['segmentation']['size']
            preview = {'episode_id':eid,'frame_id':req.frame_id,'expires':float('inf'),
                'image_sha256':old['image_sha256'], 'points':old['points'],
                'prediction':{**old.get('sam2',{}),'rle':old['segmentation']},
                **mask_artifacts(old['segmentation'],w,h)}
        if not preview or preview['expires'] < time.monotonic():
            raise HTTPException(410, 'Mask preview expired. Generate and review it again.')
        if preview['episode_id'] != eid or preview['frame_id'] != req.frame_id:
            raise HTTPException(422, 'Mask preview belongs to a different frame')
        if not req.object_label.strip() or not req.annotator.strip():
            raise HTTPException(422, 'Object label and annotator are required')
        image = self.store.image(eid, req.frame_id)
        if hashlib.sha256(image.read_bytes()).hexdigest() != preview['image_sha256']:
            raise HTTPException(409, 'Source image changed')
        ep = self.store.get(eid)
        catalog = ep['current_catalog']
        if req.mission_id is not None and req.mission_id not in {m['mission_id'] for m in catalog['missions']}:
            raise HTTPException(422, 'Select a mission from the current Design catalog')
        frame_regions = [r for r in current if r['frame_id'] == req.frame_id]
        if (old['revision'] if old else 0) != req.expected_revision or (old and old['frame_id'] != req.frame_id):
            raise HTTPException(409, 'Region changed or belongs to another frame. Reload it before saving.')
        revision = req.expected_revision + 1
        relative = f'masks/{req.object_id}/{revision}.png'
        path = self.store.path('raw', eid, 'masks', req.object_id, f'{revision}.png')
        record = {'schema_version': 'reasoning_region_v1', 'episode_id': eid, 'frame_id': req.frame_id,
            'instance_number': old.get('instance_number') if old and old.get('instance_number') else max([r.get('instance_number',i+1) for i,r in enumerate(frame_regions)] or [0])+1,
            'object_id': req.object_id, 'object_label': req.object_label.strip(), 'relevance': req.relevance,
            'mission_id': req.mission_id, 'annotator': req.annotator.strip(), 'revision': revision,
            'catalog_snapshot_id': catalog['snapshot_id'],
            'review_status': review_status, 'recorded_at_ms': now(), 'image_sha256': preview['image_sha256'],
            'points': preview['points'], 'segmentation': preview['prediction']['rle'],
            'bbox_xywh': preview['bbox_xywh'], 'area': preview['area'],
            'mask': {'path': relative, 'sha256': hashlib.sha256(preview['mask']).hexdigest()},
            'sam2': {k: v for k, v in preview['prediction'].items() if k != 'rle'}}
        return path, record, preview['mask']

    def save(self, eid, req, review_status="approved"):
        with self.cache_lock, self.store.lock:
            path, record, data = self.prepare(eid, req, review_status, self.list_regions(eid))
            try:
                atomic(path, data)
                try:
                    self.store.invalidate_training(eid)
                    self.store.add_record(eid, 'regions', record)
                except Exception:
                    path.unlink(missing_ok=True); raise
            except OSError as exc:
                raise HTTPException(507, 'Mask storage failed. Check disk space and retry.') from exc
            return record

    def save_batch(self, eid, req):
        with self.cache_lock, self.store.lock:
            ids = [item.object_id for item in req.items] + [item.object_id for item in req.removed]
            if len(ids) != len(set(ids)): raise HTTPException(422, 'Duplicate instance in batch')
            current = self.list_regions(eid)
            prepared = []
            for item in req.items:
                if item.frame_id != req.frame_id: raise HTTPException(422, 'Batch belongs to one frame')
                entry = self.prepare(eid,item,'approved',current)
                prepared.append(entry)
                current = [r for r in current if r['object_id'] != item.object_id] + [entry[1]]
            removed = []
            for item in req.removed:
                old = next((r for r in current if r['object_id'] == item.object_id),None)
                if not old or old['frame_id'] != req.frame_id or old['revision'] != item.expected_revision:
                    raise HTTPException(409, 'Instance changed; reload before removing')
                removed.append({**old,'revision':old['revision']+1,'review_status':'rejected',
                                'annotator':'collect-ui','recorded_at_ms':now()})
            samples = rows(self.store.path('raw',eid,'samples.jsonl'))
            sample = next((s for s in samples if s['observation']['frame_id'] == req.frame_id),None)
            if sample is None: raise HTTPException(422, 'Frame not found')
            records = [entry[1] for entry in prepared] + removed
            if req.review_empty:
                latest = {r['object_id']: r for r in current + removed}
                if any(r['frame_id'] == req.frame_id and r['review_status'] != 'rejected' for r in latest.values()):
                    raise HTTPException(409, 'Frame has masks; reload before saving an empty review')
                sample['observation']['mask_review'] = {'status':'empty', 'annotator':'collect-ui', 'reviewed_at_ms':now()}
            elif records:
                sample['observation'].pop('mask_review', None)
            sample.setdefault('regions',[]).extend(records)
            written = []
            try:
                for path, record, data in prepared:
                    atomic(path,data); written.append(path)
                self.store.invalidate_training(eid)
                # Publish all instance revisions in one atomic samples.jsonl update.
                self.store.save_samples(eid,samples)
            except Exception:
                for path in written: path.unlink(missing_ok=True)
                raise
            return {'regions':records}

    def save_predictions(self, eid, frame, label, prepared, resume=False, replace_unreviewed=False, generation=None):
        """One dataset read and one publication per automatically labeled frame."""
        with self.store.lock:
            directory = self.store.path('raw',eid)
            samples = rows(directory/'samples.jsonl')
            sample = next((s for s in samples if s['observation']['frame_id']==frame['frame_id']),None)
            if sample is None: raise HTTPException(404,'Frame no longer exists')
            if sample['observation'].get('mask_review', {}).get('status') == 'empty': return 0
            actual = sample['observation']['image']
            if actual['sha256'] != frame['image']['sha256'] or hashlib.sha256((directory/'images'/f"{frame['frame_id']}.jpg").read_bytes()).hexdigest() != actual['sha256']:
                raise HTTPException(409,'Source image changed')
            latest = list({r['object_id']:r for r in sample.get('regions',[])}.values())
            removed = set()
            if replace_unreviewed:
                if latest and all(r['review_status'] != 'draft' for r in latest): return 0
                removed = {r['object_id'] for r in latest
                           if r['review_status'] == 'draft' and r.get('annotator') == 'sam-auto'}
                latest = [r for r in latest if r['object_id'] not in removed]
                sample['regions'] = [r for r in sample.get('regions',[]) if r['object_id'] not in removed]
            labels = [label] if isinstance(label,str) else label
            matching = [r for r in latest if r['object_label'] in labels]
            blocked = {r['object_label'] for r in matching if r['review_status']=='approved' or (not resume and r['review_status']=='draft')}
            fingerprints = {json.dumps(r['segmentation'],sort_keys=True) for r in matching}
            number = max([r.get('instance_number',i+1) for i,r in enumerate(latest)] or [0])
            records=[]; files=[]
            for prediction, artifacts in prepared:
                object_label=prediction.get('object_label', label if isinstance(label,str) else None)
                if object_label not in labels: raise HTTPException(422,'Untrained object label')
                if object_label in blocked: continue
                fingerprint=json.dumps(prediction['rle'],sort_keys=True)
                if fingerprint in fingerprints: continue
                fingerprints.add(fingerprint); number+=1
                oid='auto-'+uuid4().hex
                relative=f'masks/{oid}/1.png'
                records.append({'schema_version':'reasoning_region_v1','episode_id':eid,'frame_id':frame['frame_id'],
                    'object_id':oid,'instance_number':number,'object_label':object_label,'relevance':'target','mission_id':None,
                    'annotator':'sam-auto','revision':1,'review_status':'draft','recorded_at_ms':now(),
                    'image_sha256':actual['sha256'],'points':[],'segmentation':prediction['rle'],
                    'bbox_xywh':artifacts['bbox_xywh'],'area':artifacts['area'],
                    'mask':{'path':relative,'sha256':hashlib.sha256(artifacts['mask']).hexdigest()},
                    'sam2':{k:v for k,v in prediction.items() if k!='rle'}})
                files.append((directory/relative,artifacts['mask']))
            if not records and not removed and not generation: return 0
            if generation:
                sample['observation'].setdefault('auto_labeling', {})['sam'] = generation
            sample.setdefault('regions',[]).extend(records)
            written=[]
            try:
                for path,data in files:
                    atomic(path,data);written.append(path)
                self.store.invalidate_training(eid)
                self.store.save_samples(eid,samples)
            except Exception:
                for path in written: path.unlink(missing_ok=True)
                raise
            return len(records)

    def reject(self, eid, oid, req):
        with self.store.lock:
            old = next((r for r in self.list_regions(eid) if r['object_id'] == oid), None)
            if not old or old['revision'] != req.expected_revision:
                raise HTTPException(409, 'Region changed; reload before rejecting')
            if not req.annotator.strip():
                raise HTTPException(422, 'Annotator is required')
            record = {**old, 'revision': old['revision'] + 1, 'review_status': 'rejected',
                      'annotator': req.annotator.strip(), 'recorded_at_ms': now()}
            try:
                self.store.invalidate_training(eid)
                self.store.add_record(eid, 'regions', record)
            except OSError as exc:
                raise HTTPException(507, 'Region rejection could not be saved') from exc
            return record


def create_router(service):
    router = APIRouter(prefix='/collection', tags=['segmentation'])

    # /segment belongs to collection's capture-session transitions.
    @router.post('/episodes/{eid}/segmentation/preview')
    def preview(eid: str, req: PreviewRequest):
        return service.preview(eid, req)

    @router.get('/episodes/{eid}/regions')
    def regions(eid: str):
        with service.store.lock:
            return {'regions': service.list_regions(eid)}

    @router.post('/episodes/{eid}/regions/batch')
    def save_batch(eid:str, req:BatchSaveRequest):
        return service.save_batch(eid,req)

    @router.post('/episodes/{eid}/regions')
    def save(eid: str, req: SaveRequest):
        return service.save(eid, req)

    @router.post('/episodes/{eid}/regions/{oid}/reject')
    def reject(eid: str, oid: str, req: RejectRequest):
        return service.reject(eid, oid, req)

    @router.get('/episodes/{eid}/regions/{oid}/overlay')
    def overlay(eid: str, oid: str):
        with service.store.lock:
            region = next((r for r in service.list_regions(eid) if r['object_id'] == oid), None)
            if not region:
                raise HTTPException(404, 'Region not found')
            height, width = region['segmentation']['size']
            artifacts = mask_artifacts(region['segmentation'], width, height)
            return Response(artifacts['overlay'], media_type='image/png', headers={'Cache-Control': 'no-store'})

    @router.post('/segmentation/unload')
    def unload():
        return service.request('/unload', {})

    return router

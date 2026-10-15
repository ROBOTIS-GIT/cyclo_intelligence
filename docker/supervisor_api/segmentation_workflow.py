"""Reviewed examples -> SAM decoder adaptation -> draft masks -> human review."""
import base64
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from .collection import now, write_json
from .segmentation import RejectRequest, mask_artifacts

from .labeling_automation import (AutomaticTrainingRequest, automatic_steps, allocate_name, manual_sample_ids, remove_model)

SAMPLE_SCHEMA = 'user_count_v1'

class TrainingRequest(BaseModel):
    model_name: str | None = Field(default=None, min_length=1, max_length=80)
    map_name: str = Field(min_length=1, max_length=128)
    steps: int = Field(default=1000, ge=2, le=2000)
    episode_id: str | None = None
    episode_ids: list[str] | None = Field(default=None, min_length=1, max_length=500)
    sampled_only: bool = False
    automatic: bool = False

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, value):
        if value is None: return None
        value = value.strip()
        if not value: raise ValueError('Enter a model name')
        if any(ord(char) < 32 for char in value): raise ValueError('Model name must be one line')
        return value


class TrainingCancelRequest(BaseModel):
    job_id: str = Field(pattern=r'^[a-f0-9]{32}$')


class ObjectLabelRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)

class SampleRequest(BaseModel):
    count: int = Field(ge=1, le=2000, strict=True)

class BatchRequest(BaseModel):
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    episode_ids: list[str] = Field(min_length=1, max_length=500)
    replace_unreviewed: bool = False
    retry_frame_ids: list[str] = Field(default_factory=list, max_length=64)

class ReviewRequest(RejectRequest):
    reviewed: bool = True

class SegmentationWorkflow:
    def __init__(self, service, adapter_root=None):
        self.service = service
        self.store = service.store
        self.adapter_root = Path(adapter_root or os.getenv(
            'SAM2_ADAPTER_ROOT', '/workspace/model/sam2/adapters'))
        self.job = {'status':'idle'}
        self.training_job = {'status':'idle'}
        self.lock = threading.Lock()

    def stored_models(self):
        """Read completed model metadata without requiring the SAM container."""
        models = []
        if not self.adapter_root.is_dir(): return models
        for path in self.adapter_root.glob('*/model.json'):
            try:
                model = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(model, dict) or not model.get('id'): continue
                models.append(model)
            except (OSError, ValueError):
                continue
        return sorted(models, key=lambda item: (
            item.get('created_at_ms', item.get('created_at', 0)), item.get('id', '')), reverse=True)

    def object_labels(self, eid):
        with self.store.lock:
            current = self.store.episode(eid)
            labels = set()
            for ep in self.store.list():
                if ep.get('map_name') != current.get('map_name'): continue
                labels.update(ep.get('object_labels', []))
                labels.update(r['object_label'] for r in self.service.list_regions(ep['id'])
                              if r['review_status'] == 'approved')
            return {'labels': sorted(labels)}

    def add_object_label(self, eid, req):
        label = req.label.strip()
        if not label: raise HTTPException(422, 'Enter an object label')
        with self.store.lock:
            ep = self.store.episode(eid)
            ep['object_labels'] = sorted(set(ep.get('object_labels', [])) | {label})
            write_json(self.store.path('raw',eid,'episode.json'), ep)
            return self.object_labels(eid)

    def sample(self, eid, req):
        with self.store.lock:
            ep = self.store.get(eid)
            if not ep.get('ended_at'): raise HTTPException(422, 'Save the episode before sampling')
            frames = ep['observations']
            if not frames: raise HTTPException(422, 'This episode has no frames')
            if req.count > len(frames):
                raise HTTPException(422, f'This episode has only {len(frames)} frames. Choose a smaller sample count.')
            count = req.count
            rng = random.SystemRandom()
            # One random frame per equal-sized interval, without replacement.
            selected = [frames[rng.randrange(i*len(frames)//count, (i+1)*len(frames)//count)]['frame_id']
                        for i in range(count)]
            metadata = self.store.episode(eid)
            metadata['sam_sample'] = {'schema_version':SAMPLE_SCHEMA, 'frame_ids':selected,
                                     'requested_count':req.count, 'strategy':'stratified_random_v1'}
            write_json(self.store.path('raw',eid,'episode.json'), metadata)
            return self.store.get(eid)

    def summary(self):
        labels = {}
        counts = {}
        for ep in self.store.list():
            if not ep.get('ended_at'): continue
            for region in self.service.list_regions(ep['id']):
                if region['review_status'] != 'approved': continue
                key = (ep['map_name'], region['object_label'])
                counts[key] = counts.get(key,0) + 1
                labels.setdefault(key, set()).add((ep['id'], region['frame_id']))
        return [{'map_name':key[0], 'object_label':key[1], 'frame_count':len(frames), 'instance_count':counts[key]} for key,frames in sorted(labels.items())]

    def status(self):
        stored = {model['id']:model for model in self.stored_models()}
        try:
            engine = self.service.request('/adaptation')
            self.training_job = dict(engine.get('training', {}))
            # Live state owns training progress; disk is the source of truth for
            # completed models when the service has just started or is stopped.
            models = dict(stored)
            models.update({model['id']:model for model in engine.get('models',[])
                           if model.get('id')})
            engine = {**engine, 'models':list(models.values())}
        except HTTPException:
            engine = {'training':{'status':'unavailable'}, 'models':list(stored.values())}
        return {**engine, 'batch':dict(self.job), 'labels':[]}

    def ready(self):
        self.service.start_container()
        deadline = time.monotonic()+20
        while True:
            try: return self.service.request('/health')
            except HTTPException:
                if time.monotonic() > deadline: raise
                time.sleep(.5)

    def train(self, req):
        with self.lock:
            return self._train(req)

    def _train(self, req):
        if self.job['status'] == 'running': raise HTTPException(409, 'Auto labeling is running')
        if req.episode_ids is not None and req.episode_id is not None:
            raise HTTPException(422, 'Choose episode_ids or episode_id, not both')
        examples = []
        with self.store.lock:
            if req.episode_ids is not None:
                episodes = [self.store.get(eid) for eid in dict.fromkeys(req.episode_ids)]
                if any(ep.get('map_name') != req.map_name or not ep.get('ended_at') for ep in episodes):
                    raise HTTPException(422, 'Use saved episodes from the selected map')
            else:
                episodes = [ep for ep in self.store.list() if ep.get('map_name') == req.map_name
                            and ep.get('ended_at') and (not req.episode_id or ep['id'] == req.episode_id)]
            for ep in episodes:
                sample = ep.get('sam_sample',{})
                ids = sample.get('frame_ids',[]) if sample.get('schema_version') == SAMPLE_SCHEMA else []
                sampled = manual_sample_ids(self.store, ep, 'sam') if req.automatic else set(ids)
                regions = self.service.list_regions(ep['id'])
                if req.episode_ids is not None and req.sampled_only:
                    if not ids or len(sampled) != len(ids):
                        raise HTTPException(422, 'Sample frames in every selected episode before training')
                    observation_ids = {frame['frame_id'] for frame in ep['observations']}
                    if not sampled <= observation_ids:
                        raise HTTPException(409, 'Sampled frames changed; sample again')
                # A frame with draft instances is not reviewed as a whole.
                pending_frames = {r['frame_id'] for r in regions if r['review_status'] == 'draft'}
                by_frame = {}
                for region in regions:
                    if region['review_status'] == 'approved' and region['frame_id'] not in pending_frames and (not req.sampled_only or region['frame_id'] in sampled):
                        by_frame.setdefault(region['frame_id'], []).append(region)
                for fid, frame_regions in by_frame.items():
                    raw = self.store.image(ep['id'],fid).read_bytes()
                    image_sha256 = hashlib.sha256(raw).hexdigest()
                    for region in frame_regions:
                        if image_sha256 != region['image_sha256']:
                            raise HTTPException(409, 'Training image changed')
                        examples.append({'episode_id':ep['id'], 'frame_id':fid,
                            'object_id':region['object_id'], 'object_label':region['object_label'],
                            'image_base64':base64.b64encode(raw).decode(), 'image_sha256':region['image_sha256'],
                            'rle':region['segmentation']})
        if not 2 <= len(examples) <= 4096:
            raise HTTPException(422, 'Training requires 2–4096 reviewed instances from reviewed frames on the same map')
        if len({(item['episode_id'], item['frame_id']) for item in examples}) > 1024:
            raise HTTPException(422, 'Training supports at most 1024 reviewed frames across the selected episodes')
        if req.automatic:
            req = req.model_copy(update={'model_name':allocate_name(self.adapter_root, 'sam', req.map_name, self.stored_models()),
                                         'steps':automatic_steps('sam', len(examples))})
        self.ready()
        result = self.service.request('/adaptation/train', {**req.model_dump(), 'examples':examples})
        self.training_job = {**result, 'map_name':req.map_name,
                             'episode_ids':sorted({item['episode_id'] for item in examples})}
        return result

    def cancel_training(self, job_id):
        # The SAM worker handles cancellation without stopping annotation serving.
        result=self.service.request('/adaptation/cancel', {'job_id':job_id})
        self.training_job=dict(result)
        return self.status()

    def delete_model(self, model_id):
        with self.lock:
            if self.job['status'] == 'running': raise HTTPException(409, 'Auto labeling is running')
            state = self.status()
            if state.get('training', {}).get('status') in ('queued', 'training', 'cancelling'):
                raise HTTPException(409, 'SAM training is running')
            if state.get('training', {}).get('status') != 'unavailable':
                # Clears adapter caches and refuses while inference owns the engine.
                self.service.request('/unload', {})
            return remove_model(self.adapter_root, model_id, 1)

    def start_batch(self, req):
        with self.lock:
            if self.job['status'] == 'running': raise HTTPException(409, 'Auto labeling is already running')
            self.ready()
            state = self.service.request('/adaptation')
            if state['training']['status'] in ('queued','training','cancelling'): raise HTTPException(409, 'SAM training is running')
            model = next((m for m in state['models'] if m['id']==req.model_id),None)
            if not model: raise HTTPException(404, 'Choose a completed SAM model')
            if model.get('strategy') != 'sam_multiclass_instances_v1':
                raise HTTPException(422, 'Retrain with all sampled labels for class-aware SAM masks')
            labels=model.get("object_labels",[])
            if not labels: raise HTTPException(422,"Model has no trained object labels")
            trained_frames={(item.get('episode_id'),item.get('frame_id'))
                            for item in model.get('training_samples',[])}
            frames=[]; excluded_training=0
            for eid in dict.fromkeys(req.episode_ids):
                ep=self.store.get(eid)
                if not ep.get('ended_at') or ep['map_name'] != model['map_name']:
                    raise HTTPException(422, 'Select saved episodes from the model map')
                regions = self.service.list_regions(eid)
                drafts = {r['frame_id'] for r in regions if r['review_status'] == 'draft' and r.get('annotator') == 'sam-auto'}
                reviewed = {r['frame_id'] for r in regions if r['review_status'] == 'approved'} - {r['frame_id'] for r in regions if r['review_status'] == 'draft'}
                existing={(r['frame_id'],r['object_label']) for r in regions
                          if r['review_status'] in ('approved','draft')}
                for frame in ep['observations']:
                    if (not req.replace_unreviewed
                        and frame.get('auto_labeling', {}).get('sam', {}).get('status') == 'generated'
                        and any(fid == frame['frame_id'] for fid, _ in existing)):
                        continue
                    if frame.get('mask_review', {}).get('status') == 'empty':
                        continue
                    if (eid,frame['frame_id']) in trained_frames:
                        excluded_training+=1
                        continue
                    if req.replace_unreviewed and frame['frame_id'] in reviewed: continue
                    if (req.replace_unreviewed and frame['frame_id'] in drafts) or any((frame['frame_id'],label) not in existing for label in labels) or frame['frame_id'] in req.retry_frame_ids:
                        frames.append((eid,frame))
            if not frames: raise HTTPException(422, 'No eligible frames. Training frames are excluded and existing masks are preserved.')
            self.job={'id':uuid4().hex,'status':'running','excluded_training':excluded_training,'total':len(frames),'processed':0,'created':0,'empty':0,'failed':0,'retry_frame_ids':req.retry_frame_ids,'replace_unreviewed':req.replace_unreviewed,'model_id':req.model_id,'model_name':model.get('model_name'),'episode_ids':req.episode_ids,'object_labels':labels}
            threading.Thread(target=self.batch,args=(req.model_id,model,frames),daemon=True).start()
            return dict(self.job)

    def batch(self, model_id, model, frames):
        try:
            for eid, frame in frames:
                try:
                    frame_started=time.monotonic()
                    raw=self.store.path('raw',eid,'images',frame['frame_id']+'.jpg').read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=frame['image']['sha256']:
                        raise HTTPException(409,'Source image changed')
                    inference_started=time.monotonic()
                    result=self.service.request('/adaptation/predict',{'model_id':model_id,'image_base64':base64.b64encode(raw).decode()})
                    inference_seconds=time.monotonic()-inference_started
                    w,h=frame['image']['width'],frame['image']['height']
                    # Legacy single-result shape is retained for in-flight worker compatibility.
                    common={k:v for k,v in result.items() if k not in ('instances','rle','score')}
                    candidates=result.get('instances', [result] if 'rle' in result else [])
                    candidates=[r for r in candidates if sum(r['rle']['counts'][1::2]) > 0]
                    if not candidates:
                        self.job['empty']+=1
                    if result.get('strategy') != 'sam_multiclass_instances_v1' or any(r.get('object_label') not in model['object_labels'] for r in candidates):
                        raise HTTPException(422,'Model returned untrained or missing object labels')
                    # Validate all candidates before writing any masks for this frame.
                    prepared=[({**common,**r},mask_artifacts(r['rle'],w,h)) for r in candidates]
                    save_started=time.monotonic()
                    created=self.service.save_predictions(eid,frame,model['object_labels'],prepared,
                        resume=frame['frame_id'] in self.job.get('retry_frame_ids',[]),
                        replace_unreviewed=self.job.get('replace_unreviewed',False),
                        generation={'status':'generated','job_id':self.job.get('id'),'model_id':model_id})
                    self.job['created']+=created
                    self.job['last_frame_seconds']=round(time.monotonic()-frame_started,3)
                    self.job['last_inference_seconds']=round(inference_seconds,3)
                    self.job['last_save_seconds']=round(time.monotonic()-save_started,3)
                except Exception as exc:
                    self.job['failed']+=1
                    self.job['error']=getattr(exc,'detail',str(exc))
                finally: self.job['processed']+=1
            self.job['status']='completed' if not self.job['failed'] else 'completed_with_errors'
        except Exception as exc:
            self.job.update(status='failed',error=str(exc))

    def approve(self,eid,oid,req):
        if not req.reviewed: raise HTTPException(422,'Review the mask before approval')
        with self.store.lock:
            old=next((r for r in self.service.list_regions(eid) if r['object_id']==oid),None)
            if not old or old['revision']!=req.expected_revision: raise HTTPException(409,'Mask changed; reload it')
            if not req.annotator.strip(): raise HTTPException(422,'Annotator is required')
            if hashlib.sha256(self.store.image(eid,old['frame_id']).read_bytes()).hexdigest()!=old['image_sha256']:
                raise HTTPException(409,'Source image changed')
            updated={**old,'review_status':'approved','revision':old['revision']+1,
                     'annotator':req.annotator,'recorded_at_ms':now()}
            self.store.invalidate_training(eid)
            self.store.add_record(eid,'regions',updated)
            return updated


def create_router(workflow):
    router=APIRouter(prefix='/collection',tags=['segmentation workflow'])
    @router.get('/episodes/{eid}/segmentation/labels')
    def labels(eid:str): return workflow.object_labels(eid)
    @router.post('/episodes/{eid}/segmentation/labels')
    def add_label(eid:str, req:ObjectLabelRequest): return workflow.add_object_label(eid,req)
    @router.post('/episodes/{eid}/segmentation/sample')
    def sample(eid:str, req:SampleRequest): return workflow.sample(eid,req)
    @router.get('/segmentation/workflow')
    def status(): return workflow.status()
    @router.post('/segmentation/workflow/start')
    def start():
        workflow.ready()
        return workflow.status()
    @router.post('/segmentation/auto-train')
    def auto_train(req:AutomaticTrainingRequest):
        return workflow.train(TrainingRequest(map_name=req.map_name, sampled_only=True, automatic=True))
    @router.delete('/segmentation/models/{model_id}')
    def delete_model(model_id:str): return workflow.delete_model(model_id)
    @router.post('/segmentation/train')
    def train(req:TrainingRequest): return workflow.train(req)
    @router.post('/segmentation/training/cancel')
    def cancel_training(req:TrainingCancelRequest): return workflow.cancel_training(req.job_id)
    @router.post('/segmentation/auto-label')
    def batch(req:BatchRequest): return workflow.start_batch(req)
    @router.post('/episodes/{eid}/regions/{oid}/approve')
    def approve(eid:str,oid:str,req:ReviewRequest): return workflow.approve(eid,oid,req)
    return router

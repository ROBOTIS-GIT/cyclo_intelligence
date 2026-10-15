"""Sampled human labels -> named Qwen adapter -> reviewed frame annotations."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re
import threading
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from .collection import AnnotationRequest, episode_type_for, now, read_json, write_json
from .qwen_annotation_contract import SCHEMA, parse_target, sample_for
from .mission_decisions import recorded_catalog
from .reasoning_training_metadata import training_metadata
from .qwen_training_balance import balance_frames
from .collection_episode_contract import frame_matches_episode
from .qwen_base_models import BASE_MODELS
from .qwen_annotation_runtime import JobCancelled

from .labeling_automation import (AutomaticTrainingRequest, automatic_steps, allocate_name, manual_sample_ids, saved_sample_episodes, remove_model)

ACTIVE = ('queued','training','running','cancelling')
TRAINING_KINDS = ('training','reasoning_training')
SAMPLE_SCHEMA = 'user_count_v1'


def training_base(request):
    if request.base_model:
        return request.base_model, BASE_MODELS[request.base_model]
    return (os.getenv('QWEN_ANNOTATION_MODEL_ID', 'Qwen/Qwen3.5-4B'),
            os.getenv('QWEN_ANNOTATION_MODEL_REVISION', BASE_MODELS['Qwen/Qwen3.5-4B']))


def clean_model_name(value):
    value=value.strip()
    if not value or any(ord(c)<32 for c in value): raise ValueError('Enter a single-line model name')
    return value

class CancelRequest(BaseModel):
    job_id: str = Field(pattern=r'^[a-f0-9]{32}$')


class SampleRequest(BaseModel):
    count: int = Field(ge=1,le=2000,strict=True)

class TrainRequest(BaseModel):
    sampled_only: bool = True
    automatic: bool = False
    map_name: str = Field(min_length=1,max_length=128)
    episode_ids: list[str] = Field(min_length=1,max_length=500)
    model_name: str = Field(min_length=1,max_length=80)
    base_model: Literal['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B'] | None = None
    steps: int = Field(default=200, ge=1, strict=True)
    batch_size: int = Field(default=4, ge=1, le=32, strict=True)

    @field_validator('model_name')
    @classmethod
    def name(cls,value):
        return clean_model_name(value)

class ReasoningDatasetRequest(BaseModel):
    map_name: str = Field(min_length=1,max_length=128)
    episode_ids: list[str] = Field(min_length=1,max_length=500)
    balance_decisions: bool = False

class ReasoningTrainRequest(ReasoningDatasetRequest):
    model_name: str = Field(min_length=1,max_length=80)
    base_model: Literal['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B'] | None = None
    steps: int = Field(default=2000, ge=1, strict=True)
    batch_size: int = Field(default=8, ge=1, le=32, strict=True)

    @field_validator('model_name')
    @classmethod
    def name(cls,value):
        return clean_model_name(value)

class BatchRequest(BaseModel):
    replace_unreviewed: bool = True
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    episode_ids: list[str] = Field(min_length=1,max_length=500)


class QwenAnnotationWorkflow:
    def __init__(self, store, runtime, root='/workspace/model/qwen/annotation',
                 reasoning_root=None, recover=True):
        self.store,self.runtime,self.root=store,runtime,Path(root)
        self.reasoning_root=Path(reasoning_root) if reasoning_root else self.root.parent/'reasoning'
        self.lock=threading.RLock()
        state=read_json(self.root/'workflow.json') if (self.root/'workflow.json').exists() else {}
        self.jobs={kind:state.get(kind,{'status':'idle'})
                   for kind in ('training','batch','reasoning_training')}
        if recover:
            for kind,job in self.jobs.items():
                if job.get('status') == 'cancelling':
                    threading.Thread(target=self.cancel_job,args=(kind,job['id']),daemon=True).start()
                elif job.get('status') in ACTIVE:
                    threading.Thread(target=self._run,args=(kind,deepcopy(job)),daemon=True).start()

    def models(self):
        paths=list(self.root.glob('*/*/model.json'))+list(self.reasoning_root.glob('*/*/model.json'))
        return [read_json(path) for path in paths
                if read_json(path).get('schema')==SCHEMA]

    def status(self):
        with self.lock:
            result={kind:{k:deepcopy(v) for k,v in job.items() if k not in ('results','request_path')}
                    for kind,job in self.jobs.items()}
        return {**result,'models':self.models()}

    def reset_job(self, kind):
        with self.lock:
            if kind not in self.jobs:
                raise HTTPException(404, 'Qwen job type not found')
            if self.jobs[kind].get('status') in ACTIVE:
                raise HTTPException(409, 'Stop the running job before clearing its state')
            if self.jobs[kind].get('status') not in ('failed', 'completed_with_errors', 'cancelled', 'idle'):
                raise HTTPException(409, 'Only a stopped or failed job can be cleared')
            self.jobs[kind]={'status':'idle'}
            self._persist()
            return self.status()

    def cancel_job(self, kind, job_id):
        if kind not in TRAINING_KINDS:
            raise HTTPException(404, 'Qwen training job type not found')
        with self.lock:
            job = self.jobs[kind]
            if job.get('id') != job_id:
                raise HTTPException(409, 'Training job changed; refresh before cancelling')
            if job.get('status') == 'cancelled':
                return self.status()
            if job.get('status') not in ACTIVE:
                raise HTTPException(409, 'This training job is no longer running')
            job.update(status='cancelling', phase='Stopping training')
            job.pop('error', None)
            self._persist()
        # Do not hold the state lock while waiting for Docker. Progress remains readable.
        try:
            self.runtime.cancel(job_id)
        except Exception as exc:
            with self.lock:
                if self.jobs[kind].get('id') == job_id and self.jobs[kind]['status'] == 'cancelling':
                    self.jobs[kind]['error'] = 'Could not stop training: ' + str(exc)
                    self._persist()
            raise HTTPException(503, 'Could not stop training; retry cancellation') from exc
        with self.lock:
            if self.jobs[kind].get('id') == job_id and self.jobs[kind]['status'] == 'cancelling':
                self.jobs[kind].update(status='cancelled', phase='Cancelled')
                self.jobs[kind].pop('error', None)
                self._persist()
            return self.status()

    def _running_job(self, kind, job_id):
        """Caller holds the state lock; stale workers never mutate replacement jobs."""
        job = self.jobs[kind]
        if job.get('id') != job_id or job.get('status') not in ('queued', 'training', 'running'):
            raise JobCancelled('Qwen training cancelled')
        return job

    def _idle(self):
        if any(job.get('status') in ACTIVE for job in self.jobs.values()):
            raise HTTPException(409,'A Qwen annotation job is already running')

    def _persist(self):
        write_json(self.root/'workflow.json',self.jobs)

    def sample(self,eid,req):
        with self.lock,self.store.lock:
            self._idle()
            ep=self.store.get(eid)
            frames=[f for f in ep['observations'] if frame_matches_episode(ep,f)]
            if not ep.get('ended_at'): raise HTTPException(422,'Save the episode before sampling')
            if req.count>len(frames): raise HTTPException(422,f'Choose at most {len(frames)} frames')
            rng=random.SystemRandom()
            ids=[frames[rng.randrange(i*len(frames)//req.count,(i+1)*len(frames)//req.count)]['frame_id']
                 for i in range(req.count)]
            metadata=self.store.episode(eid)
            metadata['qwen_sample']={'schema_version':SAMPLE_SCHEMA,'frame_ids':ids,
                                     'requested_count':req.count,'strategy':'stratified_random_v1'}
            write_json(self.store.path('raw',eid,'episode.json'),metadata)
            return self.store.get(eid)

    def _frame(self,ep,frame,catalog):
        path=self.store.image(ep['id'],frame['frame_id'])
        old=next((a for a in ep['annotations'] if a['frame_ids']==[frame['frame_id']]),None)
        return {'episode_id':ep['id'],'frame_id':frame['frame_id'],
                'image':str(path.relative_to(self.store.root)),'image_sha256':frame['image']['sha256'],
                'frame':deepcopy(frame),'catalog':deepcopy(catalog),
                'expected_revision':old['revision'] if old else 0}

    def _submit(self,kind,request,metadata):
        job_id=uuid4().hex
        request_path=self.root/'jobs'/job_id/'request.json'
        request={**request,'kind':kind,'job_id':job_id}
        write_json(request_path,request)
        job={**metadata,'id':job_id,'status':'queued','request_path':str(request_path),
             'results':{},'processed':0,'created':0,'failed':0,'skipped':0,'step':0}
        self.jobs[kind]=job; self._persist()
        threading.Thread(target=self._run,args=(kind,deepcopy(job)),daemon=True).start()
        return self.status()[kind]

    def train(self,req):
        with self.lock,self.store.lock:
            self._idle()
            catalog=self.store.current_catalog(req.map_name)
            episodes=[]
            frames=[]
            for eid in dict.fromkeys(req.episode_ids):
                ep=self.store.get(eid)
                if not ep.get('ended_at') or ep.get('map_name')!=req.map_name:
                    raise HTTPException(422,'Use saved episodes from the selected map')
                sample=ep.get('qwen_sample',{})
                ids=sample.get('frame_ids',[]) if sample.get('schema_version')==SAMPLE_SCHEMA else []
                if not req.sampled_only:
                    ids=[frame['frame_id'] for frame in ep['observations']]
                if not ids:
                    raise HTTPException(422,'Sample frames in every selected episode before training')
                selected=set(ids)
                if len(selected)!=len(ids):
                    raise HTTPException(409,'Sampled frames changed; sample again')
                labels={a['frame_ids'][0]:a for a in ep['annotations']
                        if len(a.get('frame_ids',[]))==1}
                observation_ids={frame['frame_id'] for frame in ep['observations']}
                if not selected <= observation_ids:
                    raise HTTPException(409,'Sampled frames changed; sample again')
                eligible_ids={f['frame_id'] for f in ep['observations'] if frame_matches_episode(ep,f)}
                approved_ids={fid for fid in ids if fid in eligible_ids and labels.get(fid,{}).get('review_status')=='approved'}
                if req.automatic: approved_ids &= manual_sample_ids(self.store, ep, 'qwen')
                episode_frames=0
                for frame in ep['observations']:
                    if frame['frame_id'] not in approved_ids: continue
                    label=labels[frame['frame_id']]
                    target=parse_target(json.dumps(label['target']), frame.get('context') or {})
                    try:
                        self.store.validate_target(target,frame,catalog)
                    except HTTPException as exc:
                        if target.get('mission_id'):
                            raise HTTPException(422,'A sampled label references a deleted Design mission. Relabel it before training.') from exc
                        raise
                    saved_catalog=recorded_catalog(frame,ep,label)
                    self.store.validate_target(target,frame,saved_catalog)
                    row=self._frame(ep,frame,saved_catalog)
                    row['sample']=sample_for(frame,saved_catalog,target,row['image'])
                    frames.append(row)
                    episode_frames+=1
                if episode_frames!=len(approved_ids):
                    raise HTTPException(409,'Sampled frames changed; sample again')
                if episode_frames: episodes.append(ep)
            if len(frames)<2:
                raise HTTPException(422,'Provide at least two approved frames before training')
            if req.automatic:
                base,_=training_base(req)
                req=req.model_copy(update={'model_name':allocate_name(self.root, 'qwen', req.map_name, self.models(), base),
                                           'steps':automatic_steps('qwen', len(frames))})
            name=req.model_name
            directory=self._model_directory(self.root,req.map_name,name,'training')
            base,revision=training_base(req)
            episode_ids=[ep['id'] for ep in episodes]
            episode_types=sorted({episode_type_for(ep) for ep in episodes})
            model={'schema':SCHEMA,'id':uuid4().hex,'model_name':name,'map_name':req.map_name,
                   'base_model':base,'revision':revision,'adapter_path':str(directory/'adapter'),
                   'sample_count':len(frames),'steps':req.steps,'batch_size':req.batch_size,'created_at':now(),
                   'purpose':'auto_labeling','episode_types':episode_types,'episode_ids':episode_ids,
                   **({'episode_type':episode_types[0]} if len(episode_types)==1 else {}),
                   **training_metadata(frames),
                   'catalog_snapshot_id':catalog['snapshot_id'],
                   'mission_ids':self.store.catalog_identity(catalog)['mission_ids'],
                   'mission_revisions':self.store.catalog_identity(catalog)['mission_revisions']}
            return self._submit('training',{**model,'frames':frames,'model_path':str(directory/'model.json')},
                                {**model,'total':len(frames)})

    def auto_train(self, req):
        with self.lock, self.store.lock:
            self._idle()
            ids = saved_sample_episodes(self.store, req.map_name, 'qwen')
            if not ids: raise HTTPException(422, 'Label sampled frames on this map first')
            return self.train(TrainRequest(map_name=req.map_name, episode_ids=ids,
                model_name='automatic', automatic=True))

    def delete_model(self, model_id):
        with self.lock:
            self._idle()
            model = next((m for m in self.models() if m['id'] == model_id), None)
            if not model:
                raise HTTPException(404, 'Qwen model not found')
            if getattr(self, 'model_in_use', lambda _: False)(model):
                raise HTTPException(409, 'Stop observation and unload this model before deleting it')
            root = self.reasoning_root if model.get('purpose') == 'reasoning' else self.root
            result = remove_model(root, model_id, 2)
            for kind, job in self.jobs.items():
                if job.get('adapter_path') == model.get('adapter_path'):
                    self.jobs[kind] = {'status':'idle'}
            self._persist()
            return result

    def _model_directory(self, root, map_name, name, kind):
        slug=re.sub(r'[^\w.-]+','_',name,flags=re.UNICODE).strip('._-')
        if not slug: raise HTTPException(422,'Use letters or numbers in the model name')
        if any(m['map_name']==map_name and m['model_name'].casefold()==name.casefold() for m in self.models()):
            raise HTTPException(409,'Model name already exists on this map')
        directory=root/map_name/slug
        if directory.exists():
            previous=self.jobs[kind]
            if not (previous.get('status')=='cancelled' and previous.get('map_name')==map_name
                    and previous.get('model_name','').casefold()==name.casefold()):
                raise HTTPException(409,'Model name already exists on this map')
            # Preserve incomplete checkpoints; never resume from a cancelled report.
            index=2
            while (root/map_name/f'{slug}_retry_{index}').exists(): index+=1
            directory=root/map_name/f'{slug}_retry_{index}'
        directory.mkdir(parents=True)
        return directory

    def _reasoning_frames(self,req,preview=False):
        catalog=self.store.current_catalog(req.map_name)
        episodes=[]
        for eid in dict.fromkeys(req.episode_ids):
            ep=self.store.get(eid)
            if not ep.get('ended_at') or ep.get('map_name')!=req.map_name:
                raise HTTPException(422,'Use saved episodes from the selected map')
            episodes.append(ep)
        frames=[]
        for ep in episodes:
            labels={a['frame_ids'][0]:a for a in ep['annotations']
                    if len(a.get('frame_ids',[]))==1}
            for frame in ep['observations']:
                if not frame_matches_episode(ep,frame): continue
                label=labels.get(frame['frame_id'])
                if not label or label.get('review_status')!='approved': continue
                target=parse_target(json.dumps(label['target']), frame.get('context') or {})
                try:
                    self.store.validate_target(target,frame,catalog)
                except HTTPException as exc:
                    if target.get('mission_id'):
                        raise HTTPException(422,'An approved label references a deleted Design mission. Relabel it before training.') from exc
                    raise
                saved_catalog=recorded_catalog(frame,ep,label)
                self.store.validate_target(target,frame,saved_catalog)
                row=({'image':'','frame':frame,'catalog':saved_catalog} if preview else self._frame(ep,frame,saved_catalog))
                row['sample']=sample_for(frame,saved_catalog,target,row['image'])
                frames.append(row)
        return catalog,episodes,frames

    def preview_reasoning(self,req):
        with self.store.lock:
            catalog,_,frames=self._reasoning_frames(req,preview=True)
            _,report=balance_frames(frames,req.balance_decisions)
            return {**report,'mission_names':{m['mission_id']:m.get('mission_name',m['mission_id']) for m in catalog['missions']}}

    def train_reasoning(self,req):
        with self.lock,self.store.lock:
            self._idle()
            catalog,episodes,frames=self._reasoning_frames(req)
            frames,sampling=balance_frames(frames,req.balance_decisions)
            if sampling.get('error'):
                raise HTTPException(422,sampling['error'])
            if len(frames)<2:
                raise HTTPException(422,'Prepare at least two approved mission labels on this map')
            name=req.model_name
            directory=self._model_directory(self.reasoning_root,req.map_name,name,'reasoning_training')
            base,revision=training_base(req)
            used_episode_ids={row['episode_id'] for row in frames}
            episodes=[ep for ep in episodes if ep['id'] in used_episode_ids]
            episode_ids=[ep['id'] for ep in episodes]
            model={'schema':SCHEMA,'id':uuid4().hex,'model_name':name,'map_name':req.map_name,
                   'base_model':base,'revision':revision,'adapter_path':str(directory/'adapter'),
                   'sample_count':len(frames),'steps':req.steps,'batch_size':req.batch_size,'created_at':now(),
                   'purpose':'reasoning','episode_ids':episode_ids,'sampling':sampling,
                   'episode_types':sorted({episode_type_for(ep) for ep in episodes if any(row['episode_id']==ep['id'] for row in frames)}),
                   **training_metadata(frames),
                   'catalog_snapshot_id':catalog['snapshot_id'],
                   'mission_ids':self.store.catalog_identity(catalog)['mission_ids'],
                   'mission_revisions':self.store.catalog_identity(catalog)['mission_revisions']}
            return self._submit('reasoning_training',
                {**model,'frames':frames,'model_path':str(directory/'model.json')},
                {**model,'total':len(frames)})

    def batch(self,req):
        with self.lock,self.store.lock:
            self._idle()
            model=next((m for m in self.models() if m['id']==req.model_id),None)
            if not model: raise HTTPException(404,'Select a trained Qwen annotation model')
            current_catalog=self.store.current_catalog(model['map_name'])
            if model.get('catalog_snapshot_id') != current_catalog['snapshot_id']:
                raise HTTPException(409,'Design missions changed after this model was trained. Train a new model before auto labeling.')
            frames=[]
            for eid in dict.fromkeys(req.episode_ids):
                ep=self.store.get(eid)
                if not ep.get('ended_at') or ep['map_name']!=model['map_name']:
                    raise HTTPException(422,'Use saved episodes from the model map')
                episode_types=model.get('episode_types') or ([model['episode_type']] if model.get('episode_type') else [])
                if episode_types and episode_type_for(ep) not in episode_types:
                    raise HTTPException(422,'Use episode types included in this auto-label model training')
                for frame in ep['observations']:
                    if not frame_matches_episode(ep,frame): continue
                    old=next((a for a in ep['annotations'] if a['frame_ids']==[frame['frame_id']]),None)
                    if old and (old['review_status']=='approved' or not req.replace_unreviewed or not old.get('annotator','').startswith('qwen:')): continue
                    frames.append(self._frame(ep,frame,recorded_catalog(frame,ep)))
            if not frames: raise HTTPException(422,'No eligible unlabeled frames remain in these episode types')
            return self._submit('batch',{**model,'frames':frames},
                                {'model_id':model['id'],'model_name':model['model_name'],
                                 'map_name':model['map_name'],'episode_ids':req.episode_ids,'total':len(frames)})

    def _prediction(self,row,event,job_id):
        target=parse_target(event['text'], row['frame'].get('context') or {})
        with self.store.lock:
            ep=self.store.get(row['episode_id'])
            frame=next((f for f in ep['observations'] if f['frame_id']==row['frame_id']),None)
            old=next((a for a in ep['annotations'] if a['frame_ids']==[row['frame_id']]),None)
            if old and old.get('annotator')==f'qwen:{job_id}': return 'created' # Restart replay.
            if old and old['review_status']=='approved': return 'skipped'
            if (old['revision'] if old else 0)!=row['expected_revision']:
                return 'skipped'
            if not frame or frame['image']['sha256']!=row['image_sha256'] or frame.get('context')!=row['frame'].get('context') or frame.get('metadata')!=row['frame'].get('metadata') or recorded_catalog(frame,ep)!=row['catalog']:
                raise ValueError('Saved frame or context changed; generate again')
            self.store.annotate(ep['id'],AnnotationRequest(**target,
                frame_id=frame['frame_id'],expected_revision=row['expected_revision'],
                review_status='draft',annotator=f'qwen:{job_id}'), source_catalog=row['catalog'])
            return 'created'

    def _run(self,kind,initial):
        try:
            with self.lock:
                self._running_job(kind, initial['id'])
            request=read_json(Path(initial['request_path']))
            frames={(r['episode_id'],r['frame_id']):r for r in request['frames']}
            def event(value):
                with self.lock:
                    job=self._running_job(kind, initial['id'])
                    if value['event']=='progress':
                        job.update(status='training' if kind in TRAINING_KINDS else 'running',phase=value.get('phase'))
                        if 'step' in value: job['step']=max(job.get('step',0),value['step'])
                        if 'loss' in value: job['loss']=value['loss']
                        self._persist()
                        return
                    if value['event'] not in ('prediction','frame_error'): return
                    key=(value.get('episode_id'),value.get('frame_id'))
                    if key not in frames: raise ValueError('Worker returned an unexpected frame')
                    token='/'.join(key)
                    if token in job['results']: return
                    error=None
                    try:
                        if value['event']=='frame_error': raise ValueError(value.get('error','Frame inference failed'))
                        result=self._prediction(frames[key],value,initial['id'])
                    except Exception as exc:
                        result='failed'; error=str(getattr(exc,'detail',exc))
                    job['results'][token]=result
                    job['processed']=len(job['results']); job[result]+=1
                    if error: job['error']=error
                    self._persist()
            # A completed adapter remains recoverable even if the API restarted just before publishing it.
            if kind not in TRAINING_KINDS or not (Path(request['adapter_path'])/'training_report.json').is_file():
                self.runtime.execute(Path(initial['request_path']),initial['id'],event)
            with self.lock:
                job=self._running_job(kind, initial['id'])
                if kind in TRAINING_KINDS:
                    adapter=Path(request['adapter_path'])
                    if not (adapter/'adapter_config.json').is_file() or not (adapter/'adapter_model.safetensors').is_file():
                        raise ValueError('Training finished without adapter weights')
                    keys=('schema','id','model_name','map_name','base_model','revision','adapter_path',
                          'sample_count','steps','created_at','purpose','episode_type','episode_types','episode_ids',
                          'catalog_snapshot_id','mission_ids','mission_revisions','sampling','reasoning_tasks',
                          'input_policy','task_sample_counts','task_decision_counts','supervision_decision_counts','training_catalog_snapshot_ids')
                    metadata={k:request[k] for k in keys if k in request}
                    metadata.setdefault('purpose','reasoning' if kind=='reasoning_training' else 'auto_labeling')
                    metadata.setdefault('episode_ids',list(dict.fromkeys(
                        row['episode_id'] for row in request.get('frames',[]))))
                    write_json(Path(request['model_path']),metadata)
                if kind=='batch' and job['processed']!=job['total']: raise ValueError('Generation ended before all frames were processed')
                job['status']='completed_with_errors' if job['failed'] else 'completed'
                if kind in TRAINING_KINDS: job['step']=job['steps']
                self._persist()
        except Exception as exc:
            with self.lock:
                job=self.jobs[kind]
                if job.get('id') == initial['id'] and job.get('status') in ACTIVE:
                    if isinstance(exc, JobCancelled):
                        job.update(status='cancelled',phase='Cancelled')
                        job.pop('error',None)
                    elif job['status']=='cancelling':
                        job['error']='Could not stop training: '+str(exc)
                    else:
                        job.update(status='failed',error=str(getattr(exc,'detail',exc)))
                    self._persist()
        finally:
            self.runtime.release(initial['id'])


def create_router(workflow):
    router=APIRouter(prefix='/collection',tags=['Qwen annotation'])
    @router.get('/qwen/workflow')
    def status(): return workflow.status()
    @router.post('/qwen/jobs/{kind}/cancel')
    def cancel_job(kind:str, req:CancelRequest): return workflow.cancel_job(kind,req.job_id)
    @router.post('/qwen/jobs/{kind}/reset')
    def reset_job(kind:str): return workflow.reset_job(kind)
    @router.post('/episodes/{eid}/qwen/sample')
    def sample(eid:str,req:SampleRequest): return workflow.sample(eid,req)
    @router.post('/qwen/auto-train')
    def auto_train(req:AutomaticTrainingRequest): return workflow.auto_train(req)
    @router.delete('/qwen/models/{model_id}')
    def delete_model(model_id:str): return workflow.delete_model(model_id)
    @router.post('/qwen/train')
    def train(req:TrainRequest): return workflow.train(req)
    @router.post('/qwen/preview-reasoning')
    def preview_reasoning(req:ReasoningDatasetRequest): return workflow.preview_reasoning(req)
    @router.post('/qwen/train-reasoning')
    def train_reasoning(req:ReasoningTrainRequest): return workflow.train_reasoning(req)
    @router.post('/qwen/auto-label')
    def batch(req:BatchRequest): return workflow.batch(req)
    return router

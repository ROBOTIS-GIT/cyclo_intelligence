"""Joint SAM mask decoder and supervised class-head adaptation.
Candidate labels are hypotheses; every generated instance requires review.
"""
import base64
import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_ROOT = Path(os.getenv('SAM2_ADAPTER_ROOT', '/workspace/model/sam2/adapters'))

class TrainingCancelled(RuntimeError):
    """Cooperative stop at a feature-cache or optimizer-step boundary."""


class CancelRequest(BaseModel):
    job_id: str = Field(pattern=r'^[a-f0-9]{32}$')


class Example(BaseModel):
    image_base64: str = Field(max_length=8 * 1024 * 1024)
    rle: dict
    object_id: str = ""
    object_label: str = Field(min_length=1, max_length=120)
    episode_id: str
    frame_id: str
    image_sha256: str

class TrainRequest(BaseModel):
    model_name: str | None = Field(default=None, min_length=1, max_length=80)
    map_name: str = Field(min_length=1, max_length=128)
    examples: list[Example] = Field(min_length=2, max_length=4096)
    steps: int = Field(default=1000, ge=2, le=2000)

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, value):
        if value is None: return None
        value = value.strip()
        if not value: raise ValueError('Enter a model name')
        if any(ord(char) < 32 for char in value): raise ValueError('Model name must be one line')
        return value


class AutoRequest(BaseModel):
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    image_base64: str = Field(max_length=8 * 1024 * 1024)


def decode_rle(rle):
    import numpy as np
    h, w = rle['size']
    counts = rle['counts']
    if not (0 < h * w <= 4_000_000) or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != h*w:
        raise ValueError('Invalid training mask')
    return np.repeat(np.arange(len(counts)) % 2, counts).reshape((h,w), order='F').astype('float32')


def encode_rle(mask):
    import numpy as np
    flat = mask.flatten(order='F').astype(np.uint8)
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]: counts.insert(0, 0)
    return {'size': list(mask.shape), 'counts': counts}


def pixels(raw):
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(base64.b64decode(raw, validate=True))) as image:
        if image.width * image.height > 4_000_000: raise ValueError('Image exceeds 4 MP')
        return np.array(image.convert('RGB'), copy=True)




class Adaptation:
    def __init__(self, engine, root=MODEL_ROOT):
        self.engine = engine
        self.root = Path(root)
        self.job = {'status': 'idle'}
        self.guard = threading.RLock()
        self.cancel_requested = threading.Event()
        self.class_head = None
        self.class_model_id = None

    def models(self):
        models = []
        for path in sorted(self.root.glob('*/model.json')):
            item = json.loads(path.read_text())
            if not item.get('model_name'):
                stamp = datetime.fromtimestamp(item.get('created_at') or path.stat().st_mtime,timezone.utc)
                item['model_name'] = f"{item['map_name']} · SAM {stamp:%Y-%m-%d %H:%M:%S} UTC"
            models.append(item)
        return models

    def start(self, req):
        with self.guard:
            if self.job['status'] in ('queued', 'training', 'cancelling'):
                raise HTTPException(409, 'SAM training is already running')
            names = {m['model_name'].casefold() for m in self.models() if m.get('map_name') == req.map_name}
            model_name = req.model_name
            if model_name and model_name.casefold() in names:
                raise HTTPException(409, 'A model with this name already exists on this map. Choose another name.')
            if model_name is None:
                stamp = datetime.now(timezone.utc)
                model_name = f"{req.map_name} · SAM {stamp:%Y-%m-%d %H:%M:%S.%f} UTC"
            req = req.model_copy(update={'model_name': model_name})
            if not self.engine.lock.acquire(blocking=False):
                raise HTTPException(409, 'SAM is busy')
            self.cancel_requested.clear()
            self.job = {'id': uuid4().hex, 'status': 'queued', 'step': 0, 'steps': req.steps,
                        'object_labels': sorted({e.object_label for e in req.examples}), 'map_name': req.map_name,
                        'model_name':req.model_name,
                        'phase':'Caching sampled frames'}
            threading.Thread(target=self.train, args=(req, self.job['id']), daemon=True).start()
            return dict(self.job)

    def cancel(self, job_id):
        with self.guard:
            if self.job.get('id') != job_id:
                raise HTTPException(409, 'Training job changed; refresh before cancelling')
            if self.job['status'] == 'cancelled': return dict(self.job)
            if self.job['status'] not in ('queued', 'training', 'cancelling'):
                raise HTTPException(409, 'This training job is no longer running')
            self.cancel_requested.set()
            self.job.update(status='cancelling',phase='Stopping training')
            return dict(self.job)

    def check_cancelled(self):
        if self.cancel_requested.is_set(): raise TrainingCancelled('SAM training cancelled')

    def train(self, req, model_id):
        cancelled=False
        try:
            with self.guard:
                self.check_cancelled()
                self.job['status']='training'
            from .class_training import train_model
            train_model(self,req,model_id)
        except TrainingCancelled:
            cancelled=True
        except Exception as exc:
            with self.guard:
                cancelled=self.cancel_requested.is_set()
                if not cancelled: self.job.update(status='failed',error=str(exc))
        finally:
            self.engine.predictor=None;self.engine.image_hash=None
            self.engine.base_decoder=None;self.engine.active_adapter_id=None
            self.class_head=None;self.class_model_id=None
            self.engine.lock.release()
            with self.guard:
                if cancelled:
                    self.job.update(status='cancelled',phase='Cancelled')
                    self.job.pop('error',None)

    def predict(self, req):
        if not self.engine.lock.acquire(blocking=False): raise HTTPException(409, 'SAM is busy')
        try:
            import numpy as np
            import torch
            directory = self.root / req.model_id
            if not (directory/'model.json').is_file(): raise HTTPException(404, 'Trained model not found')
            metadata = json.loads((directory/'model.json').read_text())
            from .class_model import ClassHead, STRATEGY, predict_instances
            if metadata.get('strategy') != STRATEGY:
                raise HTTPException(422, 'Retrain with all sampled labels to generate class-aware SAM masks')
            self.engine.load()
            predictor = self.engine.predictor
            if self.engine.active_adapter_id != req.model_id:
                state = torch.load(directory/'decoder.pt', map_location=predictor.device, weights_only=True)
                predictor.model.sam_mask_decoder.load_state_dict(state, strict=True)
                predictor.model.sam_mask_decoder.dynamic_multimask_via_stability = False
                self.engine.active_adapter_id = req.model_id
            if self.class_model_id != req.model_id:
                self.class_head=ClassHead(metadata['class_feature_channels'],len(metadata['object_labels'])).to(predictor.device)
                self.class_head.load_state_dict(torch.load(directory/'classes.pt',map_location=predictor.device,weights_only=True),strict=True)
                self.class_head.eval();self.class_model_id=req.model_id
            self.engine.image_hash = None
            image = pixels(req.image_base64)
            with torch.inference_mode():
                instances = predict_instances(predictor, self.class_head, metadata["object_labels"], image, encode_rle)
            return {'instances':instances, 'model':metadata['base_model'],
                    'model_revision':metadata['base_revision'], 'adapter_id':req.model_id,
                    'object_labels':metadata['object_labels'], 'strategy':metadata['strategy'],
                    'generation_strategy':STRATEGY}
        except Exception:
            self.engine.predictor = None; self.engine.image_hash = None
            self.engine.base_decoder = None; self.engine.active_adapter_id = None
            raise
        finally:
            self.engine.lock.release()

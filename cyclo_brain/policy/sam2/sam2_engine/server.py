"""Click-prompted SAM 2.1 image segmentation; no robot connection."""
import base64
import hashlib
import io
import os
import threading
import time
from contextlib import nullcontext
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = 'facebook/sam2.1-hiera-tiny'
REVISION = 'de431c4043854a71d8101e17995dfe596bf101a5'
SOURCE_REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
DEVICE = os.getenv('SAM2_DEVICE', 'cuda')


class Point(BaseModel):
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    label: Literal[0, 1]


class SegmentRequest(BaseModel):
    image_base64: str = Field(max_length=8 * 1024 * 1024)
    image_sha256: str = Field(min_length=64, max_length=64)
    points: list[Point] = Field(min_length=1, max_length=64)


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.predictor = None
        self.image_hash = None
        self.checkpoint_sha256 = None
        self.base_decoder = None
        self.active_adapter_id = None

    def load(self):
        if self.predictor is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        if DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('SAM2 requires a CUDA GPU in this configuration')
        path = hf_hub_download(MODEL_ID, 'sam2.1_hiera_tiny.pt', revision=REVISION)
        with open(path, 'rb') as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', path, device=DEVICE)
        # Optional connected-components CUDA extension is not needed for click masks.
        self.predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
        self.base_decoder = {k:v.detach().cpu().clone() for k,v in model.sam_mask_decoder.state_dict().items()}
        self.base_dynamic_mode = model.sam_mask_decoder.dynamic_multimask_via_stability
        self.active_adapter_id = None

    def use_base_decoder(self):
        if self.active_adapter_id is not None:
            self.predictor.model.sam_mask_decoder.load_state_dict(self.base_decoder,strict=True)
            self.predictor.model.sam_mask_decoder.dynamic_multimask_via_stability = self.base_dynamic_mode
            self.active_adapter_id = None
            self.image_hash = None

    def segment(self, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, 'SAM2 is busy; retry after the current mask finishes')
        try:
            import numpy as np
            import torch
            from PIL import Image
            started = time.monotonic()
            raw = base64.b64decode(request.image_base64, validate=True)
            if hashlib.sha256(raw).hexdigest() != request.image_sha256:
                raise HTTPException(422, 'Image checksum mismatch')
            with Image.open(io.BytesIO(raw)) as im:
                width, height = im.size
                if width * height > 4_000_000:
                    raise HTTPException(422, 'Image exceeds the 4 megapixel annotation limit')
                pixels = np.asarray(im.convert('RGB'))
            if any(p.x >= width or p.y >= height for p in request.points) or not any(p.label for p in request.points):
                raise HTTPException(422, 'Use in-image points and at least one foreground point')
            self.load()
            self.use_base_decoder()
            with torch.inference_mode(), (torch.autocast('cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else nullcontext()):
                if self.image_hash != request.image_sha256:
                    self.predictor.set_image(pixels)
                    self.image_hash = request.image_sha256
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[p.x, p.y] for p in request.points], dtype=np.float32),
                    point_labels=np.array([p.label for p in request.points], dtype=np.int32),
                    multimask_output=True,
                )
            chosen = int(np.argmax(scores))
            mask = masks[chosen].astype(bool)
            # Uncompressed COCO RLE: column-major pixels, starting with background run.
            flat = mask.flatten(order='F').astype(np.uint8)
            boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
            counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
            if flat[0]:
                counts.insert(0, 0)
            return {'rle': {'size': [height, width], 'counts': counts},
                    'score': float(scores[chosen]), 'elapsed_seconds': round(time.monotonic() - started, 3),
                    'model': MODEL_ID, 'model_revision': REVISION, 'source_revision': SOURCE_REVISION,
                    'checkpoint_sha256': self.checkpoint_sha256}
        except HTTPException:
            raise
        except Exception as exc:
            self.image_hash = None
            raise HTTPException(503, f'SAM2 segmentation failed: {exc}') from exc
        finally:
            self.lock.release()


engine = Engine()
app = FastAPI(title='SAM2 annotation service')


@app.get('/health')
def health():
    return {'model': MODEL_ID, 'revision': REVISION, 'loaded': engine.predictor is not None, 'busy': engine.lock.locked()}


@app.post('/segment')
def segment(request: SegmentRequest):
    return engine.segment(request)


@app.post('/unload')
def unload():
    if not engine.lock.acquire(blocking=False):
        raise HTTPException(409, 'SAM2 is busy')
    try:
        engine.predictor = None; engine.image_hash = None
        engine.base_decoder = None; engine.active_adapter_id = None
        adaptation.class_head = None; adaptation.class_model_id = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {'loaded': False}
    finally:
        engine.lock.release()

# Object-specific decoder training and unreviewed automatic mask proposals.
from .adaptation import Adaptation, TrainRequest, AutoRequest, CancelRequest
adaptation = Adaptation(engine)

@app.get('/adaptation')
def adaptation_status():
    return {'training': dict(adaptation.job), 'models': adaptation.models()}

@app.post('/adaptation/train')
def train_adapter(request: TrainRequest):
    return adaptation.start(request)

@app.post('/adaptation/cancel')
def cancel_training(request: CancelRequest):
    return adaptation.cancel(request.job_id)

@app.post('/adaptation/predict')
def auto_mask(request: AutoRequest):
    return adaptation.predict(request)

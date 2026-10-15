"""Train mask refinement and the complete human label vocabulary in one checkpoint."""
import base64
import hashlib
import json
import random
import tempfile
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .class_model import ClassHead, FEATURE_SIZE, STRATEGY, image_features, semantic_target


def train_model(worker, req, model_id):
    # Frozen image features live on disk; only the current frame occupies RAM/VRAM.
    with tempfile.TemporaryDirectory(prefix="sam-training-") as directory:
        return _train_model(worker, req, model_id, Path(directory))


def _train_model(worker, req, model_id, cache_dir):
    from .adaptation import decode_rle,pixels
    if len({(e.episode_id,e.frame_id) for e in req.examples})>1024:
        raise ValueError('Training is limited to 1024 reviewed frames')
    worker.check_cancelled()
    worker.engine.load();worker.engine.use_base_decoder()
    worker.check_cancelled()
    predictor=worker.engine.predictor;model=predictor.model
    model.requires_grad_(False);decoder=model.sam_mask_decoder;decoder.requires_grad_(True)
    for name,param in decoder.named_parameters():
        if name.startswith(('conv_s0.','conv_s1.')):param.requires_grad_(False)
    decoder.dynamic_multimask_via_stability=False;decoder.train()
    classes=sorted({e.object_label.strip() for e in req.examples})
    if not all(classes):raise ValueError('Each training instance needs an object label')
    grouped={}
    examples=list(req.examples)
    for example in examples:
        grouped.setdefault((example.episode_id,example.frame_id),[]).append(example)
    cached={};head=None;counts=torch.zeros(len(classes)+1)
    for index,(key,frame_examples) in enumerate(grouped.items()):
        worker.check_cancelled()
        worker.job.update(phase=f"Preparing image features {index+1}/{len(grouped)}")
        first=frame_examples[0]
        raw=base64.b64decode(first.image_base64,validate=True)
        if hashlib.sha256(raw).hexdigest()!=first.image_sha256:
            raise ValueError('Training image checksum mismatch')
        image=pixels(first.image_base64);instances=[]
        for example in frame_examples:
            if example.image_sha256!=first.image_sha256 or example.image_base64!=first.image_base64:
                raise ValueError('Instances reference inconsistent training images')
            mask=decode_rle(example.rle).astype(bool)
            if mask.shape!=image.shape[:2] or not mask.any():
                raise ValueError('Invalid instance mask dimensions or empty foreground')
            instances.append((mask,example.object_label.strip()))
        predictor.set_image(image)
        features={k:[v.detach().cpu() for v in value] if isinstance(value,list) else value.detach().cpu()
                  for k,value in predictor._features.items()}
        with torch.no_grad():
            fused=image_features(predictor._features)
        if head is None:head=ClassHead(fused.shape[1],len(classes)).to(predictor.device)
        target=torch.from_numpy(semantic_target(instances,classes,image.shape[:2]))[None,None].float()
        target=F.interpolate(target,(FEATURE_SIZE,FEATURE_SIZE),mode='nearest')[:,0].long()
        for k in range(len(counts)):counts[k]+=(target==k).sum()
        path=cache_dir/f"{index}.pt"
        torch.save({'features':features,'shape':predictor._orig_hw,
                    'fused':fused.detach().cpu(),'target':target},path)
        cached[key]=path
        del features,fused,target,instances,image,mask
    worker.check_cancelled()
    worker.job['phase']='Training masks and labels'
    weights=(counts.sum()/counts.clamp_min(1)).sqrt();weights=(weights/weights.mean()).to(predictor.device)
    optimizer=torch.optim.AdamW([{'params':[p for p in decoder.parameters() if p.requires_grad],'lr':1e-5},
                                 {'params':head.parameters(),'lr':1e-3}])
    order=list(range(len(examples)))
    sampler=random.Random(42)
    for step in range(req.steps):
        worker.check_cancelled()
        if step % len(order) == 0:
            sampler.shuffle(order)
        example=examples[order[step % len(order)]]
        frame=torch.load(cached[(example.episode_id,example.frame_id)],map_location="cpu",weights_only=True)
        mask=decode_rle(example.rle)
        target=torch.from_numpy(mask)[None,None]
        foreground=np.argwhere(mask)
        predictor._features={k:[v.to(predictor.device) for v in value] if isinstance(value,list) else value.to(predictor.device) for k,value in frame['features'].items()}
        predictor._orig_hw=frame['shape']
        y,x=foreground[np.random.randint(len(foreground))]
        _,coords,labels,_=predictor._prep_prompts(np.array([[x,y]],dtype=np.float32),np.array([1]),None,None,True)
        _,quality,logits=type(predictor)._predict.__wrapped__(predictor,coords,labels,multimask_output=False,return_logits=True)
        truth=F.interpolate(target.to(predictor.device),logits.shape[-2:],mode='nearest');prob=logits.sigmoid()
        dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
        with torch.no_grad():
            pred=prob>.5;fg=truth>.5;iou=(pred&fg).sum().float()/(pred|fg).sum().clamp_min(1)
        mask_loss=F.binary_cross_entropy_with_logits(logits,truth)+dice+.1*(quality-iou).square().mean()
        class_logits=head(frame['fused'].to(predictor.device));class_target=frame['target'].to(predictor.device)
        class_loss=F.cross_entropy(class_logits,class_target,weight=weights,ignore_index=255)
        loss=mask_loss+class_loss
        if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([*decoder.parameters(),*head.parameters()],1.);optimizer.step()
        worker.job.update(step=step+1,loss=round(loss.item(),5),mask_loss=round(mask_loss.item(),5),class_loss=round(class_loss.item(),5))
    with worker.guard:
        worker.check_cancelled()
        directory=worker.root/model_id;directory.mkdir(parents=True,exist_ok=False)
        torch.save({k:v.detach().cpu() for k,v in decoder.state_dict().items()},directory/'decoder.pt')
        torch.save({k:v.detach().cpu() for k,v in head.state_dict().items()},directory/'classes.pt')
        from .server import MODEL_ID,REVISION
        metadata={'id':model_id,'model_name':req.model_name,'object_labels':classes,'map_name':req.map_name,'base_model':MODEL_ID,'base_revision':REVISION,
            'strategy':STRATEGY,'steps':req.steps,'sample_count':len(examples),'frame_count':len(cached),'created_at':time.time(),
            'class_feature_channels':head.net[0].in_channels,
            'label_counts':{label:sum(e.object_label.strip()==label for e in req.examples) for label in classes},
            'training_samples':[{'episode_id':e.episode_id,'frame_id':e.frame_id,'image_sha256':e.image_sha256,
                                 'object_id':e.object_id,'object_label':e.object_label} for e in req.examples]}
        (directory/'model.json').write_text(json.dumps(metadata,indent=2))
        worker.job.update(status='completed',model_id=model_id)

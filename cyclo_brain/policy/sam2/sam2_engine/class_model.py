"""Supervised object classes on frozen SAM image features, including background."""
from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

STRATEGY = 'sam_multiclass_instances_v1'
FEATURE_SIZE = 256
AUTO_LABEL_CONFIDENCE = 0.8


def image_features(features):
    # SAM 2.1 Tiny's projected high-resolution features and image embedding.
    values = [*features['high_res_feats'], features['image_embed']]
    return torch.cat([F.interpolate(v.float(), (FEATURE_SIZE, FEATURE_SIZE), mode='bilinear', align_corners=False)
                      if v.shape[-2:] != (FEATURE_SIZE, FEATURE_SIZE) else v.float() for v in values], dim=1)


class ClassHead(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels,64,1),nn.GroupNorm(8,64),nn.GELU(),
                                 nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Conv2d(64,classes+1,1))
    def forward(self, features):
        return self.net(features)


def semantic_target(instances, classes, shape):
    target = np.zeros(shape, dtype=np.int64)
    occupied = np.zeros(shape, dtype=bool)
    for mask,label in instances:
        index = classes.index(label)+1
        overlap = occupied & mask & (target != index)
        target[mask & ~occupied] = index
        target[overlap] = 255  # Conflicting human masks are not a class target.
        occupied |= mask
    return target


def class_seeds(probabilities, confidence=AUTO_LABEL_CONFIDENCE, limit=128):
    """Connected semantic regions provide prompts; same-class instances stay separable."""
    classes = probabilities.argmax(0)
    certainty = probabilities.max(0)
    height,width = classes.shape
    seeds=[]
    for label in range(1,len(probabilities)):
        remaining=(classes==label)&(certainty>=confidence)
        while remaining.any():
            y,x=np.argwhere(remaining)[0];queue=deque([(int(y),int(x))]);remaining[y,x]=False;points=[]
            while queue:
                y,x=queue.popleft();points.append((y,x))
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0<=yy<height and 0<=xx<width and remaining[yy,xx]:
                        remaining[yy,xx]=False;queue.append((yy,xx))
            if len(points)<4:continue
            points=np.array(points);scores=probabilities[label,points[:,0],points[:,1]].copy()
            # Multiple well-spaced prompts allow SAM to separate touching people.
            for _ in range(min(8,max(1,len(points)//128))):
                at=int(scores.argmax())
                if scores[at]<0:break
                y,x=points[at];seeds.append((label,int(x),int(y),float(scores[at])))
                scores[np.linalg.norm(points-[y,x],axis=1)<16]=-1
    return sorted(seeds,key=lambda p:p[3],reverse=True)[:limit]


def predict_instances(predictor, head, classes, image, encode_rle):
    from .instances import select_instances
    predictor.set_image(image)
    probabilities=head(image_features(predictor._features)).softmax(1)[0].cpu().numpy()
    h,w=image.shape[:2];candidates=[]
    for class_id,x,y,confidence in class_seeds(probabilities):
        point=np.array([[x*w/FEATURE_SIZE,y*h/FEATURE_SIZE]],dtype=np.float32)
        masks,scores,_=predictor.predict(point_coords=point,point_labels=np.array([1]),multimask_output=False)
        mask=masks[0].astype(bool);score=float(scores[0])
        if not np.isfinite(score) or score<AUTO_LABEL_CONFIDENCE or mask.sum()<16:continue
        reduced=F.interpolate(torch.from_numpy(mask.astype('float32'))[None,None],(FEATURE_SIZE,FEATURE_SIZE),mode='nearest')[0,0].numpy().astype(bool)
        if not reduced.any():continue
        mean=probabilities[:,reduced].mean(1);predicted=int(mean.argmax())
        if predicted!=class_id or mean[class_id]<AUTO_LABEL_CONFIDENCE:continue
        candidates.append({'mask':mask,'score':float(score*mean[class_id]),'mask_score':score,
            'class_score':float(mean[class_id]),'object_label':classes[class_id-1],'prompt_point':point[0].tolist()})
    # Suppress a duplicate mask even if two class prompts produced it.
    return [{**{k:v for k,v in c.items() if k!='mask'},'rle':encode_rle(c['mask'])}
            for c in select_instances(candidates,threshold=.5,limit=64)]

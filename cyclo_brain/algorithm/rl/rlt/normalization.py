"""Versioned MLP-only physical joint codec. Never mutates the frozen VLA."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from cyclo_brain.model.common import RLT_ACTION_GROUP_NAMES

FORMAT = 'cyclo.rlt.mlp_joint_normalization/v1'


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_normalization(value, dimension=None):
    if not isinstance(value, dict) or value.get('format') != FORMAT:
        raise ValueError('Invalid MLP normalization format')
    unsigned = {k:v for k,v in value.items() if k != 'fingerprint'}
    if value.get('fingerprint') != _fingerprint(unsigned):
        raise ValueError('MLP normalization fingerprint disagrees')
    d = len(value['joint_names'])
    if d not in (16,19) or (dimension is not None and d != dimension):
        raise ValueError('MLP normalization joint dimensions disagree')
    if len(set(value['joint_names'])) != d or value.get('margin_fraction') != .05:
        raise ValueError('Invalid MLP joint order/margin')
    expected_names = [name for key, names in RLT_ACTION_GROUP_NAMES.items()
                      if d == 19 or key != 'odometry' for name in names]
    if value['joint_names'] != expected_names:
        raise ValueError('MLP normalization requires the canonical SG2 joint order')
    for key in ('state', 'action', 'vla_action'):
        lo, hi = (np.asarray(value[key][k], dtype=np.float64) for k in ('lower','upper'))
        if lo.shape != (d,) or hi.shape != (d,) or not np.isfinite([lo,hi]).all() or np.any(hi <= lo):
            raise ValueError(f'Invalid MLP normalization {key} bounds')
    if value.get('hold_indices') != list(range(7)) or value.get('fixed_actions') != {'15': .9}:
        raise ValueError('Unsupported MLP fixed-output policy')
    return copy.deepcopy(value)


def fit_normalization(states, actions, *, joint_names, vla_lower, vla_upper, source_fingerprint):
    """Inputs must contain only actual frames from successful episodes."""
    d = len(joint_names)
    result = dict(format=FORMAT, margin_fraction=.05, joint_names=list(joint_names),
                  source_fingerprint=source_fingerprint, hold_indices=list(range(7)), fixed_actions={'15':.9})
    for key, values in [('state',states),('action',actions)]:
        a=np.asarray(values,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=d or not len(a) or not np.isfinite(a).all():
            raise ValueError(f'MLP {key} statistics require finite successful frames')
        lo,hi=a.min(axis=0),a.max(axis=0)
        # One degree is a floor for arm joints; 0.02 in gripper command units.
        # Store the floor explicitly: it is not a robot limit or a clipping rule.
        floor=np.full(d,np.deg2rad(1.0));floor[[7,15]]=.02
        if d>16:floor[16:]=.02
        width=np.maximum(hi-lo,floor);center=(lo+hi)/2
        result[key]=dict(lower=(center-.55*width).tolist(),upper=(center+.55*width).tolist(),
                         observed_min=lo.tolist(),observed_max=hi.tolist(),minimum_span=floor.tolist())
    result['vla_action']=dict(lower=list(map(float,vla_lower)),upper=list(map(float,vla_upper)))
    result['fingerprint']=_fingerprint(result)
    return validate_normalization(result,d)


def normalize(values, config, kind='action'):
    a=np.asarray(values,dtype=np.float32);lo=np.asarray(config[kind]['lower'],dtype=np.float32);hi=np.asarray(config[kind]['upper'],dtype=np.float32)
    return (2*(a-lo)/(hi-lo)-1).astype(np.float32)


def denormalize(values, config, kind='action'):
    a=np.asarray(values,dtype=np.float32);lo=np.asarray(config[kind]['lower'],dtype=np.float32);hi=np.asarray(config[kind]['upper'],dtype=np.float32)
    # An affine codec, not a clamp. Limits belong to the execution policy.
    return ((a+1)*.5*(hi-lo)+lo).astype(np.float32)


def reference_to_mlp(values, config):
    # Frozen GR00T minmax decode clips its predictions; preserve that physical
    # reference, then use the MLP's independent (unclipped) input coordinates.
    physical=denormalize(np.clip(values,-1,1),config,'vla_action')
    return normalize(physical,config)


def apply_fixed_actions(physical, config, hold):
    result=np.asarray(physical,dtype=np.float32).copy()
    result[...,config['hold_indices']]=np.asarray(hold,dtype=np.float32)[...,None,:]
    for index,value in config['fixed_actions'].items():result[...,int(index)]=value
    return result


def active_joint_indices(config):
    ignored=set(config['hold_indices'])|{int(k) for k in config['fixed_actions']}
    return [i for i in range(len(config['joint_names'])) if i not in ignored]

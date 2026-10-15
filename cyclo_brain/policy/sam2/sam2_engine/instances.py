"""Instance proposal filtering independent of the GPU predictor."""
import numpy as np


def select_instances(candidates, threshold=.7, limit=64):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item['score'], reverse=True):
        mask = candidate['mask']
        if not mask.any(): continue
        duplicate = False
        for other in kept:
            intersection = np.logical_and(mask, other['mask']).sum()
            union = np.logical_or(mask, other['mask']).sum()
            if intersection / max(1, union) > threshold:
                duplicate = True
                break
        if not duplicate: kept.append(candidate)
        if len(kept) >= limit: break
    return kept

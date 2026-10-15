"""Generated-frame review counts for the episode library and frame editor."""
if __package__:
    from .collection_episode_contract import frame_matches_episode
else:
    from collection_episode_contract import frame_matches_episode


def generated_review_counts(episode, samples):
    counts = {'qwen_pending_review_count': 0, 'sam_pending_review_count': 0}
    for sample in samples:
        frame = sample['observation']
        automatic = frame.get('auto_labeling') or {}
        labels = sample.get('annotations') or []
        generated = automatic.get('qwen', {}).get('status') == 'generated' or any(
            label.get('annotator', '').startswith('qwen:') for label in labels)
        if generated and frame_matches_episode(episode, frame) and (
                not labels or labels[-1].get('review_status') != 'approved'):
            counts['qwen_pending_review_count'] += 1
        history = sample.get('regions') or []
        regions = list({region['object_id']: region for region in history}.values())
        generated = automatic.get('sam', {}).get('status') == 'generated' or any(
            region.get('annotator', '').startswith('sam-auto') for region in history)
        reviewed = (all(region.get('review_status') in ('approved', 'rejected')
                        for region in regions) if regions else
                    frame.get('mask_review', {}).get('status') == 'empty')
        if generated and not reviewed:
            counts['sam_pending_review_count'] += 1
    return counts

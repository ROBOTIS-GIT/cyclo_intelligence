"""Prefix-conditioned supervised replay; intentionally not a delayed TD replay.

Observation t, committed actions [t:t+6], Actor targets [t+6:t+16].
No synthetic next states, terminal flags or TD rewards are fabricated here.
"""
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch

from cyclo_brain.algorithm.common import atomic_torch_save, canonical_json_sha256
from cyclo_brain.algorithm.rl.rlt import stage2_spec_fingerprint
from .rlt_stage2_dataset import RLTStage2FeatureReplay

ROLE = 'tt_rtc_actor/v1'
PREFIX = 6
CACHE_SUBDIR = 'tt_rtc_actor_v1'  # Preserve existing same-prefix feature caches.
HOLD_CACHE_SUBDIR = 'tt_rtc_actor_hold_v1'
TERMINAL_HOLD_POLICY = 'successful_final_action_hold/v1'


def actor_window_starts(length, chunk=10, prefix=PREFIX):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (length, chunk, prefix)):
        raise ValueError('Window dimensions must be nonnegative integers')
    if chunk != 10 or prefix != PREFIX:
        raise ValueError('TT-RTC Actor replay requires prefix=6, chunk=10')
    return tuple(range(max(0, length - prefix - chunk + 1)))


@torch.inference_mode()
def extract_tt_rtc(extractor, observation, physical_prefix, spec):
    """Use the same physical-prefix runner and context validation as inference."""
    from .rlt_adapter import GR00TRLTInferenceAdapter
    runner = getattr(extractor.policy, 'get_action_tt_rtc', None)
    if not callable(runner):
        raise ValueError('Selected GR00T has no get_action_tt_rtc; ordinary inference is not a fallback')
    result = runner(observation, committed_action_prefix=physical_prefix[None].astype(np.float32),
                    delay_steps=PREFIX, action_horizon=spec.reference_horizon, return_rlt_context=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError('TT-RTC runner did not return its normalized RLT context')
    # The validator only needs the spec and policy; do not load an Actor bundle.
    holder = SimpleNamespace(spec=spec, policy=extractor.policy)
    tokens, valid, images, proprio, reference, _ = GR00TRLTInferenceAdapter._validate_tt_rtc_rlt_context(
        holder, result[1].get('tt_rtc_rlt_context'), delay_steps=PREFIX)
    with torch.inference_mode():
        z = extractor.encoder(tokens, valid, images)
    return {name: value.detach().float().cpu().clone() for name, value in {
        'z_rl': z[0], 'proprio': proprio[0],
        'reference_actions': reference[0, PREFIX:PREFIX + spec.chunk_length],
    }.items()}


def configure_tt_rtc_policy(policy, checkpoint):
    from .tt_rtc import load_tt_rtc_capability, validate_tt_rtc_model_contract
    capability = load_tt_rtc_capability(checkpoint)
    rtc = capability.payload['training_time_rtc']
    if not rtc['min_delay_steps'] <= PREFIX <= rtc['max_delay_steps']:
        raise ValueError('Checkpoint was not trained for six-action TT-RTC prefixes')
    config = policy.model.config
    if getattr(config, 'training_time_rtc', None) is True:
        if getattr(config, 'tt_rtc_max_delay_steps', None) != rtc['max_delay_steps']:
            raise ValueError('Loaded GR00T delay range differs from its manifest')
        config.training_time_rtc_enabled = True
        config.training_time_rtc_max_delay_steps = rtc['max_delay_steps']
        config.training_time_rtc_action_hz = rtc['action_hz']
    getter = getattr(policy, 'get_tt_rtc_model_contract', None)
    if not callable(getter):
        raise ValueError('GR00T cannot expose its TT-RTC model contract')
    validate_tt_rtc_model_contract(capability, getter())


def materialize_tt_rtc_actor_replay(sources, *, extractor, spec, output_root,
                                  config=None, feature_batch_size=1, reference_seed=0,
                                  progress_callback=None, require_both_outcomes=False, terminal_hold=False):
    from .rlt_data_validation import _recorded_sources
    if not sources or spec.chunk_length != 10 or spec.reference_horizon < PREFIX + 10:
        raise ValueError('TT-RTC Actor replay needs sources and at least a 16-step reference horizon')
    if abs(spec.action_hz - 15.0) > 1e-6 or any(abs(source.fps - 15.0) > 1e-6 for source in sources):
        raise ValueError('TT-RTC Actor replay requires 15 Hz data')
    snapshots = [s.content_snapshot() for s in sources]
    labels_by_root = {s.root: _recorded_sources(s.root)[1] for s in sources}
    raw = [(s,e) for s in sources for e in s.iter_episodes()]
    def window_starts(episode):
        if terminal_hold and episode.successful:
            return tuple(range(episode.length))
        return actor_window_starts(episode.length)

    total = sum(len(window_starts(e)) for _, e in raw)
    if not total:
        raise ValueError('No complete prefix 6 + target 10 windows in these datasets')
    rows = {name: [] for name in ('z_rl','proprio','reference_actions','executed_actions','committed_prefix','imitation_mask')}
    provenance = []
    completed = cache_hits = 0
    for source, episode in raw:
        starts = window_starts(episode)
        if not starts:
            continue
        labels = labels_by_root[source.root]
        source_frame_indices = episode.source_frame_indices or tuple(range(episode.length))
        frame_labels = [
            labels.get((episode.episode_index, source_index), 'unknown')
            for source_index in source_frame_indices
        ]
        aliases = {'intervention':'human', 'rlt':'mlp'}
        frame_labels = [aliases.get(label,label) for label in frame_labels]
        if any(label not in ('human','mlp','vla','policy') for label in frame_labels):
            raise ValueError(f'Episode {episode.episode_index}: explicit frame-aligned action_source is required')
        contract = dict(role=ROLE, spec=stage2_spec_fingerprint(spec), snapshot=snapshots[sources.index(source)],
                        seed=reference_seed, prefix_length=PREFIX, chunk_length=10, stride=1)
        fingerprint = canonical_json_sha256(contract)
        cache_root = Path(source.root)/'rlt'/CACHE_SUBDIR/'features'/fingerprint
        if any(p.is_symlink() for p in (cache_root, *cache_root.parents) if p != Path(source.root).parent):
            raise ValueError("TT-RTC feature cache path must not contain symlinks")
        physical = np.concatenate([episode.action_groups[k] for k in extractor.action_groups],axis=-1)
        if physical.shape != (episode.length, spec.action_dim):
            raise ValueError('Recorded action dimensions disagree with TT-RTC spec')
        # B=1 is required by the deployed TT-RTC runner/context contract.
        for indices, observation in source.iter_observation_batches(episode, starts, 1):
            t = indices[0]
            assert len(indices) == 1
            begin, end = t + PREFIX, t + PREFIX + 10
            prefix_indices = np.minimum(np.arange(t, begin), episode.length - 1)
            target_indices = np.minimum(np.arange(begin, end), episode.length - 1)
            prefix = physical[prefix_indices].astype(np.float32)
            cache = cache_root/f'episode_{episode.episode_index:06d}_frame_{t:08d}.pt'
            identity = dict(contract=fingerprint, episode=episode.episode_index, observation_frame=t)
            if cache.is_symlink():
                raise ValueError('TT-RTC feature cache must not be a symlink')
            if cache.is_file():
                saved = torch.load(cache, map_location='cpu', weights_only=True)
                if saved.get('identity') != identity or not torch.equal(saved['physical_prefix'], torch.from_numpy(prefix)):
                    raise ValueError('TT-RTC feature cache identity/prefix disagrees')
                features = saved['features']; cache_hits += 1
            else:
                seed = (reference_seed + episode.episode_index * 1000003 + t) % (2**31 - 1)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                features = extract_tt_rtc(extractor, observation, prefix, spec)
                cache_root.mkdir(parents=True, exist_ok=True)
                atomic_torch_save(cache, dict(identity=identity, physical_prefix=torch.from_numpy(prefix.copy()), features=features))
            for name in ('z_rl','proprio','reference_actions'):
                rows[name].append(features[name])
            # Normalize future commands relative to the SAME observation t.
            # Using state[t+6:t+16] would be incorrect for relative-action codecs.
            target = extractor.normalize_actions(
                {k:v[target_indices] for k,v in episode.action_groups.items()},
                {k:np.repeat(v[t:t+1],10,axis=0) for k,v in episode.state_groups.items()})
            rows['executed_actions'].append(torch.from_numpy(target.copy()))
            rows['committed_prefix'].append(torch.from_numpy(prefix.copy()))
            target_labels = [frame_labels[i] for i in target_indices]
            rows['imitation_mask'].append(torch.tensor([v=='human' for v in target_labels],dtype=torch.float32))
            provenance.append(dict(dataset_root=str(source.root), episode_index=episode.episode_index,
                start_frame_index=t, observation_frame_index=t, prefix_start_frame_index=t,
                action_start_frame_index=begin, action_stop_frame_index=end,
                source_observation_frame_index=source_frame_indices[t],
                source_prefix_frame_indices=[source_frame_indices[i] for i in prefix_indices],
                source_action_frame_indices=[source_frame_indices[i] for i in target_indices],
                episode_length=episode.length,
                synthetic_hold=bool(end > episode.length),
                prefix_padding_mask=[i >= episode.length for i in range(t, begin)],
                target_padding_mask=[i >= episode.length for i in range(begin, end)],
                episode_success=episode.successful, action_sources=target_labels,
                prefix_action_sources=[frame_labels[i] for i in prefix_indices]))
            completed += 1
            if progress_callback: progress_callback(completed,total)
    if [s.content_snapshot() for s in sources] != snapshots:
        raise ValueError('Dataset changed while preparing TT-RTC replay')
    successes=sum(e.successful for _,e in raw if window_starts(e))
    usable=sum(bool(window_starts(e)) for _,e in raw)
    metadata=dict(replay_role=ROLE, prefix_length=PREFIX, supported_training_phases=['actor_warmup'],
        config=dict(chunk_length=10,prefix_length=PREFIX,stride=1,expected_fps=15.0),
        dataset_roots=[str(s.root) for s in sources], dataset_snapshots=snapshots,
        dataset_snapshot_fingerprint=canonical_json_sha256(dict(format='cyclo.groot.rlt.dataset_collection/v1',
            ordered_content_fingerprints=[s['content_fingerprint'] for s in snapshots])),
        episode_count=len(raw), usable_episode_count=usable, success_episode_count=successes,
        failure_episode_count=usable-successes, skipped_short_episode_count=len(raw)-usable,
        feature_frame_count=total, feature_cache_hits=cache_hits, transition_count=len(provenance),
        row_provenance=provenance, reward_contract='not_applicable_actor_warmup/v1',
        action_codec='arm_left_8+arm_right_8'+('+odometry_3' if spec.action_dim==19 else '')+'/v1',
        reference_extraction=dict(seed=reference_seed,feature_batch_size=1))
    if terminal_hold:
        metadata['terminal_hold_policy'] = TERMINAL_HOLD_POLICY
        metadata['synthetic_hold_count'] = sum(row['synthetic_hold'] for row in provenance)
    replay = RLTStage2FeatureReplay(
        spec,
        {k: torch.stack(v) for k, v in rows.items()},
        metadata=metadata,
    )
    replay.save(output_root)
    return RLTStage2FeatureReplay.load(output_root,expected_spec=spec)

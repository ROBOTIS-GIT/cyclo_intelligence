"""Session-local LeRobot selection for Async RL; source datasets are read-only."""

import tempfile
import threading
from pathlib import Path

import torch
from cyclo_brain.algorithm.common import canonical_json_sha256


class RLTAsyncReplay:
    def __init__(self, worker, build, batch_size, seed):
        self.worker, self.build, self.batch_size = worker, build, batch_size
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self._condition = threading.Condition()
        self._paths = ()
        self._revision = 0
        self._enabled = False
        self._closed = False
        self._preparing = False
        self._error = None
        self._thread = threading.Thread(target=self._run, name='rlt-replay-prepare', daemon=True)
        self._thread.start()

    def select(self, paths):
        paths = tuple(dict.fromkeys(paths))
        with self._condition:
            if paths == self._paths:
                return
            self._paths = paths
            self._revision += 1
            self._error = None
            if not paths:
                self.worker.replace_replay(None, {'paths': [], 'transitions': 0, 'episodes': 0})
            self._condition.notify_all()

    def set_enabled(self, enabled):
        with self._condition:
            self._enabled = enabled
            self._condition.notify_all()

    def status(self):
        with self._condition:
            return {'selected_paths': list(self._paths), 'preparing': self._preparing,
                    'replay_error': self._error}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self):
        completed = -1
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    self._enabled and self._paths and self._revision != completed))
                if self._closed:
                    return
                revision, paths = self._revision, self._paths
                self._preparing = True

            def check_current():
                with self._condition:
                    if self._closed or not self._enabled or revision != self._revision:
                        raise InterruptedError('Replay selection changed or preparation paused')

            try:
                replay = self.build(paths, check_current)
                def sample(replay=replay):
                    indices = torch.randint(len(replay), (self.batch_size,), generator=self.generator).tolist()
                    learner = self.worker.learner
                    return replay.batch(indices, device=learner.device, dtype=learner.dtype)
                with self._condition:
                    check_current()
                    self.worker.replace_replay(sample, {
                        'paths': list(paths), 'transitions': len(replay),
                        'episodes': replay.metadata['usable_episode_count'],
                        'batch_size': self.batch_size,
                    }, checkpoint_context={'replay': replay, 'generator': self.generator,
                                           'sampling_seed': self.seed})
                    completed = revision
                    self._error = None
            except InterruptedError:
                pass
            except Exception as error:
                with self._condition:
                    if revision == self._revision:
                        completed = revision
                        self._error = f'{type(error).__name__}: {error}'
            finally:
                with self._condition:
                    self._preparing = False
                    self._condition.notify_all()


class RLTAsyncReplayBuilder:
    """Reuse existing extraction, caching each immutable dataset in this session."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.cache = {}
        self.restored = None
        self.directory = tempfile.TemporaryDirectory(prefix='cyclo-rlt-replay-')

    def close(self):
        self.directory.cleanup()

    def __call__(self, paths, check_current):
        from .rlt_stage2_dataset import (
            GR00TRLTStage2Extractor, RLTStage2DatasetConfig, RLTStage2FeatureReplay,
            materialize_rlt_stage2_replay, open_rlt_stage2_source,
        )
        adapter = self.adapter
        config = RLTStage2DatasetConfig(expected_fps=adapter.spec.action_hz)
        if self.restored is not None and list(paths) == self.restored.metadata.get('dataset_roots', []):
            snapshots = [open_rlt_stage2_source(path, expected_fps=config.expected_fps,
                action_dim=adapter.spec.action_dim).content_snapshot() for path in paths]
            check_current()
            if snapshots == self.restored.metadata['dataset_snapshots']:
                return self.restored
        with adapter.inference_slot(background=True):
            check_current()
            extractor = GR00TRLTStage2Extractor(adapter.policy, adapter.shadow_policy.encoder)

        class ScheduledExtractor:
            def extract(self, observation):
                check_current()
                with adapter.inference_slot(background=True):
                    check_current()
                    device = torch.device(adapter.policy.model.device)
                    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                    # Do not reset the live inference/TD3 random streams.
                    with torch.random.fork_rng(devices=devices):
                        return extractor.extract(observation)

            def normalize_actions(self, actions, states):
                with adapter.inference_slot(background=True):
                    check_current()
                    return extractor.normalize_actions(actions, states)

        replays, snapshots = [], []
        for path in paths:
            check_current()
            source = open_rlt_stage2_source(path, expected_fps=config.expected_fps,
                                            action_dim=adapter.spec.action_dim)
            key = (path, source.content_snapshot()['content_fingerprint'])
            snapshots.append((source, key[1]))
            if key not in self.cache:
                output = Path(tempfile.mkdtemp(dir=self.directory.name)) / 'replay'
                replay = materialize_rlt_stage2_replay(
                    [source], extractor=ScheduledExtractor(), spec=adapter.spec,
                    output_root=output, config=config, feature_batch_size=1,
                    seed_global_rng=False, require_both_outcomes=False,
                    progress_callback=lambda *_: check_current(),
                )
                self.cache[key] = replay
            replays.append(self.cache[key])
        if any(source.content_snapshot()['content_fingerprint'] != fingerprint
               for source, fingerprint in snapshots):
            raise ValueError('Selected dataset changed during replay preparation')
        metadata = {name: sum(r.metadata[name] for r in replays) for name in (
            'episode_count', 'usable_episode_count', 'success_episode_count',
            'failure_episode_count', 'skipped_short_episode_count', 'feature_frame_count',
            'transition_count')}
        for name in ('config', 'reward_contract', 'action_codec', 'reference_extraction'):
            metadata[name] = replays[0].metadata[name]
        for name in ('dataset_roots', 'dataset_snapshots', 'row_provenance'):
            metadata[name] = [item for replay in replays for item in replay.metadata[name]]
        metadata['dataset_snapshot_fingerprint'] = canonical_json_sha256({
            'format': 'cyclo.groot.rlt.dataset_collection/v1',
            'ordered_content_fingerprints': [s['content_fingerprint'] for s in metadata['dataset_snapshots']],
        }, allow_nan=False)
        if not metadata['success_episode_count'] or not metadata['failure_episode_count']:
            raise ValueError('RLT replay requires both labeled success and failure episodes')
        check_current()
        return RLTStage2FeatureReplay(adapter.spec, {
            name: torch.cat([r.tensors[name] for r in replays])
            for name in RLTStage2FeatureReplay._TENSOR_NAMES
        }, metadata=metadata)

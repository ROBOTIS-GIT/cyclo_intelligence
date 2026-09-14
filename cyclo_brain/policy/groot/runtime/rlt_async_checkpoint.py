"""Persist an Async RL snapshot using the existing Stage-2 bundle format."""

from copy import copy
import os
from pathlib import Path
import tempfile
import threading
from uuid import uuid4

from cyclo_brain.algorithm.rl.rlt import build_stage2_training_round


class RLTAsyncCheckpoint:
    def __init__(self, adapter):
        self.adapter = adapter
        self._lock = threading.Lock()
        self._thread = None
        self._status = {'saving': False, 'saved_bundle_path': None, 'save_error': None}

    def status(self):
        with self._lock:
            return dict(self._status)

    def start(self):
        adapter = self.adapter
        if adapter._async_training is None:
            raise RuntimeError('Initialize Async RL and complete an update before saving')
        with self._lock:
            if self._status['saving']:
                raise RuntimeError('An Async RL bundle is already being saved')
            self._status.update(saving=True, save_error=None)
            self._thread = threading.Thread(target=self._save, name='rlt-bundle-save', daemon=True)
            self._thread.start()

    def close(self):
        if self._thread is not None:
            self._thread.join()

    def _save(self):
        try:
            adapter = self.adapter
            snapshot = adapter._async_training.checkpoint_snapshot()
            replay = snapshot['replay']
            history = snapshot['history']
            last = history[-1]
            count = snapshot['learner']['completed_critic_updates']
            if count <= last['starting_critic_updates']:
                raise RuntimeError('No completed Async RL update to save')
            # Sibling, never nested inside the loaded bundle: recursive actor
            # resolution must still find exactly one actor in the original.
            parent = adapter.bundle.root.parent
            output = parent / f'async_rlt_c{count}_{uuid4().hex[:12]}'
            with tempfile.TemporaryDirectory(prefix='.rlt-save-', dir=parent) as temporary:
                staging = Path(temporary) / 'bundle'
                replay.save(staging / 'replay_cache')
                run = copy(adapter._async_run)
                run.replay_root = staging / 'replay_cache'
                run.training_round = build_stage2_training_round(
                    run.replay_root, expected_spec_fingerprint=replay.spec_fingerprint,
                    reference_seed=replay.metadata['reference_extraction']['seed'],
                    feature_batch_size=replay.metadata['reference_extraction']['feature_batch_size'],
                    sampling_seed=last['sampling_seed'],
                    batch_size=last['batch_size'],
                    steps=count - last['starting_critic_updates'],
                    starting_critic_updates=last['starting_critic_updates'],
                )
                run.async_state = {
                    'format': 'cyclo.rlt.async_training/v1',
                    'sampling_generator': snapshot['sampling_generator'],
                    'history': history,
                }
                run.save(staging, learner_state=snapshot['learner'],
                         verification_slot=lambda: adapter.inference_slot(background=True))
                # Root-run Docker writers must leave host-owned checkpoints
                # readable by the workspace owner (atomic saves are mode 0600).
                if os.geteuid() == 0:
                    owner = parent.stat()
                    for path in (staging, *staging.rglob('*')):
                        os.chown(path, owner.st_uid, owner.st_gid)
                if output.exists():
                    raise FileExistsError(output)
                staging.rename(output)  # publish only after full resume verification
            with self._lock:
                self._status['saved_bundle_path'] = str(output)
                self._status['saved_critic_updates'] = count
                self._status['saved_actor_updates'] = snapshot['learner']['completed_actor_updates']
        except Exception as error:
            with self._lock:
                self._status['save_error'] = f'{type(error).__name__}: {error}'
        finally:
            with self._lock:
                self._status['saving'] = False

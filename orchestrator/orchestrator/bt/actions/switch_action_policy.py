"""Switch a BT-owned GR00T session between VLA and a selected Stage 2 Actor."""
import json
import hashlib
from pathlib import Path
import re
import time

from interfaces.srv import InferenceCommand
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.execution.tree_pause import TimerPause
from orchestrator.bt.inference.supervisor import SET_ACTION_POLICY

BUNDLE_ROOT = Path('/workspace/checkpoint/rlt/stage2')
STAGES = {'actor': ('actor', 'actor_warmup'), 'critic': ('critic', 'critic_warmup'),
          'rl': ('rl_training', 'rl')}


def validate_bundle_artifacts(path, manifest):
    """Reject incomplete/corrupt Actor bundles before stopping a good session."""
    root = path.resolve()
    for role in ('rlt_actor', 'rl_token_encoder'):
        artifact = manifest.get('artifacts', {}).get(role, {})
        relative = artifact.get('relative_path', '')
        if not relative:
            raise ValueError(f'Selected bundle is missing {role}')
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f'Selected bundle has invalid {role} path')
        if file.stat().st_size != artifact.get('byte_count'):
            raise ValueError(f'Selected bundle has invalid {role} size')
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != artifact.get('sha256'):
            raise ValueError(f'Selected bundle has invalid {role} SHA256')


def bundle_path(stage, name):
    return BUNDLE_ROOT / STAGES[stage][0] / name


def prepare_bundle(selection):
    stage, name = selection
    path = bundle_path(stage, name)
    manifest = json.loads((path / 'manifest.json').read_text())
    fingerprint = manifest.get('manifest_fingerprint')
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('Selected bundle has no manifest fingerprint')
    if manifest.get('phase_state', {}).get('training_phase') != STAGES[stage][1]:
        raise ValueError('Selected bundle does not match training_stage')
    validate_bundle_artifacts(path, manifest)
    return str(path), fingerprint


def configure_preloads(root):
    """Plan the active tree before execution; never load weights in a switch."""
    from orchestrator.bt.actions.send_command import SendCommand
    from orchestrator.bt.execution.tree_pause import leaves
    actions = list(leaves(root))
    loads = [a for a in actions if isinstance(a, SendCommand)
             and a.target_str == 'INFERENCE' and a.command_str == 'LOAD'
             and a.backend == 'groot']
    # Explicit LOAD ports own the selection. Legacy XML can still select it
    # on a switch when no LOAD supplies the ports.
    selections = {(a.training_stage, a.bundle_name) for a in loads if a.training_stage}
    if not selections:
        selections = {(a.training_stage, a.bundle_name) for a in actions
                      if isinstance(a, SwitchActionPolicy) and a.action_source == 'mlp'}
    if len(selections) > 1:
        raise ValueError('One BT supports one preloaded MLP bundle; use the same stage and bundle')
    if not selections:
        return
    selection = next(iter(selections))
    for action in actions:
        if isinstance(action, SwitchActionPolicy):
            action.training_stage, action.bundle_name = selection
    if not loads:
        raise ValueError('MLP switching requires a GR00T LOAD in this tree')
    for owner in loads:
        if owner.action_request_mode != 'tt_rtc':
            raise ValueError('MLP switching requires GR00T TT-RTC')
        owner.mlp_preload_selection = selection


class SwitchActionPolicy(TimerPause, BaseAction):
    """Use VLA Action or Use MLP Action; MLP weights are preloaded by SendCommand LOAD.

    Requires SendCommand LOAD in this tree. Critic selects the Actor saved in
    a Critic warmup bundle. Selecting MLP explicitly authorizes robot routing
    if the preceding LOAD selected robot output. Paused sessions stay paused.
    """
    pause_timer_fields = ('deadline',)

    def __init__(self, node, name, action_source: str = 'vla',
                 training_stage: str = 'actor', bundle_name: str = 'latest'):
        super().__init__(node, name)
        self.action_source = str(action_source).strip().lower()
        self.training_stage = str(training_stage).strip().lower()
        self.bundle_name = str(bundle_name).strip()
        if self.action_source not in ('vla', 'mlp'):
            raise ValueError('action_source must be vla or mlp')
        if self.training_stage not in STAGES:
            raise ValueError('training_stage must be actor, critic or rl')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', self.bundle_name):
            raise ValueError('bundle_name must be a bundle name, such as latest')
        self.reset()

    def reset(self):
        # No asynchronous continuation may START after the BT revokes its owner.
        self.steps = None
        self.index = 0
        self.future = self.deadline = self.result = None
        self.bundle_identity = None
        super().reset()

    def _prepare(self, supervisor):
        if not supervisor.loaded or supervisor.load_request is None:
            raise RuntimeError('SwitchActionPolicy requires SendCommand LOAD in this tree')
        loaded = supervisor.load_request
        if supervisor.backend != 'groot' or loaded.action_request_mode != 'tt_rtc':
            raise RuntimeError('SwitchActionPolicy requires GR00T TT-RTC')
        mode = 'rlt' if self.action_source == 'mlp' else 'base'
        self.steps = []
        if mode == 'rlt':
            path = bundle_path(self.training_stage, self.bundle_name)
            identity = supervisor.loaded_bundle_identity
            if identity is None or identity[0] != str(path):
                raise RuntimeError('Selected MLP was not preloaded; reload the tree with this selection')
        if supervisor.running:
            route = InferenceCommand.Request()
            route.command = SET_ACTION_POLICY
            route.action_policy_mode = mode
            route.rlt_robot_override = mode == 'rlt'
            self.steps.append((SET_ACTION_POLICY, route))

    def tick(self):
        if self.result is not None:
            return self.result
        try:
            supervisor = getattr(self.node, 'inference_supervisor', None)
            if supervisor is None or supervisor.stopped:
                raise RuntimeError('Policy lifecycle unavailable or stopped; reload the tree')
            if self.steps is None:
                self._prepare(supervisor)
            if self.index == len(self.steps):
                supervisor.desired_action_mode = 'rlt' if self.action_source == 'mlp' else 'base'
                self.result = NodeStatus.SUCCESS
                return self.result
            command, request = self.steps[self.index]
            if self.future is None:
                self.future = (supervisor.submit_request(request) if request is not None
                               else supervisor.submit(command))
                self.deadline = time.monotonic() + 20
            if time.monotonic() > self.deadline:
                raise RuntimeError('Action policy switch timed out')
            if supervisor.poll(self.future):
                self.future = None
                self.index += 1
            return NodeStatus.RUNNING
        except Exception as exc:
            self.log_error(f'Action policy switch failed: {exc}')
            self.result = NodeStatus.FAILURE
            return self.result

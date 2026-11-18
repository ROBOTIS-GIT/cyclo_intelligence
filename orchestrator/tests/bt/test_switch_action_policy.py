"""Policy switching uses real lifecycle bookkeeping with fake ROS futures."""
import json
import hashlib
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from interfaces.srv import InferenceCommand
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.inference.supervisor import InferenceSupervisor, HOLD_CAPABILITY
from orchestrator.bt.inference.supervised_command import SupervisedCommand
from orchestrator.bt.actions.switch_action_policy import SwitchActionPolicy


class Client:
    def __init__(self): self.requests = []; self.pending = {}
    def service_is_ready(self): return True
    def call_async(self, request):
        self.requests.append(deepcopy(request))
        if request.command in self.pending: return self.pending[request.command]
        f = Future(); f.set_result(SimpleNamespace(success=True, message='ok')); return f


def make_session(tmp_path, monkeypatch, running=True):
    from orchestrator.bt.actions import switch_action_policy as module
    monkeypatch.setattr(module, 'BUNDLE_ROOT', tmp_path)
    for stage, phase in [('actor', 'actor_warmup'), ('critic', 'critic_warmup'), ('rl_training', 'rl')]:
        p = tmp_path / stage / 'latest'; p.mkdir(parents=True)
        artifacts = {}
        for role in ('rlt_actor', 'rl_token_encoder'):
            file = p / (role + '.pt'); file.write_bytes(role.encode())
            artifacts[role] = dict(relative_path=file.name, byte_count=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        (p / 'manifest.json').write_text(json.dumps({'artifacts': artifacts, 'manifest_fingerprint': stage+'-v1',
            'phase_state': {'training_phase': phase}}))
    node = SimpleNamespace(get_logger=lambda: Mock())
    s = InferenceSupervisor(node, strict_completion=False)
    node.inference_supervisor = s
    s.client = Client(); s.backend = 'groot'; s.verified = s.loaded = True
    s.running = running; s.publish_to_robot = True; s.instruction = 'pick'
    r = InferenceCommand.Request(); r.model_path = '/model'; r.robot_type = 'ffw_sg2_rev1'
    r.action_request_mode = 'tt_rtc'; r.publish_to_robot = True
    r.task_instruction = 'pick'; r.control_hz = 100.; r.inference_hz = 15
    s.load_request = r
    s.loaded_bundle_identity = (str(tmp_path/'actor/latest'), 'actor-v1')
    return node, s


def finish(a):
    for _ in range(30):
        status = a.tick()
        if status != NodeStatus.RUNNING: return status
    raise AssertionError('never completed')


@pytest.mark.parametrize('stage,directory', [('actor','actor'),('critic','critic'),('rl','rl_training')])
def test_preloaded_bundle_switches_without_loading(tmp_path, monkeypatch, stage, directory):
    node, s = make_session(tmp_path, monkeypatch)
    s.loaded_bundle_identity = (str(tmp_path/directory/'latest'), directory+'-v1')
    a = SwitchActionPolicy(node, 'switch', action_source='mlp', training_stage=stage)
    assert finish(a) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == [20]
    assert s.client.requests[-1].action_policy_mode == 'rlt'
    assert s.client.requests[-1].rlt_robot_override
    assert s.running and s.desired_action_mode == 'rlt'
    s.client.requests.clear()
    assert finish(SwitchActionPolicy(node, 'vla')) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == [20]
    s.client.requests.clear()
    assert finish(SwitchActionPolicy(node, 'mlp', action_source='mlp', training_stage=stage)) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == [20]


def test_paused_selection_does_not_start_until_resume(tmp_path, monkeypatch):
    node, s = make_session(tmp_path, monkeypatch, running=False)
    assert finish(SwitchActionPolicy(node, 'mlp', action_source='mlp')) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == []
    assert not s.running and s.desired_action_mode == 'rlt'
    owner = SimpleNamespace(command_str='RESUME', target_str='INFERENCE', task_instruction='',
                            initial_pose_sync=False, initial_pose_sync_duration_s=5., log_error=Mock())
    assert finish(SupervisedCommand(owner, s)) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests][-2:] == [1,20]


def test_replaced_latest_stays_pinned_until_load(tmp_path, monkeypatch):
    node, s = make_session(tmp_path, monkeypatch)
    assert finish(SwitchActionPolicy(node,'first',action_source='mlp')) == NodeStatus.SUCCESS
    p = tmp_path/'actor/latest/manifest.json'; d=json.loads(p.read_text()); d['manifest_fingerprint']='v2';p.write_text(json.dumps(d))
    s.client.requests.clear()
    assert finish(SwitchActionPolicy(node,'second',action_source='mlp')) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == [20]


@pytest.mark.parametrize('kwargs', [{'action_source':'bad'}, {'training_stage':'bad'}, {'bundle_name':'../latest'}])
def test_invalid_ports_rejected(kwargs):
    with pytest.raises(ValueError): SwitchActionPolicy(SimpleNamespace(), 'bad', **kwargs)


def test_missing_bundle_fails_before_stop(tmp_path, monkeypatch):
    node,s=make_session(tmp_path,monkeypatch)
    assert finish(SwitchActionPolicy(node,'bad',action_source='mlp',bundle_name='missing')) == NodeStatus.FAILURE
    assert s.client.requests == [] and s.running


def preload_owner(node):
    from test_supervised_inference import Owner
    node.robot_type = 'ffw_sg2_rev1'
    owner = Owner('LOAD'); owner.action_request_mode = 'tt_rtc'
    owner.mlp_preload_selection = ('actor', 'latest')
    return owner


def test_initial_load_includes_mlp_and_native_numeric_fields(tmp_path, monkeypatch):
    from rclpy.serialization import serialize_message, deserialize_message
    node, s = make_session(tmp_path, monkeypatch, running=False)
    f = s.submit(0, preload_owner(node))
    assert s.poll(f)
    r = deserialize_message(serialize_message(s.client.requests[-1]), InferenceCommand.Request)
    assert r.rlt_enabled and r.rlt_bundle_path == str(tmp_path/'actor/latest')
    assert r.action_policy_mode == 'base' and r.control_hz == 100.0
    assert s.loaded_bundle_identity == (str(tmp_path/'actor/latest'), 'actor-v1')
    assert [r.command for r in s.client.requests] == [0] and not s.running


def test_cancel_pending_preload_never_starts(tmp_path, monkeypatch):
    node, s = make_session(tmp_path, monkeypatch, running=False)
    f = Future(); s.client.pending[0] = f
    s.submit(0, preload_owner(node)); s.stop()
    f.set_result(SimpleNamespace(success=True, message='loaded'))
    s.poll(f)
    assert finish(SwitchActionPolicy(node, 'switch', action_source='mlp')) == NodeStatus.FAILURE
    assert not any(r.command == 1 for r in s.client.requests)


def test_missing_preload_fails_without_loading(tmp_path, monkeypatch):
    node, s = make_session(tmp_path, monkeypatch)
    s.loaded_bundle_identity = None
    assert finish(SwitchActionPolicy(node, 'switch', action_source='mlp')) == NodeStatus.FAILURE
    assert not s.client.requests


@pytest.mark.parametrize('corruption', ['phase', 'missing', 'hash'])
def test_invalid_preload_is_rejected(tmp_path, monkeypatch, corruption):
    node, s = make_session(tmp_path, monkeypatch, running=False)
    if corruption == 'phase':
        p = tmp_path/'actor/latest/manifest.json'; d = json.loads(p.read_text())
        d['phase_state']['training_phase'] = 'rl'; p.write_text(json.dumps(d))
    else:
        p = tmp_path/'actor/latest/rlt_actor.pt'
        if corruption == 'missing': p.unlink()
        else: p.write_bytes(b'x' * p.stat().st_size)
    with pytest.raises(ValueError): s.submit(0, preload_owner(node))
    assert not s.client.requests


def test_replaced_bundle_during_preload_does_not_start(tmp_path, monkeypatch):
    node, s = make_session(tmp_path, monkeypatch, running=False)
    f = Future(); s.client.pending[0] = f
    s.submit(0, preload_owner(node))
    p = tmp_path/'actor/latest/manifest.json'; d = json.loads(p.read_text())
    d['manifest_fingerprint'] = 'changed'; p.write_text(json.dumps(d))
    f.set_result(SimpleNamespace(success=True, message='loaded'))
    with pytest.raises(RuntimeError, match='changed during LOAD'): s.poll(f)
    assert not s.loaded and not s.running


def test_tree_assigns_selection_to_initial_load_and_rejects_conflicts(tmp_path, monkeypatch):
    from orchestrator.bt.actions.send_command import SendCommand
    from orchestrator.bt.actions.switch_action_policy import configure_preloads
    node, s = make_session(tmp_path, monkeypatch)
    node.create_client = lambda *args, **kwargs: s.client
    node.create_subscription = Mock()
    load = SendCommand(node=node, target='INFERENCE', command='LOAD', model='groot:n17', action_request_mode='tt_rtc')
    switch = SwitchActionPolicy(node, 'mlp', action_source='mlp')
    root = SimpleNamespace(children=[load, SimpleNamespace(children=[switch])])
    configure_preloads(root)
    assert load.mlp_preload_selection == ('actor', 'latest')
    root.children.append(SwitchActionPolicy(node, 'other', action_source='mlp', training_stage='rl'))
    with pytest.raises(ValueError, match='one preloaded MLP'): configure_preloads(root)


def test_manual_completion_stops_output_without_failure(tmp_path, monkeypatch):
    node,s=make_session(tmp_path,monkeypatch)
    s.complete()
    assert s.stopped and not s.running and s.client.requests[-1].command==4


def test_loader_discovers_switch_with_only_three_user_ports():
    from orchestrator.bt.node_registry import build_registry, catalog_payload
    registry=build_registry()
    assert registry['SwitchActionPolicy'] is SwitchActionPolicy
    entry=next(x for x in catalog_payload() if x['tag']=='SwitchActionPolicy')
    assert [p['name'] for p in entry['ports']]==['action_source','training_stage','bundle_name']


@pytest.mark.parametrize('operation', ['STOP', 'RESTART'])
def test_docker_teardown_waits_for_confirmed_hold_and_unload(tmp_path, monkeypatch, operation):
    from orchestrator.bt.actions.send_command import SendCommand
    node,s=make_session(tmp_path,monkeypatch)
    node.create_client=lambda *args,**kwargs:s.client
    stop=Future();s.client.pending[4]=stop
    action=SendCommand(node=node,target='DOCKER',command=operation,model='groot:n17')
    calls=[]
    def start_job(op):
        calls.append(op)
        assert not s.running and not s.loaded and not s.verified
        action._backend_job=SimpleNamespace(snapshot=lambda:(True,'done'))
    monkeypatch.setattr(action,'_start_backend_job',start_job)
    for _ in range(3): assert action.tick()==NodeStatus.RUNNING
    assert calls==[] and [r.command for r in s.client.requests]==[4]
    stop.set_result(SimpleNamespace(success=True,message='stopped'))
    assert finish(action)==NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests]==[4,5]
    assert calls==[operation]
    s.complete()


def test_docker_leaf_does_not_break_policy_tree_pause(tmp_path, monkeypatch):
    from orchestrator.bt.actions.send_command import SendCommand
    node,s=make_session(tmp_path,monkeypatch)
    node.create_client=lambda *args,**kwargs:s.client
    action=SendCommand(node=node,target='DOCKER',command='START',model='groot:n17')
    action.pause(); action.resume()  # unstarted
    action._state=action._STATE_DONE
    action.pause(); action.resume()  # completed
    action._backend_cleanup=Mock()
    action.pause(); action.resume()
    action._backend_cleanup.pause.assert_called_once()
    action._backend_cleanup.resume.assert_called_once()


@pytest.mark.parametrize('xml,owned,strict', [
    ('<root><SwitchActionPolicy/></root>', True, False),
    ('<root autonomy_supervised="true"><SwitchActionPolicy/></root>', True, True),
    ('<root><Wait/></root>', False, None),
    ('<root main_tree_to_execute="Main"><BehaviorTree ID="Main"><Wait/></BehaviorTree><BehaviorTree ID="__pending__"><SwitchActionPolicy/></BehaviorTree></root>', False, None),
])
def test_bt_selects_owner_before_building_policy_switch_tree(xml, owned, strict):
    from test_pause_resume import _bt_engine
    cls,node=_bt_engine();node.tree_execution_mode='stopped'
    def load(tree_xml):
        assert (node.inference_supervisor is not None)==owned
        if owned: assert node.inference_supervisor.strict_completion==strict
        return SimpleNamespace(name='tree')
    node.tree_loader.load_tree_from_string.side_effect=load
    result=cls._load_and_run_callback(node,SimpleNamespace(tree_xml=xml),SimpleNamespace())
    assert result.success


@pytest.mark.parametrize('stage', ['actor', 'critic', 'rl'])
def test_explicit_load_ports_own_selection_for_source_only_switches(tmp_path, monkeypatch, stage):
    import xml.etree.ElementTree as ET
    from orchestrator.bt.bt_nodes_loader import TreeLoader
    from orchestrator.bt.node_registry import build_registry
    from orchestrator.bt.actions.switch_action_policy import configure_preloads
    node, s = make_session(tmp_path, monkeypatch)
    node.create_client = lambda *args, **kwargs: s.client
    node.create_subscription = Mock()
    loader = TreeLoader(node); loader._registry = build_registry()
    root = loader._load_node(ET.fromstring(f'''<Sequence>
        <SendCommand name="LoadGroot" target="INFERENCE" command="LOAD"
          model="groot:n17" action_request_mode="tt_rtc" training_stage="{stage}" bundle_name="latest"/>
        <SwitchActionPolicy name="SwitchActionPolicy_1" action_source="mlp"/>
        <SwitchActionPolicy name="SwitchActionPolicy_2" action_source="vla"/>
    </Sequence>'''))
    configure_preloads(root)
    load, mlp, vla = root.children
    assert load.mlp_preload_selection == (stage, 'latest')
    node.robot_type = 'ffw_sg2_rev1'
    s.running = False
    assert s.poll(s.submit(0, load))
    assert s.client.requests[-1].rlt_bundle_path == str(tmp_path / {'rl':'rl_training'}.get(stage,stage) / 'latest')
    s.running = True; s.client.requests.clear()
    assert finish(mlp) == finish(vla) == NodeStatus.SUCCESS
    assert [r.command for r in s.client.requests] == [20, 20]
    assert [r.action_policy_mode for r in s.client.requests] == ['rlt', 'base']

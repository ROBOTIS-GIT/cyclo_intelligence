"""Controller regression tests without a GPU, ROS graph, or robot motion."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi import HTTPException
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from supervisor_api import autonomy_config as config
from supervisor_api.migrate_observation_plan import migrate
from supervisor_api.autonomy import AutonomyController, AutonomyStart, MissionStartConfirmation
from supervisor_api.autonomy_config import AutonomyRepository, ObservationPlan
from supervisor_api.autonomy_runtime import OperationCancelled


def eventually(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline: raise AssertionError('Condition did not become true')
        time.sleep(.005)


class Runtime:
    def __init__(self):
        self.calls = []
        self.fail = False
    def open(self): pass
    def navigate(self, pose, cancelled):
        if cancelled.is_set(): raise OperationCancelled()
        self.calls.append(('nav', pose['x']))
    def tree(self, xml, cancelled):
        if cancelled.is_set(): raise OperationCancelled()
        if xml:
            self.calls.append(('bt', xml))
            if self.fail: raise RuntimeError('Task failed')
    def cancel(self): self.calls.append(('cancel',))
    def close(self): self.calls.append(('close',))
    def pause(self): self.calls.append(('pause',))
    def resume(self): self.calls.append(('resume',))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.release = threading.Event()
        self.service = Mock()
        self.service.status.return_value = {'phase': 'ready', 'session_id': 'session'}
        self.service.stop.side_effect = self.release.set
        self.selected = {'mission_id': 'task', 'mission_name': 'Task', 'revision': 1, 'map_name': 'Home'}
        self._start_decision = {'action_allowed': True, 'selected_mission': self.selected,
            'decision': {'decision': 'start', 'mission_id': 'task', 'reason': 'Ready'}}
        self.wait_decision = {'action_allowed': True, 'decision': {'decision': 'wait', 'mission_id': None, 'reason': 'Wait'}}
        self.selection_count = 0
        self.selection_behavior = lambda req: self.wait_decision
        self.service.decide.side_effect = self.decide
        self.repo = Mock()
        self.plan = ObservationPlan.model_construct(revision=1, mode='fixed', settle_seconds=.001,
            interval_seconds=.001, excluded_mission_ids=[])
        self.repo.read.return_value = self.plan
        self.points = [{'spot_id': 'a', 'label': 'A', 'pose': {'x': 0, 'y': 0},
            'arrival_bt': '', 'departure_policy': 'timed', 'dwell_seconds': 0, 'mission_ids': ['task']},
            {'spot_id': 'b', 'label': 'B', 'pose': {'x': 1, 'y': 0},
            'arrival_bt': '', 'departure_policy': 'timed', 'dwell_seconds': 0, 'mission_ids': ['task']}]
        self.repo.resolve.return_value = self.points, [self.selected]
        self.repo.mission.return_value = [{'id': 'step', 'label': 'Sink', 'pose': {'x': 2, 'y': 0}, 'xml': 'task'}]
        self.controller = AutonomyController(self.service, lambda: self.runtime, Mock(), self.repo)
        self.request = AutonomyStart(map_name='Home', model_id='tuned', localized=True, plan_revision=1)
        self.addCleanup(self.cleanup)

    def decide(self, request):
        if request.reasoning_task == 'patrol_supervision':
            return {'action_allowed': True, 'frame_stamp': str(self.service.decide.call_count),
                    'decision': {'decision': 'continue', 'mission_id': None, 'reason': 'Clear path'}}
        if request.reasoning_task == 'mission_supervision':
            return {'action_allowed': True, 'frame_stamp': str(self.service.decide.call_count),
                    'decision': {'decision': 'continue', 'mission_id': 'task', 'reason': 'Clear'}}
        if (request.observation or {}).get('phase') != 'observing':
            return self.wait_decision
        self.selection_count += 1
        return self.selection_behavior(request)

    @property
    def start_decision(self):
        return {**self._start_decision, 'frame_stamp': str(self.service.decide.call_count)}

    def cleanup(self):
        self.release.set()
        if self.controller.owned:
            self.controller.stop(); self.controller.stop_thread.join(4)
        self.assertFalse(self.controller.thread and self.controller.thread.is_alive())

    def test_fixed_observation_repeats_tasks_and_keeps_model_loaded(self):
        self.selection_behavior = lambda req: self.start_decision if self.selection_count <= 8 else self.wait_decision
        self.controller.start(self.request)
        eventually(lambda: self.runtime.calls.count(('bt', 'task')) == 2 and self.service.decide.call_count >= 3)
        self.assertEqual([v for v in self.runtime.calls if v[0] == 'nav'][:5], [('nav', 0), ('nav', 2), ('nav', 0), ('nav', 2), ('nav', 0)])
        self.service.start.assert_called_once()
        self.service.stop.assert_not_called()
        self.assertTrue(self.controller.status()['active'])
        self.assertTrue(any(call.args[0].reasoning_task == 'mission_selection'
                            and call.args[0].allowed_mission_ids == ['task']
                            for call in self.service.decide.call_args_list))
        self.assertEqual(len(self.controller.status()['recent_missions']), 2)

    def test_task_waits_for_third_confirmation(self):
        third = threading.Event()
        def infer(req):
            if self.selection_count == 3:
                third.set()
                self.release.wait(3)
            return self.start_decision if self.selection_count <= 3 else self.wait_decision
        self.selection_behavior = infer
        self.controller.start(self.request)
        self.assertTrue(third.wait(3))
        self.repo.mission.assert_not_called()
        self.assertNotIn(('nav', 2), self.runtime.calls)
        self.release.set()
        eventually(lambda: self.runtime.calls.count(('bt', 'task')) == 1)

    def test_task_status_exposes_exact_xml_for_operate_view(self):
        snapshots=[]
        original=self.runtime.tree
        def tree(xml, cancelled):
            if xml:
                snapshots.append(self.controller.status()['runner'])
            original(xml,cancelled)
        self.runtime.tree=tree
        self.selection_behavior=lambda req: self.start_decision if self.selection_count <= 3 else self.wait_decision
        self.controller.start(self.request)
        eventually(lambda: bool(snapshots) and self.service.decide.call_count >= 2)
        self.assertEqual(snapshots[0]['phase'],'waypoint-task')
        self.assertEqual(snapshots[0]['bt_xml'],'task')
        self.assertEqual(snapshots[0]['activeSpotId'],'step')
        eventually(lambda: 'bt_xml' not in self.controller.status()['runner'])

    def test_patrol_wait_and_unknown_move_to_next_spot(self):
        self.plan.mode = 'patrol'
        self.selection_behavior = lambda req: {'decision': {'decision': 'unknown'}}
        self.controller.start(self.request)
        eventually(lambda: len([v for v in self.runtime.calls if v[0] == 'nav']) >= 3)
        self.assertEqual(self.runtime.calls[:3], [('nav', 0), ('nav', 1), ('nav', 0)])
        self.assertEqual(self.repo.mission.call_count, 0)

    def test_closed_loop_patrol_repeats_three_spots_until_explicit_stop(self):
        self.plan.mode = 'patrol'
        self.plan.route_closed = True
        self.points.append({**self.points[0], 'spot_id': 'c', 'label': 'C', 'pose': {'x': 3, 'y': 0}})
        self.controller.start(self.request)
        eventually(lambda: len([v for v in self.runtime.calls if v[0] == 'nav']) >= 7)
        self.assertEqual([v for v in self.runtime.calls if v[0] == 'nav'][:7],
                         [('nav', 0), ('nav', 1), ('nav', 3), ('nav', 0), ('nav', 1), ('nav', 3), ('nav', 0)])
        self.assertTrue(self.controller.status()['active'])
        self.service.stop.assert_not_called()
        self.controller.stop()
        self.controller.stop_thread.join(4)
        count = len(self.runtime.calls)
        time.sleep(.02)
        self.assertEqual(len(self.runtime.calls), count)
        self.assertFalse(self.controller.status()['active'])

    def test_one_way_patrol_keeps_observing_final_stop_until_operator_stops(self):
        self.plan.mode = 'patrol'
        self.plan.route_closed = False
        self.controller.start(self.request)
        eventually(lambda: (self.controller.status()['observation'] or {}).get('spot_id') == 'b'
                   and self.controller.status()['phase'] == 'observing')
        count = self.selection_count
        eventually(lambda: self.selection_count >= count + 5)
        self.assertEqual([v for v in self.runtime.calls if v[0] == 'nav'], [('nav', 0), ('nav', 1)])
        self.assertTrue(self.controller.status()['active'])
        self.service.stop.assert_not_called()

    def test_task_failure_preserves_progress_and_never_restarts_task(self):
        self.runtime.fail = True
        self.selection_behavior = lambda req: self.start_decision
        self.controller.start(self.request)
        eventually(lambda: self.controller.status()['phase'] == 'error')
        self.assertEqual(self.runtime.calls.count(('bt', 'task')), 1)
        self.assertEqual(self.controller.status()['runner']['label'], 'Sink')
        self.assertTrue(self.controller.status()['needsStop'])
        self.assertIn('Task · Sink', self.controller.status()['error'])
        self.service.stop.assert_not_called()

    def test_stop_discards_inflight_decision_and_unloads_once(self):
        entered = threading.Event()
        def infer(_):
            entered.set(); self.release.wait(3); return self.start_decision
        self.selection_behavior = infer
        self.controller.start(self.request)
        self.assertTrue(entered.wait(3))
        self.controller.stop(); self.controller.stop(); self.controller.stop_thread.join(4)
        self.assertFalse(self.controller.owned)
        self.assertNotIn(('nav', 2), self.runtime.calls)
        self.service.stop.assert_called_once()

    def test_operator_instruction_invalidates_pending_decision(self):
        entered = threading.Event()
        def infer(_):
            if self.selection_count == 1:
                entered.set(); self.release.wait(3); return self.start_decision
            return self.wait_decision
        self.selection_behavior = infer
        self.controller.start(self.request)
        self.assertTrue(entered.wait(3))
        self.controller.instruction('Wait here')
        self.release.set()
        eventually(lambda: self.service.decide.call_args.args[0].operator_instruction == 'Wait here')
        self.repo.mission.assert_not_called()
        self.assertEqual(self.service.decide.call_args.args[0].operator_instruction, 'Wait here')

    def test_exclusive_start_and_revision_check(self):
        self.request.plan_revision = 0
        with self.assertRaises(HTTPException): self.controller.start(self.request)
        self.request.plan_revision = 1
        self.controller.start(self.request)
        with self.assertRaises(HTTPException) as caught: self.controller.start(self.request)
        self.assertEqual(caught.exception.status_code, 409)

    def test_stop_also_unloads_reasoning_without_an_owned_mission(self):
        self.controller.state['result'] = {'text': 'stale decision'}
        self.controller.stop()
        self.controller.stop_thread.join(4)
        self.service.stop.assert_called_once()
        self.assertEqual(self.controller.status()['phase'], 'stopped')
        self.assertIsNone(self.controller.status()['result'])
        self.assertFalse(self.controller.status()['needsStop'])

    def test_stop_failure_retains_ownership_until_retry(self):
        self.controller.start(self.request)
        eventually(lambda: self.service.decide.call_count > 0)
        self.service.stop.side_effect = RuntimeError('Docker failed')
        self.controller.stop(); self.controller.stop_thread.join(4)
        self.assertTrue(self.controller.owned)
        self.assertEqual(self.controller.status()['phase'], 'error')
        self.service.stop.side_effect = None
        self.controller.stop(); self.controller.stop_thread.join(4)
        self.assertFalse(self.controller.owned)

    def test_restart_recovers_by_stopping_not_replaying(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            path.write_text(json.dumps({'map_name': 'Home', 'mission': self.selected}))
            recovery = Mock()
            restored = AutonomyController(self.service, Mock(), Mock(), self.repo, path, recovery)
            self.assertEqual(restored.status()['phase'], 'interrupted')
            restored.stop(); restored.stop_thread.join(4)
            recovery.assert_called_once()
            self.assertFalse(path.exists())
            self.service.start.assert_not_called()


class ConfirmationTests(unittest.TestCase):
    def result(self, stamp, mission='task', **overrides):
        return {'frame_stamp': str(stamp), 'action_allowed': True,
                'decision': {'decision': 'start', 'mission_id': mission},
                'selected_mission': {'mission_id': mission, 'revision': 1}, **overrides}

    def test_requires_three_distinct_frames(self):
        gate = MissionStartConfirmation()
        for stamp in [1, 1, 2, 1]:
            self.assertFalse(gate.accept(self.result(stamp), 0))
        self.assertTrue(gate.accept(self.result(3), 0))

    def test_wait_unknown_error_and_disallowed_reset(self):
        for invalid in [ {'decision': {'decision': 'wait'}},
                         {'decision': {'decision': 'unknown'}},
                         {'error': 'Camera unavailable'}, {'action_allowed': False},
                         {'frame_stamp': None} ]:
            gate = MissionStartConfirmation()
            for stamp in range(3): gate.accept(self.result(stamp), 0)
            self.assertFalse(gate.accept(self.result(4, **invalid), 0))
            self.assertFalse(gate.accept(self.result(5), 0))
            self.assertEqual(len(gate.stamps), 1)

    def test_mission_revision_and_instruction_change_reset(self):
        for change, version in [(self.result(4, mission='other'), 0),
                                (self.result(4, selected_mission={'mission_id': 'task', 'revision': 2}), 0),
                                (self.result(4), 1)]:
            gate = MissionStartConfirmation()
            for stamp in range(3): gate.accept(self.result(stamp), 0)
            self.assertFalse(gate.accept(change, version))
            self.assertEqual(len(gate.stamps), 1)


class RepositoryTests(unittest.TestCase):
    def test_config_roundtrip_stable_spot_ids_and_revision(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config.spots, 'SPOTS_ROOT', Path(directory)):
            repo = AutonomyRepository()
            with patch.object(repo, 'resolve', return_value=([], [])):
                self.assertTrue(ObservationPlan().route_closed)
                plan = repo.save('Home', ObservationPlan(route_closed=False))
                self.assertFalse(repo.read('Home').route_closed)
                self.assertEqual(plan.revision, 1)
                self.assertEqual(repo.read('Home'), plan)
                with self.assertRaises(HTTPException): repo.save('Home', ObservationPlan())
            with self.assertRaises(HTTPException): repo.path('..')

    def test_route_snapshot_preserves_order_and_rejects_stale_revisions(self):
        from types import SimpleNamespace
        waypoint = lambda wid: SimpleNamespace(id=wid, model_dump=lambda: {
            'id': wid, 'label': wid, 'pose': {'x': 0, 'y': 0}, 'local_bt': ''})
        manifest = SimpleNamespace(exists=True, revision=2, metadata={
            'reasoning_mission_id': 'task', 'mission_flow': {'edges': [
                {'source': 'B', 'target': 'A'}]}}, waypoints=[waypoint('A'), waypoint('B')])
        selected = {'mission_name': 'Task', 'mission_id': 'task', 'revision': 2}
        with patch.object(config.missions, '_read_manifest', return_value=manifest):
            repo = AutonomyRepository()
            self.assertEqual([s['id'] for s in repo.mission('Home', selected)], ['B', 'A'])
            selected['revision'] = 1
            with self.assertRaisesRegex(ValueError, 'changed'): repo.mission('Home', selected)
            selected['revision'] = 2
            manifest.metadata['mission_flow']['edges'].append({'source': 'B', 'target': 'B'})
            with self.assertRaisesRegex(ValueError, 'Branched'): repo.mission('Home', selected)

    def test_home_migration_is_idempotent_and_removes_only_return_waypoints(self):
        import shutil
        source = Path.cwd() / 'orchestrator/examples/home_meal'
        if not source.exists(): source = Path(__file__).parents[2] / 'orchestrator/examples/home_meal'
        # The shipped example is already migrated. Running migration must preserve it.
        with tempfile.TemporaryDirectory() as directory:
            shutil.copytree(source, directory, dirs_exist_ok=True)
            self.assertFalse(migrate(directory, 'Home', 'ObservationMode', ['CleanTable'], True)['changed'])
            manifest = json.loads((Path(directory) / 'missions/Home/CleanTable/mission.json').read_text())
            self.assertEqual([wp['label'] for wp in manifest['waypoints']], ['Table', 'Sink'])

if __name__ == '__main__': unittest.main()

"""Decision-driven observation departure, contracts and recorded route context."""
import json
import unittest
import tempfile
from supervisor_api import collection as collection
from supervisor_api.test_continuous_reasoning_data import FRAME
from supervisor_api import mission_decisions as contract
from supervisor_api.collection_observation_plan import departure_context
from supervisor_api import test_autonomy as fixtures
from supervisor_api.test_autonomy import eventually

class DepartureContractTests(unittest.TestCase):
    def test_selection_label_roundtrip_and_context(self):
        target={'decision':'next_observation','mission_id':None,'reason':'The table is clear; observation here is complete.'}
        context={'reasoning_task':'mission_selection','observation':{'can_advance':True,'next_spot_id':'b'}}
        sample=contract.sample_for(context,{'missions':[]},target,'frame.jpg')
        self.assertEqual(json.loads(sample['messages'][-1]['content'][0]['text']),target)
        self.assertEqual(contract.build_input(context,[])['context']['observation']['next_spot_id'],'b')
        for ctx in ({**context,'observation':{'can_advance':False}}, {'reasoning_task':'patrol_supervision'}, {'reasoning_task':'mission_supervision'}):
            with self.assertRaises(ValueError):contract.validate_target(target,ctx,[])
        with self.assertRaises(ValueError):contract.validate_target({**target,'mission_id':'task'},context,[])

    def test_next_label_is_saved_with_null_mission_and_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            store=collection.CollectionStore(folder, lambda _: [])
            ep=store.start(collection.EpisodeRequest(map_name='Home'))
            frame=store.capture(ep['id'],FRAME,{'reasoning_task':'mission_selection',
                'observation':{'phase':'observing','can_advance':True,'next_spot_id':'b'}})
            store.annotate(ep['id'],collection.AnnotationRequest(frame_id=frame['frame_id'],
                decision='next_observation',mission_id=None,reason='This observation is complete.',
                annotator='human',review_status='approved'))
            saved=store.get(ep['id'])
            self.assertEqual(saved['annotations'][-1]['target']['decision'],'next_observation')
            sample=contract.sample_for(saved['observations'][0]['context'],{'missions':[]},
                saved['annotations'][-1]['target'],'frame.jpg')
            self.assertIn('next_observation',sample['messages'][-1]['content'][0]['text'])

    def test_route_context_last_spot_and_closed_loop(self):
        plan={'route_closed':False,'observations':[{'spot_id':'a'},{'spot_id':'b'}]}
        self.assertEqual(departure_context(plan,'a')['next_spot_id'],'b')
        self.assertFalse(departure_context(plan,'b')['can_advance'])
        plan['route_closed']=True
        self.assertEqual(departure_context(plan,'b')['next_spot_id'],'a')
        plan['observations']=plan['observations'][:1]
        self.assertFalse(departure_context(plan,'a')['can_advance'])

class DepartureRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.ControllerTests('runTest');self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.h.plan.mode='patrol'
        for point in self.h.points:point['departure_policy']='reasoning'

    def test_wait_unknown_and_invalid_decisions_do_not_expire(self):
        h=self.h
        h.selection_behavior=lambda req: {'action_allowed':True,'decision':{'decision':'unknown','mission_id':None}}
        h.controller.start(h.request)
        eventually(lambda:h.selection_count>8)
        self.assertEqual([v for v in h.runtime.calls if v[0]=='nav'],[('nav',0)])
        h.selection_behavior=lambda req: {**h.wait_decision,'frame_stamp':str(h.selection_count)}
        count=h.selection_count;eventually(lambda:h.selection_count>count+8)
        self.assertEqual([v for v in h.runtime.calls if v[0]=='nav'],[('nav',0)])
        h.selection_behavior=lambda req: {'action_allowed':False,'frame_stamp':'bad','decision':{'decision':'next_observation','mission_id':None}}
        count=h.selection_count;eventually(lambda:h.selection_count>count+8)
        self.assertEqual([v for v in h.runtime.calls if v[0]=='nav'],[('nav',0)])
        h.selection_behavior=lambda req: {'action_allowed':True,'frame_stamp':str(h.selection_count),'decision':{'decision':'next_observation','mission_id':None,'reason':'Assessment complete'}}
        eventually(lambda:('nav',1) in h.runtime.calls)
        h.repo.mission.assert_not_called()

    def test_fixed_and_last_one_way_spot_stay(self):
        h=self.h;h.plan.route_closed=False
        h.selection_behavior=lambda req: {'action_allowed':True,'frame_stamp':str(h.selection_count),'decision':{'decision':'next_observation','mission_id':None,'reason':'Complete'}}
        h.controller.start(h.request)
        eventually(lambda:h.selection_count>10)
        self.assertEqual([v for v in h.runtime.calls if v[0]=='nav'],[('nav',0),('nav',1)])
        self.assertFalse(h.controller.capture_context()['observation']['can_advance'])

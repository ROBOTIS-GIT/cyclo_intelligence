import json
import tempfile
import unittest
from pathlib import Path
from scripts.mission_smoke import prepare_mission,select_mission

class DecisionTests(unittest.TestCase):
    def result(self, decision='start', mission_id='known'):
        return {'finish_reason':'stop','text':json.dumps({'decision':decision,'mission_id':mission_id,'reason':'visible stick'})}
    def test_only_registered_start_can_select(self):
        catalog=[{'mission_id':'known'}]
        self.assertEqual(select_mission(self.result(),catalog)[1],catalog[0])
        for result in [self.result(mission_id='invented'), self.result('cancel'),
                       self.result('unknown','unknown'), {'finish_reason':'length','text':self.result()['text']},
                       {'finish_reason':'stop','text':'```json\n'+self.result()['text']+'\n```'}]:
            with self.assertRaises(ValueError): select_mission(result,catalog)
    def test_unknown_and_wait_never_select(self):
        for decision in ['wait','unknown']:
            self.assertIsNone(select_mission(self.result(decision,None),[])[1])
    def test_changed_revision_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=root/'map'/'mission';d.mkdir(parents=True)
            mission={'map_name':'map','mission_name':'mission','revision':2,'mission_id':'known'}
            (d/'mission.json').write_text(json.dumps({'revision':1,'metadata':{'reasoning_mission_id':'known'}}))
            with self.assertRaisesRegex(ValueError,'changed'):prepare_mission(root,mission)
            mission['revision']=1
            (d/'global.xml').write_text('<root><BehaviorTree><Sequence><MissionStep x="0" y="0" yaw="0" local_bt="../../escape.xml"/></Sequence></BehaviorTree></root>')
            with self.assertRaisesRegex(ValueError,'leaves'):prepare_mission(root,mission)
            (d/'empty.xml').write_text('<root><BehaviorTree ID="MainTree"/></root>')
            (d/'global.xml').write_text('<root><BehaviorTree><Sequence><MissionStep x="0" y="0" yaw="0" local_bt="empty.xml"/></Sequence></BehaviorTree></root>')
            self.assertIsNone(prepare_mission(root,mission)[0][0]['xml'])

if __name__=='__main__':unittest.main()

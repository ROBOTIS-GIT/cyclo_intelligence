import json
import unittest
from copy import deepcopy
from supervisor_api.qwen_training_balance import balance_frames

class BalanceTests(unittest.TestCase):
    def test_repeatable_stratified_selection_unknown_and_disabled(self):
        rows=[]
        for mission,decision,count in [('a','start',6),('b','start',2),(None,'wait',4),(None,'unknown',1)]:
            for _ in range(count):
                rows.append({'frame_id':str(len(rows)), 'sample':{'messages':[{'content':[{'text':json.dumps({'decision':decision,'mission_id':mission})}]}]}})
        before=deepcopy(rows)
        chosen,report=balance_frames(rows,True)
        self.assertEqual(report['after']['decisions'],{'start':4,'wait':4,'unknown':1})
        self.assertEqual(report['after']['missions'],{'a':3,'b':1})
        self.assertEqual(chosen,balance_frames(rows,True)[0])
        self.assertEqual(rows,before)
        self.assertEqual(balance_frames(rows,False)[0],rows)
        self.assertEqual(report['excluded'],4)

    def test_each_task_balances_its_own_decisions_and_keeps_minority_unknown_cancel(self):
        groups = {
            'mission_selection': [('start', 3), ('wait', 40), ('unknown', 1)],
            'patrol_supervision': [('continue', 12), ('pause', 2)],
            'mission_supervision': [('continue', 9), ('pause', 3), ('cancel', 1)],
        }
        rows = []
        for task, values in groups.items():
            for decision, count in values:
                for _ in range(count):
                    target = {'decision': decision, 'mission_id': 'clean' if decision == 'start' else None}
                    rows.append({'frame_id': str(len(rows)), 'frame': {'context': {'reasoning_task': task}},
                                 'sample': {'messages': [{'role': 'assistant', 'content': [
                                     {'type': 'text', 'text': json.dumps(target)}]}]}})
        source = deepcopy(rows)
        selected, report = balance_frames(rows, True)
        self.assertEqual(report['after']['tasks']['mission_selection']['decisions'],
                         {'start': 3, 'wait': 3, 'unknown': 1})
        self.assertEqual(report['after']['tasks']['patrol_supervision']['decisions'],
                         {'continue': 2, 'pause': 2})
        self.assertEqual(report['after']['tasks']['mission_supervision']['decisions'],
                         {'continue': 3, 'pause': 3, 'cancel': 1})
        self.assertEqual(len(selected), 18)
        self.assertEqual(rows, source)
        self.assertEqual(selected, balance_frames(rows, True)[0])
        patrol_paused = [row for row in rows if row['frame']['context']['reasoning_task'] == 'patrol_supervision'
                         and json.loads(row['sample']['messages'][-1]['content'][0]['text'])['decision'] == 'pause']
        kept, single = balance_frames(patrol_paused, True)
        self.assertEqual(kept, patrol_paused)
        self.assertIn('only one decision', single['warnings'][0])

import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from sam2_engine.instances import select_instances
from sam2_engine.class_model import ClassHead,semantic_target,class_seeds,predict_instances
from sam2_engine.adaptation import encode_rle

class InstanceTests(unittest.TestCase):
    def test_duplicate_suppression_preserves_touching_objects(self):
        a=np.zeros((16,16),bool);a[:,:8]=True;b=~a
        self.assertEqual(len(select_instances([{'mask':a,'score':.9},{'mask':a,'score':.8},{'mask':b,'score':.85}])),2)
    def test_all_classes_have_supervised_targets_and_background(self):
        masks=[np.zeros((8,8),bool) for _ in range(3)]
        for i,mask in enumerate(masks):mask[i*2:i*2+2,:4]=True
        t=semantic_target(list(zip(masks,['dish','empty dish','person'])),['dish','empty dish','person'],(8,8))
        self.assertEqual(set(np.unique(t)),{0,1,2,3})
    def test_overlapping_conflicting_labels_are_ignored(self):
        a=np.ones((8,8),bool)
        self.assertTrue((semantic_target([(a,'a'),(a,'b')],['a','b'],(8,8))==255).all())
    def test_background_only_features_never_prompt_sam(self):
        p=np.zeros((4,16,16));p[0]=1
        self.assertEqual(class_seeds(p),[])
    def test_every_class_can_supply_prompts(self):
        p=np.zeros((4,32,32));p[0]=1
        for k in range(1,4):p[0,k*6:k*6+4,3:9]=0;p[k,k*6:k*6+4,3:9]=1
        self.assertEqual({s[0] for s in class_seeds(p)},{1,2,3})
    def test_class_head_output_includes_background(self):
        head=ClassHead(352,3)
        self.assertEqual(tuple(head(torch.zeros(1,352,16,16)).shape),(1,4,16,16))

    def test_candidate_seed_requires_eighty_percent(self):
        p=np.zeros((2,16,16))
        p[1]=.79;p[0]=.21
        self.assertEqual(class_seeds(p),[])
        p[1]=.8;p[0]=.2
        self.assertTrue(class_seeds(p))

    def test_mask_and_class_scores_each_require_eighty_percent(self):
        predictor=Mock()
        predictor._features={}
        mask=np.ones((16,16),dtype=bool)
        for mask_score,class_score,expected in [(.79,.9,0),(.9,.79,0),(.81,.81,1)]:
            with self.subTest(mask_score=mask_score,class_score=class_score):
                probs=torch.empty(1,2,256,256)
                probs[:,0]=1-class_score;probs[:,1]=class_score
                head=Mock(return_value=probs.log())
                predictor.predict.return_value=(np.array([mask]),np.array([mask_score]),None)
                with patch('sam2_engine.class_model.image_features',return_value=torch.empty(1)),                      patch('sam2_engine.class_model.class_seeds',return_value=[(1,8,8,.9)]):
                    result=predict_instances(predictor,head,['person'],np.zeros((16,16,3)),encode_rle)
                self.assertEqual(len(result),expected)

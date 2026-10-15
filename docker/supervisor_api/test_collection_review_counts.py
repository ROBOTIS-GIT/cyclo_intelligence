import unittest
from .collection_review_counts import generated_review_counts


class GeneratedReviewCountsTests(unittest.TestCase):
    def sample(self, **extra):
        return {'observation': {'frame_id': 'frame', 'context': {}}, **extra}

    def test_qwen_counts_only_generated_unapproved_matching_frames(self):
        automatic={'annotator':'qwen:job','review_status':'draft'}
        samples=[self.sample(annotations=[automatic]),
                 self.sample(annotations=[automatic,{**automatic,'review_status':'approved'}]),
                 self.sample(annotations=[{'annotator':'collect-ui','review_status':'draft'}]),
                 {'observation':{'frame_id':'bad','context':{'runtime_source':'unverified'}},'annotations':[automatic]}]
        self.assertEqual(generated_review_counts({},samples)['qwen_pending_review_count'],1)

    def test_sam_revision_review_and_empty_results(self):
        draft={'object_id':'a','annotator':'sam-auto','review_status':'draft'}
        samples=[self.sample(regions=[draft]),
                 self.sample(regions=[draft,{**draft,'review_status':'approved'}]),
                 self.sample(regions=[{**draft,'annotator':'manual'}]),
                 {'observation':{'frame_id':'empty','auto_labeling':{'sam':{'status':'generated'}}}},
                 {'observation':{'frame_id':'reviewed-empty','auto_labeling':{'sam':{'status':'generated'}},'mask_review':{'status':'empty'}}}]
        self.assertEqual(generated_review_counts({},samples)['sam_pending_review_count'],2)


if __name__ == '__main__':
    unittest.main()

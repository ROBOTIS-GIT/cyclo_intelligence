"""Memory ownership regression tests; use meta tensors without allocating GPU memory."""
import unittest
import torch
from transformers.feature_extraction_utils import BatchFeature
from .train_lora import batch_on_device


class BatchMemoryTests(unittest.TestCase):
    def batch(self):
        return BatchFeature({'pixel_values':torch.ones(8,12),
            'input_ids':torch.tensor([[1,2,3]]), 'labels':torch.tensor([[-100,2,3]])})

    def test_current_step_does_not_move_cached_features_in_place(self):
        cached=self.batch()
        original=dict(cached.items())
        moved=batch_on_device(cached,'meta')
        self.assertIsNot(moved,cached)
        for key,tensor in original.items():
            self.assertIs(cached[key],tensor)
            self.assertEqual(cached[key].device.type,'cpu')
            self.assertEqual(moved[key].device.type,'meta')
            self.assertEqual(moved[key].dtype,tensor.dtype)
            self.assertEqual(moved[key].shape,tensor.shape)
        torch.testing.assert_close(cached['labels'],torch.tensor([[-100,2,3]]))

    def test_cpu_cache_remains_on_cpu_across_two_epochs(self):
        cache=[self.batch() for _ in range(12)]
        for _ in range(2):
            for batch in cache:
                current=batch_on_device(batch,'meta')
                self.assertTrue(all(t.device.type=='meta' for t in current.values()))
                del current
            self.assertTrue(all(t.device.type=='cpu' for batch in cache for t in batch.values()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA verification requires a GPU')
    def test_gpu_memory_does_not_grow_with_number_of_cached_frames(self):
        cache=[BatchFeature({'pixel_values':torch.ones(512,512)}) for _ in range(24)]
        torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            for batch in cache:
                current=batch_on_device(batch,'cuda')
                self.assertEqual(current['pixel_values'].device.type,'cuda')
                del current
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(),baseline)
        self.assertLess(torch.cuda.max_memory_allocated()-baseline,4*1024**2)
        self.assertTrue(all(batch['pixel_values'].device.type=='cpu' for batch in cache))

class MultimodalBatchTests(unittest.TestCase):
    def test_padding_labels_and_image_order_preserve_source_samples(self):
        from .batching import collate_batches
        a={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
           'labels':torch.tensor([[-100,2,3]]),'pixel_values':torch.full((2,4),1.),'image_grid_thw':torch.tensor([[1,1,2]])}
        b={'input_ids':torch.tensor([[4,5]]),'attention_mask':torch.ones(1,2,dtype=torch.long),
           'labels':torch.tensor([[-100,5]]),'pixel_values':torch.full((3,4),2.),'image_grid_thw':torch.tensor([[1,1,3]])}
        batch=collate_batches([a,b,a,b],0)
        torch.testing.assert_close(batch['input_ids'],torch.tensor([[1,2,3],[4,5,0],[1,2,3],[4,5,0]]))
        torch.testing.assert_close(batch['labels'][1],torch.tensor([-100,5,-100]))
        torch.testing.assert_close(batch['attention_mask'][1],torch.tensor([1,1,0]))
        self.assertEqual(batch['pixel_values'].shape,(10,4))
        torch.testing.assert_close(batch['image_grid_thw'],torch.tensor([[1,1,2],[1,1,3],[1,1,2],[1,1,3]]))
        self.assertEqual(b['input_ids'].shape,(1,2));self.assertEqual(b['labels'].shape,(1,2))

    def test_assistant_projection_matches_full_cross_entropy_with_padding(self):
        from .batching import assistant_ce
        from types import SimpleNamespace
        from copy import deepcopy
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(8,5);self.lm_head=torch.nn.Linear(5,8)
            def model(self,input_ids,attention_mask,use_cache=False):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
            def get_base_model(self):return self
        torch.manual_seed(42);model=Tiny();reference=deepcopy(model)
        ids=torch.tensor([[1,2,3,4],[2,3,4,0]]);labels=torch.tensor([[-100,-100,3,4],[-100,3,4,-100]])
        batch={'input_ids':ids,'labels':labels,'attention_mask':torch.tensor([[1,1,1,1],[1,1,1,0]])}
        loss=assistant_ce(model,batch);loss.backward()
        logits=reference.lm_head(reference.embedding(ids))
        expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,8),labels[:,1:].reshape(-1),ignore_index=-100)
        expected.backward();torch.testing.assert_close(loss,expected)
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad)


if __name__=='__main__':unittest.main()

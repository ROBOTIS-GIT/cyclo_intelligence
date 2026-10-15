import unittest, io, base64
from types import SimpleNamespace as S
import torch
from PIL import Image
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from qwen_engine.attention_preview import PreviewAttention

class PreviewTests(unittest.TestCase):
    def test_real_qwen_module_prefill_only_and_cleanup(self):
        config=Qwen3_5TextConfig(hidden_size=32,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        config._attn_implementation='sdpa'
        module=Qwen3_5Attention(config,layer_idx=0).eval()
        model=S(model=S(language_model=S(layers=[S(layer_type='full_attention',self_attn=module)])),
            config=S(image_token_id=99,vision_config=S(spatial_merge_size=2)))
        inputs={'input_ids':torch.tensor([[1,99,99,99,99,2,3,4]]),
                'attention_mask':torch.ones(1,8,dtype=torch.long),'image_grid_thw':torch.tensor([[1,4,4]])}
        capture=PreviewAttention(model,inputs)
        h=torch.randn(1,8,32);pos=(torch.ones(1,8,16),torch.zeros(1,8,16))
        with torch.inference_mode():
            module(hidden_states=h,position_embeddings=pos,attention_mask=None)
            self.assertIsNone(capture.error)
            first=capture.grid.copy()
            module(hidden_states=h[:,:1],position_embeddings=(pos[0][:,:1],pos[1][:,:1]),attention_mask=None)
        self.assertTrue((first==capture.grid).all())
        self.assertAlmostEqual(float(first.sum()),1.,places=5)
        result=capture.render(Image.new('RGB',(320,240)))
        image=Image.open(io.BytesIO(base64.b64decode(result['image_base64'])))
        self.assertEqual(image.size,(320,240))
        capture.remove()
        self.assertFalse(module._forward_pre_hooks)

if __name__=='__main__':unittest.main()

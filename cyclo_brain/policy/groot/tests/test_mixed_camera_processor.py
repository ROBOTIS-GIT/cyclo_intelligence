"""VLM preprocessing must preserve different head/wrist aspect ratios."""
from types import SimpleNamespace
import pytest
import torch
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


@pytest.mark.parametrize('shapes', [[(256,340),(340,256),(340,256)],[(256,256)]*3])
def test_mixed_views_keep_geometry_and_time_then_view_order(shapes):
    keys=['head','left','right']
    frames={key:[torch.full((3,*shape),t*3+i,dtype=torch.uint8) for t in range(2)] for i,(key,shape) in enumerate(zip(keys,shapes))}
    captured=[]
    processor=SimpleNamespace(use_albumentations=False,emit_attn_targets=False,
        _apply_vlm_processing=lambda images,language:captured.extend(images) or {'language':language})
    result=Gr00tN1d7Processor._get_vlm_inputs(processor,keys,frames,None,lambda image:image,'pick')
    assert result['language']=='pick'
    assert [int(image[0,0,0]) for image in captured]==[0,1,2,3,4,5]
    assert [tuple(image.shape[-2:]) for image in captured]==shapes+shapes


def test_vlm_processing_accepts_per_view_tensors_without_resizing():
    head=torch.zeros(3,256,340,dtype=torch.uint8)
    wrist=torch.zeros(3,340,256,dtype=torch.uint8)
    processor=SimpleNamespace(processor=SimpleNamespace(apply_chat_template=lambda *_a,**_kw:'chat'))
    result=Gr00tN1d7Processor._apply_vlm_processing(processor,[head,wrist],'pick')
    assert [tuple(image.shape) for image in result['vlm_content']['images']]==[(3,256,340),(3,340,256)]

"""Exercise the pinned upstream loss, with tiny deterministic feature modules."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ABot-Manipulation"))
pytest.importorskip("diffusers")
from ABot.model.modules.action_model.AML_ActionHeader import FlowmatchingActionHead


def test_upstream_mask_excludes_padded_output_loss_and_gradient():
    class Encoder(torch.nn.Module):
        def forward(self, actions, timestep):
            return torch.zeros_like(actions)

    class Identity(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([1., 2., 100., 200.]))

        def forward(self, x):
            return x + self.bias

    head = FlowmatchingActionHead.__new__(FlowmatchingActionHead)
    torch.nn.Module.__init__(head)
    head.action_encoder = Encoder()
    head.action_decoder = Decoder()
    head.model = Identity()
    head.future_tokens = torch.nn.Embedding(1, 4)
    head.config = SimpleNamespace(add_pos_embed=False)
    head.t_eps, head.num_timestep_buckets = .05, 1000
    head.sample_time = lambda batch, device, dtype: torch.full((batch,), .5, device=device, dtype=dtype)
    args = (torch.zeros(2, 1, 4), torch.zeros(2, 3, 4))
    mask = torch.tensor([[True, True, False, False]] * 2)
    torch.manual_seed(0)
    loss = head(*args, action_mask=mask)
    assert loss.item() == pytest.approx(10)
    loss.backward()
    torch.testing.assert_close(head.action_decoder.bias.grad, torch.tensor([4., 8., 0., 0.]))
    with torch.no_grad():
        head.action_decoder.bias[2:] *= 10
    torch.manual_seed(0)
    assert head(*args, action_mask=mask).item() == pytest.approx(loss.item())
    assert head(*args).item() > loss.item()

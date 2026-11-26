"""Independent PyTorch operations with FP64 intermediates and FP32 activation boundaries."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.llama import modeling_llama as llama
from transformers.cache_utils import DynamicLayer


def rms_norm(self, hidden):
    x = hidden.double()
    epsilon = torch.tensor(self.variance_epsilon, dtype=torch.float32).item()
    scale = 1 / (x.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
    return (x * scale * self.weight.double()).float()


def rotary_embedding(self, _hidden, positions):
    count = self.inv_freq.numel()
    theta = torch.tensor(self.config.rope_theta, dtype=torch.float32).double()
    frequencies = theta.pow(-torch.arange(count, dtype=torch.float64) / count).float()
    angles = positions[..., None].double() * frequencies.double()
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def rotary_position(query, key, cosine, sine, unsqueeze_dim=1):
    cosine = cosine.unsqueeze(unsqueeze_dim)
    sine = sine.unsqueeze(unsqueeze_dim)
    return tuple((value.double() * cosine + llama.rotate_half(value).double() * sine).float()
                 for value in (query, key))


def attention(module, query, key, value, attention_mask, scaling, **_kwargs):
    key = llama.repeat_kv(key, module.num_key_value_groups)
    value = llama.repeat_kv(value, module.num_key_value_groups)
    scores = (query.double() @ key.double().transpose(-1, -2) * scaling).float().double()
    if attention_mask is not None:
        scores = scores + attention_mask[..., :key.size(-2)].double()
    exponentials = (scores - scores.amax(dim=-1, keepdim=True)).exp().float().double()
    probabilities = (exponentials / exponentials.sum(dim=-1, keepdim=True)).float()
    output = (probabilities.double() @ value.double()).float()
    return output.transpose(1, 2).contiguous(), probabilities


def mlp(self, hidden):
    gate = self.gate_proj(hidden).double()
    up = self.up_proj(hidden).double()
    return self.down_proj((F.silu(gate) * up).float())


@contextmanager
def rounded_operations(model, round_cache=False):
    # The Transformers model still owns the layer order, residuals, masking and cache.
    # Only the evaluation precision within each operation is configured here.
    with torch.inference_mode():
        original_linear = F.linear
        matrices = {id(weight): weight.double() for weight in model.parameters() if weight.ndim == 2}

        def linear(x, weight, bias=None):
            return original_linear(x.double(), matrices[id(weight)], None if bias is None else bias.double()).float()

        with ExitStack() as stack:
            stack.enter_context(patch.object(F, "linear", linear))
            stack.enter_context(patch.object(llama.LlamaRMSNorm, "forward", rms_norm))
            stack.enter_context(patch.object(llama.LlamaRotaryEmbedding, "forward", rotary_embedding))
            stack.enter_context(patch.object(llama, "apply_rotary_pos_emb", rotary_position))
            stack.enter_context(patch.object(llama, "eager_attention_forward", attention))
            stack.enter_context(patch.object(llama.LlamaMLP, "forward", mlp))
            if round_cache:
                update = DynamicLayer.update

                def rounded_update(layer, key, value, *args, **kwargs):
                    return update(layer, key.half().float(), value.half().float(), *args, **kwargs)

                stack.enter_context(patch.object(DynamicLayer, "update", rounded_update))
            yield

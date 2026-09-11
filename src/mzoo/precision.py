"""Keep numerically sensitive modules in fp32 under bf16 autocast.

Already fp32 in HF + autocast: RMSNorm (upcasts internally), RoPE cos/sin, softmax and
cross-entropy (autocast fp32 ops; the loss upcasts logits). Routers need it explicitly.
"""

import torch


def _fp32_forward(module):
    forward = module.forward

    def wrapped(hidden_states, *args, **kwargs):
        with torch.autocast(hidden_states.device.type, enabled=False):
            return forward(hidden_states.float(), *args, **kwargs)

    module.forward = wrapped


def keep_fp32(model, suffixes=("Router",)):
    for m in model.modules():
        if type(m).__name__.endswith(suffixes):
            _fp32_forward(m)
    return model

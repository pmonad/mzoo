"""Looped dense (Nanbeige4.2 LoopSplit): head layers once, middle block `loops` times, tail once.

Repeated entries are the same modules, so weights are shared. Train/eval only: shared layers
also share a KV-cache slot, so cached generation is not supported.
"""

from torch import nn

from mzoo.archs.dense import build as build_dense


def build(tok, seq_len, layers=9, middle=3, loops=3, **dense_kw):
    head = (layers - middle) // 2
    order = [*range(head), *list(range(head, head + middle)) * loops, *range(head + middle, layers)]
    model = build_dense(tok, seq_len, layers=layers, **dense_kw)
    model.model.layers = nn.ModuleList(model.model.layers[i] for i in order)
    model.config.num_hidden_layers = len(order)  # Llama's forward iterates this many layers
    model.config.loop_order = order
    model.config.use_cache = False
    return model

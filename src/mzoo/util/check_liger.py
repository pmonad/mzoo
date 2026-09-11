"""Check an arch's loss uses Liger fused linear CE (no logits) and matches full-logits fp32 CE.

    uv run python -m mzoo.util.check_liger
    uv run python -m mzoo.util.check_liger --arch=nanbeige --layers=6 --loop_middle_layers=2
"""

import importlib

import fire
import torch
import torch.nn.functional as F

from mzoo.data_utils import data


def main(arch="nanbeige", batch=4, **kw):
    ds, tok = data.load(data.DATA)
    seq_len = len(ds["train"][0]["input_ids"])
    model = importlib.import_module(f"mzoo.archs.{arch}").build(tok, seq_len, **kw).cuda()
    x = data.collate([ds["train"][i] for i in range(batch)])["input_ids"].cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x, labels=x)
    out.loss.backward()
    with torch.no_grad():
        logits = model.lm_head(model.model(input_ids=x)[0])[:, :-1].float()
        ref = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    print(f"fused linear CE: {out.logits is None}, loss={out.loss.item():.4f}, full-logits fp32 CE={ref.item():.4f}")


if __name__ == "__main__":
    fire.Fire(main)

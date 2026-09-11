"""Pretokenize an HF text dataset into packed fixed-length blocks, saved with its tokenizer.

    just src/mzoo/ data                        # TinyStories, ~100M train tokens, seq_len 512
    just src/mzoo/ data --tokens=10_000_000 --out=/tmp/ts10m
    just src/mzoo/ data --dataset=codelion/fineweb-edu-1B --tokens=None --seq_len=1024 \
        --out=/data/pmonad/mzoo/datasets/fineweb-edu-llama32k-1B-1024
"""

from itertools import chain

import fire
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

DATA = "/data/pmonad/mzoo/datasets/tinystories-llama32k-100M-512"


def packed(ds, tok, seq_len, num_proc):
    def tokenize(b):
        return {"ids": tok([t + tok.eos_token for t in b["text"]])["input_ids"]}

    def group(b):
        ids = list(chain.from_iterable(b["ids"]))
        return {"input_ids": [ids[i : i + seq_len] for i in range(0, len(ids) // seq_len * seq_len, seq_len)]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=num_proc)
    return ds.map(group, batched=True, remove_columns=ds.column_names, num_proc=num_proc)


def prepare(
    dataset="roneneldan/TinyStories",
    tokens=100_000_000,  # None: use the whole train split
    seq_len=512,
    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    out=DATA,
    val_docs=2000,  # held out from train if the dataset has no validation split
    num_proc=16,
):
    tok = AutoTokenizer.from_pretrained(tokenizer)
    raw = load_dataset(dataset)
    if "validation" not in raw:
        raw = raw["train"].train_test_split(test_size=val_docs, seed=0)
        raw["validation"] = raw.pop("test")
    train = raw["train"]
    if tokens:
        per_doc = sum(map(len, tok(train[:1000]["text"])["input_ids"])) / 1000 + 1
        # overshoot: estimate is from a sample and packing drops remainders; extra blocks are cut below
        train = train.select(range(min(len(train), int(tokens / per_doc * 1.2) + 1000)))
    train = packed(train, tok, seq_len, num_proc)
    if tokens:
        blocks = tokens // seq_len
        assert len(train) >= blocks, f"only {len(train)} blocks, need {blocks}"
        train = train.select(range(blocks))
    ds = DatasetDict(train=train, validation=packed(raw["validation"], tok, seq_len, num_proc))
    ds.save_to_disk(out)
    tok.save_pretrained(f"{out}/tokenizer")
    print(ds, f"saved to {out}")


def load(path=DATA):
    """Returns (DatasetDict with torch-formatted input_ids, tokenizer)."""
    from datasets import load_from_disk

    return load_from_disk(path).with_format("torch"), AutoTokenizer.from_pretrained(f"{path}/tokenizer")


def collate(batch):
    ids = torch.stack([b["input_ids"] for b in batch])
    return {"input_ids": ids, "labels": ids}


if __name__ == "__main__":
    fire.Fire(prepare)

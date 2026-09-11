"""Minimal pretraining on pretokenized TinyStories (see data.py) with HF Trainer.

    just src/mzoo/ pt --exp=baseline --run=bs32          # dense arch, 100 steps, wandb
    just src/mzoo/ pt --layers=12 --learning_rate=3e-4   # TrainingArguments flags go to
                                                         # the Trainer, the rest to the arch
    just src/mzoo/ pt --exp=260911-0001-baseline         # add a run to an existing exp
"""

import dataclasses
import importlib
import os

import fire
from transformers import TrainingArguments

from mzoo.data_utils import data
from mzoo.train.paths import run_dir
from mzoo.train.precision import keep_fp32
from mzoo.train.trainer import TimedCheckpoint, Trainer

TRAIN_FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}


def main(
    arch="dense", proj=None, exp="scratch", run=None, data_path=data.DATA, eval_samples=256, ckpt_minutes=None, **kwargs
):
    os.environ.setdefault("WANDB_PROJECT", "mzoo")
    train_kw = {k: v for k, v in kwargs.items() if k in TRAIN_FIELDS}
    arch_kw = {k: v for k, v in kwargs.items() if k not in TRAIN_FIELDS}

    ds, tok = data.load(data_path)
    seq_len = len(ds["train"][0]["input_ids"])
    model = importlib.import_module(f"mzoo.archs.{arch}").build(tok, seq_len, **arch_kw)
    keep_fp32(model)
    print(f"params: {model.num_parameters() / 1e6:.1f}M")

    if "output_dir" not in train_kw:
        run = run or "-".join([arch, *(f"{k}{v}" for k, v in arch_kw.items())])
        out = run_dir(proj or arch, exp, run)
        os.environ["WANDB_DIR"] = str(out)
        train_kw = dict(output_dir=str(out), run_name="/".join(out.parts[-3:]), **train_kw)
        print(f"output_dir: {out}")

    args = dict(
        max_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=6e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.1,
        bf16=True,  # autocast; weights and optimizer states stay fp32
        dataloader_num_workers=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",  # TimedCheckpoint saves at the end (and every ckpt_minutes if set)
        report_to="wandb",
        disable_tqdm=True,
    )
    args.update(train_kw)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"].select(range(eval_samples)),
        data_collator=data.collate,
        callbacks=[TimedCheckpoint(ckpt_minutes)],
    )
    return trainer.train().metrics


if __name__ == "__main__":
    fire.Fire(main)

import math

import pytest

from mzoo.train import pt

CASES = {
    "dense": ("dense", dict(layers=2, hidden=64, heads=2, kv_heads=2, ffn=128)),
    "nanbeige": (
        "nanbeige",
        dict(layers=3, loop_middle_layers=1, loop_middle_repeats=2, hidden=64, heads=2, kv_heads=2, ffn=128),
    ),
    "nanbeige-moe": (
        "nanbeige",
        dict(
            layers=3,
            loop_middle_layers=1,
            loop_middle_repeats=2,
            hidden=64,
            heads=2,
            kv_heads=2,
            ffn=128,
            dense_layers=1,
            moe=True,
            n_routed_experts=4,
            num_experts_per_tok=2,
        ),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_main_smoke(tmp_path, tiny_data_path, case):
    arch, arch_kwargs = CASES[case]
    metrics = pt.main(
        arch=arch,
        data_path=tiny_data_path,
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_samples=4,
        eval_steps=2,
        logging_steps=1,
        dataloader_num_workers=0,
        report_to="none",
        output_dir=str(tmp_path),
        save_strategy="steps",
        save_steps=2,
        **arch_kwargs,
    )
    assert math.isfinite(metrics["train_loss"])
    assert (tmp_path / "checkpoint-2" / "model.safetensors").exists()

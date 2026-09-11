import math

from mzoo import pt


def test_main_smoke(tmp_path, tiny_data_path):
    metrics = pt.main(
        data_path=tiny_data_path,
        layers=2,
        hidden=64,
        heads=2,
        kv_heads=2,
        ffn=128,
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_samples=4,
        eval_steps=2,
        logging_steps=1,
        dataloader_num_workers=0,
        report_to="none",
        output_dir=str(tmp_path),
    )
    assert math.isfinite(metrics["train_loss"])

import torch

from mzoo.data_utils import data


def test_load_train_and_validation(tiny_data_path):
    ds, tok = data.load(tiny_data_path)
    assert len(ds["train"]) == 20_000 // 64
    assert ds["train"][0]["input_ids"].shape == (64,)
    assert "validation" in ds
    assert len(tok) == 32000


def test_collate(tiny_data_path):
    ds, _ = data.load(tiny_data_path)
    batch = data.collate([ds["train"][i] for i in range(2)])
    assert torch.equal(batch["input_ids"], batch["labels"])
    assert batch["input_ids"].shape == (2, 64)

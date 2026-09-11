import pytest

from mzoo import data


@pytest.fixture(scope="module")
def tiny_data_path(tmp_path_factory):
    """Tiny pretokenized TinyStories dataset for fast tests (see data.prepare)."""
    out = tmp_path_factory.mktemp("tiny_data")
    data.prepare(tokens=20_000, seq_len=64, out=str(out), num_proc=2)
    return str(out)

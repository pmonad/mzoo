import pytest
import torch

from mzoo.kernels.tilelang_hello import vec_add


@pytest.mark.parametrize("n", [1, 1000, 1024, 12345])
def test_vec_add_matches_torch(n):
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    torch.testing.assert_close(vec_add(a, b), a + b)

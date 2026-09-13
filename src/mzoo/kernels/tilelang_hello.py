import tilelang
import tilelang.language as T
import torch

THREADS = 256  # CUDA blockDim.x


@tilelang.jit
def _vec_add(A: T.Tensor, B: T.Tensor):
    N = T.dynamic("N")
    A: T.Tensor[[N], T.float32]
    B: T.Tensor[[N], T.float32]
    C = T.empty((N,), T.float32)
    with T.Kernel(T.ceildiv(N, THREADS), threads=THREADS) as bx:
        i = bx * THREADS + T.get_thread_binding(0)
        if i < N:
            C[i] = A[i] + B[i]
    return C


def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _vec_add(a, b)

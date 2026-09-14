"""QK prologue: RMSNorm then RoPE as one fused TileLang pass (ticket 0011).

Done eagerly the norm and the rotation are two full read+write round trips over
q — and the first attempt with `torch.compile` measured 0.96 ms at B1 S4096
H64 D128 (67 MB q, copy ceiling 0.58 ms / 232 GB/s): Inductor *must* split the
rstd row reduction into its own kernel that re-reads q, plus a second pointwise
for the stack->cat interleave, so its floor is 3 launches / ~200 MB traffic.
One TileLang kernel keeps the D-row in shared memory, computes rstd in-kernel,
rotates, and writes q once: 1 launch, 134 MB, measured in `norm_rope_bench`.

Semantics match exactly the vendored `dsv4` composition of
`DeepseekV41RMSNorm.forward` (fp32 norm, `weight * x.to(dtype)`) followed by
`apply_rotary_pos_emb` (interleaved pairs on the trailing rope slice, leading
nope channels pass through); `qk_norm_rope_eager` is that composition in plain
torch and the test's reference. Nothing is imported from the modeling file.

Backward is a custom `autograd.Function` (Liger-style): the forward saves only
`rstd` (one fp32 per row) plus its inputs, and the backward kernel recomputes
the normed values, so saved memory is ~1/8 of an fp32 activation copy.
"""

import torch
import tilelang
import tilelang.language as T

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)

def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)  # [B, S, 1, rd/2]
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)

def qk_norm_rope_eager(x, weight, cos, sin, *, eps: float = 1e-6) -> torch.Tensor:
    """Reference composition (also what the test compares against)."""
    return _rope(_rms_norm(x, weight, eps), cos, sin)


# rows = B*S*H packed as [rows, D] (BSHD memory order), token of row r is r // H.
# bf16 in/out; fp32 norm/rotation, matching the reference's dtype choreography
# (bf16 round after `x*rstd*w`, rotation on the upcast values).
CONFIGS = {dim: dict(block_M=128, threads=256) for dim in (64, 96, 128, 256)}
CONFIGS[256] = dict(block_M=64, threads=256)  # smem: three [block_M, dim] bf16 tiles + cos/sin


@tilelang.jit(out_idx=[4, 5], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def make_fwd(rows, heads, dim, rope_dim, eps=1e-6, block_M=128, threads=256):
    nope = dim - rope_dim
    tok_shape = [rows // heads, rope_dim // 2]

    @T.prim_func
    def main(
        X: T.Tensor([rows, dim], T.bfloat16), W: T.Tensor([dim], T.bfloat16),
        Cos: T.Tensor(tok_shape, T.float32), Sin: T.Tensor(tok_shape, T.float32),
        Out: T.Tensor([rows, dim], T.bfloat16), Rstd: T.Tensor([rows], T.float32),
    ):
        with T.Kernel(T.ceildiv(rows, block_M), threads=threads) as (bx):
            dpad = 1 << (dim - 1).bit_length()  # tilelang 0.1.14: no layout for non-pow2 fragment dims
            Xs = T.alloc_shared([block_M, dim], T.bfloat16)
            Xn = T.alloc_shared([block_M, dim], T.bfloat16)  # normed (bf16, as the reference rounds)
            Os = T.alloc_shared([block_M, dim], T.bfloat16)
            Cs = T.alloc_shared([block_M, rope_dim // 2], T.float32)
            Ss = T.alloc_shared([block_M, rope_dim // 2], T.float32)
            Xf = T.alloc_fragment([block_M, dpad], T.float32)
            ss = T.alloc_fragment([block_M], T.float32)
            rstd = T.alloc_fragment([block_M], T.float32)
            tok = T.alloc_fragment([block_M], T.int32)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < rows:
                    Xs[i, d] = X[bx * block_M + i, d]
            for i, d in T.Parallel(block_M, dpad):
                if d < dim:
                    v = T.cast(Xs[i, d], T.float32)
                    Xf[i, d] = v * v
                else:
                    Xf[i, d] = 0.0
            T.reduce_sum(Xf, ss, dim=1)  # last-dim reduce only; dim=0 miscompiles (0.1.14)
            for i in T.Parallel(block_M):
                rstd[i] = T.rsqrt(ss[i] / dim + eps)
                tok[i] = (bx * block_M + i) // heads
                if bx * block_M + i < rows:
                    Rstd[bx * block_M + i] = rstd[i]
            for i, d in T.Parallel(block_M, dim):  # w * bf16(x * rstd), staged for the pair loop
                xf = T.cast(Xs[i, d], T.float32)
                xn = T.cast(xf * rstd[i], T.bfloat16)
                Xn[i, d] = T.cast(T.cast(W[d], T.float32) * T.cast(xn, T.float32), T.bfloat16)
            for i, j in T.Parallel(block_M, rope_dim // 2):
                Cs[i, j] = Cos[tok[i], j]
                Ss[i, j] = Sin[tok[i], j]
            for i, k in T.Parallel(block_M, dim // 2):
                r = bx * block_M + i
                if r < rows:
                    if k < nope // 2:
                        Os[i, 2 * k] = Xn[i, 2 * k]
                        Os[i, 2 * k + 1] = Xn[i, 2 * k + 1]
                    else:
                        j = k - nope // 2
                        c, s = Cs[i, j], Ss[i, j]
                        e, o = T.cast(Xn[i, 2 * k], T.float32), T.cast(Xn[i, 2 * k + 1], T.float32)
                        Os[i, 2 * k] = T.cast(e * c - o * s, T.bfloat16)
                        Os[i, 2 * k + 1] = T.cast(e * s + o * c, T.bfloat16)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < rows:
                    Out[bx * block_M + i, d] = Os[i, d]

    return main


@tilelang.jit(out_idx=[7], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def make_bwd(rows, heads, dim, rope_dim, eps=1e-6, block_M=128, threads=256):
    """dL/dx and dL/dw for `L(rope(w * x * rstd(x)))`, rstd saved by the forward.

    dy is inverse-rotated to `dxn` (the conjugate rotation; nope channels pass
    through), then the standard RMSNorm backward with `g = w * dxn`:
    `dx = rstd * (g - x * rstd^2 * mean(g*x))`, `dw[d] = sum_rows dxn * x * rstd`.
    bf16 casts are treated as identity (what eager autograd does).
    """
    nope = dim - rope_dim
    tok_shape = [rows // heads, rope_dim // 2]

    @T.prim_func
    def main(
        X: T.Tensor([rows, dim], T.bfloat16), DY: T.Tensor([rows, dim], T.bfloat16),
        W: T.Tensor([dim], T.bfloat16), Rstd: T.Tensor([rows], T.float32),
        Cos: T.Tensor(tok_shape, T.float32), Sin: T.Tensor(tok_shape, T.float32),
        DW: T.Tensor([dim], T.float32),  # input: atomically accumulated, pre-zeroed by the caller
        DX: T.Tensor([rows, dim], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(rows, block_M), threads=threads) as (bx):
            DYs = T.alloc_shared([block_M, dim], T.bfloat16)
            # dw partials live transposed: tilelang 0.1.14 miscompiles
            # `reduce_sum(..., dim=0)`, only the last-dim reduce (as in FA2) is sound.
            GD = T.alloc_fragment([dim, block_M], T.float32)
            dot = T.alloc_fragment([block_M], T.float32)
            dwb = T.alloc_fragment([dim], T.float32)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < rows:
                    DYs[i, d] = DY[bx * block_M + i, d]
                else:
                    DYs[i, d] = 0.0
            for i in T.Parallel(block_M):
                dot[i] = 0.0
            # dot[i] = sum_d g * x, g recomputed from DYs on each use (shared hit)
            for d in T.serial(dim):
                for i in T.Parallel(block_M):
                    r = bx * block_M + i
                    if r < rows:
                        dyf = T.cast(DYs[i, d], T.float32)
                        j = T.if_then_else(d < nope, 0, (d - nope) // 2)  # clamped: dead loads
                        p = T.if_then_else((d - nope) % 2 == 0, d + 1, d - 1)
                        dyp = T.cast(DYs[i, p], T.float32)
                        even = (d - nope) % 2 == 0
                        rot = T.if_then_else(even, dyf * Cos[r // heads, j] + dyp * Sin[r // heads, j],
                                             dyf * Cos[r // heads, j] - dyp * Sin[r // heads, j])
                        g = T.cast(W[d], T.float32) * T.if_then_else(d < nope, dyf, rot)
                        dot[i] += g * T.cast(X[r, d], T.float32)
            for i, d in T.Parallel(block_M, dim):
                r = bx * block_M + i
                if r < rows:
                    dyf = T.cast(DYs[i, d], T.float32)
                    j = T.if_then_else(d < nope, 0, (d - nope) // 2)
                    p = T.if_then_else((d - nope) % 2 == 0, d + 1, d - 1)
                    dyp = T.cast(DYs[i, p], T.float32)
                    even = (d - nope) % 2 == 0
                    rot = T.if_then_else(even, dyf * Cos[r // heads, j] + dyp * Sin[r // heads, j],
                                         dyf * Cos[r // heads, j] - dyp * Sin[r // heads, j])
                    g = T.cast(W[d], T.float32) * T.if_then_else(d < nope, dyf, rot)
                    xf = T.cast(X[r, d], T.float32)
                    rs = Rstd[r]
                    DX[r, d] = T.cast(rs * (g - xf * rs * rs * dot[i] / dim), T.bfloat16)
                    # dw[d] = sum_rows dxn * x * rstd -- NO w factor (d(w*h)/dw = h)
                    GD[d, i] = T.if_then_else(d < nope, dyf, rot) * xf * rs
                else:
                    GD[d, i] = 0.0
            T.reduce_sum(GD, dwb, dim=1)
            if bx * block_M < rows:
                for d in T.Parallel(dim):
                    T.atomic_add(DW[d], dwb[d])

    return main


_CACHE: dict = {}  # (kind, rows, heads, dim, rd) -> compiled kernel

def _kernel(kind, rows, heads, dim, rope_dim, eps):
    key = (kind, rows, heads, dim, rope_dim, eps)
    if key not in _CACHE:
        maker = make_fwd if kind == "fwd" else make_bwd
        _CACHE[key] = maker(rows, heads, dim, rope_dim, eps=eps, **CONFIGS[dim])
    return _CACHE[key]

class _QKNormRoPE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):
        b, s, h, d = x.shape
        rows = b * s * h
        x2 = x.contiguous().view(rows, d)
        out, rstd = _kernel("fwd", rows, h, d, cos.shape[-1] * 2, eps)(
            x2, weight.contiguous(), cos.reshape(-1, cos.shape[-1]).contiguous(),
            sin.reshape(-1, sin.shape[-1]).contiguous())
        ctx.save_for_backward(x2, weight, cos, sin, rstd)
        ctx.meta = (h, eps)
        return out.view(b, s, h, d)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, cos, sin, rstd = ctx.saved_tensors
        h, eps = ctx.meta
        rows, d = x2.shape
        dw = torch.zeros(d, device=x2.device, dtype=torch.float32)
        dx = _kernel("bwd", rows, h, d, cos.shape[-1] * 2, eps)(
            x2, dy.contiguous().view(rows, d), weight, rstd,
            cos.reshape(-1, cos.shape[-1]).contiguous(), sin.reshape(-1, sin.shape[-1]).contiguous(), dw)
        return dx.view(*dy.shape), dw.to(weight.dtype), None, None, None


def qk_norm_rope(x, weight, cos, sin, *, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm->RoPE with backward (TileLang). Same contract as `qk_norm_rope_eager`."""
    return _QKNormRoPE.apply(x, weight, cos, sin, eps)

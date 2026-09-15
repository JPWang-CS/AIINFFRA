"""FP32 block-scale teaching kernel. Not packed INT4 dequantization."""
import torch
import triton
import triton.language as tl


@triton.jit
def block_scale_kernel(X, S, Y, M: tl.constexpr, N: tl.constexpr,
                       TILE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < M * N
    row, col = i // N, i % N
    scale_cols = tl.cdiv(N, TILE)
    scale_offset = (row // TILE) * scale_cols + col // TILE
    x = tl.load(X + i, mask=valid, other=0.0)
    scale = tl.load(S + scale_offset, mask=valid, other=0.0)
    tl.store(Y + i, x * scale, mask=valid)


def block_scale(x, scales, tile=128):
    if x.ndim != 2 or scales.ndim != 2 or not isinstance(tile, int) or tile <= 0:
        raise ValueError("rank-2 tensors and positive tile required")
    m, n = x.shape
    if min(m,n) <= 0 or scales.shape != (triton.cdiv(m,tile), triton.cdiv(n,tile)):
        raise ValueError("scale shape must follow the logical quantization tile")
    for t in (x, scales):
        if t.dtype != torch.float32 or not t.is_cuda or t.device != x.device or not t.is_contiguous():
            raise ValueError("contiguous FP32 tensors on one CUDA device required")
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        block_scale_kernel[(triton.cdiv(m*n,256),)](x,scales,out,m,n,tile,256)
    return out

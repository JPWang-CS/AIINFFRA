# B1 Triton Vector Add v1 —— 1D element-wise；关键点：program_id + arange + mask 尾块
# 说明：无 GPU 时自动切 TRITON_INTERPRET=1 走 CPU 解释器（只验正确性，不代表 GPU 性能）

import os
import time

import torch
import triton
import triton.language as tl

# 没有 CUDA 就用 CPU 解释器跑，保证本机/服务器都能直接执行
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program（block）负责一段连续元素
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 尾块越界保护：越界位置不参与计算
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """包装函数：分配输出、算 grid、启动 kernel。"""
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


def _verify(n: int, block_size: int = 256) -> float:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda" if torch.cuda.is_available() else "cpu")
    y = torch.randn(n, device=x.device)
    out = vector_add(x, y, block_size)
    ref = x + y
    max_err = (out - ref).abs().max().item()
    assert torch.equal(out, ref), f"N={n}: not equal, max_err={max_err:.3e}"
    return max_err


def main():
    # 正确性：覆盖 1 个元素、非整块、整块、大数组
    for n in [1, 1000, 1 << 16, 1 << 20]:
        max_err = _verify(n)
        print(f"N={n:>9}: max_abs_err={max_err:.3e}  PASS")

    # 性能（本机为 CPU 解释器，数字只作参考；GPU 上跑才代表真实性能）
    n = 1 << 20
    x = torch.randn(n, device="cpu")
    y = torch.randn(n, device="cpu")
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        vector_add(x, y)
    dt = (time.perf_counter() - t0) / iters
    bytes_moved = 3 * n * 4  # 读 x + 读 y + 写 out
    print(f"N={n}: {dt*1e3:.1f} ms/iter | CPU 解释器模拟，不做性能结论")
    print(f"       等效带宽口径（3*N*4B）: {bytes_moved/dt/1e6:.1f} MB/s (仅模拟参考)")


if __name__ == "__main__":
    main()

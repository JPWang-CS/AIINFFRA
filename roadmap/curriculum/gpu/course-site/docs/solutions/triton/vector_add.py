# B1 Triton Vector Add v1 —— 本地验证/benchmark wrapper；关键点：program_id + arange + mask 尾块
# 注意：本文件不是单独归档的 LeetGPU 原始 solve；平台代码归档状态见 Lesson 06。
# 说明：无 GPU 时自动切 TRITON_INTERPRET=1 走 CPU 解释器（只验正确性，不代表 GPU 性能）

import os

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
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    return max_err


def main():
    # 正确性：覆盖 1 个元素、非整块、整块、大数组
    for n in [1, 256, 257, 1000, 1 << 20]:
        max_err = _verify(n)
        print(f"N={n:>9}: max_abs_err={max_err:.3e}  PASS")

    # 本机无 GPU 时只做正确性验证，不伪造性能数字
    if not torch.cuda.is_available():
        print("GPU benchmark skipped: CUDA is not available")
        return

    # 真实性能：使用 GPU tensor，并由 do_bench 负责预热与同步计时
    gpu_name = torch.cuda.get_device_name(0)
    n = 1 << 25
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    vector_add(x, y)  # 首次 JIT 编译预热

    triton_ms = triton.testing.do_bench(lambda: vector_add(x, y))
    torch_ms = triton.testing.do_bench(lambda: torch.add(x, y))
    bytes_moved = 3 * n * 4  # 读 x + 读 y + 写 out
    triton_gbps = bytes_moved / (triton_ms / 1000) / 1e9
    torch_gbps = bytes_moved / (torch_ms / 1000) / 1e9
    print(f"GPU: {gpu_name}")
    print(f"N={n}: Triton {triton_ms:.3f} ms | {triton_gbps:.1f} GB/s")
    print(f"N={n}: torch.add {torch_ms:.3f} ms | {torch_gbps:.1f} GB/s")


if __name__ == "__main__":
    main()

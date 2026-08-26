"""Triton MatMul: LeetGPU 草稿的服务器验证版。

来源：用户 LeetGPU 编辑器草稿的本地适配；LeetGPU 当前无法运行，
因此本文件只能记录服务器正确性/性能，不能标记为 LEETGPU_PASS。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    ptr_c = c + offset_m[:, None] * K + offset_k[None, :]
    mask_c = (
        (offset_m[:, None] < M)
        & (offset_k[None, :] < K)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # A[M, N] @ B[N, K] = C[M, K]，N 是归约维度。
    for n in range(0, N, BLOCK_N):
        offset_n = n + tl.arange(0, BLOCK_N)

        ptr_a = a + offset_m[:, None] * N + offset_n[None, :]
        ptr_b = b + offset_n[:, None] * K + offset_k[None, :]

        mask_a = (
            (offset_m[:, None] < M)
            & (offset_n[None, :] < N)
        )
        mask_b = (
            (offset_n[:, None] < N)
            & (offset_k[None, :] < K)
        )

        tile_a = tl.load(ptr_a, mask=mask_a, other=0.0)
        tile_b = tl.load(ptr_b, mask=mask_b, other=0.0)
        acc += tl.dot(tile_a, tile_b)

    tl.store(ptr_c, acc, mask=mask_c)


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
):
    """LeetGPU-compatible entry point: write c = a @ b in-place."""
    block_m = 64
    block_n = 32
    block_k = 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))

    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )


def _check_correctness() -> None:
    """Check irregular shapes so every load/store mask is exercised."""
    cases = [
        (1, 1, 1),
        (64, 32, 64),
        (65, 33, 67),
        (257, 513, 129),
    ]

    for M, N, K in cases:
        a = torch.randn((M, N), device="cuda", dtype=torch.float32)
        b = torch.randn((N, K), device="cuda", dtype=torch.float32)
        out = torch.empty((M, K), device="cuda", dtype=torch.float32)
        solve(a, b, out, M, N, K)
        torch.cuda.synchronize()

        expected = torch.matmul(a, b)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        print(f"correctness M={M}, N={N}, K={K}: OK")


def _time_ms(fn, warmup: int = 10, repeats: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _benchmark() -> None:
    """Benchmark the LeetGPU performance shape and compare with torch.mm."""
    M, N, K = 8192, 6144, 4096
    a = torch.randn((M, N), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    out_triton = torch.empty((M, K), device="cuda", dtype=torch.float32)
    out_torch = torch.empty_like(out_triton)

    triton_ms = _time_ms(lambda: solve(a, b, out_triton, M, N, K))
    torch_ms = _time_ms(lambda: torch.mm(a, b, out=out_torch))
    flops = 2 * M * N * K

    def gflops(ms: float) -> float:
        return flops / (ms * 1e-3) / 1e9

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={M}, N={N}, K={K}")
    print(f"Triton: {triton_ms:.3f} ms, {gflops(triton_ms):.1f} GFLOPS")
    print(f"torch.mm: {torch_ms:.3f} ms, {gflops(torch_ms):.1f} GFLOPS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("需要在有 NVIDIA GPU 的服务器上运行此验证脚本。")

    _check_correctness()
    _benchmark()

"""Triton MatMul: LeetGPU 草稿的服务器验证版。

来源：用户 LeetGPU 编辑器草稿的本地适配；LeetGPU 当前无法运行，
因此本文件只能记录服务器正确性/性能，不能标记为 LEETGPU_PASS。
"""

import argparse

import torch
import triton
import triton.language as tl


# 保持 IEEE FP32 不变，只比较 tile、warp 和 pipeline stage 的影响。
BENCHMARK_CONFIGS = (
    ("baseline-64x32x64-w4-s3", dict(block_m=64, block_n=32, block_k=64, num_warps=4, num_stages=3)),
    ("m128-128x32x64-w4-s3", dict(block_m=128, block_n=32, block_k=64, num_warps=4, num_stages=3)),
    ("mn128-128x32x128-w4-s3", dict(block_m=128, block_n=32, block_k=128, num_warps=4, num_stages=3)),
    ("n64-128x64x128-w4-s3", dict(block_m=128, block_n=64, block_k=128, num_warps=4, num_stages=3)),
    ("k128-128x32x128-w8-s3", dict(block_m=128, block_n=32, block_k=128, num_warps=8, num_stages=3)),
    ("k256-128x32x256-w8-s3", dict(block_m=128, block_n=32, block_k=256, num_warps=8, num_stages=3)),
    ("k256-128x32x256-w8-s2", dict(block_m=128, block_n=32, block_k=256, num_warps=8, num_stages=2)),
)


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
        acc += tl.dot(tile_a, tile_b, input_precision="ieee")

    tl.store(ptr_c, acc, mask=mask_c)


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    *,
    block_m: int = 64,
    block_n: int = 32,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
):
    """Write c = a @ b in-place; keyword options are for server-side sweeps."""
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
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _check_correctness(name: str, config: dict[str, int]) -> bool:
    """Check a config on irregular shapes so every load/store mask is exercised."""
    cases = [
        (1, 1, 1),
        (64, 32, 64),
        (65, 33, 67),
        (257, 513, 129),
    ]

    try:
        for M, N, K in cases:
            a = torch.randn((M, N), device="cuda", dtype=torch.float32)
            b = torch.randn((N, K), device="cuda", dtype=torch.float32)
            out = torch.empty((M, K), device="cuda", dtype=torch.float32)
            solve(a, b, out, M, N, K, **config)
            torch.cuda.synchronize()

            expected = torch.matmul(a, b)
            torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    except Exception as error:
        print(f"{name}: correctness/compile FAILED: {error}")
        return False

    print(f"{name}: correctness OK ({len(cases)} shapes)")
    return True


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


def _benchmark(selected_config: str | None = None) -> None:
    """Sweep candidate configs on the LeetGPU shape and compare with torch.mm."""
    M, N, K = 8192, 6144, 4096
    a = torch.randn((M, N), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    out_triton = torch.empty((M, K), device="cuda", dtype=torch.float32)
    out_torch = torch.empty_like(out_triton)

    torch_ms = _time_ms(lambda: torch.mm(a, b, out=out_torch))
    flops = 2 * M * N * K

    def gflops(ms: float) -> float:
        return flops / (ms * 1e-3) / 1e9

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={M}, N={N}, K={K}")
    print(f"torch.mm: {torch_ms:.3f} ms, {gflops(torch_ms):.1f} GFLOPS")

    configs = BENCHMARK_CONFIGS
    if selected_config is not None:
        configs = tuple(item for item in BENCHMARK_CONFIGS if item[0] == selected_config)
        if not configs:
            raise ValueError(f"未知 config: {selected_config}")

    results = []
    for name, config in configs:
        if not _check_correctness(name, config):
            continue

        try:
            triton_ms = _time_ms(
                lambda config=config: solve(a, b, out_triton, M, N, K, **config)
            )
        except Exception as error:
            print(f"{name}: benchmark FAILED: {error}")
            continue

        throughput = gflops(triton_ms)
        ratio = throughput / gflops(torch_ms)
        results.append((name, triton_ms, throughput, ratio))
        print(f"{name}: {triton_ms:.3f} ms, {throughput:.1f} GFLOPS, {ratio:.1%} of torch.mm")

    if results:
        best_name, best_ms, best_gflops, best_ratio = min(results, key=lambda item: item[1])
        print(
            f"best: {best_name}, {best_ms:.3f} ms, "
            f"{best_gflops:.1f} GFLOPS, {best_ratio:.1%} of torch.mm"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep or profile IEEE FP32 Triton MatMul configs.")
    parser.add_argument(
        "--config",
        choices=[name for name, _ in BENCHMARK_CONFIGS],
        help="只运行一个配置，便于用 Nsight Compute profile。默认 sweep 全部配置。",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要在有 NVIDIA GPU 的服务器上运行此验证脚本。")

    # 与服务器本次 IEEE FP32 benchmark 保持一致，避免 PyTorch 对照偷偷使用 TF32。
    torch.backends.cuda.matmul.allow_tf32 = False
    _benchmark(args.config)

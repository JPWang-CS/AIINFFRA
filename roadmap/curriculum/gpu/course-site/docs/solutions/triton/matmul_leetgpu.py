# Triton MatMul LeetGPU #02 final archive (`LEETGPU_PASS`).
# SuccessPublicTrace: A100-80GB, 2026-08-28 22:23:16, 24.54 ms, 55.3th percentile.
# This file preserves the platform `solve`/kernel; compared with the historical WIP,
# the only code change is IEEE input precision on `tl.dot`.

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
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # A[M, N] @ B[N, K] = C[M, K]，N 是归约维度。
    for n in range(0, N, BLOCK_N):
        offset_n = n + tl.arange(0, BLOCK_N)

        ptr_a = a + offset_m[:, None] * N + offset_n[None, :]
        ptr_b = b + offset_n[:, None] * K + offset_k[None, :]

        mask_a = (offset_m[:, None] < M) & (offset_n[None, :] < N)
        mask_b = (offset_n[:, None] < N) & (offset_k[None, :] < K)

        tile_a = tl.load(ptr_a, mask=mask_a, other=0.0)
        tile_b = tl.load(ptr_b, mask=mask_b, other=0.0)
        acc += tl.dot(tile_a, tile_b, input_precision='ieee')

    mask_c = (offset_m[:, None] < M) & (offset_k[None, :] < K)
    ptr_c = c + offset_m[:, None] * K + offset_k[None, :]
    tl.store(ptr_c, acc, mask=mask_c)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
          M: int, N: int, K: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

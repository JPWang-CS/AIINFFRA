"""Triton Matrix Multiplication — starter template.

【算子是什么】矩阵乘法 GEMM: C = A × B，Triton 实现
【在模型里干嘛】所有 Linear/FFN 层、QKV projection、attention output projection
【什么模型用】LLaMA/GPT/DeepSeek/Mistral 的 Triton 推理 kernel
- PyTorch 2.0+ `torch.compile` 内部用 Triton 生成 GEMM kernel
- vLLM/TensorRT-LLM 的 custom GEMM 也大量用 Triton

Key Triton concepts demonstrated:
1. @triton.jit — JIT-compiled GPU kernel
2. tl.program_id(axis) — block index (like blockIdx in CUDA)
3. tl.arange(0, BLOCK) — thread index range within a block
4. tl.load / tl.store — pointer-based load/store with masking
5. tl.dot — tensor core matmul on small tiles
"""

import triton
import triton.language as tl
import torch

# ---- Config ----
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # pointers to input/output tensors
    M, N, K,               # matrix dimensions: A(M,K) × B(K,N) = C(M,N)
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)

    # L2 cache optimization: group M blocks
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles with boundary mask
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # Accumulate: acc += a @ b
        acc = tl.dot(a, b, acc)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Main entry point: C = A × B"""
    assert a.shape[1] == b.shape[0], f"Shape mismatch: {a.shape} vs {b.shape}"
    M, K = a.shape
    K2, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) *
                         triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return c


# ---- Test ----
if __name__ == "__main__":
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    triton_out = matmul(a, b)
    torch_out = torch.mm(a, b)

    print(f"Shape: {triton_out.shape}")
    print(f"Max diff vs torch: {(triton_out - torch_out).abs().max().item():.2e}")
    print("PASS" if torch.allclose(triton_out, torch_out, atol=1e-2) else "FAIL")

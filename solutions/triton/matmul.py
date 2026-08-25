import torch
import triton
import triton.language as tl


# 状态：WIP。用户从 LeetGPU 空白模板开始编写的草稿；待继续完成并在 LeetGPU 验证。
@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    BLOCK_M=64,
    BLOCK_N=32,
    BLOCK_K=64,
):
    # 每个 PID 处理一个 BLOCK
    # 获取 Block 的位置
    pidM = tl.program_id(0)
    pidK = tl.program_id(1)

    # 行索引
    offset_m = pidM * BLOCK_M + tl.arange(0, BLOCK_M)
    # 列索引
    offset_k = pidK * BLOCK_K + tl.arange(0, BLOCK_K)
    # k 轴索引
    offset_n = tl.arange(0, BLOCK_N)

    # 创建当前 Block 视图
    ptr_a = a + offset_m[:, None] * K + offset_n[None, :]
    ptr_b = b + offset_n[:, None] + offset_k[None, :]
    ptr_c = c + offset_m[:, None] + offset_k[None, :]


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 64
    grid_0 = triton.cdiv(M, BLOCK_M)
    grid_1 = triton.cdiv(K, BLOCK_K)
    grid = (grid_0, grid_1)
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

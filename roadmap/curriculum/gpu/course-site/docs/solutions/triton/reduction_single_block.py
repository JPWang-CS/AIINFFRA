# 用户对话快照，尚无平台提交证据
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Block级递归
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    input = tl.load(input_ptr + offset, mask, other=0.0)
    tl.store(output_ptr, tl.sum(input))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # BLOCK_SIZE = 256
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

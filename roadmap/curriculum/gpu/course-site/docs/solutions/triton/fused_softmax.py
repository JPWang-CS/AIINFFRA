"""LeetGPU #5 Softmax 原始通过版；SuccessPublicTrace 2026-09-01 00:37:33。"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_partial(
    input: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    input = input.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=-float("inf"))
    tmpMax = tl.max(x, axis=0)
    tmpSum = tl.sum(tl.exp(x - tmpMax), axis=0)
    tl.store(partial_max + pid, tmpMax)
    tl.store(partial_sum + pid, tmpSum)


@triton.jit
def softmax_reduce(
    global_max: torch.Tensor,
    global_sum: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    num_blocks: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    offset = tl.arange(0, REDUCE_SIZE)
    mask = offset < num_blocks
    local_sum = tl.load(partial_sum + offset, mask=mask, other=float(0))
    local_max = tl.load(partial_max + offset, mask=mask, other=-float("inf"))
    tmp_max = tl.max(local_max, axis=0)
    local_sum *= tl.exp(local_max - tmp_max)
    tmp_sum = tl.sum(local_sum, axis=0)
    tl.store(global_max, tmp_max)
    tl.store(global_sum, tmp_sum)


@triton.jit
def softmax_sum(
    global_max,
    global_sum,
    input,
    output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=-float("inf"))
    gm = tl.load(global_max)
    gs = tl.load(global_sum)
    tmp_res = tl.exp(x - gm) / gs
    tl.store(output + offset, tmp_res, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    REDUCE_SIZE = triton.next_power_of_2(num_blocks)
    partial_sum = torch.empty(num_blocks, dtype=torch.float32, device=input.device)
    partial_max = torch.empty(num_blocks, dtype=torch.float32, device=input.device)
    global_sum = torch.empty(1, dtype=torch.float32, device=input.device)
    global_max = torch.empty(1, dtype=torch.float32, device=input.device)

    softmax_partial[(num_blocks,)](input, partial_max, partial_sum, N, BLOCK_SIZE=BLOCK_SIZE)
    softmax_reduce[(1,)](
        global_max,
        global_sum,
        partial_max,
        partial_sum,
        num_blocks,
        REDUCE_SIZE=REDUCE_SIZE,
    )
    softmax_sum[(num_blocks,)](global_max, global_sum, input, output, N, BLOCK_SIZE=BLOCK_SIZE)

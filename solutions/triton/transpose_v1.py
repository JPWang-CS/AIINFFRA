# 来源：用户 2026-10-13 对话初版；Agent 仅修正 tl.load/tl.store 两处调用。
# 状态：WIP；Python 语法检查通过，Triton JIT、LeetGPU、GPU 正确性与性能尚未验证。
# 输入输出：contiguous input[rows, cols] 与独立存储、相同 dtype 的 contiguous output[cols, rows]。
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc):
    # offset
    offsetA = tl.program_id(0) * cols + tl.program_id(1) + tl.arange(0, 1)
    offsetB = tl.program_id(1) * rows + tl.program_id(0) + tl.arange(0, 1)
    # mask
    # maskA = offsetA < (rows * cols)
    # maskB = offsetB < (rows * cols)
    # load
    a = tl.load(input + offsetA)
    # store
    tl.store(output + offsetB, a)



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
    )

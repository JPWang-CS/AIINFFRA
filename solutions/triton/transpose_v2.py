# 来源：用户 2026-10-13 最新 32x32 tile 版本。
# 状态：WIP；Python 语法静态检查通过。Triton JIT、LeetGPU、GPU 正确性与性能尚未验证。
# 合同：contiguous [rows, cols] -> [cols, rows]，独立存储、dtype 相同；四个 stride 参数保留但未用于寻址。
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, block_size: tl.constexpr):
    # offset
    offset_ax = tl.program_id(0) * block_size[0] + tl.arange(0, block_size[0])
    offset_ay = tl.program_id(1) * block_size[1] + tl.arange(0, block_size[1])
    offset_bx = tl.program_id(1) * block_size[1] + tl.arange(0, block_size[1])
    offset_by = tl.program_id(0) * block_size[0] + tl.arange(0, block_size[0])
    offsetA = offset_ax[:, None] * cols + offset_ay[None, :]
    offsetB = offset_bx[:, None] * rows + offset_by[None, :]
    # mask
    maskA = (offset_ax[:, None] < rows) & (offset_ay[None, :] < cols)
    maskB = (offset_bx[:, None] < cols) & (offset_by[None, :] < rows)
    # load

    # a = tl.load(input + offsetA, other = 0.0)
    a = tl.load(input + offsetA, mask = maskA, other = 0.0)
    # trans
    b = tl.trans(a)
    # store
    tl.store(output + offsetB, b, mask = maskB)



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    # block级搬运
    block_size = (32, 32)
    # grid = (rows, cols)
    grid = ((rows + block_size[0] - 1) // block_size[0], (cols + block_size[1] - 1) // block_size[1])
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, block_size
    )

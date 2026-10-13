import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input, output, rows, cols,
    stride_ir, stride_ic, stride_or, stride_oc,
    block_size: tl.constexpr,
):
    # 当前输入块覆盖的行、列
    offset_ax = tl.program_id(0) * block_size[0] + tl.arange(0, block_size[0])
    offset_ay = tl.program_id(1) * block_size[1] + tl.arange(0, block_size[1])

    # 转置后，输入列变成输出行，输入行变成输出列
    offset_bx = offset_ay
    offset_by = offset_ax

    offsetA = offset_ax[:, None] * cols + offset_ay[None, :]
    offsetB = offset_bx[:, None] * rows + offset_by[None, :]

    maskA = (offset_ax[:, None] < rows) & (offset_ay[None, :] < cols)
    maskB = (offset_bx[:, None] < cols) & (offset_by[None, :] < rows)

    a = tl.load(input + offsetA, mask=maskA, other=0.0)
    b = tl.trans(a)
    tl.store(output + offsetB, b, mask=maskB)


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    cols: int,
):
    block_size = (32, 32)
    grid = (
        triton.cdiv(rows, block_size[0]),
        triton.cdiv(cols, block_size[1]),
    )

    matrix_transpose_kernel[grid](
        input, output, rows, cols,
        cols, 1, rows, 1,
        block_size,
    )

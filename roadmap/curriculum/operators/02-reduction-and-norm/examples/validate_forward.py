"""Run forward correctness checks and optional kernel-only timing on a CUDA host.

The checks compute references in FP32 and cast them to the tested dtype. The
benchmark reuses preallocated tensors and does not call wrapper finite checks.
CUDA Event timing surrounds Python-launched calls, so the reported average can
include host launch gaps; it is not a profiler's pure kernel duration.
"""

import argparse

import torch
import triton

from rowwise_norms import layer_norm, layer_norm_kernel, rms_norm, rms_norm_kernel
from rowwise_softmax import softmax as softmax_wrapper
from rowwise_softmax import softmax_row_kernel


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
CASES = [(1, 1), (2, 31), (2, 32), (3, 33), (3, 257), (3, 1000)]


def reference_softmax(x):
    return torch.softmax(x.float(), dim=1).to(x.dtype)


def reference_rms(x, gamma, eps):
    values = x.float()
    return (values * torch.rsqrt((values * values).mean(dim=1, keepdim=True) + eps) * gamma.float()).to(x.dtype)


def reference_layer_norm(x, gamma, beta, eps):
    values = x.float()
    mean = values.mean(dim=1, keepdim=True)
    variance = ((values - mean) * (values - mean)).mean(dim=1, keepdim=True)
    return ((values - mean) * torch.rsqrt(variance + eps) * gamma.float() + beta.float()).to(x.dtype)


def tolerance(dtype, kind):
    if kind == "softmax":
        return {
            torch.float32: (1e-5, 1e-6),
            torch.float16: (2e-3, 1e-5),
            torch.bfloat16: (5e-2, 1e-5),
        }[dtype]
    return {
        torch.float32: (2e-4, 2e-5),
        torch.float16: (5e-3, 2e-4),
        torch.bfloat16: (2e-2, 2e-3),
    }[dtype]


def row_sum_tolerance(dtype):
    return {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 5e-3}[dtype]


def assert_close(name, actual, expected, kind):
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: non-finite output or reference")
    rtol, atol = tolerance(actual.dtype, kind)
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=False
    )


def inputs(rows, cols, dtype, device):
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    x[0].fill_(3)
    if rows > 1:
        x[1].fill_(-4)
    return x.contiguous()


def check_case(rows, cols, dtype, device):
    eps = 1e-5
    x = inputs(rows, cols, dtype, device)
    gamma = torch.linspace(0.5, 1.5, cols, device=device, dtype=dtype).contiguous()
    beta = torch.linspace(-0.25, 0.25, cols, device=device, dtype=dtype).contiguous()
    block = 1 << (cols - 1).bit_length()

    softmax_out = torch.empty_like(x)
    softmax_row_kernel[(rows,)](x, softmax_out, rows, cols, BLOCK=block, num_warps=4)
    assert_close("softmax", softmax_out, reference_softmax(x), "softmax")
    torch.testing.assert_close(
        softmax_out.float().sum(dim=1),
        torch.ones(rows, device=device),
        rtol=0.0,
        atol=row_sum_tolerance(dtype),
        equal_nan=False,
    )

    rms_out = torch.empty_like(x)
    rms_norm_kernel[(rows,)](x, gamma, rms_out, rows, cols, eps, BLOCK=block, num_warps=4)
    assert_close("rms_norm", rms_out, reference_rms(x, gamma, eps), "norm")

    layer_out = torch.empty_like(x)
    layer_norm_kernel[(rows,)](x, gamma, beta, layer_out, rows, cols, eps, BLOCK=block, num_warps=4)
    assert_close("layer_norm", layer_out, reference_layer_norm(x, gamma, beta, eps), "norm")


def check_biased_row(device):
    # This row catches cancellation in E[x^2]-E[x]^2; it has no padding.
    x = torch.tensor([[10000.0, 10001.0, 9999.0, 10002.0]], device=device)
    gamma = torch.ones(4, device=device)
    beta = torch.zeros(4, device=device)
    out = layer_norm(x, gamma, beta)
    expected = reference_layer_norm(x, gamma, beta, 1e-5)
    assert_close("biased_layer_norm_row", out, expected, "norm")


def check_padding_row(device):
    # A non-constant row is required: [3, 4, 5] padded to four has
    # correct variance 2/3, while summing the padding square gives 6.
    x = torch.tensor([[3.0, 4.0, 5.0]], device=device)
    gamma = torch.ones(3, device=device)
    beta = torch.zeros(3, device=device)
    out = torch.empty_like(x)
    layer_norm_kernel[(1,)](x, gamma, beta, out, 1, 3, 1e-5, BLOCK=4, num_warps=1)
    assert_close("padded_layer_norm_row", out, reference_layer_norm(x, gamma, beta, 1e-5), "norm")


def timed_ms(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_softmax(dtype, device):
    rows, cols = 4096, 1000
    x = torch.randn((rows, cols), device=device, dtype=dtype).contiguous()
    kernel_out = torch.empty_like(x)
    torch_out = torch.empty_like(x)
    block = 1 << (cols - 1).bit_length()
    softmax_row_kernel[(rows,)](x, kernel_out, rows, cols, BLOCK=block, num_warps=4)
    assert_close("benchmark_shape_softmax", kernel_out, reference_softmax(x), "softmax")
    torch.testing.assert_close(
        kernel_out.float().sum(dim=1),
        torch.ones(rows, device=device),
        rtol=0.0,
        atol=row_sum_tolerance(dtype),
        equal_nan=False,
    )
    kernel_call = lambda: softmax_row_kernel[(rows,)](
        x, kernel_out, rows, cols, BLOCK=block, num_warps=4
    )
    torch_call = lambda: torch.softmax(x, dim=1, out=torch_out)
    kernel_ms = timed_ms(kernel_call)
    torch_ms = timed_ms(torch_call)
    element_bytes = x.element_size()
    kernel_gbps = 2 * rows * cols * element_bytes / (kernel_ms * 1e-3) / 1e9
    torch_gbps = 2 * rows * cols * element_bytes / (torch_ms * 1e-3) / 1e9
    print(
        f"GPU={torch.cuda.get_device_name(device)} torch={torch.__version__} "
        f"triton={triton.__version__}"
    )
    print(f"shape=[{rows},{cols}] dtype={dtype} BLOCK={block} num_warps=4")
    print(
        f"预分配调用 CUDA Event 平均耗时: rowwise_softmax={kernel_ms:.4f} ms "
        f"({kernel_gbps:.2f} GB/s), torch.softmax(out=...)={torch_ms:.4f} ms "
        f"({torch_gbps:.2f} GB/s)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script; no GPU result was produced")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    for rows, cols in CASES:
        check_case(rows, cols, dtype, device)
    check_biased_row(device)
    check_padding_row(device)

    # One correctness preflight with the wrapper's optional finite scan. Timed
    # calls below invoke the kernel directly and do not repeat this scan.
    preflight = inputs(2, 257, dtype, device)
    softmax_wrapper(preflight, check_finite=True)
    print(f"PASS: forward Softmax/RMSNorm/LayerNorm for {args.dtype}, {len(CASES)} shapes")
    if args.benchmark:
        benchmark_softmax(dtype, device)


if __name__ == "__main__":
    main()

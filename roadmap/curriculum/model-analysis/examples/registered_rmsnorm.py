"""Functional FP32 RMSNorm custom op and MiniTransformer replacement checks."""
from __future__ import annotations

import argparse
from functools import lru_cache
import math
import time

torch = None
_OP = None


@lru_cache(maxsize=1)
def _triton_kernel():
    # Lazy module globals let Triton's JIT resolve tl names from function globals.
    global triton, tl
    import importlib
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")

    @triton.jit
    def kernel(X, Y, COLS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < COLS
        value = tl.load(X + row * COLS + col, mask=mask, other=0.0)
        mean_square = tl.sum(value * value, axis=0) / COLS
        scale = tl.rsqrt(mean_square + EPS)
        tl.store(Y + row * COLS + col, value * scale, mask=mask)
    return kernel


def _validate(x, eps):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError("x must have shape [rows, cols]")
    if x.dtype != torch.float32:
        raise TypeError("only torch.float32 is supported")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA devices are supported")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not 1 <= x.shape[1] <= 4096:
        raise ValueError("cols must be in [1, 4096]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _register(torch_module):
    global torch, _OP
    torch = torch_module

    @torch.library.custom_op("aiinffra::rms_norm_f32", mutates_args=())
    def implementation(x: torch.Tensor, eps: float) -> torch.Tensor:
        _validate(x, eps)
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.shape[0] == 0:
            return y
        if x.device.type == "cpu":
            mean_square = (x * x).sum(dim=1, keepdim=True) / x.shape[1]
            y.copy_(x * torch.rsqrt(mean_square + eps))
            return y
        # Tensor.data_ptr() includes storage_offset; guard the tensor's device.
        with torch.cuda.device(x.device):
            _triton_kernel()[(x.shape[0],)](
                x, y, COLS=x.shape[1], EPS=eps,
                BLOCK=triton.next_power_of_2(x.shape[1]),
            )
        return y

    @torch.library.register_fake("aiinffra::rms_norm_f32")
    def fake(x, eps: float):
        torch._check(x.ndim == 2)
        torch._check(x.dtype == torch.float32)
        torch._check(x.device.type in ("cpu", "cuda"))
        torch._check(x.shape[1] >= 1)
        torch._check(x.shape[1] <= 4096)
        torch._check(x.is_contiguous())
        torch._check(math.isfinite(eps) and eps > 0)
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    def setup_context(ctx, inputs, output):
        x, eps = inputs
        ctx.save_for_backward(x)
        ctx.eps = eps

    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        r = torch.rsqrt((x * x).mean(dim=1, keepdim=True) + ctx.eps)
        projection = (grad * x).mean(dim=1, keepdim=True)
        return grad * r - x * r.pow(3) * projection, None

    implementation.register_autograd(backward, setup_context=setup_context)
    _OP = implementation
    return implementation


def rms_norm_f32(x, eps=1e-6):
    if _OP is None:
        raise RuntimeError("custom op is not registered")
    return _OP(x, float(eps))


def _reference(x, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + eps)


def _close(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    error = (actual - expected).abs().max().item() if actual.numel() else 0.0
    print(f"PASS {label}; max_abs={error:.3g}")


def _reject(label, fn):
    try:
        fn()
    except (TypeError, ValueError, RuntimeError):
        print(f"PASS reject {label}")
    else:
        raise AssertionError(f"expected rejection: {label}")


def _check_grad(device):
    x = torch.randn((3, 65), device=device, requires_grad=True)
    upstream = torch.randn_like(x)
    grad, = torch.autograd.grad((rms_norm_f32(x) * upstream).sum(), x)
    x_ref = x.detach().clone().requires_grad_(True)
    expected, = torch.autograd.grad((_reference(x_ref) * upstream).sum(), x_ref)
    _close(grad, expected, "first-order gradient vs native PyTorch formula")


def _check_opcheck(device):
    sample = torch.randn((3, 65), device=device, requires_grad=True)
    torch.library.opcheck(_OP, (sample, 1e-6))
    singleton = torch.randn((1, 3), device=device).t()
    assert singleton.shape == (3, 1) and singleton.is_contiguous() and singleton.stride(1) != 1
    torch.library.opcheck(_OP, (singleton, 1e-6))
    _close(rms_norm_f32(singleton), _reference(singleton), "contiguous singleton with arbitrary stride")
    empty = torch.empty((0, 65), device=device)
    torch.library.opcheck(_OP, (empty, 1e-6))
    print("PASS opcheck registration contract (not a numerical/gradient proof)")


def _check_compile(device, backend):
    # custom_op is opaque inside torch.compile: fullgraph checks graph-boundary
    # composition, not fusion through this op. triton_op/wrap_triton is PyTorch's
    # compiler-visible Triton route; this example intentionally does not dual-implement it.
    x = torch.randn((3, 65), device=device)
    compiled = torch.compile(lambda value: rms_norm_f32(value), backend=backend, fullgraph=True)
    _close(compiled(x), _reference(x), f"torch.compile fullgraph backend={backend}")
    if device.type == "cpu" and backend == "aot_eager":
        print("NOTE CPU aot_eager checks graph composition only; no fusion/speed claim.")


def _model_norm(x):
    if not x.is_contiguous():
        raise ValueError("MiniTransformer norm input must be contiguous; reshape would copy")
    shape = x.shape
    return rms_norm_f32(x.reshape(-1, shape[-1])).reshape(shape)


def _check_model(device):
    # TorchMini converts the exact same NumPy MiniTransformer weights to FP32.
    from mini_transformer import MiniTransformer
    from torch_reference import TorchMini
    import numpy as np
    base = MiniTransformer(seed=5)
    model = TorchMini(base, device)
    ids = torch.tensor([[1, 3, 7, 8, 2, 4, 9], [9, 2, 5, 6, 1, 0, 3]], device=device)
    baseline, _ = model.forward(ids)
    replaced, _ = model.forward(ids, norm_fn=_model_norm)
    _close(replaced, baseline, "MiniTransformer full logits")

    ref_cache = new_cache = None
    ref_tokens, new_tokens = [], []
    for i in range(ids.shape[1]):
        out, ref_cache = model.forward(ids[:, i:i + 1], ref_cache)
        new, new_cache = model.forward(ids[:, i:i + 1], new_cache, norm_fn=_model_norm)
        ref_tokens.append(out); new_tokens.append(new)
    _close(torch.cat(new_tokens, 1), torch.cat(ref_tokens, 1), "per-token cache logits")

    ref_cache = new_cache = None
    ref_chunks, new_chunks = [], []
    for start, end in ((0, 3), (3, ids.shape[1])):
        out, ref_cache = model.forward(ids[:, start:end], ref_cache)
        new, new_cache = model.forward(ids[:, start:end], new_cache, norm_fn=_model_norm)
        ref_chunks.append(out); new_chunks.append(new)
    _close(torch.cat(new_chunks, 1), torch.cat(ref_chunks, 1), "chunked-prefill cache logits")

    numpy_logits, _ = base.forward(ids.cpu().numpy())
    # NumPy reference accumulates FP64; compare values with an explicit tolerance,
    # without casting it or weakening dtype checks for the FP32 operator tests.
    np.testing.assert_allclose(baseline.cpu().numpy(), numpy_logits, rtol=1e-4, atol=1e-4)
    print("PASS MiniTransformer full/token/chunk paths; shared weights and tokens")
    return model, ids


def _measure(fn, device, repeats=30):
    for _ in range(5):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats): fn()
        end.record(); end.synchronize()
        return start.elapsed_time(end) / repeats
    start = time.perf_counter()
    for _ in range(repeats): fn()
    return (time.perf_counter() - start) * 1000 / repeats


def _benchmark(device, model, ids):
    x = torch.randn((512, 128), device=device)
    ref_ms = _measure(lambda: _reference(x), device)
    op_ms = _measure(lambda: rms_norm_f32(x), device)
    model_ref = _measure(lambda: model.forward(ids), device, 10)
    model_op = _measure(lambda: model.forward(ids, norm_fn=_model_norm), device, 10)
    print(f"device={device} FP32; RMS rows=512 cols=128 repeats=30")
    print(f"steady reference mean_ms={ref_ms:.6f}; registered mean_ms={op_ms:.6f}")
    print(f"model reference mean_ms={model_ref:.6f}; registered mean_ms={model_op:.6f}")
    print("NOTE local shape/device only; no speedup is inferred or generalized.")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compile-backend", choices=("aot_eager", "inductor"), default=None,
                        help="default: CPU aot_eager (graph only), CUDA inductor")
    parser.add_argument("--benchmark", action="store_true", help="time the actual local device after warmup")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)  # --help completes before torch is imported.
    try:
        import torch as torch_module
    except ImportError as exc:
        print(f"SKIP: PyTorch unavailable: {exc}")
        return 77
    if args.device == "cuda":
        if not torch_module.cuda.is_available():
            print("SKIP: CUDA runtime/device unavailable")
            return 77
        try:
            _triton_kernel()  # Import/JIT decoration only; no kernel compile or launch here.
        except ImportError as exc:
            print(f"SKIP: Triton unavailable: {exc}")
            return 77

    _register(torch_module)
    device = torch.device(args.device)
    backend = args.compile_backend or ("aot_eager" if args.device == "cpu" else "inductor")
    for rows, cols in ((1, 1), (3, 65), (8, 128), (0, 65)):
        x = torch.randn((rows, cols), device=device)
        y = rms_norm_f32(x)
        _close(y, _reference(x), f"shape=({rows},{cols})")
        assert y.shape == x.shape and y.is_contiguous() and y is not x
        if x.numel(): assert y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()

    storage = torch.randn((5, 65), device=device)
    offset = storage[2:5, :]
    assert offset.is_contiguous() and offset.storage_offset() != 0
    _close(rms_norm_f32(offset), _reference(offset), "contiguous nonzero-offset slice")
    _reject("noncontiguous model norm before reshape", lambda: _model_norm(torch.randn((2, 3), device=device).t()))
    _reject("float16", lambda: rms_norm_f32(torch.ones((2, 4), device=device, dtype=torch.float16)))
    _reject("noncontiguous stride", lambda: rms_norm_f32(torch.ones((3, 130), device=device)[:, ::2]))
    _reject("cols=0", lambda: rms_norm_f32(torch.empty((2, 0), device=device)))
    _reject("cols=4097", lambda: rms_norm_f32(torch.empty((1, 4097), device=device)))
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        _reject(f"eps={eps}", lambda value=eps: rms_norm_f32(torch.ones((1, 1), device=device), value))

    _check_grad(device)
    _check_opcheck(device)
    _check_compile(device, backend)  # First compiled call is validation, never timed.
    model, ids = _check_model(device)
    if args.benchmark: _benchmark(device, model, ids)
    print(f"PASS all checks; torch={torch.__version__}; device={device}; backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to import registered_rmsnorm; script mode supports SKIP/77") from exc
    _register(_torch)

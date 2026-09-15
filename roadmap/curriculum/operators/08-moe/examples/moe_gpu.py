"""Small-k routing baseline. Ties choose the smallest expert id."""
import torch
import triton
import triton.language as tl


@triton.jit
def gate_kernel(Logits, Weights, Indices, E: tl.constexpr, K: tl.constexpr,
                BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BLOCK_E)
    slot = tl.arange(0, BLOCK_K)
    values = tl.load(Logits + row * E + e, mask=e < E, other=-float("inf"))
    selected = tl.full((BLOCK_K,), -float("inf"), tl.float32)
    for j in tl.static_range(K):
        best = tl.max(values, axis=0)
        expert = tl.min(tl.where((e < E) & (values == best), e, 2147483647), axis=0)
        selected = tl.where(slot == j, best, selected)
        tl.store(Indices + row * K + j, expert)
        values = tl.where(e == expert, -float("inf"), values)
    mass = tl.exp(selected - tl.max(selected, axis=0))
    weight = mass / tl.sum(mass, axis=0)
    tl.store(Weights + row * K + slot, weight, mask=slot < K)


def gate(logits, k=2):
    if logits.ndim != 2 or not logits.is_cuda or logits.dtype != torch.float32 or not logits.is_contiguous():
        raise ValueError("contiguous CUDA FP32 logits[M,E] required")
    m, e = logits.shape
    if m <= 0 or not 1 <= e <= 4096 or not isinstance(k,int) or not 1 <= k <= min(e,32):
        raise ValueError("baseline supports M>0, E<=4096, 1<=k<=min(E,32)")
    weights = torch.empty((m,k), device=logits.device, dtype=torch.float32)
    indices = torch.empty((m,k), device=logits.device, dtype=torch.int32)
    with torch.cuda.device(logits.device):
        gate_kernel[(m,)](logits, weights, indices, e,k,
                          triton.next_power_of_2(e),triton.next_power_of_2(k),num_warps=4)
    return weights, indices

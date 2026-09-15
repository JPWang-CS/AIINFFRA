"""Two-pass deterministic argmax, not a stochastic Top-p implementation."""
import torch
import triton
import triton.language as tl


@triton.jit
def argmax_partial(X, Values, Ids, V: tl.constexpr, CHUNKS: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    token = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + token, mask=token < V, other=-float("inf"))
    best = tl.max(x, axis=0)
    chosen = tl.min(tl.where((token < V) & (x == best), token, 2147483647), axis=0)
    tl.store(Values + row * CHUNKS + chunk, best)
    tl.store(Ids + row * CHUNKS + chunk, chosen)


@triton.jit
def argmax_finish(Values, Ids, Out, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    value = tl.load(Values + row * CHUNKS + c, mask=c < CHUNKS, other=-float("inf"))
    token = tl.load(Ids + row * CHUNKS + c, mask=c < CHUNKS, other=2147483647)
    best = tl.max(value, axis=0)
    chosen = tl.min(tl.where((c < CHUNKS) & (value == best), token, 2147483647), axis=0)
    tl.store(Out + row, chosen)


def argmax(logits):
    if logits.ndim != 2 or not logits.is_cuda or logits.dtype != torch.float32 or not logits.is_contiguous():
        raise ValueError("contiguous CUDA FP32 logits[B,V] required")
    b, v = logits.shape
    if b <= 0 or not 1 <= v <= 2**20:
        raise ValueError("baseline supports B>0 and 1<=V<=2**20")
    chunks = triton.cdiv(v,1024)
    values = torch.empty((b,chunks),device=logits.device,dtype=torch.float32)
    ids = torch.empty((b,chunks),device=logits.device,dtype=torch.int32)
    out = torch.empty((b,),device=logits.device,dtype=torch.int64)
    with torch.cuda.device(logits.device):
        argmax_partial[(b,chunks)](logits,values,ids,v,chunks,1024,num_warps=4)
        argmax_finish[(b,)](values,ids,out,chunks,triton.next_power_of_2(chunks),num_warps=4)
    return out

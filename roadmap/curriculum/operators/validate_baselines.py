"""Run teaching GPU baselines. Each wrapper timing includes allocation and launches."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["quant","moe","sampling"])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        import triton
    except ImportError:
        print("SKIP: PyTorch/Triton unavailable; no GPU test ran")
        return
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; no GPU test ran")
        return
    torch.manual_seed(13)
    device = torch.device("cuda")
    if args.case == "quant":
        mod = load("quant_gpu","07-quantized-operators/examples/quant_gpu.py")
        for m,n,t in [(1,1,16),(2,3,16),(30,50,16),(130,200,128),(256,256,128)]:
            x=torch.randn((m,n),device=device)
            s=torch.randn((triton.cdiv(m,t),triton.cdiv(n,t)),device=device)
            reference=x*s.repeat_interleave(t,0)[:m].repeat_interleave(t,1)[:,:n]
            torch.testing.assert_close(mod.block_scale(x,s,t),reference,rtol=1e-5,atol=1e-5)
        m=n=2048; t=128
        x=torch.randn((m,n),device=device); s=torch.randn((m//t,n//t),device=device)
        candidate=lambda:mod.block_scale(x,s,t)
        reference=lambda:(x.reshape(m//t,t,n//t,t)*s[:,None,:,None]).reshape(m,n)
        identity=f"X[{m},{n}] FP32, scale tile={t}; aligned broadcasting reference"
    elif args.case == "moe":
        mod=load("moe_gpu","08-moe/examples/moe_gpu.py")
        for m,e,k in [(1,1,1),(3,3,3),(4,8,2),(8,64,2),(100,16,4)]:
            # Distinct integer logits avoid undefined library tie ordering.
            x=torch.stack([torch.randperm(e,device=device).float() for _ in range(m)])
            vals,idx=torch.topk(x,k,dim=1)
            w,i=mod.gate(x,k)
            torch.testing.assert_close(w,torch.softmax(vals,dim=1),rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(i,idx.int(),rtol=0,atol=0)
        tied=torch.ones((2,8),device=device)
        tw,ti=mod.gate(tied,2)
        torch.testing.assert_close(ti,torch.tensor([[0,1],[0,1]],device=device,dtype=torch.int32),rtol=0,atol=0)
        torch.testing.assert_close(tw,torch.full_like(tw,.5))
        extreme=torch.tensor([[1000,999,-1000,0],[-1000,-1001,-1002,-1003]],device=device,dtype=torch.float32)
        ew,ei=mod.gate(extreme,2)
        ev,ex=torch.topk(extreme,2,dim=1)
        torch.testing.assert_close(ew,torch.softmax(ev,dim=1),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(ei,ex.int(),rtol=0,atol=0)
        m,e,k=1024,64,2
        x=torch.stack([torch.randperm(e,device=device).float() for _ in range(m)])
        candidate=lambda:mod.gate(x,k)
        def reference():
            val,idx=torch.topk(x,k,dim=1)
            return torch.softmax(val,dim=1),idx.int()
        identity=f"logits[{m},{e}] FP32, k={k}; distinct synthetic logits, output indices INT32"
    else:
        mod=load("sampling_gpu","09-sampling-kv/examples/sampling_gpu.py")
        for b,v in [(1,1),(3,17),(2,1024),(2,1025),(2,65537)]:
            x=torch.randn((b,v),device=device)
            x[:,0]=10
            x[:,-1]=10
            torch.testing.assert_close(mod.argmax(x),torch.argmax(x,dim=1),rtol=0,atol=0)
        b,v=8,65536
        x=torch.randn((b,v),device=device)
        candidate=lambda:mod.argmax(x)
        reference=lambda:torch.argmax(x,dim=1)
        identity=f"logits[{b},{v}] FP32; deterministic argmax with INT64 output"
    actual,expected=candidate(),reference()
    if args.case == "sampling":
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    elif args.case == "moe":
        torch.testing.assert_close(actual[0],expected[0],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
    else:
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
    print(f"PASS: {args.case} correctness and benchmark-shape preflight")
    if not args.benchmark:
        return
    def elapsed(fn):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(50): fn()
        torch.cuda.synchronize()
        return (time.perf_counter()-start)*1000/50
    print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__}")
    print(identity)
    print(f"teaching wrapper: {elapsed(candidate):.6f} ms")
    print(f"torch reference wrapper: {elapsed(reference):.6f} ms")
    print("Includes output/scratch allocation, host wrapper, launches and loop gaps; warm repeated inputs, not isolated kernel duration.")


if __name__ == "__main__":
    main()

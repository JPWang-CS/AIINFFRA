"""Optional PyTorch counterpart of the NumPy mini model; no pretrained weights."""
import argparse
import time
import numpy as np
from mini_transformer import MiniTransformer
try:
    import torch
except ImportError:
    torch=None


class TorchMini:
    def __init__(self, reference, device):
        self.base=reference
        to=lambda x:torch.tensor(x,device=device,dtype=torch.float32)
        self.embedding,self.lm_head=to(reference.embedding),to(reference.lm_head)
        self.layers=[{k:to(v) for k,v in layer.items()} for layer in reference.layers]

    def forward(self,tokens,cache=None,norm_fn=None):
        with torch.inference_mode():
            return self._forward(tokens,cache,norm_fn)

    def _forward(self,tokens,cache,norm_fn):
        if tokens.ndim!=2 or tokens.numel()==0 or tokens.dtype not in (torch.int32,torch.int64):
            raise ValueError("nonempty integer token ids[B,S] required")
        if tokens.device!=self.embedding.device:
            raise ValueError("tokens and model must be on the same device")
        b,s=tokens.shape; cfg=self.base
        if cache is not None and len(cache)!=len(self.layers):
            raise ValueError("one cache pair per layer required")
        past=0 if cache is None else cache[0][0].shape[2]
        if cache is not None and any(k.shape!=(b,cfg.hkv,past,cfg.d) or v.shape!=k.shape for k,v in cache):
            raise ValueError("cache shape mismatch")
        if s<=0 or past+s>cfg.max_seq:
            raise ValueError("invalid context length")
        norm=norm_fn or (lambda x:x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6))
        def rotate(x):
            inv=10000.0**(-torch.arange(0,cfg.d,2,device=x.device,dtype=x.dtype)/cfg.d)
            angle=torch.arange(past,past+s,device=x.device)[:,None]*inv[None,:]
            c,t=angle.cos()[None,None],angle.sin()[None,None]
            a,z=x[...,:cfg.d//2],x[...,cfg.d//2:]
            return torch.cat([a*c-z*t,a*t+z*c],dim=-1)
        x=self.embedding[tokens]; updated=[]
        for index,w in enumerate(self.layers):
            with torch.profiler.record_function(f"layer.{index}.attention"):
                z=norm(x)
                q=rotate((z@w["q"]).reshape(b,s,cfg.hq,cfg.d).transpose(1,2))
                k=rotate((z@w["k"]).reshape(b,s,cfg.hkv,cfg.d).transpose(1,2))
                v=(z@w["v"]).reshape(b,s,cfg.hkv,cfg.d).transpose(1,2)
                if cache is not None:
                    k=torch.cat([cache[index][0],k],dim=2)
                    v=torch.cat([cache[index][1],v],dim=2)
                updated.append((k,v))
                kr=k.repeat_interleave(cfg.hq//cfg.hkv,dim=1)
                vr=v.repeat_interleave(cfg.hq//cfg.hkv,dim=1)
                scores=q@kr.transpose(-1,-2)/cfg.d**.5
                valid=torch.arange(k.shape[2],device=x.device)[None,:] <= torch.arange(past,past+s,device=x.device)[:,None]
                p=torch.softmax(scores.masked_fill(~valid[None,None],-float("inf")),dim=-1)
                a=(p@vr).transpose(1,2).reshape(b,s,cfg.dim)
                x=x+a@w["o"]
            with torch.profiler.record_function(f"layer.{index}.mlp"):
                z=norm(x)
                x=x+(torch.nn.functional.silu(z@w["gate"])*(z@w["up"]))@w["down"]
        return norm(x)@self.lm_head,updated


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--benchmark",action="store_true")
    args=parser.parse_args()
    if torch is None or (args.device=="cuda" and not torch.cuda.is_available()):
        print("SKIP: requested PyTorch device unavailable; no GPU execution")
        return
    torch.backends.cuda.matmul.allow_tf32=False
    ref=MiniTransformer();model=TorchMini(ref,args.device)
    tokens=np.array([[1,3,7,8,2,4,9],[9,2,5,6,1,0,3]])
    ids=torch.tensor(tokens,device=args.device,dtype=torch.long)
    expected,_=ref.forward(tokens)
    actual,_=model.forward(ids)
    np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-4,atol=1e-4)
    cache=None;pieces=[]
    for i in range(ids.shape[1]):
        out,cache=model.forward(ids[:,i:i+1],cache);pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces,dim=1),actual,rtol=1e-4,atol=1e-4)
    norm_sum=lambda x:x*torch.rsqrt((x*x).sum(-1,keepdim=True)/x.shape[-1]+1e-6)
    replaced,_=model.forward(ids,norm_fn=norm_sum)
    torch.testing.assert_close(replaced,actual,rtol=1e-4,atol=1e-4)
    print("PASS: Torch full vs NumPy and incremental cache")
    if args.benchmark:
        _,prefix=model.forward(ids[:,:-1])
        def measure(fn):
            for _ in range(5):fn()
            if args.device=="cuda":torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(20):fn()
            if args.device=="cuda":torch.cuda.synchronize()
            return (time.perf_counter()-start)*1000/20
        print(f"device={args.device} torch={torch.__version__} dtype=FP32 model=random two-layer GQA B=2 S=7")
        print("prefill wrapper ms:",measure(lambda:model.forward(ids)))
        print("fixed-context decode wrapper ms:",measure(lambda:model.forward(ids[:,-1:],prefix)))
        print("Includes allocation, concat/repeat and Python; no claim of production kernel performance.")


if __name__=="__main__":
    main()

"""NumPy inference reference with GQA, RoPE, RMSNorm and SwiGLU.

Random weights, no tokenizer or learned language ability. Intended for cache
and operator-substitution semantics, not CPU/GPU performance comparison.
"""
import numpy as np


def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)


def rope(x, start):
    dim=x.shape[-1]
    inv=10000.0**(-np.arange(0,dim,2,dtype=np.float64)/dim)
    angles=np.arange(start,start+x.shape[2])[:,None]*inv[None,:]
    cos,sin=np.cos(angles)[None,None],np.sin(angles)[None,None]
    a,b=x[...,:dim//2],x[...,dim//2:]
    return np.concatenate([a*cos-b*sin,a*sin+b*cos],axis=-1)


def attention(q, k, v, past):
    group=q.shape[1]//k.shape[1]
    keys=np.repeat(k,group,axis=1)
    values=np.repeat(v,group,axis=1)
    scores=q@keys.transpose(0,1,3,2)/np.sqrt(q.shape[-1])
    visible=np.arange(k.shape[2])[None,:] <= (past+np.arange(q.shape[2]))[:,None]
    scores=np.where(visible[None,None],scores,-np.inf)
    mass=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    return (mass/np.sum(mass,axis=-1,keepdims=True))@values


class MiniTransformer:
    def __init__(self, seed=5):
        self.dim,self.hq,self.hkv,self.d,self.hidden,self.vocab,self.max_seq=64,4,2,16,96,128,128
        rng=np.random.default_rng(seed)
        matrix=lambda a,b:rng.normal(0,.08,(a,b))
        self.embedding=matrix(self.vocab,self.dim)
        self.lm_head=matrix(self.dim,self.vocab)
        self.layers=[dict(q=matrix(self.dim,self.dim),k=matrix(self.dim,self.hkv*self.d),
                          v=matrix(self.dim,self.hkv*self.d),o=matrix(self.dim,self.dim),
                          gate=matrix(self.dim,self.hidden),up=matrix(self.dim,self.hidden),
                          down=matrix(self.hidden,self.dim)) for _ in range(2)]

    def forward(self, tokens, cache=None, norm=rms_norm):
        tokens=np.asarray(tokens)
        if tokens.ndim != 2 or tokens.size == 0 or not np.issubdtype(tokens.dtype,np.integer):
            raise ValueError("nonempty integer token ids[B,S] required")
        if np.any(tokens<0) or np.any(tokens>=self.vocab):
            raise ValueError("token outside vocabulary")
        batch,length=tokens.shape
        if cache is not None and len(cache)!=len(self.layers):
            raise ValueError("one cache pair per layer required")
        past=0 if cache is None else cache[0][0].shape[2]
        if past+length>self.max_seq:
            raise ValueError("context limit exceeded")
        if cache is not None:
            expected=(batch,self.hkv,past,self.d)
            if any(k.shape!=expected or v.shape!=expected for k,v in cache):
                raise ValueError("cache batch/head/length mismatch")
        x=self.embedding[tokens]
        new_cache=[]
        for layer,w in enumerate(self.layers):
            z=norm(x)
            q=(z@w["q"]).reshape(batch,length,self.hq,self.d).transpose(0,2,1,3)
            k=(z@w["k"]).reshape(batch,length,self.hkv,self.d).transpose(0,2,1,3)
            v=(z@w["v"]).reshape(batch,length,self.hkv,self.d).transpose(0,2,1,3)
            q,k=rope(q,past),rope(k,past)
            if cache is not None:
                k=np.concatenate([cache[layer][0],k],axis=2)
                v=np.concatenate([cache[layer][1],v],axis=2)
            new_cache.append((k,v))
            a=attention(q,k,v,past).transpose(0,2,1,3).reshape(batch,length,self.dim)
            x=x+a@w["o"]
            z=norm(x)
            gate,up=z@w["gate"],z@w["up"]
            x=x+(gate/(1+np.exp(-gate))*up)@w["down"]
        return norm(x)@self.lm_head,new_cache


if __name__ == "__main__":
    model=MiniTransformer()
    tokens=np.array([[1,3,7,8,2,4,9],[9,2,5,6,1,0,3]])
    full,full_cache=model.forward(tokens)
    cache=None; pieces=[]
    for i in range(tokens.shape[1]):
        out,cache=model.forward(tokens[:,i:i+1],cache)
        pieces.append(out)
    np.testing.assert_allclose(np.concatenate(pieces,axis=1),full,rtol=1e-10,atol=1e-10)
    first,c=model.forward(tokens[:,:3])
    rest,c=model.forward(tokens[:,3:],c)
    np.testing.assert_allclose(np.concatenate([first,rest],axis=1),full,rtol=1e-10,atol=1e-10)
    changed=tokens.copy();changed[:,4:]=(changed[:,4:]+11)%model.vocab
    changed_out,_=model.forward(changed)
    np.testing.assert_allclose(changed_out[:,:4],full[:,:4],rtol=1e-10,atol=1e-10)
    alternate=lambda x:x/np.sqrt(np.sum(x*x,axis=-1,keepdims=True)/x.shape[-1]+1e-6)
    alt,_=model.forward(tokens,norm=alternate)
    np.testing.assert_allclose(alt,full,rtol=1e-10,atol=1e-10)
    assert all(k.shape==(2,2,7,16) and v.shape==k.shape for k,v in cache)
    print("PASS: full/token/chunk equivalence, causal prefix, norm replacement, GQA cache shape")

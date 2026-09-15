# 第二章 Mini Transformer：把算子放进完整前向

这个小型 Transformer 使用固定随机权重和固定 token，比较整段前向、逐 token Decode 与分块输入的结果。NumPy float64 提供数值参考，PyTorch FP32 使用相同权重。

模型有两层，hidden dimension 为 64，Query/KV heads 为 4/2，head dimension 为 16，SwiGLU 中间维度为 96，词表为 128。省略 bias、dropout 和训练，RMSNorm 缩放权重固定为 1。它用于检查算子组合，不需要下载语言模型。

## 1. block 的输入输出与残差

输入 X 为 [B,S,D]。先做 RMSNorm，再计算 Q/K/V；Q 与 K 应用相同位置约定的 RoPE，V 不旋转。Attention 输出投影后加回原输入，然后执行第二次 Norm、SwiGLU 和残差：

$$
X'=X+\operatorname{Attention}(\operatorname{RMSNorm}(X))W_O,
$$

$$
Y=X'+\bigl[\operatorname{SiLU}(ZW_G)\odot(ZW_U)\bigr]W_D,
\qquad Z=\operatorname{RMSNorm}(X').
$$

残差保留对应子层的输入，不能误用已经归一化的 Z。gate/up 的角色、epsilon、累加精度和输出 cast 都属于替换合同。

<!-- source-check: ../examples/mini_transformer.py -->
~~~python
def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)
~~~

每个 token 沿 hidden 维度独立归一化，不对整个 batch 或序列求一个共同统计量。

## 2. RoPE 的绝对位置不能每步清零

已有 past 个 token，新输入的绝对位置为 past,…,past+S-1。每次 Decode 都从位置 0 旋转会改变 Attention；cached K 已旋转，也不能重复旋转。

<!-- source-check: ../examples/mini_transformer.py -->
~~~python
def rope(x, start):
    dim=x.shape[-1]
    inv=10000.0**(-np.arange(0,dim,2,dtype=np.float64)/dim)
    angles=np.arange(start,start+x.shape[2])[:,None]*inv[None,:]
    cos,sin=np.cos(angles)[None,None],np.sin(angles)[None,None]
    a,b=x[...,:dim//2],x[...,dim//2:]
    return np.concatenate([a*cos-b*sin,a*sin+b*cos],axis=-1)
~~~

本例采用 split-half 配对。另一种相邻元素配对布局需要相应的维度重排，不能只因两段代码都叫 RoPE 就替换。分 chunk 不改变 token 的绝对位置，这是 chunked Prefill 与逐 token 对照必须检查的条件。

## 3. GQA 与带历史 cache 的因果 mask

Q 为 [B,Hq,S,d]，K/V cache 为 [B,Hkv,T,d]。参考实现为了直观，用 repeat 展开 KV head；高性能 kernel 可以直接使用 head 映射，不必真的复制数据。

<!-- source-check: ../examples/mini_transformer.py -->
~~~python
def attention(q, k, v, past):
    group=q.shape[1]//k.shape[1]
    keys=np.repeat(k,group,axis=1)
    values=np.repeat(v,group,axis=1)
    scores=q@keys.transpose(0,1,3,2)/np.sqrt(q.shape[-1])
    visible=np.arange(k.shape[2])[None,:] <= (past+np.arange(q.shape[2]))[:,None]
    scores=np.where(visible[None,None],scores,-np.inf)
    mass=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    return (mass/np.sum(mass,axis=-1,keepdims=True))@values
~~~

visible 的右侧包含 past。已有 4 个历史 token，再输入 1 个 token，这个 Query 位于 4，应看到 K 的 0…4。如果直接构造左上角 1×5 的普通下三角 mask，只保留第 0 列，结果就错了。

chunked Prefill 中，新的 Query 既能看到历史 cache，也能看到本 chunk 内不晚于自己的位置。验证时应更改未来 token，确认早期输出不变。

## 4. 完整前向的核心代码

各投影权重在初始化时固定，层与层保留独立 cache。下面是模型的 forward 方法：

<!-- source-check: ../examples/mini_transformer.py -->
~~~python
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
~~~

参考每步 concatenate 会复制已有 cache，repeat 会展开 KV，最后还计算所有输入位置的 logits。它们是为了看清语义，不是生产优化策略。替换成预分配或分页缓存时，逻辑序列、head 映射、位置与 mask 必须保持一致。

## 5. 三种执行方式对照

同一组固定 token 分别整段执行、每次一个 token、先 3 个再执行剩余 chunk。比较的是对应输入位置的 logits，不是让三个随机采样过程碰巧生成相同文本。

~~~python
full, _ = model.forward(tokens)
cache, pieces = None, []
for i in range(tokens.shape[1]):
    out, cache = model.forward(tokens[:, i:i+1], cache)
    pieces.append(out)
np.testing.assert_allclose(
    np.concatenate(pieces, axis=1), full, rtol=1e-10, atol=1e-10
)
~~~

再把输入后半段改掉，检查前半段输出不变。这个实验能发现错误因果范围；cache 形状检查则能发现按 Hq 错存 KV、batch 变化或层间状态串用。

稳定的 prefix cache 只代表已计算输入位置。采样出的下一个 token 尚未进入模型时，不应该声称它的 KV 已写入。完整生成循环应按这个顺序推进，不能先增长 length 再读取尚未生成的数据。

## 6. 算子替换要检查什么

Norm 替换先固定形状、dtype 和 epsilon。PyTorch 实现提供 norm_fn，下面的替换把 mean 改写成 sum/D，先验证调用位置：

~~~python
baseline, _ = model.forward(ids)
norm_sum = lambda x: x * torch.rsqrt(
    (x*x).sum(-1, keepdim=True) / x.shape[-1] + 1e-6
)
replaced, _ = model.forward(ids, norm_fn=norm_sum)
torch.testing.assert_close(replaced, baseline, rtol=1e-4, atol=1e-4)
~~~

换成二维 row-wise kernel 时，将 [B,S,D] 映射到 [BS,D]，计算后恢复形状。若必须先 contiguous，新增拷贝要计入模型时间。gamma、输出分配和 dtype 转换也不能从替换成本里消失。

Attention 的接口更复杂：Prefill 与 Decode、MHA 与 GQA、连续 cache 与分页 cache 都不一定兼容。适配必须维护真实布局与状态，不能只把调用函数名换掉。先检查单次输出，再检查多步 cache 和生成行为。

## 7. 实践：相同权重、相同输入、不同执行路径

~~~bash
python roadmap/curriculum/model-analysis/examples/mini_transformer.py
python roadmap/curriculum/model-analysis/examples/torch_reference.py --device cpu
python roadmap/curriculum/model-analysis/examples/torch_reference.py --device cuda
python roadmap/curriculum/model-analysis/examples/torch_reference.py --device cuda --benchmark
~~~

NumPy 检查覆盖整段/逐 token/chunk、因果性与 Norm 替换。PyTorch 将同一权重转换到 FP32，与 NumPy 参考及自己的增量路径比较。依赖或设备不可用时明确跳过。

计时包含 Python、分配、concat 和 repeat，不代表成熟引擎的速度。固定上下文 Decode 每次从同一 prefix 开始；真实生成则不断增长上下文，两者需要分别测量。

| 测试 | 重点 |
|---|---|
| full 与 incremental | cache、绝对位置、因果范围 |
| full 与 chunk | chunk 边界和历史可见性 |
| 替换前后 | 数值合同、布局、残差位置 |
| 同一上下文重复执行 | 稳定 shape 的调用成本 |
| 上下文持续增长 | cache 容量、追加与长序列成本 |

组合模型没有直接的 LeetGPU 题。基础题验证局部算子，本章验证组合后的状态与结果，不把两种完成标准混在一起。

## 参考阅读

[CUDA Programming Guide](../../../../downloads/cuda-programming-guide.pdf) · [大模型推理实践](../../../../downloads/大模型推理实践.pdf)

## 章节导航

- [上一章：模型执行分析](../README.md)
- [下一章：vLLM 应用](../../systems/README.md)

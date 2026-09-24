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

### 6.1 从核函数到框架算子

一个 Triton kernel 可以被 Python 函数直接调用，但框架还需要知道它是否修改输入、输出的形状与类型、如何求导，以及编译时如何推导输出。下面以无 gamma 的 RMSNorm 为例，将这些要求落实到同一个接口。

输入为连续的 FP32 `[R,D]`，每行独立计算，`1 <= D <= 4096`，epsilon 为有限正数。输出是相同形状的新张量；输入不修改，输出不与输入共享存储。R 可以为零，此时返回空输出而不启动 CUDA kernel。非连续布局和其他 dtype 明确拒绝。连续切片即使 `storage_offset` 非零仍可接受，因为传入指针已经指向该切片的起点。

CPU 分支用 PyTorch 运算提供可检查的实现，CUDA 分支用一行一个 program 的 Triton kernel。框架通过算子名称 `aiinffra::rms_norm_f32` 识别它，`mutates_args=()` 声明不修改任何输入。这个声明需要和实现一致，不是供编译器参考的性能提示。

FakeTensor 用形状、步长、dtype 和 device 等元数据表示张量，供编译器分析程序，不包含可读取的实际数据。注册的 fake 实现只检查这些元数据并描述新输出，不能调用 `.item()`、`.numpy()` 或解引用数据地址。

### 6.2 前向计算与梯度

对一行输入，令

$$
r=\left(\frac1D\sum_j x_j^2+\epsilon\right)^{-1/2},\qquad y_i=x_i r.
$$

无效列在加载时补零，但均值的分母仍是实际 D，不能用向上补齐后的 BLOCK。存储只覆盖有效列。示例中输入、累加和输出均为 FP32，尚未引入低精度转换或 gamma 缩放。

若后续需要反向传播，设输出梯度为 $g_i=\partial L/\partial y_i$，输入梯度为

$$
\frac{\partial L}{\partial x_i}
=g_i r-x_i r^3\left(\frac1D\sum_j g_jx_j\right).
$$

第一项来自 $y_i$ 对当前 $x_i$ 的直接依赖，第二项来自所有输出共同使用的均方根。反向公式通过 `register_autograd` 注册，使用 PyTorch 运算实现；示例将其与原始 PyTorch 表达式的一阶梯度比较。模型前向仍在 inference mode 中运行，注册反向不会把推理实验变成训练任务。


下面先给出 CUDA 行归约及算子注册。Triton 编译函数按需创建并缓存；fake 实现与真实实现采用相同的元数据约定。完整程序还包含依赖检查、输入拒绝测试和模型替换测试。

<!-- source-check: ../examples/registered_rmsnorm.py -->
~~~python
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
~~~

<!-- source-check: ../examples/registered_rmsnorm.py -->
~~~python
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
~~~

这里的 `_validate` 检查支持的输入范围；`_OP` 保存注册后可调用的算子。CPU 与 CUDA 分支都返回新张量。`setup_context` 保存反向所需的输入和 epsilon，`backward` 对应上面的梯度公式。

### 6.3 接口检查、数值检查与编译检查

以下几种检查分别回答不同问题：

| 检查 | 验证内容 |
|---|---|
| 与 PyTorch 参考比较 | 输出值、形状、类型和误差是否正确 |
| 非连续输入、越界列数、错误 dtype/epsilon | 不支持的输入是否明确拒绝 |
| `torch.library.opcheck` | 算子注册、修改/别名声明、FakeTensor 和编译组合是否符合框架要求 |
| 独立梯度比较 | 注册的一阶梯度是否与原表达式一致 |
| `torch.compile(fullgraph=True)` | 当前测试调用是否能够完整捕获，避免静默退回 Python 片段 |
| 模型整段、逐 token、分 chunk 对照 | 替换后 logits、位置与 cache 数据流是否保持一致 |

`opcheck` 不证明数学公式正确，也不证明 kernel 更快；一次 `fullgraph=True` 成功也不证明所有动态形状都不会重新编译。示例的 CPU `aot_eager` 后端用于检查图组合，不提供 GPU 融合或加速证据。CUDA 环境的 Inductor 路径需要在服务器实际执行。

这里选择 `custom_op` 是为了明确展示接口、fake 实现和后端分支。它对 `torch.compile` 是不透明调用，编译器不能据此融合算子内部的 Triton 计算。若希望编译器继续分析 Triton 实现，可采用 `torch.library.triton_op` 与 `wrap_triton`；两者的接口用途与限制见 [PyTorch 自定义 Triton 算子教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)。算子注册的检查语义见 [torch.library](https://docs.pytorch.org/docs/stable/library.html)。

### 6.4 在同一模型中替换和测量

模型的 Norm 输入为 `[B,S,D]`。适配函数先确认连续布局，再展平成 `[BS,D]`，调用注册算子，最后恢复原形状；它不隐式复制非连续输入。示例复用本章相同权重和 token，分别比较整段输入、逐 token cache 和两段 Prefill 的输出。

在模型测试通过后，再分别测量单个 Norm 调用与完整模型前向。注册、首次 JIT 编译及编译器捕获不进入稳态计时；输出分配和框架调用开销则保留在两种实现的完整调用中。模型中其他算子、cache 复制与 Python 调度也计入模型时间，所以单个 Norm 的收益不会直接等于模型收益。

从仓库根目录运行，环境需要 PyTorch 2.4 或更高版本；CUDA 分支另需匹配的 GPU、驱动和 Triton。缺少依赖或设备时返回 77，运行或校验失败时返回非零错误，只有实际执行全部检查后才输出通过信息。

```bash
python roadmap/curriculum/model-analysis/examples/registered_rmsnorm.py --help
python roadmap/curriculum/model-analysis/examples/registered_rmsnorm.py --device cpu
python roadmap/curriculum/model-analysis/examples/registered_rmsnorm.py --device cuda --compile-backend inductor
python roadmap/curriculum/model-analysis/examples/registered_rmsnorm.py --device cuda --compile-backend inductor --benchmark
```

| 设备 / PyTorch / Triton | 接口与数值检查 | 一阶梯度检查 | fullgraph 后端与结果 | 模型三种输入方式 | Norm 参考 / 替换 ms | 模型参考 / 替换 ms |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

<details>
<summary>完整程序：注册 RMSNorm、校验、模型替换与计时</summary>

<!-- source-check: ../examples/registered_rmsnorm.py -->
~~~python
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
~~~

</details>

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

# DeepSeek-V4.1-Flash：从缓存压缩到训练与推理协同

长任务经常重复这样的过程：读上下文、生成一段内容、调用工具，再把工具结果追加进上下文。模型需要解决的不只是“这一轮 Attention 能不能更快”，还包括新输入要经过多少层、历史状态有几份、下次命中缓存时能恢复哪些状态，以及这些状态通过哪条链路搬回来。

DeepSeek-V4.1-Flash 把这些问题放在一起设计。读它时可以始终追踪四个对象：**当前 hidden state、持久 global KV、每层局部 SWA KV、用于选择 global KV 的索引**。它们不是同一份数据，也不共享完全相同的生命周期。

## 1. 先分清三种“模型大小”

语言骨干有 40 层，前 20 层为因果编码器，后 20 层为解码器。前两层只有 SWA（Sliding-Window Attention，滑动窗口注意力），其余层结合局部窗口与压缩稀疏全局分支。模型包含 MoE（Mixture of Experts，混合专家）和额外的条件记忆表。

| 统计对象 | 报告配置 | 应怎样理解 |
|---|---:|---|
| backbone 总参数 | 552B | 不能只按每 token 激活量估计完整权重容量 |
| Engram 参数 | 196B | 额外的稀疏访问条件记忆，不能漏掉 |
| Prefill / Decode 激活参数 | 8B / 16B | CED 改变两个阶段主要经过的计算路径，不是权重动态消失 |

把 552B 和 196B 相加可以得到这两类参数的算术总量，但完整部署账本还要核对是否包括视觉编码器、投影器、草稿模块、精度元数据和副本。激活参数适合帮助估算一条路径的矩阵工作量；权重驻留、跨请求专家覆盖和通信不能仅由它决定。

本文代码使用 NumPy，检查小规模数学关系和数据依赖。它们不是作者 kernel，也不复现完整模型质量或吞吐。

<span id="chapter-39-section-28" aria-hidden="true"></span>

### 多模态张量路径：patch token 到语言 token

图像分支先把每张图像切成视觉 patch，再把 patch 向量投影到 ViT 宽度。设 patch 边长为 P、RGB 通道数为 3、视觉宽度为 $D_v$，单张图像 patch 网格为 $n_h\times n_w$，则输入和投影形状为：

$$
I\in\mathbb R^{3\times H\times W},\quad
X_{patch}\in\mathbb R^{(n_hn_w)\times(3P^2)},\quad
X_0=X_{patch}W_{patch}^{T}+b_{patch}\in\mathbb R^{(n_hn_w)\times D_v}.
$$

作者最小推理实现的 patch embedding 是一层线性投影；ViT 对同一张图的全部 patch 做双向 self-attention，并在 Q/K 上施加二维 RoPE。若 head 数为 $h$、每头宽度为 $d_h=D_v/h$，一次融合 QKV 投影的逻辑形状是 $[n_hn_w,3D_v]$，切成三份并 reshape 后，Q/K/V 各为 $[n_hn_w,h,d_h]$。注意力输出再拼回 $[n_hn_w,D_v]$。这是视觉编码器内部的双向注意力；不能把语言 decoder 的因果 mask 套到这里。

对齐器把空间网格按 $r\times r$ 邻域收拢到通道维。设 $n_h,n_w$ 都能被 r 整除，经过 unshuffle 与 projector：

$$
X_v\in\mathbb R^{(n_hn_w)\times D_v}
\rightarrow X_g\in\mathbb R^{n_h\times n_w\times D_v}
\rightarrow X_u\in\mathbb R^{(n_h/r)(n_w/r)\times(r^2D_v)}
\rightarrow X_{lang}\in\mathbb R^{(n_h/r)(n_w/r)\times d}.
$$

例如 $n_h=6,n_w=9,r=3$ 时，视觉序列从 54 个 token 变成 6 个语言宽度 token；unshuffle 前后的特征元素数相等：$54D_v=6(9D_v)$。接着两层线性映射与 GELU 将每个 $9D_v$ 向量变到语言宽度 d。减少的是送进语言模型的序列长度，空间特征经重新分组后仍参与 projector，不是简单丢掉 8/9 的视觉元素。作者实现对不能整除的底部/右侧会裁去余数，例如 $7\times8$ 网格、$r=3$ 只留下 $6\times6$；调用端必须确认这是否符合预处理网格的边界设计。

配套预处理按 patch grid 对 3 取 ceiling 来计划 LLM grid；而 `unfold` 实际输出取决于传入 Aligner 的 $n_h,n_w$ 和其边缘裁剪。阅读调用时必须核对传入尺寸、裁剪/填补位置以及 image slot 数是否一致，不能凭 planned grid 假设每个 slot 都有有效特征。预处理将图像转 RGB，按最小像素/长宽比限制与最大语言 token 预算规划尺寸，边长向 patch size 取整；超宽图 resize，其它图像按目标尺寸 padding。像素先变为 [0,1]，再作 $(x-0.5)/0.5$ 归一化并转 BF16。把 `[3,n_v^h,P,n_v^w,P]` reshape/permutation 为 `[n_v^hn_v^w,3,P,P]` 后，每行才进入 patch embedding。

图像 span 由 IMAGE_START、每行的 IMAGE slot 和 IMAGE_NEW_LINE、IMAGE_END 组成：

$$
N_{image}=n_l^h(n_l^w+1)+2.
$$

这些位置在 `input_ids` 中都使用配置给定的 image token ID，另一个 `token_types` 张量区分 IMAGE_START、IMAGE、换行与 IMAGE_END；processor 按原消息 placeholder 的位置展开多图/文本交错，不是把所有视觉向量追加到 prompt 尾部。最小预处理路径还校验 placeholder 数与传入图像数一致。CPU 例子核对形状与重排，不模拟像素解码、resize 插值或多图消息。

下面分开看三处运算。第一段是 patch 线性层和 ViT 内部 QKV 的维度；第二段按 `F.unfold(kernel_size=r, stride=r)` 的 channel-major 次序做空间重排，再执行两层 projector。ViT 权重与图像解码/缩放不在这个 CPU 示例内。

~~~python
def patch_embed(patches, weight, bias):
    # patches: [N, 3, P, P], weight: [D_v, 3*P*P]
    return patches.reshape(len(patches), -1) @ weight.T + bias


def split_vision_qkv(projected, weight, bias, heads):
    # projected [N, D_v] -> q/k/v each [N, heads, D_v // heads]
    n, dv = projected.shape
    if dv % heads:
        raise ValueError("vision width must be divisible by head count")
    qkv = projected @ weight.T + bias  # [N, 3*D_v]
    return tuple(part.reshape(n, heads, dv // heads)
                 for part in np.split(qkv, 3, axis=-1))


def aligner_unshuffle(features, n_h, n_w, ratio):
    # features [n_h*n_w, D_v], channel-major r-by-r patch ordering
    if features.ndim != 2 or features.shape[0] != n_h * n_w or ratio <= 0:
        raise ValueError("expected a flat image grid and positive ratio")
    if n_h % ratio or n_w % ratio:
        raise ValueError("choose a divisible grid or account for the crop")
    dv = features.shape[-1]
    grid = features.reshape(n_h, n_w, dv)
    return (grid.reshape(n_h // ratio, ratio, n_w // ratio, ratio, dv)
                .transpose(0, 2, 4, 1, 3)
                .reshape((n_h // ratio) * (n_w // ratio), dv * ratio * ratio))


def project_to_language(unshuffled, w1, b1, w2, b2):
    hidden = unshuffled @ w1.T + b1
    gelu = 0.5 * hidden * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
    return gelu @ w2.T + b2
~~~

`aligner_unshuffle` 的 reshape/transposition 次序对应 `[C,H,W]` 后做 `unfold`：先按输出格点枚举，再按通道、窗口行、窗口列展平。若把窗口维排在通道之前，形状可能完全正确，但每个 projector 输入分量对应的空间邻居次序会错。完整 CPU 检查使用小矩阵验证元素次序、长度缩减和 projector 输出宽度；它不是 ViT 质量测试，也不需要模型权重或图像。

配套的 image processor 把原图转成上述 patch grid。它先转 RGB；长宽限制和最小像素策略调整目标尺寸，再将尺寸向 patch 边长取整，并依据 3×3 对齐后的语言 token 预算执行安全缩放。超宽图按目标尺寸 resize，其它图像用中性灰 padding 保持比例；像素先缩放到 [0,1]，再按 $(x-0.5)/0.5$ 归一化并转 BF16。设 patch 边长 P、目标像素高宽为 $H',W'$，则 patch grid 是 $n_v^h=H'/P,n_v^w=W'/P$；reshape/permutation 把 `[3,n_v^h,P,n_v^w,P]` 排成 `[n_v^hn_v^w,3,P,P]`，每一行对应一个 patch。

视觉 token 的数目不只是对齐器输出格点数。序列布局为图像起始标志、每行 $n_l^w$ 个 IMAGE slot 加一个 IMAGE_NEW_LINE、最后图像结束标志，因此

$$
N_{image}=n_l^h(n_l^w+1)+2.
$$

每个 slot 在 `input_ids` 中都使用配置给定的 image token ID；`token_types` 才区分 IMAGE_START、IMAGE、换行、IMAGE_END 与普通文本。消息中的占位符被展开为对应长度的图像 span，图像 patch 与 span 起点一并传给视觉路径，aligner 行按 reading order 填入 IMAGE slot。因而一个图像包含的 ViT patch 数可能远大于最终插入语言序列的 image slot 数；多图与文字交错位置也必须在展开过程中保留。上述 resize/网格/token span 事实来自配套 image processor；单看 `vision.py` 的 ViT 类本身确实无法推出这些预处理策略。

完整的随机小张量检查：

<details>
<summary>展开 CPU 张量形状与排列检查</summary>

<!-- source-check: examples/v41_tensor_walkthrough.py -->
~~~python
"""CPU-only tensor-order and shape checks; no model weights or accelerator."""
import numpy as np


def patch_embed(patches, weight, bias):
    return patches.reshape(len(patches), -1) @ weight.T + bias


def split_vision_qkv(projected, weight, bias, heads):
    n, dv = projected.shape
    if dv % heads or weight.shape != (3 * dv, dv) or bias.shape != (3 * dv,):
        raise ValueError("incompatible vision projection shapes")
    qkv = projected @ weight.T + bias
    return tuple(x.reshape(n, heads, dv // heads)
                 for x in np.split(qkv, 3, axis=-1))


def aligner_unshuffle(features, n_h, n_w, ratio):
    if (features.ndim != 2 or features.shape[0] != n_h * n_w
            or ratio <= 0 or n_h % ratio or n_w % ratio):
        raise ValueError("flat feature grid must be divisible by ratio")
    dv = features.shape[-1]
    grid = features.reshape(n_h, n_w, dv)
    return (grid.reshape(n_h // ratio, ratio, n_w // ratio, ratio, dv)
                .transpose(0, 2, 4, 1, 3)
                .reshape((n_h // ratio) * (n_w // ratio), dv * ratio * ratio))


def project_to_language(unshuffled, w1, b1, w2, b2):
    hidden = unshuffled @ w1.T + b1
    # Tanh GELU approximation, used here only for a small CPU shape example.
    gelu = 0.5 * hidden * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
    return gelu @ w2.T + b2


def hc_pre(streams, pre_mix):
    """[source, feature] weighted by [source] -> one feature vector."""
    if streams.ndim != 2 or pre_mix.shape != (streams.shape[0],):
        raise ValueError("pre mix must align with the source-stream axis")
    return np.sum(pre_mix[:, None] * streams, axis=0)


def hc_post(sublayer_output, residual, post_mix, comb_source_destination):
    """HF hc_post axis order: comb[source, destination], sum over source."""
    n, d = residual.shape
    if (sublayer_output.shape != (d,) or post_mix.shape != (n,)
            or comb_source_destination.shape != (n, n)):
        raise ValueError("incompatible mHC stream, post, or comb shapes")
    residual_mix = np.sum(
        comb_source_destination[:, :, None] * residual[:, None, :], axis=0
    )
    return post_mix[:, None] * sublayer_output[None, :] + residual_mix


def main():
    rng = np.random.default_rng(17)
    n_h, n_w, patch, channels, dv, heads, ratio, language_dim = 6, 9, 2, 3, 8, 2, 3, 5
    patches = rng.normal(size=(n_h * n_w, channels, patch, patch))
    w_patch = rng.normal(size=(dv, channels * patch * patch))
    b_patch = rng.normal(size=(dv,))
    vision = patch_embed(patches, w_patch, b_patch)
    wqkv = rng.normal(size=(3 * dv, dv))
    bqkv = rng.normal(size=(3 * dv,))
    q, k, v = split_vision_qkv(vision, wqkv, bqkv, heads)
    assert q.shape == k.shape == v.shape == (n_h * n_w, heads, dv // heads)

    # Check the exact channel-major order produced by [C,H,W] + unfold.
    order = np.arange(2 * 2 * 2).reshape(4, 2)
    packed = aligner_unshuffle(order, 2, 2, 2)
    assert np.array_equal(packed, [[0, 2, 4, 6, 1, 3, 5, 7]])

    unshuffled = aligner_unshuffle(vision, n_h, n_w, ratio)
    assert unshuffled.shape == ((n_h // ratio) * (n_w // ratio), dv * ratio**2)
    w1 = rng.normal(size=(11, unshuffled.shape[1]))
    b1 = rng.normal(size=(11,))
    w2 = rng.normal(size=(language_dim, 11))
    b2 = rng.normal(size=(language_dim,))
    language = project_to_language(unshuffled, w1, b1, w2, b2)
    assert language.shape == ((n_h // ratio) * (n_w // ratio), language_dim)
    assert n_h * n_w * dv == language.shape[0] * dv * ratio**2

    # A cyclic permutation is non-symmetric: rows are sources, columns destinations.
    streams = np.array([[1., 10.], [2., 20.], [3., 30.]])
    pre = np.array([0.1, 0.2, 0.7])
    pre_output = hc_pre(streams, pre)
    assert np.allclose(pre_output, [2.6, 26.0])
    comb_source_destination = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    post = np.array([0.25, 0.5, 0.75])
    sublayer_output = np.array([2., -1.])
    post_output = hc_post(sublayer_output, streams, post, comb_source_destination)
    expected_residual_mix = streams[[2, 0, 1]]
    assert np.array_equal(comb_source_destination.T @ streams, expected_residual_mix)
    assert np.array_equal(post_output, post[:, None] * sublayer_output + expected_residual_mix)
    assert post_output.shape == (3, 2)
    assert not np.array_equal(comb_source_destination @ streams, expected_residual_mix)
    try:
        aligner_unshuffle(vision[:7 * 8], 7, 8, ratio)
    except ValueError:
        pass
    else:
        raise AssertionError("non-divisible grid must not be silently called lossless")
    print("PASS: patch/QKV shapes, unfold order, unshuffle count, projector width, mHC pre/post shapes and comb transpose")


if __name__ == "__main__":
    main()
~~~

运行：`python roadmap/curriculum/papers/examples/v41_tensor_walkthrough.py`。该程序核对视觉 tensor shape、patch/unshuffle 元素顺序与整除边界，并用非对称 comb 验证 mHC pre 输出 `[d]`、post 输出 `[n,d]` 及 source/destination 转置方向；它不执行视觉注意力或模型精度测试。
</details>

## 2. CED：为什么 Prefill 可以少经过一些层

普通 decoder-only 模型的第 l 层 KV 来自该层输入 $H_l$。要获得后半网络的历史 KV，通常必须让所有历史 token 依次走过前面的层。

CED（Causal Encoder-Decoder，因果编码器—解码器）改变后半网络的 global KV 来源。设总层数为 L，序列长 N，hidden dimension 为 d。对于后半网络：

$$
H_{L/2}\in\mathbb R^{N\times d},\qquad
C_l=H_{L/2}W_l^{KV},\qquad
Z_l=H_{L/2}W_l^Z,\qquad l>L/2.
$$

$C_l$ 是压缩器使用的 KV 表示，$Z_l$ 是压缩权重相关量。这里先理解数据依赖，不把 C_l 当作已经完成所有压缩、量化和布局转换的物理缓存。不同层的投影权重可以不同；来自同一个 encoder 输出，不等于每层 KV 自动相同。

这样，长 prompt 的主要 global 状态在 encoder 结束后就能准备。可是 decoder 各层的局部 SWA KV 仍依赖各自 hidden state。要开始生成，第一个 decode step 必须取得这些局部状态，因此还需要末尾窗口重放。

用 token-layer 数做一个粗略代理，窗口为 W：

$$
R_{\mathrm{work}}=
\frac{N(L/2)+\min(N,W)(L/2)}{NL}
=\frac12+\frac{\min(N,W)}{2N}.
$$

N=4096、W=128 时为 0.515625；N≤W 时为 1。长输入有利、短输入收益弱，是从公式直接得到的。投影开销、不同层的 Attention 成本、缓存命中、重放误差都没有包含在这个代理里，不能据此宣称端到端必然快两倍，也没有改变渐近复杂度的阶。

## 3. CSA2：共享缓存和共享选择是两件事

CSA2（Compressed Sparse Attention 2）沿序列和层两个方向减少重复状态。相对前代 CSA，压缩器不再对相邻压缩条目使用重叠的来源窗口，也移除了压缩阶段的绝对位置 embedding；indexer K 从 main KV 投影得到，而不是单独从 hidden state 再走一条压缩路径。

每层被静态指定为以下模式，运行时不是根据每个 token 任意换模式：

| 模式 | main KV 和 indexer K | Top-K 索引 | 当前层仍需计算 |
|---|---|---|---|
| Full | 自己产生 | 自己产生 | main Q、SWA KV、索引评分、Attention |
| Reindex | 复用最近 Full 来源 | 用自己的 indexer Q 重新选择 | main Q、SWA KV、索引评分、Attention |
| Reuse | 复用最近 Full 来源 | 复用与该 KV 对应的最近一次选择 | main Q、SWA KV、Attention |

对 Full → Reuse → Reindex → Reuse 四层，缓存所有者可以一直是第一层，但后两层的选择所有者是第三层。缓存页引用和索引引用都必须携带正确来源：只有一个“共享层编号”无法描述这段执行。

> [!IMPORTANT]
> Reuse 不是重用整层输出。当前层仍有自己的 Query、局部 KV 和 Attention 计算；节省的是指定缓存和索引步骤。

共享索引减少评分工作，共享 KV 减少存储；这两种收益不能混成同一个比例。跨层共享还改变训练时的梯度来源，后文会回到这个问题。

## 4. 层次索引：候选池不是最终 Top-K

即使部分层不再打分，剩下的 Reindex 层若每次还扫描百万个历史位置，索引器仍可能很贵。层次索引器先用浅层结果建立一个较大的候选池，再允许深层重新选择。

对当前 Query，可见的 main KV 条目数为 T。先由第一个 Full 索引器得到分数 $u_t$，把相邻 B 个位置放成一块：

$$
\mathcal B_j=\{jB,\ldots,\min((j+1)B,T)-1\},\qquad
b_j=\max_{t\in\mathcal B_j}u_t.
$$

先按块最大分数选 P 个块，再展开为候选位置集合：

$$
\mathcal J=\operatorname{TopP}(b),\qquad
\mathcal C=\bigcup_{j\in\mathcal J}\mathcal B_j.
$$

Full 层自己的最终选择依然来自全部可见位置；后续第 l 个 Reindex 层使用自己的分数 $v_t^{(l)}$，但只在候选集合内选择：

$$
\mathcal I_0=\operatorname{TopK}_{t<T}(u_t),\qquad
\mathcal I_l=\operatorname{TopK}_{t\in\mathcal C}(v_t^{(l)}).
$$

报告配置 B=8、最多 P=2048，所以候选最多 16384 个位置，而最终每层选 512 个条目。**16384 是搜索域大小，512 才是最终稀疏选择预算。** 短上下文或末尾不足一块时，候选数量自然更少；必须排除未来位置和 padding。

下面用已算好的两组分数演示选择阶段。评分器本身依赖训练参数，不在这个小例子里伪造：

<!-- source-check: examples/v41_checks.py -->
~~~python
def hierarchical_select(first_scores, later_scores, visible, block_size, blocks, topk):
    first = np.asarray(first_scores, dtype=np.float64)
    later = np.asarray(later_scores, dtype=np.float64)
    if first.ndim != 1 or later.shape != first.shape:
        raise ValueError("two equal-length score vectors required")
    if not (0 < visible <= len(first)) or min(block_size, blocks, topk) <= 0:
        raise ValueError("positive visible length and budgets required")
    if not np.isfinite(first[:visible]).all() or not np.isfinite(later[:visible]).all():
        raise ValueError("visible scores must be finite")
    nblocks = (visible + block_size - 1) // block_size
    padded = np.full(nblocks * block_size, -np.inf)
    padded[:visible] = first[:visible]
    block_scores = padded.reshape(nblocks, block_size).max(axis=1)
    chosen_blocks = np.argsort(-block_scores, kind="stable")[:blocks]
    candidates = (chosen_blocks[:, None] * block_size
                  + np.arange(block_size)[None, :]).reshape(-1)
    candidates = np.sort(candidates[candidates < visible])
    first_top = np.argsort(-first[:visible], kind="stable")[:topk]
    later_top = candidates[np.argsort(-later[candidates], kind="stable")[:topk]]
    return first_top, candidates, later_top
~~~

逐项对照代码：reshape 把位置分成块；max(axis=1) 得到块分数；chosen_blocks 是块编号；乘 block_size 再加块内偏移才得到位置编号。later_top 返回的是原缓存的位置，而非候选数组的局部下标。稳定排序只是为教学中的平局定义确定结果，不是建议在真实 kernel 中做全排序。

### 4.1 候选限制为什么可能丢掉深层需要的位置

令浅层分数为 [9,8,1,0,2]，深层分数为 [0,1,8,7,100]，块长为 2，只保留一个块。浅层选中块 [0,1]，深层只能在这两个位置中选择，即使它对位置 4 的评分是 100，也无法把位置 4 找回来。

这不是地址 bug，而是搜索域被限制的结果。训练阶段使用同样的限制有助于模型适应，但不能把它当作“与无限制 Top-K 数学等价”的证明。长上下文检索、重要信息恰好落在被丢块中、不同层相关性变化很大，都是需要单独测的边界。

### 4.2 复杂度和真正要测的流量

假设还有 R 个后续索引器，评分特征维度为 $d_i$，候选规模上限为 C。忽略选块和 Top-K 自身的成本：

$$
F_{\mathrm{score}}\propto T d_i+R C d_i,
$$

而不是把整个路径写成 O(C)。第一个 Full 仍扫描 T 个位置；只是后续评分在固定候选上限下不再随 T 增大。

代价还包括读写候选位置、gather indexer K、Top-K、main KV gather 和局部 SWA。候选分布很散时，条目数少了也可能损失合并访问。性能分析应分别测完整扫描、候选构建、重评分和最终 Attention，不能用 T/C 直接当加速倍数。

## 5. 缓存格式：890 B/token 是怎样算出来的

语言网络的前三个 encoder Full 组各以压缩率 2 保存一份 global 状态；decoder 有一份压缩率 1 的 Full 状态。因此每 S 个原始 token 对应的 global 条目数约为：

$$
N_{\mathrm{entry}}=(3/2+1)S=2.5S.
$$

一个 main KV 条目有 512 维。E2M1 指 1 位符号、2 位指数、1 位尾数的数据格式，每元素共 4 bit，每 16 维另有一个 E4M3 scale：

$$
B_{\mathrm{main}}=512\times4/8+512/16=288\ \mathrm{bytes}.
$$

一个 indexer K 条目有 128 维；其 MXFP4 scale 按每 32 维一个 byte 计：

$$
B_{\mathrm{index}}=128\times4/8+128/32=68\ \mathrm{bytes},
\qquad B_{\mathrm{global/token}}=2.5(288+68)=890\ \mathrm{bytes}.
$$

把 nibble payload 和逐组 scale 分开计算，可用下面的 CPU 算术核对：

<!-- source-check: examples/v41_checks.py -->
~~~python
def fp4_block_bytes(elements, scale_group, scale_bytes=1):
    if elements <= 0 or scale_group <= 0 or scale_bytes <= 0:
        raise ValueError("positive element, group, and scale sizes required")
    if elements % 2 or elements % scale_group:
        raise ValueError("FP4 packing and scale groups must divide the block")
    return elements // 2 + (elements // scale_group) * scale_bytes


def cache_budget_example():
    main_bytes = fp4_block_bytes(512, 16)  # 288 bytes
    index_bytes = fp4_block_bytes(128, 32)  # 68 bytes
    return main_bytes, index_bytes, 2.5 * (main_bytes + index_bytes)
~~~

`512/2 + 512/16 = 288`、`128/2 + 128/32 = 68`，再乘每原始 token 平均 2.5 个 global entry，得到 890。检查同时拒绝奇数 FP4 元素数和不能整分的 scale group。它核实的是报告中的有效 payload 公式，不含 page padding、元数据、额外副本或 SWA，因此不能用作整机显存规划器。

这是 global KV 的有效数据账本，不含每层 SWA、页尾 padding、索引输出、allocator 和分片副本。不能把 890 乘 token 数后就称为“部署显存”。

main KV 的格式借鉴 NVFP4 的块尺度，但省略第二层 global scale；indexer 使用另一种格式，SWA 则保留 FP8。main KV 读取后反量化再进入 Attention，因此 **缓存是 FP4，不代表 Attention 使用原生 FP4 矩阵指令**。

### 5.1 为什么省略一层 scale 要看实际值域

RMSNorm 后若各通道权重绝对值不超过约 1，512 维向量的 L2 范数约不超过 √512。RoPE 是正交旋转，保持 L2 范数，所以任一旋转后坐标的绝对值也不超过该范数，约为 22.6。报告所选块尺度与 E2M1 的最大幅值乘积为 448×6=2688，有足够的范围。

这是依赖该模型规范化权重与训练观察的论证，不是任意模型的普遍上界。大值域够用也不保证量化误差小：小值分辨率、离群点和后续计算的敏感性仍须看。

main KV 在 RoPE 后量化。不能把量化移动到 RoPE 前然后仅用“旋转保范数”论证等价：一般有 Q(Rx)≠R Q(x)，因为舍入不是线性变换。

## 6. Engram：地址能预先知道，不等于结果与上下文无关

MoE 的专家选择依赖当前 hidden state；Engram 的查表地址只依赖输入 token 序列。因此，不等主干网络算出当前层 hidden，就可以发起 embedding 预取。

Engram 保留 tokenizer compression、多头哈希、上下文感知门控和多分支整合，V4.1 移除了短因果卷积。两处模块共 196B 参数，位于从零计数的第 1、14 层；每处包含 2/3/4-gram，各阶 8 个哈希头，每阶总 embedding 维度 2048。表和相关 key/value 投影使用 FP8。

### 从 tokenizer 状态到门控记忆

真实地址不能从 token 字符串随手拼出。`NgramHashState` 先遍历与模型配套的 tokenizer vocab，用 Rust tokenizer backend 解码每个 ID，再应用 NFKC、NFD、去重音符、lowercase、折叠连续空白、保留单个空格的 sentinel、strip 并还原空格；规范化后等价的 token ID 映射到同一个压缩 ID。含 U+FFFD 的不完整 UTF-8 byte token 则按原始 token 字符串保留。压缩词表大小必须与模型配置一致，因为后续 hash multiplier 的取值界依赖它。每个序列位置的 int64 压缩 ID 写入 `[max_batch,max_seq]` 状态 cache，以 `start_pos` 支持 Prefill 后接 Decode。输入中的图像 span 由 `token_mask=False` 标为 DEAD；向左 look-back 在序列开头或遇到 DEAD 时停止，并以 pad ID 填充，不允许文本 n-gram 穿过图像区域。

令当前位置为 t，压缩 ID 为 $z_t$，每层 l、每个回看距离 q 有一个奇数 multiplier $a_{l,q}$；不同 head 使用不同素数桶，但 multiplier 本身不按 head 增加。对长度 j+1 的 n-gram 先滚动 XOR，再对 head h 的素数取模：

$$
r_{l,j}(t)=\bigoplus_{q=0}^{j} z_{t-q}a_{l,q},\qquad
id_{l,j,h}(t)=offset_{l,j,h}+\left(r_{l,j}(t)\bmod p_{l,j,h}\right).
$$

这里 j=1 对应 2-gram。multiplier 用按 layer ID 派生的确定性随机种子生成，保持奇数且受 int64 乘法上界约束；每个 `(ngram order, head)` 有独立素数桶，offset 令各桶区间在该层表中不重叠。模运算仍会造成哈希碰撞，不能把它误解成无碰撞 perfect hash。Decode 只需把新 ID 写到 cache 的 `start_pos`，即可带着已有历史 ID 计算跨 chunk 的 n-gram 地址。对本报告配置，每个目标位置产生 3 个 n-gram 阶 × 8 个 head = 24 个地址。

地址产生与 embedding 融合是两步。hash ID 可以在主干算出当前 hidden state 前得到，用于向设备本地或 host/RDMA 表请求行；FP8 embedding 与分组 scale 查回后反量化到 BF16，再把 $n_{hash}$ 行展平送进 `wkv`，投影成每条 hyper-connection 残差流各自的 key 和一个共享 value。设流 i 的 hidden、key 为 $x_i,k_i\in\mathbb R^d$，可学习逐通道权重为 $\gamma_{q,i},\gamma_{k,i}\in\mathbb R^d$，权重乘积写作 $w_i=\gamma_{q,i}\odot\gamma_{k,i}$，则 HF Minimal 的 gate 为

$$
ids=Hash(tokens),\quad e=Lookup(ids),\quad (k_1,\ldots,k_n,v)=W_{kv}(e),\quad
X'_i=X_i+Gate(X_i,k_i)\odot v.
$$

$$
r_i=(\operatorname{mean}(x_i^2)+\epsilon)^{-1/2}(\operatorname{mean}(k_i^2)+\epsilon)^{-1/2},\quad
u_i=\frac{\sum_c x_{i,c}w_{i,c}k_{i,c}}{\sqrt d}r_i,\quad
g_i=\sigma(\operatorname{copysign}(\sqrt{\max(|u_i|,10^{-6})},u_i)),\quad
x'_i=x_i+g_i v.
$$

数据流式的最后一式是便于追踪依赖的概括；前面的 $r_i,u_i,g_i$ 才是参考实现的门控细节。每个 token/残差流沿 d 归一化，不把多个 hc copy 合并成一个统计量；masked 位置的 gate 设为零。上下文改变 $x_i$，即使查到同一 n-gram value 也会得到不同注入量。真正的读路径还要处理 embedding row 的分片、FP8 scale、远端完成事件与多请求去重。最小实现中每个 rank 只读自己拥有的 row；非本地 ID 先映射安全下标、对应值清零，all-reduce 后才获得完整 lookup。这个 all-reduce 只说明参考实现的语义路径，不等于大规模线上 host-RDMA 的传输策略。

预取的条件也可以准确表述：在 hash 地址已知时发起异步请求；若 embedding 可读时间不晚于当前 Engram 消费点，就能隐藏等待；否则消费点必须 wait。host 表、GPU 常驻表、模型/请求版本、重复地址合并和 RDMA 排队会改变完成时间。下列程序用人为小词表验证压缩 ID、跨 Prefill/Decode 的 hash 状态、lookup、上下文门控与预取等待关系；映射、multiplier、素数和向量全是玩具值，不是作者 checkpoint 参数。

~~~python
def compress_with_map(token_ids, compressed_map):
    return [compressed_map[token] for token in token_ids]


def append_hashes(cache, position, pad_id, dead_id, multipliers, primes, offsets):
    # State-faithful sketch: one bucket for each (order, head); not production code.
    ngram_count, heads = primes.shape
    looked_back = []
    blocked = False
    for shift in range(ngram_count + 1):
        source_pos = position - shift
        source = dead_id if source_pos < 0 else cache[source_pos]
        blocked = blocked or source == dead_id or source_pos < 0
        looked_back.append(pad_id if blocked else source)
    result = []
    for order in range(1, ngram_count + 1):
        for head in range(heads):
            acc = looked_back[0] * multipliers[0]
            for shift in range(1, order + 1):
                acc ^= looked_back[shift] * multipliers[shift]
            result.append(offsets[order - 1, head] + acc % primes[order - 1, head])
    return result


def context_gate(residual, key, gamma_q, gamma_k, eps=1e-6, token_mask=None):
    # HF Minimal gate arithmetic, applied to one synthetic residual stream.
    rstd = 1.0 / np.sqrt(np.mean(residual**2, axis=-1) + eps)
    rstd *= 1.0 / np.sqrt(np.mean(key**2, axis=-1) + eps)
    weight = gamma_q * gamma_k
    dot = np.sum(residual * weight * key, axis=-1) * rstd / np.sqrt(residual.shape[-1])
    signed_root = np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)
    gate = 1.0 / (1.0 + np.exp(-signed_root))
    if token_mask is not None:
        gate = np.where(token_mask, gate, 0.0)
    return gate


def consume_memory(residual, key, value, gamma_q, gamma_k, token_mask=None):
    gate = context_gate(residual, key, gamma_q, gamma_k, token_mask=token_mask)
    return residual + gate[..., None] * value


def prefetch_wait(request_ready_ms, consumer_ready_ms):
    return max(0.0, request_ready_ms - consumer_ready_ms)
~~~

对一个固定 hash ID，改变 residual 会改变上面的教学 gate；请求完成时间 3 ms、消费点 5 ms 时等待为 0，完成时间 7 ms 则还要等 2 ms。这个算例不把延时映射成带宽，也不是线上测量。

完整 CPU 状态检查：

~~~bash
python roadmap/curriculum/papers/examples/v41_engram_state.py
~~~

程序使用合成 tokenizer map、multiplier、素数与门控权重验证地址状态机和 HF Minimal gate 算术；它不验证训练出的 gamma、embedding、wkv 参数或实际 tokenizer/checkpoint。

<!-- source-check: examples/v41_checks.py -->
~~~python
def ngram_addresses(tokens, order, table_sizes):
    # Deliberately simple teaching hash; NOT the author's hashing algorithm.
    tokens = [int(t) for t in tokens]
    if order < 1 or len(tokens) < order or any(t < 0 for t in tokens):
        raise ValueError("nonnegative tokens and a complete n-gram required")
    ids = []
    for head, size in enumerate(table_sizes):
        if size <= 1:
            raise ValueError("table size must exceed one")
        value = head + 1
        for token in tokens[-order:]:
            value = (value * 257 + token + 1) % size
        ids.append(value)
    return ids
~~~

这个简单哈希只说明“相同后缀 n-gram → 相同地址”与多头表索引，不模拟作者 tokenizer、哈希常量、表大小或碰撞处理。即使地址已知，取回的记忆怎样与当前 hidden state 融合仍受上下文门控影响；不能说相同 n-gram 在所有上下文中贡献完全相同的输出。

设本批次去重后查表字节为 $B_{\mathrm{lookup}}$，有效链路带宽为 BW，预取发起到消费之间有 Δt 可隐藏时间。一个非常粗略的等待下界是：

$$
t_{\mathrm{wait}}\gtrsim
\max\left(0,\ t_{\mathrm{latency}}+
\frac{B_{\mathrm{lookup}}}{BW}-\Delta t\right).
$$

随机访问、远端排队、FP8 scale、分片通信和重复请求会改变它。确定性地址给了预取机会，不保证传输总能隐藏。报告区分了服务推理中从 host 经 RDMA（Remote Direct Memory Access，远程直接内存访问）预取和 RL rollout 中表驻留 GPU 的放置方式，不能把其中一种写成 Engram 的唯一运行方式。

## 7. DSpark：接受率必须和验证成本一起看

DSpark 使用三个 Transformer block，滑动窗口为 128。报告配置的一次 drafter 前向为五个草稿位置产生基础 logits，再通过轻量 Markov head 建模草稿 token 之间的依赖。它不是“用五个互不相关的 argmax 同时替代五次自回归生成”。固定 HF Minimal 实现把目标模型选定层的 hidden states 拼接后投影成 draft 输入；在 `block_size=k` 个位置中用 noise token 初始化，首位替换为当前输入 token。draft block 输出 hidden `[B,k,D]` 与基础 logits `[B,k,V]`。

最后一层的 Markov head 把当前位置 token ID 映射到 rank-r embedding，再投影成词表 bias `[B,V]`；每个位置把该 bias 加到基础 logits 后采样下一候选 token，供下一个位置的 Markov head 使用：

$$
z_i=z_i^{base}+W_M E(d_i),\qquad d_{i+1}\sim\operatorname{softmax}(z_i/\tau).
$$

所以条件依赖沿候选链传递，而不是对所有候选做彼此独立的采样。confidence head 将该位置的 draft hidden 与 rank-r Markov embedding 拼接后投影为一个标量，用于估计验证收益。这个低秩 Markov logit bias 与 confidence 分数是两个不同输出，不能把 confidence 当作 token 概率或 verifier 的接受决定。

设输入上下文末尾的 hidden state 为 $h_t$，词表大小为 V，草稿长度 $k=5$。草稿网络先对 128-token 局部历史计算新的表征，通过 base head 形成各位置分布，再由 Markov head 建模候选之间的条件关系。张量账本可写成：上下文 hidden `[B,T,D]`，局部窗口输入 `[B,min(T,128),D]`，每个位置的基础 logits `[B,k,V]`，候选 token `[B,k]`；Markov 分支给出对候选序列的条件转移/评分，而不是把这 k 个位置当作独立随机变量。精确的隐藏层宽度、head 参数与分支张量以匹配 checkpoint 配置为准，不由 `k=5` 推出。

验证时目标模型接收真实前缀和草稿候选，在因果条件下并行计算每个位置的目标分布。Greedy 示例只需逐位比较候选与目标 argmax，提交最长相同前缀；首次不匹配时提交目标 token 替代候选，并丢弃该位置之后已计算的候选后缀。目标 KV 只保留到已接受候选对应的末尾，草稿 KV 也回滚到一致边界；随后把纠正 token 作为新上下文继续。随机采样场景不能用 argmax 相等代替 speculative sampling 的概率接受/拒绝校正。

```text
context KV length = T
draft proposes d[0:5] and temporary draft KV
target verifies d[0:5] in one causal block forward
if d[0]..d[a-1] match: commit only that accepted prefix
if d[a] mismatches: emit target correction at a; discard draft suffix d[a:]
rollback both temporary caches to the agreed prefix boundary
continue from committed context; no rejected suffix becomes user-visible output
```

例如草稿 `[a,b,c,d,e]`、目标 greedy 结果 `[a,b,X, ...]`：只接受 `a,b`，输出序列再接 `X`；`c,d,e` 和其临时 cache 页全部作废。若目标与五个草稿均一致，则本周期可再取目标模型的 bonus token（若输出长度、EOS 与预算允许）。这一过程区分“生成候选”“验证”“提交可见 token”“回滚暂存状态”四个阶段；只计 draft tokens/s 会把未提交工作也算成有效输出。

confidence head 预测的是逐位置条件接受概率。令 $A_i$ 表示第 i 个草稿被接受：

固定版本的 `model.py` 暴露 `forward_spec` 来产生候选 ID、logits 和 confidence；仓库内的小型 smoke 配置用 `block_size=6`、两个目标层，只检查 draft 张量前向，不是报告五候选的部署形状，也没有目标模型逐位置验证、概率校正、提交可见 token 和回滚临时 KV 的 serving loop。上文的验证/提交链是应单独验收的状态机，不应把 smoke 的 `forward_spec` 返回值报成一次投机接受轨迹。

$$
p_i=P(A_i\mid A_1,\ldots,A_{i-1}),\qquad
q_i=P(A_1,\ldots,A_i)=\prod_{j=1}^{i}p_j.
$$

$q_i$ 是前缀存活概率，来自概率链式法则，不需要额外假设各位置独立。若 $L_{\mathrm{acc}}$ 是验证 k 个草稿后连续接受的草稿数：

$$
E[L_{\mathrm{acc}}]=\sum_{i=1}^{k}P(L_{\mathrm{acc}}\geq i)
=\sum_{i=1}^{k}q_i.
$$

<!-- source-check: examples/v41_checks.py -->
~~~python
def prefix_survival(conditional_acceptance):
    p = np.asarray(conditional_acceptance, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite conditional probabilities in [0,1] required")
    return np.cumprod(p)
~~~

p=[0.9,0.8,0.5] 时，q=[0.9,0.72,0.36]，期望接受数为 1.98，不是 p 的和 2.2。第 3 个位置再可靠，也必须以前两个位置都存活为前提。

若每个验证周期还能提交一个纠正或额外 token，且没有 EOS、输出预算和容量截断，可用下式作吞吐代理：

$$
\mathrm{rate}(k)=
\frac{1+\sum_{i=1}^{k}q_i}{t_{\mathrm{cycle}}(k)}.
$$

cycle 必须包含草稿、验证、提交及实际要算入的调度成本。k=0 是关闭投机的基线，而不是免费执行。

<!-- source-check: examples/v41_checks.py -->
~~~python
def choose_length(conditional_acceptance, cycle_ms):
    survival = prefix_survival(conditional_acceptance)
    cost = np.asarray(cycle_ms, dtype=np.float64)
    if cost.shape != (len(survival) + 1,) or not np.isfinite(cost).all() or np.any(cost <= 0):
        raise ValueError("positive measured cycle cost for k=0..K required")
    # Assumes one correction/bonus token per cycle, no EOS/budget truncation.
    committed = 1.0 + np.r_[0.0, np.cumsum(survival)]
    rate = committed / cost
    return int(np.argmax(rate)), committed, rate
~~~

同样的接受概率，若 k=0..3 的周期时间为 [1,1.1,1.2,1.3] ms，较长验证有利；若负载下变成 [1,2,3,4] ms，就可能全部关闭。这里是解释调度取舍的单请求代理；真实调度要结合其他请求、batch 组成和引擎吞吐曲线，不能把这个 argmax 当作作者调度器的复现。

DSpark 在骨干预训练后单独训练，骨干冻结；后训练阶段两者继续演进，但 DSpark 目标的梯度不反传骨干。这与从骨干预训练起就联合训练的 MTP 不是同一安排。精确采样分布是否保持，还须检查草稿的实际条件分布和接受/拒绝校正，不由 confidence head 的输出自行保证。

## 8. Single-Pass mHC：不是把三个 kernel 强行拼起来

mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）维护 $n$ 条残差流 $X_l\in\mathbb R^{n\times d}$，原形式为：

$$
X_{l+1}=B_lX_l+C_lF_l(A_lX_l),
\qquad (A_l,B_l,C_l)=\mathcal H(X_l).
$$

$A_l$ 的系数预测使用 $X_l$ 的全 hidden 维统计量，必须等相关 hidden-dimension tiles 的归约完成；这里的归约是为了产生一个 `[1,n]` pre mix，不表示输入混合本身还保留 stream 输出轴。

Single-Pass 将输入混合使用的系数滞后一层：

$$
X_{l+1}=B_lX_l+C_lF_l(A_{l-1}X_l).
$$

$A_{l-1}$ 已由上一 residual update 产生，因此当前 $F_l$ 输入不依赖尚在归约的本 update $A_l$；$A_l,B_l,C_l$ 仍由当前 $X_l$ 预测，供后续 residual update 使用。**这里先改变了模型依赖，再获得融合机会，不是原方程的纯等价编译优化。**

这里的 $l$ 是一次残差更新/子层编号，而不是 Transformer block 编号；一个 HF Transformer block 有 attention 与 FFN 两个子层，因此各自发生一次 mHC 残差更新，Single-Pass 的系数滞后一个这样的更新。

在一个 token 上，$X_l\in\mathbb R^{n\times d}$。三种混合系数形状不同：pre $A_l\in\mathbb R^{1\times n}$，residual $B_l\in\mathbb R^{n\times n}$，post $C_l\in\mathbb R^{n\times1}$。因此 $A_lX_l$ 与 $F_l(A_lX_l)$ 都是 $[1,d]$，$B_lX_l$ 和 $C_lF_l(A_lX_l)$ 都是 $[n,d]$。扩展到 batch 和序列轴后，$X:[B,S,n,d]$，$A:[B,S,1,n]$，$B:[B,S,n,n]$，$C:[B,S,n,1]$。输入混合不保留 stream 输出轴 i：

$$
(A_{l-1}X_l)_{b,s,1,c}=\sum_{j=0}^{n-1}(A_{l-1})_{b,s,1,j}(X_l)_{b,s,j,c}.
$$

实际融合边界由下面固定版 API 的输入/输出形状界定：`x:[T,H]` 已在 API 外算好，API 同时接收 residual streams 与 pre/post/comb mixing state，并写出更新后的状态及归一化/可选量化输出。

固定 DeepGEMM API 注释给出的边界比 tile 示意更直接：`x:[T,H]` 是子层已经计算出的 BF16 张量，`residual:[T,n,H]` 是 n 条 residual streams；`shifted_prev_mix:[T,n,1]` 可选，`post_mix:[T,n,1]` 与 `comb_res_mix:[T,n,n]` 是 FP32。`fn` 是系数预测投影权重，不是 Transformer 子层 $F_l$。调用输出为 `new_residual`、新的 prev/post/comb mixes，以及可选的 `y_bf16` / `y_fp8` 与对应 scale。因调用不接收注意力/FFN 的权重或可调用对象，Mega mHC 的融合边界不包括计算 $F_l$ 的 GEMM/Attention；它处理的是已完成子层张量周围的 mHC 更新、mix 生成、RMSNorm 与可选 FP8 转换。

```cpp
// Focused signature excerpt; normalization and scale arguments are elided.
mega_mhc(x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn,
          mix_scales, mix_bases, hc_mult, /* norm / Sinkhorn parameters */,
          rmsnorm_weight, /* rmsnorm parameters */,
          new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
          y_bf16, y_fp8, /* scale-factor outputs */);
```

实际张量中，数学 pre row vector $A_{l-1}:[1,n]$ 以列向量存储为 `shifted_prev_mix[T,n,1]`；数学 post column $C_l:[n,1]$ 对应 `post_mix[T,n,1]`。`comb_res_mix[T,n,n]` 的第一轴是 source，第二轴是 destination，所以按通常 $X'=BX$ 写法，数学 $B_l=comb\_res\_mix^T$；shape 方阵相同并不表示方向相同。源码还强制 shifted state 成对出现：

~~~cpp
const bool is_shifted = shifted_prev_mix.has_value();
DG_HOST_ASSERT(is_shifted == new_prev_mix.has_value());
~~~

因此 normal mHC 的输入和输出 `prev_mix` 都为 `None`；Single-Pass shifted mHC 必须同时提供前一 residual update 的 `shifted_prev_mix` 并接收下一 update 的 `new_prev_mix`。一个 HF Transformer block 的 attention 与 FFN 是两个独立 residual updates；索引 $l-1\to l$ 指这两个 mHC update 之间的一步，而非跨整个 Transformer block 的滞后一层。

HF Minimal 的 block 接口用 `hc_pre` / `hc_post` 明示两次 stream 变换。若 $X\in\mathbb R^{B\times S\times n\times d}$、`pre_mix` 为 $[B,S,n]$，`hc_pre` 对 n 条 stream 加权求和得到普通 block 输入 $Y\in\mathbb R^{B\times S\times d}$；block 输出后，`hc_post` 用 `post` 将 Y 广播回 n 条流，并用 `comb\in\mathbb R^{B\times S\times n\times n}` 混合残差：

$$
Y_{b,s,c}=\sum_{source=0}^{n-1}pre_{b,s,source}X_{b,s,source,c},\qquad
X'_{b,s,dest,c}=post_{b,s,dest}Y_{b,s,c}+\sum_{source=0}^{n-1}comb_{b,s,source,dest}X_{b,s,source,c}.
$$

HF 的原式明确了 comb 的轴次序：

~~~python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
    comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
)
~~~

`residual.unsqueeze(-2)` 保留的 stream/source 轴在 dim=2；求和后剩下 comb 的最后一轴 destination。因此按常见左乘记法 $X' = BX$，有 $B=comb^{T}$。直接写成 $X'_i=\sum_j comb_{i,j}X_j$ 会转置错方向。

DeepGEMM 固定 kernel 也按 `[source,destination]` 读 comb：

~~~cpp
const uint32_t comb_idx = input_route_idx * kNumRoutes + output_route_idx;
post_value = __ffma2_rn(residual[row_idx][input_route_idx], {coeff, coeff}, post_value);
~~~

`input_route_idx` 是被求和的 source，`output_route_idx` 是累加到的 destination。它的 `test_mega_mhc.py` 把 Mega 输出与 `ref_mhc.mhc_reference` 和 baseline 逐项比较；这是测试定义，不是本机 GPU 已通过的结果。

DeepGEMM API 对 shifted FP8 保留 rounded BF16 数值边界；只请求 FP8 且启用 shifted mode 时会用 scratch 暂存 BF16：

~~~cpp
const bool needs_bf16_scratch = not y_bf16.has_value() and is_shifted;
const auto num_bf16_scratch_bytes = needs_bf16_scratch
    ? static_cast<int64_t>(num_tokens) * hidden * x.element_size() : 0;
// Shifted FP8 is derived from rounded BF16.
~~~

因此“融合”不等于所有中间物化都为零。固定测试的流量账本把 `2 × bytes(y_bf16)` 计为中间 I/O，并在 normal mode 额外计一次 residual reread；这是源码中的静态账本与测试逻辑，不是本机跑出的性能。

HF Minimal 的 `hc_pre` / `hc_post` 是展示同一语义的分步模块，不是 Mega mHC kernel，也不验证 DeepGEMM 的设备执行。报告列出的 $(4n+4)d$、$(3n+2)d$、$(2n+2)d$ 是不同 mHC 实现边界的 activation 读写模型；$n=4$ 时为 $20d,14d,10d$。它不包含权重、所有 scale、scratch 的实际事务、寄存器 spill 与 allocator 开销，不能改写成“实测零中间量”或目标设备吞吐。

## 9. Sinkhorn 更新：归一化的是更新矩阵

大 embedding 表若使用 Adam，会维护一阶、二阶状态。报告对 Engram、token embedding 和预测头采用带动量的 Sinkhorn-balanced update，保留一份动量状态；线性层使用 Muon，部分 Query/Key 权重按 head 处理，非矩阵参数仍有 AdamW 路径。它不是“整个模型换成一个优化器”。

对 $W,G,M\in\mathbb R^{m\times n}$，先更新动量，再构造 Nesterov 方向：

$$
M_t=\beta M_{t-1}+(1-\beta)G_t,\qquad
\widehat G_t=\beta M_t+(1-\beta)G_t.
$$

令 $\rho_i=\|\widehat G_{t,i,:}\|_2$。接近零的行先屏蔽，避免把极小噪声放大：

$$
U^{(0)}_{i,:}=
\begin{cases}
0,&\rho_i\leq\tau\overline\rho,\\
\widehat G_{t,i,:},&\text{otherwise}.
\end{cases}
$$

然后进行 K 次交替归一化：奇数步对每行作 L2 归一化，偶数步对每列归一化。K 取奇数，让最后一步落在行上。最后：

$$
\Delta_t=\sqrt n\,U^{(K)},\qquad
W_{t+1}=W_t-\gamma\eta_t\Delta_t.
$$

<!-- source-check: examples/v41_checks.py -->
~~~python
def sinkhorn_step(weight, gradient, momentum, beta=0.95, lr=1e-3,
                  gamma=0.18, steps=11, tau=1e-3, eps=1e-20):
    w, g, old_m = [np.asarray(x, dtype=np.float64)
                   for x in (weight, gradient, momentum)]
    if w.ndim != 2 or w.shape != g.shape or w.shape != old_m.shape:
        raise ValueError("W, G and M must have equal matrix shapes")
    if steps < 1 or steps % 2 != 1 or eps <= 0 or tau < 0 or not 0 <= beta < 1:
        raise ValueError("odd positive steps and valid numerical constants required")
    if not all(np.isfinite(x).all() for x in (w, g, old_m)):
        raise ValueError("finite inputs required")
    m = beta * old_m + (1 - beta) * g
    g_hat = beta * m + (1 - beta) * g
    rho = np.linalg.norm(g_hat, axis=1)
    active = rho > tau * rho.mean()
    u = g_hat.copy()
    u[~active] = 0.0
    for k in range(1, steps + 1):
        axis = 1 if k % 2 else 0
        u = u / (np.linalg.norm(u, axis=axis, keepdims=True) + eps)
    delta = np.sqrt(w.shape[1]) * u
    return w - gamma * lr * delta, m, delta, active
~~~

为什么有 √n？最后一次归一化使活跃行的 L2 范数约为 1；乘 √n 后，该行均方根为 √(n/n)=1。被屏蔽的零行保持零，不应强行除到 RMS=1。有限迭代只近似平衡列方向；稀疏支撑和零行也影响列 RMS，不能同时宣称所有行列精确满足目标。

它处理有正负号的更新方向，不是把权重变成行列和为 1 的概率矩阵，也不是 Muon 的正交化。11 次是报告配置，不是适用于所有矩阵的收敛定理。生产实现还避免每次写回完整 U，用行列尺度向量与融合统计减少流量；这里的 NumPy 版本优先显示每一步数学。

### 从语料组织到一次优化器更新

预训练配置同时决定模型学到了什么，以及机器上实际处理了多少数据。V4.1 报告中的文本与多模态数据先走不同管线，再合并、去重与 packing；不能只记一个总 token 数，把数据处理成本和样本身份省略掉。

| 阶段 | 处理对象 | 关键决定及其影响 |
|---|---|---|
| 文本筛选 | 文档、代码和其他文本样本 | 按质量与信息增益筛选；过滤低信息增益的生成文本或低质量翻译，不等于所有合成数据都不该使用 |
| 多模态筛选 | 图文对、交错图文和专业数据 | 先进行较便宜的文档筛选，再取图、去重和做更昂贵的图文质量判断，减少无效 I/O 与视觉处理 |
| 合并与去重 | 同一内容的纯文本和多模态版本 | 重叠样本用多模态版本替换；使用两种配置中较大的 epoch 数，而不是把重复曝光次数直接相加 |
| Packing 与长文档分片 | 不同长度的样本和超长文档 | 提高 token 槽位利用率，同时保留样本边界；不能为了少 padding 让一个样本看到另一个样本的内容 |

报告采用纯文本与多模态数据的 **7:1 token 比例**，不是图像数比例或样本条数比例；视觉 token 的数量还受图像分辨率、patch 和 unshuffle 影响。报告的 best-fit packing 将 padding 比例压到不超过 $10^{-4}$，但低 padding 只说明槽位利用率，不证明数据质量更好或有效学习信息同比增加。

假设把长度 2 与长度 3 的两个样本放进同一个长度 5 的输入，样本实例 ID 为 `[0,0,1,1,1]`。普通全序列 causal mask 会允许第 2 个位置看到位置 0、1，造成跨样本泄漏。sample-level causal mask 必须同时检查位置和样本归属：

$$
\operatorname{visible}(q,k)=(k\le q)\land(\operatorname{sample}(k)=\operatorname{sample}(q)).
$$

这里的 ID 标识一次独立样本实例，不是可在不同样本间复用的文档类别。实际稀疏实现不必构造完整的平方矩阵，但被选中的 KV 位置仍必须满足相同边界。数据 loader、索引器与 Attention 对样本边界理解不同，会使“模型能正常训练”与“训练语义正确”分离。

参数也不是仅按维数统一分给一个优化器。报告的分组如下：

| 参数职责 | 优化路径 | 需要注意的区别 |
|---|---|---|
| 语言骨干、Engram 投影与视觉语言 projector 的线性矩阵 | Muon；Q/K 权重采用 head-wise 处理 | 逐 head 的预条件方式不同，不能把所有 Q/K head 当成一个同质矩阵 |
| 大 embedding 表与预测头 | 带 Nesterov 动量的 Sinkhorn-balanced update | 保留一份动量；不施加 weight decay；报告对 Engram 使用 5 倍学习率 |
| RMSNorm 权重与其他非矩阵参数 | AdamW | Norm 权重使用 weight decay，bias 和 scaling factors 不使用；不要只按张量是否二维分组 |

视觉编码器先单独训练，再接入语言骨干；联合预训练开始阶段冻结视觉编码器主体，但最终 norm 与 projector 可训练，学习率衰减阶段再解冻视觉编码器并用较小学习率。因而“多模态数据从语言骨干预训练开始加入”并不意味着视觉编码器所有参数从第一步起都在更新。

学习率按报告的训练过程分段。全局 batch 为 100.6M tokens，前 2000 steps 线性 warmup；之后保持 $2.6\times10^{-4}$ 到 28T tokens，在 28T–40T 间按 cosine 降到 $2.6\times10^{-5}$，最后保持到 45T。上下文长度在 34T 从 64K 扩展到 1M，发生在衰减区间内，而不是一次新的从零训练。

设累计 token 数为 $t$，$t_w=2000\times100.6\mathrm{M}$，在固定 batch 的连续计数近似下，衰减段可写成：

$$
\eta(t)=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\frac{t-28\mathrm{T}}{12\mathrm{T}}\right)\right],\quad 28\mathrm{T}<t<40\mathrm{T}.
$$

34T 位于这个区间正中，因此该式给出 $1.43\times10^{-4}$。这是配置算式，不是重新取得的训练日志；实际复现仍须核对 step、有效 token、padding 与梯度累积的计数口径。以下 CPU 程序检查分段边界和跨样本隔离，不加载模型或训练数据。

<details>
<summary>展开学习率与 packing 边界检查</summary>

<!-- source-check: examples/v41_training_contracts.py -->
~~~python
"""CPU arithmetic for the reported schedule and packed-sample attention mask."""
import math

BATCH_TOKENS = 100_600_000
WARMUP_TOKENS = 2_000 * BATCH_TOKENS
PEAK_LR = 2.6e-4
FLOOR_LR = 2.6e-5


def learning_rate(tokens_seen):
    if type(tokens_seen) is not int or not 0 <= tokens_seen <= 45_000_000_000_000:
        raise ValueError("tokens_seen must be an integer within the reported 0..45T run")
    if tokens_seen < WARMUP_TOKENS:
        return PEAK_LR * tokens_seen / WARMUP_TOKENS
    if tokens_seen <= 28_000_000_000_000:
        return PEAK_LR
    if tokens_seen < 40_000_000_000_000:
        phase = (tokens_seen - 28_000_000_000_000) / 12_000_000_000_000
        return FLOOR_LR + 0.5 * (PEAK_LR - FLOOR_LR) * (1 + math.cos(math.pi * phase))
    return FLOOR_LR


def packed_causal_visible(sample_ids, query, key):
    if not (type(query) is int and type(key) is int
            and 0 <= query < len(sample_ids) and 0 <= key < len(sample_ids)):
        raise ValueError("query and key must index the packed sequence")
    return key <= query and sample_ids[key] == sample_ids[query]


def main():
    assert WARMUP_TOKENS == 201_200_000_000
    assert learning_rate(0) == 0
    assert learning_rate(WARMUP_TOKENS) == PEAK_LR
    assert learning_rate(28_000_000_000_000) == PEAK_LR
    assert math.isclose(learning_rate(34_000_000_000_000), 1.43e-4, abs_tol=1e-16)
    assert learning_rate(40_000_000_000_000) == FLOOR_LR
    assert learning_rate(45_000_000_000_000) == FLOOR_LR
    for bad in (-1, 45_000_000_000_001, 1.5, True):
        try:
            learning_rate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid schedule input accepted")
    samples = [0, 0, 1, 1, 1]
    visible = [[k for k in range(5) if packed_causal_visible(samples, q, k)]
               for q in range(5)]
    assert visible == [[0], [0, 1], [2], [2, 3], [2, 3, 4]]
    assert 0 <= 2 and not packed_causal_visible(samples, 2, 0)
    print("PASS: warmup/cosine/floor boundaries, midpoint LR, packed-sample isolation")
    print("Continuous-token arithmetic proxy; no training run or quality measurement")


if __name__ == "__main__":
    main()
~~~
</details>

~~~bash
python roadmap/curriculum/papers/examples/v41_training_contracts.py
~~~

## 10. 跨层共享到了训练阶段，会多出哪些责任

前向只读共享 KV 看起来简单，训练还需要把所有消费者的贡献送回正确参数：

| 对象 | 必须保留的信息 | 错误后果 |
|---|---|---|
| 共享参数 | 单一逻辑 owner、物理 shadow replicas、参数版本 | 重复优化或副本不同步 |
| KV 与索引中间态 | microbatch、来源层、CP 分片范围 | 不同样本或分片混用 |
| 梯度 | 各消费者贡献、聚合完成条件 | 少算或重复累加 |
| 生命周期 | 前向、重计算、反向的最后消费者 | 提前释放或显存长期滞留 |

报告用 shadow indexer 在参与的 pipeline stage 上保留轻量执行副本，但 owner 负责优化与 checkpoint。中间状态随 pipeline 点对点 payload 传递，并与 CP（Context Parallelism，上下文并行）分片一致。共享的是语义对象，不要求所有物理副本都放在同一设备。

对共享权重 W，若多个消费者损失相加，梯度也应相加：

$$
\frac{\partial\mathcal L}{\partial W}
=\sum_l\frac{\partial\mathcal L_l}{\partial W}.
$$

activation recomputation 可能重新执行前向，但不能因此把同一反向贡献额外加一次。用只读缓存推理测试，无法证明这条训练路径正确。

把它落到一次训练 step，可给共享状态分配一个完整身份：`(step_id, microbatch_id, source_layer, CP_rank, token_range, parameter_version)`。producer 只在自己的 token/CP 分片上写一次 global KV 或 index K；消费者拿到带版本和分片范围的引用，shadow indexer 做前向但不是第二份 optimizer owner。pipeline P2P 发送 tensor 和元数据后，在接收端记录 completion event；异步通信完成前不能让 consumer 读取，也不能因 host 已提交 send 就释放 buffer。反向阶段按已登记的消费者集合收集梯度贡献，owner 在全部贡献到齐后求和并应用一次更新；checkpoint 保存 owner 参数/动量及对应版本，再将新版本广播到 shadow replica。

可把反向的精确合同写为：

$$
G_{owner}=\sum_{c\in Consumers(step,microbatch)}G_c,
\qquad W^{v+1}=Optimizer(W^v,G_{owner}).
$$

如果 checkpoint recompute 再执行 producer/consumer 前向，它是同一计算图中的另一次 forward invocation，不应凭重算本身重复累加同一条 consumer gradient；若 pipeline 由于重试确实产生另一份有效样本，则它必须有不同 sample/microbatch ID。异步 all-reduce 的 buffer 还要绑定 dtype、shape、CP shard range 和对应 step，避免把上一轮尾部通信混到下一轮梯度中。

异步 OPD（On-Policy Distillation，同策略蒸馏）也必须保留行为版本，而不是只留下 prompt 与最终文本。一个最小 rollout record 至少包含 `group_id`、`sample_id`、`policy_version`、动作 token IDs、逐 token `old_logp`、loss mask、环境/验证器结果和终止原因。worker 可乱序完成，但训练端将结果归回原 prompt group 后再计算组内信号；版本年龄超出明确阈值时，系统只能选择接受陈旧样本、降低权重、等待刷新或丢弃/重采，不能把当前 checkpoint 的概率伪装成生成时的 `old_logp`。因此“异步”改变的是生产与消费调度，不是样本身份、分组语义或策略分母。

视觉路径也有自己的数据预算。3×3 pixel-unshuffle 在可整分的特征网格上把空间位置减为原来的 1/9、通道扩大为 9 倍，再由 projector 映射到语言 hidden dimension。它是重排，不等于凭空丢弃 8/9 原始特征元素；减少的是送入语言骨干的 token 数。图像多、CPU 解码慢时，GPU 算子快并不能消除输入流水线瓶颈。

## 11. Persistent KV 与重放：缓存命中不是一个布尔量

服务端至少需要区分三种情况：

| global KV | encoder SWA | 处理含义 |
|---|---|---|
| 未命中 | 不可直接依赖 | 生成缺失的 global 与局部状态 |
| 命中 | 命中 | 可以利用对应缓存继续，但仍核对版本和位置 |
| 命中 | 未命中 | 保留命中的 global，重放末尾窗口恢复近似局部状态 |

global 的复用时间分布与活跃会话 SWA 不同。报告把 global 留在长生命周期 persistent cache，把 encoder SWA 放入短 TTL（Time To Live，存活时限）的 host pool；decoder SWA 不进入同样的前缀缓存。72 小时、分钟级 TTL、10% host DRAM 是报告部署策略，不能原样作为任何服务的默认值。

窗口重放从 s 开始时，第 i 个 token 的局部可见集合为：

$$
\mathcal V_i^{\mathrm{replay}}=
\{j:\max(s,i-W+1)\leq j\leq i\}.
$$

W=4、s=100 时，位置 100 不再看到 97、98、99。即使到位置 103，两种路径都看到 100..103，窗口里这些 token 的 hidden/KV 也可能已因前层截断而不同。因而重放是近似恢复，后缀新生成的 global/SWA 也可能随缓存命中位置变化。

不能只做分页地址相等测试；还要比较完整前向与不同命中位置下的输出质量、长程检索、窗口边缘和跨轮恢复。报告自身也把极端稀疏选择和状态恢复列为需要进一步压力测试的边界。

## 12. 后训练：调度、样本分布与参数版本一起变化

V4.1 的后训练沿用 SFT（Supervised Fine-Tuning，监督微调）、RL（Reinforcement Learning，强化学习）和 OPD（On-Policy Distillation，同策略蒸馏）等路线，重点之一是任务和环境的规模化构造。一个训练任务至少包含问题、环境和验证系统。环境失败、测试泄露或奖励可被绕过时，增加 rollout 数量只是在更快地产生有问题的数据。

DSec 是报告中的弹性沙箱系统。它把长生命周期的工具环境和 GPU 训练调度分开，通过分片降低故障影响，节点端执行最终资源准入约束。宽松的全局调度一致性并不表示节点可以不检查内存上限；隔离、轨迹记录和抢占恢复都属于训练数据可靠性的一部分。

### 12.1 sample 粒度派发不改变 GRPO 的分组语义

GRPO（Group Relative Policy Optimization，组相对策略优化）让同一个 prompt 的多条回答构成一个奖励比较组。传统等整组完成再补任务，会被组内最长轨迹拖住。报告在新增完成样本数达到下一 prompt 所需组大小时派发新 prompt，完成样本可以来自不同旧组。

这改变的是补充任务的时机，**不是把不同 prompt 的回答混成一个奖励组**。训练时依然要按正确 group 标识组织优势和统计。

先结束的短样本会影响早期训练批分布，因此报告使用每数据集并发控制，并允许丢弃早返回短样本缓和过渡。它也限制过旧样本的比例，对过度陈旧 token 作 loss masking。吞吐提高、长度分布改变和策略陈旧是三件不同的事。

### 12.2 权重切换后复用状态不是数学恒等式

报告支持 token 边界中断，保留 KV 和 expert routing，跨 checkpoint 恢复，并把各段历史路由拼接用于训练。它描述的是带有陈旧样本处理的后训练机制，不能据此推论“任意更新权重后，旧 KV 都等于新权重重新算出的 KV”。

需要记录每段 token 的策略版本、行为 log probability、历史路由、环境状态和配置版本。相同 prompt 文本不够决定训练样本身份。样本完成后再释放所属状态，防止高并发下长期驻留。

OPD 的 on-policy 通常强调学生在自身生成的上下文上学习教师分布，不是教师必须和学生架构相同。异步生成进一步要求跟踪样本生成时的学生版本；“来自学生”与“恰好来自最新 checkpoint”要分开判断。

## 13. Effort：奖励改变长度倾向，不是硬上限

对推理长度 ℓ，报告加入随 effort b 改变的惩罚：

$$
r_{\mathrm{len}}(\ell,b)=
-\min\left(C_{\max},k(b)\frac{\ell}{L_{\mathrm{norm}}}\right),
\qquad
k(b)=k_0\exp\left(-\frac{b-b_{\min}}{\tau}\right).
$$

ℓ 是推理 token 数，$L_{\mathrm{norm}}$ 让长度项无量纲；$C_{\max}$ 限制最大扣分。b 增加 τ 时，k 乘 e^(-1)，表示相同长度受到更弱惩罚，而不是模型必须多输出某个固定 token 数。

<!-- source-check: examples/v41_checks.py -->
~~~python
def effort_penalty(length, effort, k0=1.0, bmin=25.0, tau=25.0,
                   length_norm=1000.0, cap=4.0):
    if length < 0 or tau <= 0 or length_norm <= 0 or min(k0, cap) < 0:
        raise ValueError("invalid length or penalty configuration")
    k = k0 * np.exp(-(effort - bmin) / tau)
    return -min(cap, k * length / length_norm)
~~~

同一问题在不同 effort 下的回答分成不同子组。组内奖励中心化可写作 $A_{b,j}=r_{b,j}-\frac1{M_b}\sum_{j'=1}^{M_b}r_{b,j'}$（$M_b$ 为子组回答数），不能把不同 b 的回答随意混组，否则长度压力的定义发生变化。

附录的简化解释在未触及惩罚上限时，把偏好长度看成：

$$
\ell^*(b)\in\arg\max_{\ell\geq0}
\left[p_x(\ell)-k(b)\ell/L_{\mathrm{norm}}\right].
$$

若内部最优点满足 $p'_x(\ell)=k(b)/L_{\mathrm{norm}}$，再假定边际收益局部近似 $a\exp(-\ell/s)$，可解出：

$$
\ell^*(b)\approx
s\log\frac{aL_{\mathrm{norm}}}{k_0}
+\frac{s}{\tau}(b-b_{\min}).
$$

这给出一个局部线性趋势的动机，而不是观测平均长度的单调定律。惩罚一旦触顶，边际长度惩罚为零，不能继续沿用内部最优条件。任务难度、采样、工具轮次和奖励归一化都可能改变实际曲线。

## 14. 怎样读实验，而不是只记压缩倍数

至少固定四层条件：模型 checkpoint 与精度；上下文与输出预算；harness、工具权限和最大轮数；指标定义与采样次数。模型不变、harness 改变也可能改变最终成绩，不能把整套系统差异都归给 Attention。

报告区分 backbone 与 Engram 参数，区分 global HBM 缓存与 persistent host/SSD 缓存，区分 Prefill 激活工作与 Decode 激活工作。比较时也必须按对应口径，不把四种数字拼成一个“整体提升”。

阅读稀疏选择、FP4、CED 和重放的质量实验时，应问：是否单独改变一个因素？是否同一数据和训练预算？是否测到长上下文边缘？是否改变了推理预算？没有控制这些条件，就不能从综合模型优于旧版推导出每项设计都独立无损。

多模态数据构造、预训练配比和长上下文扩展也同时变化。报告使用 45T token、文本与多模态 7:1 的配置；这些是这个训练运行的身份，不是适用于所有模型的配方。CPU 示例检查通过，也不能代替这些质量实验。

技术报告没有给出分别移除 CED、CSA2、主 KV FP4、Engram 等组件并报告同预算质量差值的数值消融表。它对 Single-Pass mHC 的精度影响只作“退化可忽略”的定性描述；另一次关于 ablations 的讨论针对小模型的数据传输瓶颈，并非 V4.1 组件质量消融。因此，下列结果只支持模型或评估设置层面的比较，不支持组件因果效应。

### 14.1 Base 表是同一评测设置下的模型比较，不是组件消融

Base 模型对照 Table 1 将 V4.1-Flash-Base 与 V4-Flash-Base、V4-Pro-Base 放在同一内部评测框架中。报告称三者使用相同评测设置，并把绝对差不超过 0.3 的分数视为同一水平；它没有声称三个 checkpoint 只差某一个架构组件。参数规模本身已分别为 284B、1.6T、552B，激活参数口径为 13B、49B、Prefill 8B / Decode 16B。训练数据、原生多模态训练、架构和训练配方均可能共同造成差异。

这张表适合回答“发布的 Base 模型在给定基准上的横向位置”，不适合回答“CED 单独贡献多少”或“FP4 对准确率造成多少影响”。例如 V4.1 在 MMLU-Pro 为 74.1，高于 V4-Flash 的 68.3；在 HumanEval Pass@1 为 79.4，高于 69.5；但 MGSM 为 80.2，低于 V4-Flash 的 85.7。这样的混合结果不能用一个平均数替代，也不能归因为某个单独模块。多模态行只有 V4.1 有分数（MMMU-Pro 56.5、CVBench 77.9、DocVQA 95.6、RefCOCO-avg 86.0），旧模型的短横线表示未列出可比较结果，不是零分。

内部研发语料的 held-out 评估用 bits-per-byte（BPB），越低越好；集合来自内部文档、专有代码库和学术资料，目标是推理、归因与复杂科研问题。数据未公开，所以曲线不能由外部读者独立复算，也不应与公开 benchmark 分数混排。

### 14.2 Instruct 综合成绩必须带上 reasoning effort 与 scaffold

Instruct 模型对照 Table 3 统一使用最高推理 effort，但采样设置依评测类型而异：Evaluation Setup 对 reasoning 设 temperature=1.0、top-p=1.0；代码代理与视觉代理为 temperature=1.0、top-p=0.95。代码代理总体使用 DeepSeek Harness Minimal、1M token 上下文；DeepSWE v1.1 按基准要求改用 mini-SWE；SEC-Bench Pro 用 Claude Code。视觉代理使用 Claude Code 与 512K 上下文，AutomationBench 和 Agent's Last Exam 则使用各自官方 scaffold。因此表内不是单一运行环境。

两个可直接核对的比较是：DeepSWE v1.1 Resolved 为 V4.1 74.2、V4-Flash 54.4；Terminal-Bench 2.1 Pass@1 为 90.6、82.7。它们说明报告中的完整新 checkpoint + 后训练 + 评测栈组合成绩更高，不能证明变化由注意力结构或缓存压缩导致。更广泛的 Table 3 也有非单调结果：例如 HLE Pass@1 为 36.8（括号中的 39.1 是 text-only 子集），而不是所有基准都超过旧版或所有参照模型。安全类表格亦受具体 scaffold 和基准版本约束。

代码代理还通过关闭网络、移除环境中的 Git 历史、清理临时构建及依赖缓存来降低奖励作弊风险；报告同时承认仍观察到利用环境漏洞的行为。复现时，容器和评测器安全边界也是测量条件，不只是模型参数。同页所列跨模型结果存在不同供应方 checkpoint、推理预算、工具与 scaffold 的异质性。供应方有说明时按其说明解释；不能把缺失的评测配置自行补成“完全同条件”。Pass@1 也不等价于模型单次确定性表现：需要同时知道采样数、温度、停止规则和评分器。

### 14.3 Effort sweep 与 scaffold sweep 才能分离哪些因素

Reasoning-effort Figure 9 固定 V4.1 checkpoint，扫描 effort 25–100，展示输出 token 数和准确率。reasoning-intensive 平均 Pass@1 从 67.1% 到 76.3%；DeepSWE v1.1 从 66.0% 到 74.2%；Terminal-Bench 2.1 从 82.4% 到 90.6%，输出 token 约增至 2.5 倍。报告称 60–80 已取得最高档大部分准确率，而 effort 80→100 让 agent 轨迹长度增加约 1.6–1.8 倍、正确率增益较小。reasoning-intensive 汇总由八项基准组成：AIME 2026、Apex 2025 Shortlist、GPQA Diamond、HLE、IMO-AnswerBench、LiveCodeBench、MathArena-Apex、SimpleQA-Verified。这个 sweep 能说明该 checkpoint 内调 effort 的成本—质量关系；它不等于各组件 ablation，也不能外推到不同任务、长度上限或服务负载。

Scaffold Table 4 固定最高 effort、输入窗口和运行规则：DeepSWE 的样本数 N=8，Terminal-Bench 2.1 的 N=3；Linux 容器、temperature=1.0、top-p=0.95、1M 上下文、每个 agent 最多 500 个模型生成轮次，Terminal-Bench 禁止网络。DeepSWE 从 Claude Code 69.8 到 mini-SWE 74.2、DSH Minimal 72.6；Terminal-Bench 从 88.0、90.3 到 90.6。该对照隔离的是同一 checkpoint 在已列明 scaffold 配置下的表现差异，不是对所有模型都成立的 scaffold 排名。

Claude Code 版本敏感性 Table 5 对四个版本重复评估，DeepSWE 分数 68.4、68.7、69.8、68.6，未舍入值平均后报告 68.9；Terminal-Bench 2.1 为 87.3、88.4、88.0、87.6，平均 87.8。这给出工具版本敏感性的实际量级，也提醒单版本分数有测量噪声与集成变动。

### 14.4 多 Agent 数字是初步、预算限定的系统实验

Agent Team 扩展 Figure 10 在 ProgramBench 的 172 个高置信任务与 FrontierSWE v2 的公开 no-GPU 子集比较单 Agent 和 Agent Teams。ProgramBench 每配置最多三次 rollout，即计划 516 次；指标 Almost@1 表示单个 rollout 得分至少 0.95 的比例，时间预算为 1–12 小时。FrontierSWE v2 报 Mean@5，预算为 1–20 小时。报告称它比较“观察到的最强 multi-agent 配置”和“可用的最强 single-agent 基线”，并明确称结果 preliminary；所以它不是严格同算力、同 token 预算的单因素实验。图上多 Agent 在 ProgramBench 从 1 小时 13.59% 到 8 小时 30.04%，单 Agent 对应 12.79% 到 20.39%；FrontierSWE v2 在 20 小时为 32.90% 对 28.20%。这些数值只适用于报告指定任务集、截止时间和指标，不能解释为一般并行化收益。

复核这类曲线时，应分别记录每个截止点可用的 rollout 数、工具执行时长、生成 token、并行 agent 数、失败/超时处理与统计单位。墙钟截止时间相同，不代表累计 GPU、token 或人机工具成本相同。

### 14.5 从作者最小推理实现核对数据流

作者仓库的 `inference/` 提供可读的最小推理参考实现，覆盖视觉编码、滑窗与压缩稀疏注意力、Engram、MoE、Hyper-Connections 和 DSpark 前向路径；它不是生产 serving engine。该目录配置给出 40 层、窗口 128、最终 Top-K 512；KV 来源层是 `[2, 8, 14, 20]`，索引来源层是 `[2, 8, 14, 20, 24, 28, 32, 36]`。候选来源层 20 使用 2048 个块、每块 8 个位置，候选池上限为 16384；它与最后实际选入注意力的 Top-K=512 是两级预算。配置中的 layer ID 是零起始索引。

压缩算子在 `inference/model.py` 的 `Compressor.forward` 中对组内 token 做门控加权。核心语句是：

~~~python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
~~~

`score.softmax(dim=2)` 把每个压缩组内部的权重归一化；逐元素乘法后沿同一组维求和，得到一个压缩后的 KV latent。Decode 中未填满的组保存在状态里，待组完整后再输出，因此压缩倍率也决定何时新增一个 global cache entry。

`inference/model.py` 的 Reindex 路径先取本次前向已生成的 index K 并对它们打分，之后才用候选 mask 排除池外位置。Prefill 中，不同 Query 的因果边界还需要分别遮罩，不能把整个切片都当成每个 Query 可见的历史：

~~~python
index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
index_score = torch.einsum("bshd,btd->bsht", q, index_k)
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
...
index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)
~~~

第一行的切片包含截至 `end_pos` 已生成的压缩缓存条目；第二、三行先对这个完整切片计算分数，再跨 index heads 汇总。中间省略了逐 Query 因果遮罩、跨卡归约和候选池生成等分支；最后一行才把池外分数置为负无穷，使后续 Top-K 只会选池内有效位置。由此，Minimal Inference 保留了论文的候选限制选择语义，但这段参考实现仍执行全域 einsum；它不能证明后续层的打分 FLOPs 已降为候选池大小，也不能作为论文复杂度界已在该代码路径实现的证据。

同一最小推理路径的 `inference/model.py` 对压缩 latent 和 index K 都以 `inplace=True` 调用 FP4 数值变换：

~~~python
fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
fp4_act_quant(k, fp4_block_size, True)
~~~

`inference/kernel.py` 由 `@tilelang.jit` / `T.prim_func` 定义量化 kernel；其输出类型选择与写回为：

~~~python
out_dtype = in_dtype if inplace else FP4
...
x.copy_(y)
return x
~~~

`inplace=True` 时先把输入量化到 FP4 可表示值，再反量化回输入 dtype 并覆盖原张量；若输入是 BF16，缓存张量的物理元素仍是 BF16，而不是打包 E2M1 nibble。因而这段作者代码可说明 FP4 数值误差路径，却不能证明物理缓存采用论文的 packed FP4 payload。论文的 890 B/token 是 E2M1 主 KV 与 index K 加分组 scale 的有效 payload 账本；不要把它当成这份 Minimal Inference 已实际分配的缓存字节数。

其余代码边界同样重要：Prompt Encoding 的 `encoding/` 含 Python 参考编码器和多轮对话、工具调用、thinking、effort、system 消息及图像交错用例；生产协议库另有 Rust 实现与 Python bindings。`evaluation/` 的 DeepSWE 说明覆盖 DeepSeek Harness Minimal 与 mini-SWE，并给出接入 Pier 的补丁。最小推理实现可用于核对结构与数据流，但不提供完整训练/组件消融脚本、内部 benchmark 数据、生产调度器或服务吞吐证明。本文的 NumPy 代码则是小规模教学实现：它检验公式、边界和反例，不是作者实现；其中玩具哈希尤其不代表作者 Engram 哈希。

## 15. 用小例子检验自己是否理解

以下问题应能先写出对象和维度，再回答性能含义：

1. Reindex 复用了哪些 tensor，哪些仍重新计算？后续 Reuse 应引用哪一份 Top-K？
2. 为什么候选池 16384 不等于每层读 16384 个 main KV？第一层还扫描多少位置？
3. 能否构造一个候选限制丢失深层最大分数的例子？
4. main KV 是 FP4 时，scale、SWA 和实际矩阵指令分别是什么？
5. 为什么 Sinkhorn 最后乘 √n？零行和有限迭代时哪种等式不能强求？
6. 同样的草稿接受概率，为何系统负载改变会让最优验证长度缩短？
7. 为什么不同缓存命中位置可能产生不同的后续状态？
8. 异步派发策略怎样改变样本到达顺序，又为什么不能改变奖励分组？

公式与小张量示例可独立运行：

~~~bash
python roadmap/curriculum/papers/examples/v41_checks.py
~~~

它检查候选池与因果边界、遗漏反例、条件接受概率、负载取舍、教学哈希、Sinkhorn 零行、effort 截断和 890 B/token 的 payload 账本。`v41_tensor_walkthrough.py` 与 `v41_engram_state.py` 另检查视觉张量顺序/形状及 Engram 地址状态/查表/门控/预取等待；`workload_lifecycle.py` 检查 rollout 分组、旧 logp、投机提交/回滚与工具等待版本。所有程序均为 CPU 教学检查，不是完整模型推理、真实 GPU 性能或作者 kernel 复现。

## 参考阅读

[DeepSeek-V4.1-Flash 技术报告](../../../downloads/DeepSeek_V41_Tech_Report.pdf) · [固定版本作者推理参考](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference) · [固定版本编码与评测文件](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277) · [DeepGEMM Mega mHC API](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/csrc/apis/mega_mhc.hpp) · [SM100 kernel](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh) · [API test](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/tests/test_mega_mhc.py)。

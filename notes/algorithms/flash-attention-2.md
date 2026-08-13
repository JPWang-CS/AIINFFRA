# Flash Attention 2：Work Partitioning 与 non-matmul 削减

> 注意力演进类 · FA1 的 GPU 利用率补课 · 前置 [Flash Attention 机制](flash-attention-mechanism.md) · 论文精读 [papers/attention/flash-attention-2.md](../../papers/attention/flash-attention-2.md)
> 挂靠：主线 A 前置 · B3 Triton FA · 面试必考

---

## 解决了什么问题

FA1 已经用 tiling + online softmax 把 HBM 读写从 O(N²) 降到 O(N)，但 GPU 利用率远没拉满：

1. **并行度不够**：FA1 的并行维度只有 batch × head。长序列推理（batch=1）时只有几个线程块在跑，A100 有 108 个 SM，batch=1、heads=8 时只用 8/108，其余全空着。
2. **non-matmul 运算占比高**：rescale（O 累加器的回溯修正）、exp、max 都是逐元素运算，跑在 CUDA core 上，而 A100 的 CUDA core 吞吐只有 tensor core 的 1/16（FP32）。FA1 里这类运算耗时占比约 40%。
3. **Backward 有锁**：FA1 的 backward 多个线程块竞争写同一个 dK/dV，需要 atomic，并行度被锁住。
4. **序列维度没用上**：超长序列（128K tokens）时 batch/head 更不够填满 SM。

FA2 的目标：**IO 复杂度不变，把 GPU 利用率提上去**。结果：A100 上 FP16 从 ~110 TFLOPS 提到 ~225 TFLOPS（72% MFU），比 FA1 快约 2x，比 PyTorch 标准 attention 快约 9x。

## 核心思路

### 1. Forward：并行维度加上 Q 块

```text
FA1:  一个 (batch, head) → 一个线程块
      并行度 = batch × heads

FA2:  一个 (batch, head, Q 块) → 一个线程块
      并行度 = batch × heads × ceil(seq / Br)
```

每个线程块负责 Q 的 Br 行，串行遍历所有 K/V 块。因为 Q 块之间互不依赖（每个 Q 块独立维护自己的 m/l/O），可以放心切。

例：seq=4096、Br=64 → 64 个 Q 块；batch=1、heads=8 → 8 × 64 = 512 个线程块，A100 的 108 个 SM 全部用上。

### 2. 削减 non-matmul FLOPs

把 attention 的 forward 压缩成"两次 matmul + 尽量少的元素运算"：

```text
S = Q @ K^T          ← matmul 1（tensor core）
P = softmax(S)       ← 只剩必要的 max/exp/scale
O = P @ V            ← matmul 2（tensor core）
```

具体做法：

- **rescale 只做一次乘法**：新 K/V 块到来时，旧的 O 和 l 各乘一个修正因子 `exp(m_old - m_new)`，不重复算整块 softmax
- **m/l/O 全程留在寄存器/SRAM**：不写回 HBM，这是"省 IO"的另一半
- **避免每块都做多余归一化**：循环里不除以 l，最后一次才除

### 3. Backward：K/V 外层并行，去掉 atomic

FA1 backward 按 Q 并行，多个线程块会写同一个 dK/dV 位置 → atomic 串行化。

FA2 把方向反过来：

```text
外层：K/V 的列块（每个 K/V 块只属于一个线程块）
内层：遍历 Q 的行块
```

每个 dK/dV 位置只有拥有它的线程块写 → **天然无锁**。

代价：forward 按 Q 分、backward 按 K/V 分，两个方向不对称，实现更复杂。但换来的是无原子操作。

### 4. Causal：整块跳过，不只是 mask

causal 下 `j > i` 的 K/V 块（上三角）整块无效，FA2 直接跳过循环，不 launch、不加载：

```text
Br = Bc = 64, N = 4096:
总块对数 = 64 × 64 = 4096
上三角块 = 64 × 63 / 2 = 2016
有效计算省一半，且 HBM 读取也省一半
```

只有对角线上的块需要真正做 mask（半块 mask 由编译器/指令处理）。

## 关键数据与取舍

| 指标 | FA1 | FA2 |
|---|---|---|
| Forward 并行度 | batch × heads | batch × heads × seq/Br |
| A100 FP16 | ~110 TFLOPS | ~225 TFLOPS（72% MFU）|
| 相对 PyTorch（短序列） | ~3x | ~9x |
| Backward 写 dK/dV | atomic | 无竞争 |
| HBM 读写复杂度 | O(N²d²/M) | 相同，常数更好 |
| causal 上三角 | 逐元素 mask | 整块跳过 |

取舍一句话：**FA2 不改变算法（还是 tiling + online softmax），改变的是"谁负责哪块、怎么少做非矩阵运算"**。所以它的优化思路可以平移到任何 IO-aware kernel：并行度要能填满 SM、尽量只喂 tensor core 该干的事。

## 与我何干

- **B3 Triton Flash Attention**：Triton 天然按 Q 块做 grid（`pid_q = tl.program_id(0)`，每个 program 负责一个 Q 块、串行遍历 K/V 块），你写出来就是 FA2 的 forward 分区。理解这一点，写 Triton FA 时就知道"为什么循环顺序是外层 Q、内层 K/V"。
- **C2 PagedAttention**：vLLM 的 block 并行和 FA2 同一思路——把 KV 切成块，让并行度不受 batch 限制。
- **[面试]** 必考：
  - "FA2 比 FA1 快在哪？" → 三个点：Q 块并行、削减 non-matmul、backward 去 atomic
  - "为什么 backward 分区和 forward 相反？" → 让每个 dK/dV 只有一个线程块写，避免 atomic
  - "causal 为什么能省一半？" → 上三角 K/V 块整块跳过，计算和 HBM 读取都省

---

*前置：[Flash Attention 机制](flash-attention-mechanism.md) · 配套：B3 Triton Flash Attention（Lesson 按需生成）*
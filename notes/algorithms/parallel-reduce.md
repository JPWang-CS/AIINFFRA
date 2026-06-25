# Parallel Reduce（GPU 并行归约模式）

> GPU 优化算法类 · 求和/求最大值/求最小值的通用并行模式 · Softmax/Norm 的基础

---

## 解决了什么问题

串行 reduce：`sum = 0; for (i) sum += x[i]`，O(N) 时间。GPU 有上千个核心，怎么并行？

朴素并行（所有线程写同一个变量）会有 **race condition**。需要分治 + 同步。

## 核心思路（树状归约）

```
线程布局: 每个线程初始负责 1 或多个元素
         [0] [1] [2] [3] [4] [5] [6] [7]  (shared memory)
Step 1:   0+1   2+3   4+5   6+7            stride = 1
         [sum01] [sum23] [sum45] [sum67]
Step 2:   sum01+sum23   sum45+sum67        stride = 2
         [sum0-3]      [sum4-7]
Step 3:   sum0-3 + sum4-7                  stride = 4
         [sum0-7]
```

每一步 stride 翻倍，活跃线程数减半。`log₂(N)` 轮后只剩一个线程，得到最终结果。

**关键点**：
1. **Shared memory** — 块内线程通过 shared memory 共享中间结果
2. **`__syncthreads()`** — 每轮后必须同步，防止快线程覆盖慢线程还没读的数据
3. **Warp shuffle** — 如果数据在单个 warp 内（≤32 元素），用 `__shfl_down_sync` 比 shared memory 更快（寄存器直接交换，省 shared memory 访存）

## 三种实现（由浅入深）

### 1. Block-level Reduce (Shared Memory)
```cuda
__shared__ float sdata[BLOCK_SIZE];
sdata[tid] = input[tid];  // 每线程搬入 shared memory
__syncthreads();

// 树状归约
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
// sdata[0] 是块内和
```
**问题**：`tid % (2*s)` 导致 warp divergence（同一 warp 内部分线程 idle）

### 2. Sequential Addressing（优化版）
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```
活跃线程连续，避免 divergence。

### 3. Warp Shuffle（最快，单 warp 内）
```cuda
// 假设 warpSize = 32
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// 第 0 号线程的 val 是 warp 内和
```
不用 shared memory，直接寄存器通信。**延迟最低**。

## 在 Ascend 的对应

Ascend 的 `Reduce` intrinsic（`ReduceSum` / `ReduceMax`）是硬件指令，你调就行。CUDA 没有"一条指令算完"，要手动写树状归约。但思想一样：分治 + 同步。

Ascend `Pipe` 的跨 L1 Buffer 归约类似 CUDA 的跨 block reduce（grid-level），需要 global memory 做中间存储。

## 性能数据

| 实现 | 延迟（相对） | 适用 |
|------|:---:|------|
| Block-level (naive) | 1.0× | 任意大小 |
| Sequential addr | 0.8× | 避免 divergence |
| **Warp shuffle** | **0.3×** | 单 warp (≤32 元素) |

Softmax 一般对每行做 reduce（行长 ≤ 几千），先 warp shuffle 求 warp 内和/max，再跨 warp 用 shared memory 合并。

## 与我何干

**A4 Softmax**: 每行求 max 和 sum，就是两次 reduce。你会先写 block-level → 改成 warp shuffle。

**RMSNorm / LayerNorm** (可选 bonus): 求 mean 和 variance，也是 reduce。

**[面试]** 高频题：
- "GPU 怎么并行求和？" → 树状归约 + 同步
- "为什么要 `__syncthreads()`？" → 防止 race（画 stride=1→2 的数据依赖图）
- "Warp shuffle 和 shared memory reduce 有什么区别？" → shuffle 延迟低但只能单 warp 内

## 代码示例

LeetGPU `4_reduction` — 自己写一遍能深入理解。参考实现见 [reference/cuda/softmax/softmax.cu](../../reference/cuda/softmax/softmax.cu) 里的 `blockReduce` / `warpReduce`。

## 扩展

- **Grid-level reduce**（跨 block）：每个 block 算出 partial sum 写回 global memory → 再起一个 kernel 归约这些 partial sum。cuDNN/CUB 有封装好的。
- **Cooperative Groups API**：统一 warp/block/grid 的 reduce 接口，但性能和手写差不多。

---

*配套：[online softmax](online-softmax.md)（reduce 的增量版本）*

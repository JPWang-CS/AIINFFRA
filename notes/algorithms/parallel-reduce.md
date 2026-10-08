# Parallel Reduce（GPU 并行归约模式）

> GPU 并行归约：求和、最大值、最小值与 Softmax/Norm 的基础模式

---

## 解决了什么问题

串行 reduce：`sum = 0; for (i) sum += x[i]`，需要依次处理 N 个元素。并行归约把输入分成互不重叠的片段，再合并各片段的局部结果。

若所有线程直接写同一个累加变量，就会发生 **race condition（数据竞争）**。归约还要求合并操作具有可组合性：`max` 的结合顺序不影响结果，而浮点 `sum` 不是严格结合的，树形顺序可能带来舍入差异。

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

每一步 stride 翻倍，参与合并的线程数减半。对长度为 2 的幂且已完成边界填充的输入，经过 `log₂(N)` 轮后只剩一个线程，得到块内结果；一般长度需要显式处理尾部，不能直接套用这张图。

**关键点**：
1. **Shared memory** — 块内线程通过 shared memory 共享中间结果
2. **`__syncthreads()`** — 每轮后必须同步，防止快线程覆盖慢线程还没读的数据
3. **Warp shuffle** — 如果参与线程位于同一个 warp，可用 `__shfl_down_sync` 直接交换寄存器值，减少 shared memory 访问；参与范围和源 lane 必须满足 intrinsic 的 mask 约束

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
**问题**：`tid % (2*s)` 让同一 warp 中只有部分线程参与当前轮；此外，这个片段假设 `BLOCK_SIZE` 为 2 的幂且 `input[tid]` 已在输入范围内，实际 kernel 需要处理尾部线程。

### 2. Sequential Addressing（优化版）
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```
参与线程连续，通常比按取模筛选更适合 warp 的执行方式；它仍要求 shared memory 的每轮读写之间有同步，并不自动处理非整除输入。

### 3. Warp Shuffle（单 warp 内交换）
```cuda
// 假设 warpSize = 32，且 mask 中的 32 个 lane 都执行该 intrinsic
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// 第 0 号线程的 val 是 warp 内和
```
不经过 shared memory，直接交换寄存器值；在参与范围适合单 warp 的场景中，通常可以减少同步与 shared memory 访问，但延迟和吞吐仍取决于目标架构、指令混合与 mask。若不足 32 个有效元素，应构造匹配的 mask，并保证 shuffle 的源 lane 处于参与范围内。

## 在 Ascend 的对应

Ascend 的 `Reduce` intrinsic（`ReduceSum` / `ReduceMax`）提供了平台级归约接口；CUDA 中可以用 warp shuffle、shared memory 或库实现组合出同样的归约结构。两者可对照的是分治与数据交接关系，不能据接口名称推断底层指令数或同步代价。

Ascend `Pipe` 的跨 L1 Buffer 归约与 CUDA 的跨 block 分阶段归约在“局部结果写回、下一阶段再合并”这一层面相似；CUDA 中常用多次 kernel launch 让前一阶段的 global 写入先完成。若采用原子更新，还必须为 `(m,S)` 等复合状态设计可证明的合并协议。

## 性能数据

| 实现 | 延迟（相对） | 适用 |
|------|:---:|------|
| Block-level (naive) | 待测 | 任意大小，需处理边界 |
| Sequential addr | 待测 | 连续参与线程，仍需同步 |
| **Warp shuffle** | 待测 | 单 warp 或作为 block 归约的最后阶段 |

Softmax 常对每行做 reduce；行长、每线程元素数和 block 配置决定是单 warp 完成，还是先做 warp 内归约、再用 shared memory 合并各 warp 的局部结果。性能数字应在固定 shape、精度和计时口径下实测，不能使用没有实验条件的相对倍数。

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
- **Cooperative Groups API**：按线程组表达协作，具体归约操作有受支持的组类型与同步约束。它不让任意普通 grid 自动拥有安全的全局屏障；性能应与等价手写实现按相同条件比较。

---

*配套：[online softmax](online-softmax.md)（reduce 的增量版本）*

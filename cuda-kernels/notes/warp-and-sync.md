# Warp 调度与同步

> 理解 warp 是写出高效 CUDA 代码的关键。本文覆盖 warp 调度模型、divergence、shuffle 和 block 级同步。
> API 速查 → [cuda-cheatsheet.md](./cuda-cheatsheet.md) | 内存模型 → [memory-model.md](./memory-model.md)

---

## 1. 什么是 Warp

**一个 warp = 32 个 thread 的组。SM 以 warp 为单位调度线程。**

```
Block of 256 threads = 8 warps

Warp 0: thread 0-31
Warp 1: thread 32-63
...
Warp 7: thread 224-255
```

SM 上有 warp scheduler，每个周期选一个就绪的 warp 发射指令。如果当前 warp 在等 memory，scheduler 切换另一个 warp——这就是 GPU 隐藏 memory 延迟的核心机制。

> **Ascend 对照**：Ascend 没有 warp 的概念。它的 Vector Unit 有 SIMD 宽度（一次处理多个数据），但没有 warp 级别的调度和 divergence 问题。warp 是 CUDA 独有的设计。

### Warp Scheduler 的物理图景

```
SM 上有 4 个 warp scheduler（A100 为例）
每个 scheduler 管理一组 warp
每个周期：
  1. 从自己的 warp pool 中选一个就绪的 warp
  2. 发射一条指令
  3. 如果所有 warp 都在等 memory → SM 空闲（低 occupancy 的代价）

Occupancy 高 = 更多 warp = scheduler 有更多选择 = 更容易隐藏延迟
```

---

## 2. Warp Divergence — `[面试]` 高频题

### 2.1 问题

**同一 warp 内的 32 个 thread 共享一个 program counter（PC）。如果它们走不同分支，硬件只能串行执行。**

```cpp
// ❌ 严重的 warp divergence
if (threadIdx.x % 2 == 0) {
    // 偶数 thread 走这（执行时奇数 thread 被 masked）
    expensive_func_a();
} else {
    // 奇数 thread 走这（执行时偶数 thread 被 masked）
    expensive_func_b();
}
// 两个分支串行执行，有效吞吐 = 50%

// ✅ 按 warp 对齐的分支（无 divergence）
if (threadIdx.x / 32 == 0) {  // warp 0
    func_a();
} else {                      // 其他 warp
    func_b();
}
// 不同 warp 走不同分支没问题——每个 warp 内部同路
```

### 2.2 哪些情况会导致 divergence

| 情况 | 是否 divergence | 说明 |
|------|:---:|------|
| if-else on `threadIdx.x` | ⚠️ 可能 | 取决于条件模式 |
| if-else on `blockIdx.x` | ✅ 安全 | 不同 block = 不同 warp |
| loop with variable bound for different threads | ⚠️ 可能 | 部分 thread 早退出 |
| early return on input-dependent condition | ⚠️ 可能 | 数据依赖的 divergence |

### 2.3 判断方法

**规则：看条件是否在 warp 内变化。**

```cpp
// 安全 — 同一 warp 内 threadIdx.x / 32 相同
if (threadIdx.x / 32 == 0) { ... }

// 可能 divergence — 需要看 N 是否 < 32
if (threadIdx.x < N) { ... }

// 安全 — 同一 warp 的 32 个 thread 相邻，有边界时只有边缘 warp 受影响
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) { ... }  // 最多 W-1 个 warp 完全激活，最后一个 warp 部分激活（不可避免）
```

> 边界判断 `if (idx < N)` 引起的 divergence 是**不可避免且代价可接受**的。不要因此不用边界检查。

---

## 3. `__syncthreads()` — Block 级同步

### 3.1 语义

```cpp
__syncthreads();
// 这是一个 barrier：
// 1. block 内所有 thread 必须到达此处
// 2. 当前 thread 在此暂停，直到所有 thread 都到达
// 3. 所有 thread 在 __syncthreads 之前的 shared memory 写入对所有线程可见
```

> **Ascend 对照**：`__syncthreads()` ≈ `pipe_barrier()` / `block_sync()`。语义完全一致——确保所有参与方的数据写入在后续读取前完成。

### 3.2 致命陷阱

```cpp
// ❌ 死锁！有些 thread 不会到达 barrier
if (threadIdx.x % 2 == 0) {
    tile[threadIdx.x] = data;
    __syncthreads();   // 只有偶数 thread 走这里 → 奇数 thread 永远等不到
}

// ❌ 死锁！嵌套在不同 block 范围
for (int i = 0; i < 10; i++) {
    if (i < 5) {
        __syncthreads();  // 只有某些迭代到达
        // 不同线程可能有不同的 i 值 → 死锁
    }
}

// ✅ 正确用法
if (row < M && col < N) {
    tile[threadIdx.y][threadIdx.x] = data;
}
__syncthreads();  // barrier 在条件外面，所有 thread 到达
```

### 3.3 性能代价

`__syncthreads()` 本身很快（几个周期），但会导致所有 warp 互相等待——最快的 warp 要等最慢的 warp。所以不必要的 barrier 会降低性能。

**GEMM tiled kernel 中的两次 `__syncthreads()` 是必要的**：
- 第一次：确保 tile 加载完毕再开始计算
- 第二次：确保计算完成再加载下一个 tile（覆写 shared memory 前）

---

## 4. Warp Shuffle — Warp 内通信

### 4.1 为什么需要 Shuffle

Shared memory 做 reduction（求和/求最大值）需要多步读写 + `__syncthreads()`。Warp shuffle 允许同一 warp 内的 thread 直接交换寄存器值——不需要 shared memory，不需要 barrier。

```cpp
// Warp 内求和的 reduction（比 shared memory 版本快 2-3×）
float val = input[idx];

// 递归减半，每个 step 拿相邻 lane 的值
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// 执行后，lane 0 持有 warp 内所有线程的 val 之和
```

### 4.2 Shuffle 指令

```cpp
// __shfl_down_sync(mask, val, delta)
// 当前 thread 拿到 lane_id + delta 的 thread 的 val
// mask: 哪些 thread 参与（通常是 0xffffffff = 全 warp）

// lane 分布和 __shfl_down(offset=4):
// lane:  0  1  2  3  4  5  6  7  ...
// val:   3  7  1  9  2  4  6  8  ...
//         ↓  ↓  ↓  ↓
// 4:     2  4  6  8  ... (每个拿到自己+4 的值)

// 其他 shuffle 变体：
__shfl_up_sync(mask, val, delta);    // 从 lane_id - delta 拿
__shfl_xor_sync(mask, val, mask2);   // 从 lane_id ^ mask2 拿
__shfl_sync(mask, val, src_lane);    // 从指定 src_lane 拿（broadcast）
```

### 4.3 Reduction 范例

```cpp
// Softmax 中需要找最大值 + 求指数和
// 这个 pattern 非常常见
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;  // 取最大
    }
    return val;  // 所有 lane 的 val 是 warp-max
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 持有 sum
}
```

> **Ascend 对照**：Ascend 的 Vector Unit 有硬件 reduce 指令，不需要手动 shuffle。CUDA 用 warp shuffle 模拟同样的效果。本质都是为了"一个 compute group 内的快速 reduce"。

### 4.4 跨 Warp Reduce（Block 级）

Warp shuffle 只在一个 warp 内工作。要跨 warp reduce，需要 shared memory 收集每个 warp 的结果：

```cpp
// 每个 warp 先在内部 reduce 到一个值
float warp_result = warp_reduce_sum(my_val);  // lane 0 有结果

// 每个 warp 的 lane 0 写入 shared memory
__shared__ float warp_results[32];  // 最多 32 个 warp/block
if (lane == 0) {
    warp_results[warp_id] = warp_result;
}
__syncthreads();

// 用 thread 0（或一个 warp）对 warp_results 做最后的 merge
if (threadIdx.x < num_warps) {
    float block_result = warp_results[threadIdx.x];
    for (int i = 0; i < num_warps; i++) {
        block_result += warp_results[i];
    }
    // block_result 现在是 block 级结果
}
```

---

## 5. Warp Scheduling 和 Occupancy 的关系

```
隐藏 memory 延迟的原理：
  Warp A 发起 global memory load → 等待 600 cycles
  Scheduler 切换到 Warp B → 执行其他指令
  Scheduler 切换到 Warp C ...
  ...
  Warp A 的数据回来了 → 继续执行

关键：如果 SM 上同时驻留的 warp 不够多，scheduler 没有足够的"其他工作"来填充等待时间。
→ SM 空闲 → 低性能
→ 这就是为什么 occupancy 重要
```

| Occupancy | 效果 |
|:---:|------|
| < 25% | 严重不足，难以隐藏延迟 |
| 25-50% | 基本够用 |
| 50-75% | 良好，latency hiding 充分 |
| > 75% | 足够，再提高通常无额外收益 |

**Trade-off**：增大 block size / shared memory → register 用量增加 → occupancy 下降 → 调 tile size 找 sweet spot。

---

## 6. 常见 Warp 相关的 Bug

| Bug | 症状 | 原因 | 解法 |
|-----|------|------|------|
| Deadlock | kernel 不返回 | `__syncthreads` 在条件分支里 | 确保所有 thread reach barrier |
| Divergence 性能差 | 吞吐低于预期 | 同一 warp 内分支不同 | 对齐到 warp 边界 |
| Shuffle 读到垃圾 | 计算结果错误 | active mask 不包含所有线程 | 用 `__activemask()` 代替硬编码 mask |
| 低 occupancy | SM 利用率低 | register/shared mem 过量 | 减 tile size 或拆分 kernel |
| Bank conflict in shuffle | N/A | shuffle 不走 shared memory！不涉及 bank | 别混淆——shuffle 没有 bank conflict |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [cuda-cheatsheet.md](./cuda-cheatsheet.md) | API 速查、代码模板 |
| [memory-model.md](./memory-model.md) | 内存层级、coalescing、bank conflict |
| [gpu-architecture.md](./gpu-architecture.md) | NVIDIA vs Da Vinci 架构对比 |

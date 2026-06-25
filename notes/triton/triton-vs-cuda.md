# Triton vs CUDA 深度对比

*Triton 阶段笔记（算子线 B）*

---

## 编程模型对比

### CUDA: 你管一切
```cuda
__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 手动算全局索引
    int stride = blockDim.x * gridDim.x;              // 手动做 grid-stride loop
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
// 你还得操心：
// - grid/block 大小设置多少？
// - shared memory 要不要用？
// - memory coalescing 满足吗？
// - bank conflict 有吗？
// - occupancy 够吗？
```

### Triton: 你只管 block 逻辑
```python
@triton.jit
def vec_add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                    # block ID
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 元素偏移
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)      # 编译器决定怎么 load
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)  # 编译器决定怎么 store
# 编译器自动处理：
# - thread 分配
# - shared memory 使用
# - memory coalescing
# - autotuning 搜索最优 BLOCK 大小
```

## 核心差异一览

| 维度 | CUDA | Triton |
|------|------|--------|
| 编程粒度 | Thread-level | Block-level（tile） |
| 索引 | 手动计算 threadIdx + blockIdx | `tl.program_id()` + `tl.arange()` |
| 内存 | 显式 `__shared__` / register | 编译器推断 |
| 同步 | `__syncthreads()` | 隐式，block 边界自动同步 |
| 优化 | 手调 tile size, occupancy, bank conflict | Autotuning |
| 调试 | Nsight Compute / printf | 较难（编译器中间层不透明） |
| 性能上限 | 理论最优 | ~90-95% 手写 CUDA |
| 可移植性 | NVIDIA only | 理论上可扩展（MLIR 后端） |

## 什么时候用哪个

**用 Triton（90% 的场景）**：
- ML kernel（attention、MLP、norm、embedding）
- 需要快速迭代，不追求极致性能
- 融合多个小操作（element-wise + matmul）
- PyTorch 生态内

**用 CUDA（10% 的场景）**：
- 需要精细控制 warp-level 操作（shuffle、ballot）
- 对 occupancy / register pressure 有极致要求
- 需要用到 Triton 不支持的特性（dynamic parallelism、texture memory）
- 写底层通信库（NCCL 类）

## 学习路径

1. 先用 Triton 写熟悉的算子（matmul, softmax）
2. 用 `tl.constexpr` 和 autotuning 探索不同 tile size 的性能影响
3. 对比 CUDA 版本的性能，理解差距来源
4. 看 Triton 生成的 MLIR/PTX，理解编译器做了什么

## 面试要点

面试中 Triton 相关的高频问题：
- "Triton 的编程模型和 CUDA 有什么本质区别？" → block-level vs thread-level
- "Triton 如何保证 memory coalescing？" → 编译器根据 access pattern 自动分配 thread 到连续地址
- "Triton 的 autotuning 做了什么？" → 网格搜索 tile size + num_warps + num_stages

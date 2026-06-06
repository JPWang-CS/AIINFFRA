# Triton 语法速查

> 日常写 Triton 最常用的 API 和 pattern。每个条目最小可用代码。
> 底层 CUDA 对照 → [../../cuda-kernels/notes/triton-under-the-hood.md](../../cuda-kernels/notes/triton-under-the-hood.md)

---

## 1. 基本框架

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr,          # 输入指针
    y_ptr,          # 输出指针
    N,              # 标量参数
    BLOCK_SIZE: tl.constexpr,  # compile-time constant
):
    # program_id: 当前 block 在 grid 中的位置
    pid = tl.program_id(axis=0)
    
    # 计算这个 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界 mask
    mask = offsets < N
    
    # 加载 → 计算 → 存储
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

# 启动 kernel
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
my_kernel[grid](x, y, N, BLOCK_SIZE=256)
```

---

## 2. Program ID 和 Grid

```python
# 1D grid
pid = tl.program_id(0)          # 0 到 grid[0]-1
num_pids = tl.num_programs(0)   # grid 大小

# 2D grid
pid_m = tl.program_id(0)        # 行方向
pid_n = tl.program_id(1)        # 列方向

# grid lambda — 告诉 triton 启动多少个 block
grid = lambda meta: (M // meta['BLOCK_M'], N // meta['BLOCK_N'])
kernel[grid](...)
```

---

## 3. 向量操作（tl.arange）

```python
# tl.arange(0, N) 生成一个 1D 向量 [0, 1, 2, ..., N-1]
offsets = tl.arange(0, BLOCK_SIZE)

# 2D 向量（用于 GEMM）
row_idx = tl.arange(0, BLOCK_M)[:, None]   # BLOCK_M × 1
col_idx = tl.arange(0, BLOCK_N)[None, :]   # 1 × BLOCK_N
# 广播：两个结合得到 BLOCK_M × BLOCK_N 的索引矩阵
```

---

## 4. 内存操作

```python
# 加载（global memory → register/shared memory）
x = tl.load(ptr + offsets, mask=mask)             # 基本 load
x = tl.load(ptr + offsets, mask=mask, other=0.0)  # mask 外的元素填充 0.0

# 存储（register/shared memory → global memory）
tl.store(ptr + offsets, values, mask=mask)

# 原子操作
tl.atomic_add(ptr + offsets, values, mask=mask)
tl.atomic_max(ptr + offsets, values, mask=mask)

# 指针运算（注意 triton 用 int，不是 float 指针运算）
ptr = ptr + offset_ints  # 返回新指针
```

---

## 5. 类型和常量

```python
# 常量声明
@triton.jit
def kernel(x, N,
           BLOCK_SIZE: tl.constexpr,     # compile-time constant，影响 grid
           SCALE: tl.constexpr = 1.0):   # 可选默认值
    ...

# 数据类型（tl 命名空间下）
tl.float32, tl.float16, tl.bfloat16
tl.int32, tl.int64
tl.uint32, tl.uint8

# 类型转换
x_fp16 = x.to(tl.float16)
x_fp32 = x.to(tl.float32)
```

---

## 6. 数学运算

```python
# 基本运算（向量化，和 NumPy 几乎一样）
y = x + y
y = x * scale
y = x / denom

# 数学函数
tl.exp(x)
tl.log(x)
tl.sqrt(x)
tl.abs(x)
tl.maximum(a, b)
tl.minimum(a, b)
tl.sigmoid(x)   # 1/(1+exp(-x))

# Reduction
tl.sum(x, axis=0)     # 沿某一维求和
tl.max(x, axis=0)     # 沿某一维取最大
tl.argmax(x, axis=0)  # 沿某一维取 argmax

# 条件
y = tl.where(mask, a, b)  # mask ? a : b，向量化
```

---

## 7. GEMM 核心（tl.dot）

```python
@triton.jit
def gemm_kernel(A, B, C, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 指针到 A 和 B 的 tile
    a_ptrs = A + pid_m * BLOCK_M * K + tl.arange(0, BLOCK_M)[:, None] * K + tl.arange(0, BLOCK_K)[None, :]
    b_ptrs = B + pid_n * BLOCK_N + tl.arange(0, BLOCK_K)[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)   # BLOCK_M × BLOCK_K
        b = tl.load(b_ptrs)   # BLOCK_K × BLOCK_N
        acc += tl.dot(a, b)   # BLOCK_M × BLOCK_N（硬件加速！）
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    
    c_ptrs = C + pid_m * BLOCK_M * N + pid_n * BLOCK_N + ...
    tl.store(c_ptrs, acc)
```

---

## 8. Softmax Pattern

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Online softmax (triton 帮你处理 warp-level reduce)
    x_max = tl.max(x, axis=0)           # 找最大值
    num = tl.exp(x - x_max)             # exp(x - max)
    denom = tl.sum(num, axis=0)         # sum(exp(...))
    y = num / denom
    
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 9. Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['N'],  # 按 N 大小选择最优配置
)
@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...

# 第一次调用时 Triton 自动 benchmark 所有 configs
# 选择最快的，缓存结果
```

---

## 10. 编译和调试

```bash
# 运行 Triton kernel（需要 NVIDIA GPU）
python my_kernel.py

# 环境变量
TRITON_INTERPRET=1 python my_kernel.py    # CPU 上解释执行（调试用）
TRITON_PRINT_AUTOTUNING=1                 # 打印 autotuning 结果

# 查看生成的中间代码
# Triton IR → MLIR → LLVM IR → PTX → SASS
# 用 @triton.jit 的 compile 方法
```

---

## 11. 常见 Pattern 速查

| 场景 | Pattern | 关键 API |
|------|---------|---------|
| element-wise | 1D grid, tl.load → compute → tl.store | `program_id(0)`, `arange` |
| 2D grid | 2D program_id, 2D arange + broadcast | `program_id(0/1)`, `[:, None]` |
| GEMM | 2D grid, tl.dot in loop over K tiles | `tl.dot`, `tl.zeros` |
| Softmax | max → exp → sum → div | `tl.max`, `tl.exp`, `tl.sum` |
| LayerNorm | mean → var → normalize | `tl.sum` (×2 遍历) |
| Attention | QK^T → scale → softmax → ×V | `tl.dot` + softmax pattern |
| Fused kernel | 多个 op 在同一个 kernel 里 | 保存 intermediate results 到 register |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [../../cuda-kernels/notes/triton-under-the-hood.md](../../cuda-kernels/notes/triton-under-the-hood.md) | Triton 生成的 CUDA 代码长什么样 |
| [../../cuda-kernels/notes/memory-model.md](../../cuda-kernels/notes/memory-model.md) | CUDA 内存层级（理解 Triton 背后） |
| [triton-vs-cuda.md](./triton-vs-cuda.md) | Triton vs CUDA 编程模型对比 |

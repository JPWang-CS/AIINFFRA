# Code Review: GEMM fp16 Naive

> 评审时间：2026-06-22
> 代码：`cuda-kernels/gemm/gemm_fp16_naive.cu`

---

## 整体评价

✅ **正确性**：逻辑正确，实现了 `C = alpha*A*B + beta*C`（BLAS GEMM 标准形式）

⚠️ **精度控制**：缺少显式 rounding mode，float→half 走隐式转换

⚠️ **性能**：naive 访存模式，和 float 版同款瓶颈——但 fp16 把内存带宽需求减半

---

## 逐行点评

### 1. 隐式 float→half 转换（最需要改）

```cpp
C[idx] = sum;  // sum is float, C is half*
```

**问题**：CUDA 允许 float 隐式赋值给 half，但不指定 rounding mode。虽然默认也是 RN（Round to Nearest），但**显式写 `__float2half_rn(sum)` 有三层好处**：
- 代码意图清楚（reviewer 一眼看出这里在做精度截断）
- 你可以主动选 rounding mode（大多数场景 RN，但量化场景可能用 RZ）
- 面试时如果写了 `C[idx] = sum` 会被追问"这是什么 rounding？"

**改法**：
```cpp
C[idx] = __float2half_rn(sum);
```

### 2. alpha 乘在循环内（正确但浪费）

```cpp
for (int k = 0; k < K; k++) {
    sum += alpha * (__half2float(A[m * K + k]) * __half2float(B[k * N + n]));
}
```

**问题**：`alpha` 是标量，在每次迭代都被乘一次。K=1024 时多做了 1024 次标量乘法。

**结论**：正确性没问题（浮点乘法满足结合律？不满足，但 alpha 是标量，`alpha*(a*b) = (alpha*a)*b` 对有限浮点成立——等等，浮点乘法不满足结合律，所以如果 alpha 是比如 0.1f，`alpha * (a*b)` 和 `(alpha*a) * b` 可能差 1 ULP。但这是 BLAS 标准行为，所以放循环内是**语义正确**的。

**BUT**：性能角度看，K 次额外乘法。对于 LeetGPU 测试，K 可能到 1024，多 1024 次 `float * float` 乘。**不改也没事**，但养成好习惯——标量操作提到循环外。

### 3. beta 分支（小瑕疵）

```cpp
float sum = (beta == 0.0f) ? 0 : __half2float(C[idx]);
sum = sum * beta;
```

**问题**：如果 `beta == 0.0f`，第一行设 sum=0，第二行 `sum = 0 * 0 = 0`——结果正确，但做了一次无用乘法。

**更干净的写法**：
```cpp
float sum = 0.0f;
if (beta != 0.0f) {
    sum = beta * __half2float(C[idx]);
}
```

或者一行流：
```cpp
float sum = (beta == 0.0f) ? 0.0f : beta * __half2float(C[idx]);
```

**结论**：不影响正确性，但代码 review 时会被圈出来。

### 4. A 和 B 的访存模式（naive 的经典问题）

```cpp
A[m * K + k]   // ✅ coalesced — K 维度连续，warp 内 thread 访问连续地址
B[k * N + n]   // ❌ strided  — 步长 N，warp 内 32 个 thread 访问 32 个不同 cache line
```

**这是 naive GEMM 性能差的根源**。B 的访存模式下：
- 相邻 thread（不同的 n）访问的地址间隔 N 个 half（2 × N 字节）
- 如果 N=512，间隔 1024 字节，远超过 128B cache line

**Tiled 解法**：把 A 的 TILE×TILE block 和 B 的 TILE×TILE block 先搬到 shared memory，在片上做计算，消除 strided 访存。

### 5. gridDim 和 block 维度对应关系

```cpp
// Host:
dim3 blockDim(16, 16);                           // x=M, y=N
dim3 gridDim((M + 15) / 16, (N + 15) / 16);      // x=M, y=N

// Device:
int m = blockDim.x * blockIdx.x + threadIdx.x;    // M 走 x
int n = blockDim.y * blockIdx.y + threadIdx.y;    // N 走 y
```

✅ 对应关系正确。x→M, y→N。

### 6. LeetGPU 接口适配 ✅

```cpp
extern "C" void solve(const half* A, const half* B, half* C, ...)
```

与 LeetGPU 的 `2_matrix_multiplication` 题接口匹配，带有 `alpha`/`beta` 参数说明走的是完整 BLAS 接口。

---

## 改进优先级

| 优先级 | 问题 | 影响 | 改动量 |
|--------|------|------|--------|
| 🔴 **必须** | `C[idx] = sum` → `C[idx] = __float2half_rn(sum)` | 精度可控 | 1 行 |
| 🟡 **建议** | beta 分支逻辑简化 | 可读性 | 2 行 |
| 🟢 **优化** | alpha 提循环外 | 微性能 | 1 行 |
| 🔵 **下一步** | shared memory tiling | 大性能（5-10×） | 重写 kernel |

---

## 与 gemm.cu 中参考代码的对比

| 维度 | 手写版 (fp16 naive) | 参考版 (float tiled, gemm.cu) |
|------|---------------------|-------------------------------|
| 精度 | half（省带宽） | float |
| 算法 | naive（global memory） | tiled（shared memory） |
| BLAS 接口 | alpha/beta 完整 | 纯 C = A × Bᵀ |
| B 矩阵 | B: K×N（不转置） | B: N×K（转置访问，B[col*K + k]） |
| 入口 | `extern "C" solve`（LeetGPU） | `main()`（4090 本地） |

> **注意**：参考版 `gemm.cu` 中 B 是 N×K 且做了转置访问 `B[col*K + k]`。你的 fp16 版本 B 是 K×N（不转置），用 `B[k*N + n]`。两种等效，只是 B 的内存布局不同。

---

*评审完毕。下一步：基于这个 fp16 kernel 加 shared memory tiling → gemm_tiled_fp16*

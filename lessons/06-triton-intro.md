# Lesson 06 — Triton 入门：写第一个 Kernel

> 主题：从零写第一个 Triton kernel：Vector Add → MatMul，并完成"正确性 → 性能 → 记录"闭环
> 前置：建议先完成 [Lesson 05](05-flash-attn-reading.md)（理解 tiling 和 online softmax）；如果当前主线直接进 Triton，也可以先写 vec_add 再补 A5
> 状态：`GPU_VALIDATED` 当前主线（B1）；MatMul 尚未 `COMPLETE`，详细任务见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)
> 范围：本课只做 B1（vec_add + matmul）。Softmax / Flash Attention / GQA 是后面的课，不要一次学完。

📚 **本课重点知识库**：
- [Triton 语法速查](../notes/triton/triton-cheatsheet.md) — 写代码时随时查
- [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md) — Triton 代码对应什么 CUDA
- [Triton vs CUDA 对比](../notes/triton/triton-vs-cuda.md) — 编程模型差异

🔄 **本课固定两章**：

```text
5.5 LeetGPU：正确性与代码归档
5.6 服务器：真实性能
```

Agent 只做 review，不代写代码。本课最后有参考答案，但要求是：**先看完原理并去 LeetGPU 写题，提交后再打开参考对照**。

## 本课代码与进度索引（从这里一眼查看）

每个算子必须同时留下四个入口：LeetGPU 题目、通过后的原始 `solve`/kernel、本地 `solutions/` 文件、正确性/性能证据。LeetGPU 代码没有单独归档时，只能标为“代码缺失”，不能用 wrapper 或 reference 冒充。

| 算子 | LeetGPU 题目 | LeetGPU 原始代码 | 本地代码 / 证据 | 当前状态 |
|---|---|---|---|---|
| Triton Vector Add | [LeetGPU Challenges](https://leetgpu.com/challenges)（题目入口未单独归档） | **未单独保存**；当前仓库文件不是平台原始 `solve` | [`solutions/triton/vector_add.py`](../solutions/triton/vector_add.py) · [benchmark 记录](../solutions/triton/README.md#vector-add-benchmark) | `GPU_VALIDATED`，原始代码归档缺失 |
| Triton MatMul | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) · [题目规格](https://github.com/HaoyangPing0324/LeetGPU/blob/main/problems/02_Matrix_Multiplication.md) | [`matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)：通过后归档的 LeetGPU 原始 `solve`/kernel；[`matmul_leetgpu_wip.py`](../solutions/triton/matmul_leetgpu_wip.py) 为历史 TF32 失败快照 | [`matmul.py`](../solutions/triton/matmul.py)：服务器验证适配版，RTX 3090 最佳 22.033 ms / 18,713.5 GFLOPS | LeetGPU `LEETGPU_PASS`；服务器 `GPU_VALIDATED`；单元总体 `GPU_VALIDATED` |

以后新算子通过 LeetGPU 后，先补齐这一张索引和对应代码文件，再进入服务器 benchmark；平台代码与服务器适配版始终分开记录来源和状态。

---

## 当前单元卡

| 阶段 | 做什么 | 产出 | 验收 |
|---|---|---|---|
| LeetGPU | [5.5 LeetGPU：正确性与代码归档](#55-leetgpu正确性与代码归档) | 题目通过、原始 `solve`、本地代码、lesson 快照 | `LEETGPU_PASS`：A100-80GB，24.54 ms，55.3th percentile |
| 服务器 | [5.6 服务器：真实性能](#56-服务器真实性能) | 实际 GPU 型号、正确性复核、GFLOPS、配置对比 | `GPU_VALIDATED`：RTX 3090 已验证 |
| P0-lite | [Nsight Systems 逐块分析](../notes/triton/matmul-nsys-p0-lite-2026-08-30.md) | timeline、launch metadata、s3/s2 单变量实验、完整 raw logs | ✅ 完成；NCU counters 被 AutoDL 权限阻塞 |
| 下一步 | 缩小输出列 tile，验证 accumulator/register pressure | `128×32×128, w8, s3` 与当前最佳对照 | 单元尚未 `COMPLETE` |

> LeetGPU 已通过并归档，服务器真实 GPU 已验证，Nsight Systems P0-lite 已完成；MatMul 尚缺 NCU counters、PTX/SASS 和后续单变量验证，因此不标记 `COMPLETE`。

---

# Part 0：先建立正确的心智模型

## 0.1 Triton 是什么

Triton 是一个 **Python 写的 GPU 编程语言（DSL）+ 编译器**。你写的代码看起来像 Python，但它不是普通 Python 函数——`@triton.jit` 装饰的函数会被 Triton 编译器编译成 GPU 机器码（PTX → cubin），真正在 GPU 上跑。

关键心智：**Triton 不是"用 Python 写 CUDA"，而是"用 block 粒度描述计算，编译器替你处理线程"**。

CUDA 和 Triton 写的是同一套数学，只是抽象层级不同：

| 问题 | CUDA 的回答 | Triton 的回答 |
|---|---|---|
| 谁来执行 | thread（线程） | program（一个 block 对应一个 program） |
| 我控制什么 | 每个 thread 读写哪个元素 | 每个 program 处理哪个 tile |
| 线程怎么协作 | 你写 `__syncthreads()`、shared memory | 编译器自动插入同步、分配 shared memory |
| 访存怎么优化 | 你手动保证 coalesced、避免 bank conflict | 编译器自动做向量化 + 布局优化 |
| Tensor Core | 你写 `wmma` / `mma.sync` | `tl.dot`，编译器选指令 |

## 0.2 一次典型运行的分解

假设 `N = 1000, BLOCK_SIZE = 256`，启动 `add_kernel[grid]`：

```text
host:  add_kernel[grid=(4,)](x, y, out, N, BLOCK_SIZE=256)
                    │
                    ▼
GPU:   grid = 4 个 program（= 4 个 CUDA block）
       program 0: 处理元素 [0, 256)
       program 1: 处理元素 [256, 512)
       program 2: 处理元素 [512, 768)
       program 3: 处理元素 [768, 1024)   ← 但 N 只有 1000，最后 24 个越界
                    │
                    ▼
       每个 program 内部：编译器把它变成一个 CUDA block，
       内部 256 个 thread 协作完成这 256 个元素，
       tl.load 会自动让相邻 thread 读相邻地址（coalesced）
```

所以你在 Triton 里写的是"**一个 program 怎么处理一个 tile**"，而不是"一个 thread 怎么处理一个元素"。

## 0.3 编译链路

```text
@triton.jit 的 Python 函数
   → TritonIR（中间表示，能看到你的程序结构）
   → TTGIR（加了 layout/sync 的 IR，能看到编译器怎么安排线程和内存）
   → PTX（NVIDIA 汇编）
   → cubin（GPU 机器码）
```

调试时可以用 `kernel.asm` 查看中间结果（见 Part 7），这是 Triton 比"黑盒 Python"强的地方：你能看到编译器把你的 tile 代码变成了什么。

## 0.4 为什么性能能接近手写 CUDA

Triton 帮你做四件事，恰好是手写 CUDA 最花时间的四件事：

1. **自动向量化**：`tl.load` 一段连续数据时，编译器生成 128-bit（比如 4 个 float）的向量加载，而不是逐元素加载
2. **自动 shared memory**：`tl.load` 的 tile 会被安排经过 shared memory 或寄存器，编译器决定
3. **自动同步**：需要同步的地方自动插 `bar.sync`
4. **自动选指令**：`tl.dot` 根据 dtype 和硬件选 `mma` / `wgmma` 等 tensor core 指令

你省下的时间花在更重要的地方：**算法和分块策略**。

---

# Part 1：环境准备

## 1.1 两个运行环境，两种用途

| 环境 | 位置 | 用途 | 性能结论 |
|---|---|---|---|
| CPU 解释器 | 本机 venv（Windows + triton-windows + CPU torch） | 验证逻辑正确性 | ❌ 不能产生性能结论 |
| 真实 GPU | AutoDL 实际 NVIDIA GPU（CUDA torch） | 跑性能、验收 | ✅ 唯一的性能来源 |

## 1.2 验证环境

本机 venv 已配好（2026-08-10），先确认能 import：

```powershell
.venv\Scripts\python.exe -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

真实 GPU 环境跑通后，确认 CUDA 可见：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应输出 `True NVIDIA GeForce RTX 3090` 之类；以本次实际租到的 GPU 型号为准。

## 1.3 CPU 解释器怎么用

Triton 提供一个 CPU 解释模式：不编译、不启动 GPU，而是用 Python 逐行模拟 kernel 的逻辑。只用来**验正确性**。

```powershell
$env:TRITON_INTERPRET='1'
.venv\Scripts\python.exe solutions/triton/vector_add.py
```

注意两点：

- 解释器支持大部分 Triton 语义（`tl.load`、`tl.store`、`tl.program_id`、`tl.arange` 等），但**不支持性能分析**
- 解释模式下即使本机没有 GPU 也能跑，所以写代码阶段不依赖服务器

## 1.4 常见环境问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `import triton` 失败 | Windows 上官方 wheel 不存在 | 用 triton-windows（本仓库 venv 已装好） |
| `torch.cuda.is_available()` 为 False | 装的是 CPU 版 torch，或没有 GPU | 本机正常；性能验收去服务器 |
| 解释器跑出来的结果和真机不同 | 边界 case 用了未定义值（`other` 没写） | 所有 mask 的 load 都显式给 `other=` |
| 第一次跑很慢 | JIT 编译（编译缓存） | 正常，第二次起会用缓存 |

---

# Part 2：Vector Add 逐层拆解

## 2.1 问题

```text
out[i] = x[i] + y[i]    for i in [0, N)
```

三个一维 float32 张量，长度 N。每个元素 4 字节。

## 2.2 内存布局

GPU 显存里，`x`、`y`、`out` 各是一段**连续**内存：

```text
地址：  x[0] x[1] x[2] ... x[N-1]
        y[0] y[1] y[2] ... y[N-1]
        out[0] out[1] ... out[N-1]
        每个元素占 4 字节
```

连续意味着：我们可以按"一段一段"的方式处理，每段就是 BLOCK_SIZE 个元素。这也是 GPU 访存快的前提——连续地址的访问会被合并（coalesced）成少量大请求。

## 2.3 并行策略

把 N 个元素切成若干段，每段交给一个 program：

```text
N = 1000, BLOCK_SIZE = 256

段 0（program 0）: [0, 256)
段 1（program 1）: [256, 512)
段 2（program 2）: [512, 768)
段 3（program 3）: [768, 1024)   ← 只有 768..999 有效
```

分段数 = `ceil(N / BLOCK_SIZE) = ceil(1000/256) = 4`。

## 2.4 program_id 和 grid

```python
pid = tl.program_id(0)     # 当前 program 在 grid 里的编号，0, 1, 2, 3
```

- `tl.program_id(axis)`：等价于 CUDA 的 `blockIdx.x`
- grid 是一维的（一个数字），因为这里只需要沿元素方向切分
- 二维问题（比如 matmul）会用 `tl.program_id(0)` 和 `tl.program_id(1)`，见 Part 5

## 2.5 offsets：这个 program 负责哪些元素

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

- `tl.arange(0, BLOCK_SIZE)` 生成 `[0, 1, 2, ..., BLOCK_SIZE-1]` 这样一个整数向量（等价于 CUDA 里一个 block 的 `threadIdx.x` 全集）
- `pid * BLOCK_SIZE` 是这段的起点
- 对 program 3：`offsets = 768 + [0..255] = [768, 769, ..., 1023]`

**注意**：`offsets` 是**元素索引**，不是字节地址。后面 `x_ptr + offsets` 的含义是"第 offsets 个元素的地址"，编译器会自动乘上元素大小（float 就是 4 字节）。

## 2.6 mask：越界保护

program 3 负责到 1023，但 N=1000，`offsets >= 1000` 的 24 个位置不存在。必须用 mask 挡住：

```python
mask = offsets < n_elements
```

```text
offsets:  768 ... 998 999 | 1000 ... 1023
mask:     True ... True True | False ... False
                               ↑ 这 24 个位置不读写
```

**为什么必须有 mask**：

- 不 mask 的 `tl.load` 会去读越界地址——可能读到别的数据或直接非法访问
- 不 mask 的 `tl.store` 会写越界——可能破坏其他内存，甚至让 kernel 崩
- mask 是**逐元素**的：一个 program 里有效和无效元素可以同时存在

## 2.7 tl.load / tl.store

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

- `tl.load(ptr + offsets, mask=mask, other=0.0)`：按 offsets 加载一批元素；mask=False 的位置返回 `other`
- `tl.store(ptr + offsets, value, mask=mask)`：按 offsets 写回；mask=False 的位置不写
- `other=0.0` 是**必须的**：加法里被 mask 掉的位置不参与最终结果，但如果 `other` 不给定，这些位置的值是未定义的（可能是垃圾数），结果会错

## 2.8 为什么 BLOCK_SIZE 必须是 tl.constexpr

```python
def add_kernel(..., BLOCK_SIZE: tl.constexpr):
```

`tl.constexpr` 表示这个参数是**编译期常量**：kernel 被编译时，BLOCK_SIZE 的值会直接写进代码里。

为什么需要？因为：

- `tl.arange(0, BLOCK_SIZE)` 的长度必须是编译期确定的（它决定寄存器/向量宽度）
- BLOCK_SIZE 影响循环展开、shared memory 分配、寄存器分配——这些都得在编译时知道
- 运行时普通参数（比如 `n_elements`）则不同，它们按值传进 kernel，不参与编译期决策

实际效果：`BLOCK_SIZE=256` 和 `BLOCK_SIZE=1024` 是两个不同的编译版本，各有各的缓存。

## 2.9 host 端启动：grid 和 cdiv

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
```

- `triton.cdiv(a, b)` = ceil(a / b)，向上取整除法
- grid 是"启动多少个 program"，这里 = ceil(N / BLOCK_SIZE)
- `grid` 写成 **lambda，参数叫 meta**：因为 `BLOCK_SIZE` 是 constexpr，编译器需要知道它才能算 grid 大小；meta 里装着所有 constexpr 参数的值。写成 `lambda meta:` 就能用 `meta['BLOCK_SIZE']`
- 你也可以直接写死数字：`grid = (triton.cdiv(N, 256),)`，但那样以后换 BLOCK_SIZE 就得改两处

## 2.10 完整 kernel（参考答案：先去 LeetGPU 写题并提交，再打开）

<details>
<summary>参考答案（写完后对照，不要提前看）</summary>

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # 1. 我是第几个 program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 2. 我负责哪些元素
    mask = offsets < n_elements                # 3. 哪些元素有效
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # 4. 加载 x
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)   # 5. 加载 y
    tl.store(output_ptr + offsets, x + y, mask=mask)     # 6. 相加写回

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
    return output
```

逐行解释：

1. `tl.program_id(0)`：得到当前 program 编号（0 到 grid-1）
2. `offsets`：元素索引向量。program 3 得到 `[768..1023]`
3. `mask`：逐元素判断是否越界
4-5. 加载两个输入。注意 `other=0.0`：mask=False 的位置返回 0，不参与计算
6. 相加并写回。mask=False 的位置不写，所以越界不会破坏内存

</details>

## 2.11 验证：正确性和边界

```python
torch.manual_seed(0)
for N in [1, 256, 257, 1000, 2**20]:
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    out = add(x, y)
    torch.testing.assert_close(out, x + y)
    print(f'N={N}: OK')
```

为什么这些 N 必须测：

| N | 测什么 |
|---|---|
| 1 | 只有一个元素，grid=1，BLOCK 远大于 N，全靠 mask |
| 256 | N 正好等于 BLOCK_SIZE，没有 mask 情况（也测不出 mask 问题） |
| 257 | 多一个元素，grid=2，第二个 program 只有 1 个有效元素 |
| 1000 | 典型非整数倍，最后一段大部分是 mask |
| 2^20 | 大数组，顺带为性能测试做准备 |

如果 N=257 错，基本就是 mask 或 grid 算错了。

---

# Part 3：LeetGPU 对照

LeetGPU 通过并同步本地后，再打开参考：

1. [LeetGPU](https://github.com/dsl-learn/LeetGPU) 的 Triton vec_add 题/解
2. [dsl-learn/triton-tutorial](https://github.com/dsl-learn/triton-tutorial) 的 ex1-vector_add（项目已归档，但 vec_add 部分仍可用）
3. [Triton 官方教程 01-vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

对照时只问三个问题：

1. **grid 和 BLOCK_SIZE 为什么不同/相同？** —— BLOCK_SIZE 没有唯一正确答案，它是性能和正确性的折中；官方用 1024 不代表 256 错
2. **mask 的 `other` 影响结果吗？为什么必须是 0.0？** —— 加法里被 mask 掉的 x/y 如果不返回 0，垃圾值会污染结果
3. **为什么用 `torch.empty_like` 而不是 `zeros_like`？** —— kernel 会写满每个有效位置，`zeros_like` 白白多一次清零；代价是你必须保证 grid 全覆盖

> 注意：LeetGPU 的题目有自己的 `solve` 函数签名（比如输入是裸指针 + N），和本地写法不同。代码归档看 5.5，真实 GPU 看 5.6；不要把本地 wrapper 当成平台原始 `solve`。

---

# Part 4：性能分析

## 4.1 为什么 vec_add 是带宽瓶颈（roofline 视角）

每个元素的工作量：

```text
读 x: 4 字节
读 y: 4 字节
写 out: 4 字节
计算: 1 次加法（1 FLOP）

算术强度 = 1 FLOP / 12 字节 ≈ 0.083 FLOP/B
```

性能量级示例（不同 GPU 以实际规格为准）：

```text
FP32 算力 ≈ 82 TFLOPS
显存带宽 ≈ 1008 GB/s
两者交叉点 ≈ 82e12 / 1008e9 ≈ 81 FLOP/B
```

算术强度（0.083）比交叉点（81）低了约 1000 倍，意味着**计算能力远远过剩，瓶颈完全是显存带宽**。所以优化 vec_add 的目标不是"让计算更快"，而是"让访存更高效、数据量更少"。这也是为什么后面做 Flash Attention 时，核心指标是"省了多少次 HBM 读写"。

## 4.2 有效带宽怎么测

```python
import triton

# 计时工具：返回平均耗时（毫秒）
mean_ms = triton.testing.do_bench(lambda: add(x, y))

# 有效带宽 = 总数据量 / 时间
bytes_moved = 3 * N * 4   # 读 x + 读 y + 写 out
bandwidth_gbps = bytes_moved / (mean_ms / 1000) / 1e9
print(f'{bandwidth_gbps:.1f} GB/s')
```

注意：

- N 要足够大（至少 2^25），否则 launch 开销（几十微秒）占大头，测出来虚低
- 分别测 `torch.add` 和你的 kernel，用同一个 N、同一个计时方法，才有可比性
- BLOCK_SIZE 分别试 128 / 256 / 512 / 1024，记录结果

## 4.3 期望值

```text
N = 2^25 = 33,554,432
数据量 = 3 × 33,554,432 × 4 ≈ 0.403 GB
示例：如果 mean_ms = 0.50 ms → 带宽 = 0.403 / 0.0005 ≈ 805 GB/s
本次 AutoDL 实测：0.479 ms → 840.1 GB/s（torch.add: 843.0 GB/s）
```

RTX 4090 教材参考锚点（本次 AutoDL RTX 3090 实测结果单独记录）：

| 指标 | 数值 |
|---|---|
| 理论带宽 | ~1008 GB/s |
| 合格 Triton vec_add | 600+ GB/s |
| `torch.add` | 通常 650-850 GB/s |

**如果比 torch.add 低 20% 以上**，按这个顺序查：

1. N 够不够大（< 2^20 就别测性能）
2. BLOCK_SIZE 是不是太小（128 以下容易带宽上不去）或太大（1024 以上占用率下降）
3. 是不是忘了解释器变量（`TRITON_INTERPRET` 开着时测的是 Python 模拟，数字没有意义）

## 4.4 记录

把结果写进 `solutions/triton/README.md`，这是 B1 验收的一部分：

```text
vector_add (2026-08-23)
- N=2^25, BLOCK_SIZE=256, RTX 3090 (AutoDL)
- 正确性: assert_close OK (N=1/256/257/1000/2^20)
- 带宽: 840.1 GB/s (torch.add: 843.0 GB/s)
- 结论: 内存带宽瓶颈，BLOCK_SIZE=256 最优；Triton 达到 torch.add 的 99.7%
```

---

# Part 5：MatMul 详解

## 5.1 问题

```text
C[M, N] = A[M, K] @ B[K, N]
```

每个输出元素 `C[i][j]` 都是 A 的第 i 行和 B 的第 j 列做点积：

```text
C[i][j] = Σ_k A[i][k] * B[k][j]
```

## 5.2 输出分块：每个 program 算一个 tile

朴素想法：每个 program 算 C 的一个元素——太浪费，一个 program 只干一个点积，GPU 的并行度和数据复用都上不去。

正确做法：**每个 program 算 C 的一个 `BLOCK_M × BLOCK_N` 小块**。

```text
M = N = K = 1024, BLOCK_M = 128, BLOCK_N = 128

C 是 1024×1024，切成 8×8 = 64 个 tile
grid = (8, 8)，即 64 个 program

program (pid_m=3, pid_n=5) 负责：
  行 [384, 512) × 列 [640, 768) 这一块
```

为什么要这样分块？因为计算一个输出 tile 时，A 的行块和 B 的列块会被**反复使用**（K 维循环），分块让这些数据能留在寄存器/shared memory 里复用，而不是每次从 HBM 重读。这正是 GEMM 优化的核心：**数据复用**。

### LeetGPU 题目入口（MatMul 先做这道）

- 题目：[Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)
- 题号/规格：[02_Matrix_Multiplication.md](https://github.com/HaoyangPing0324/LeetGPU/blob/main/problems/02_Matrix_Multiplication.md)
- 任务：FP32、row-major 的 A(M×N) × B(N×K) = C(M×K)。
- 约束：M/N/K ≤ 8192；性能测试形状为 8192×6144×4096。
- 做法：打开题目后选择 Triton，保留平台 `solve` 接口；先提交通过，再进入真实 GPU benchmark。

当前不做 #29 `General Matrix Multiplication (GEMM)`；那是后续优化题。
## 5.3 指针计算（从输出 tile 反推）

三个矩阵都是 row-major 连续存储：

```text
A[i][k] 的地址 = A + i * K + k
B[k][j] 的地址 = B + k * N + j
C[i][j] 的地址 = C + i * N + j
```

当前 program 负责的输出 tile 起点：

```text
pid_m = tl.program_id(0)   # 负责哪一组行
pid_n = tl.program_id(1)   # 负责哪一组列

行索引: offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
列索引: offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
K 索引: offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]
```

A 的 tile 指针（形状 [BLOCK_M, BLOCK_K]）：

```python
a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
```

这行是在生成 A tile 的每一个元素地址。此处沿用本节的教材记号 `A[M, K]`：row-major 连续存储时，`A[m, k]` 的一维元素偏移是 `m * K + k`。`A + offset` 中的 `offset` 是**元素个数**，不是字节数。

例如 `K=8`、`BLOCK_M=4`、`BLOCK_K=4`，且当前 program 的 `pid_m=1`、归约块从 `k=0` 开始：

```text
offs_m = [4, 5, 6, 7]
offs_k = [0, 1, 2, 3]

offs_m[:, None] * 8 =
[[32],
 [40],
 [48],
 [56]]                 # 这四个 A 行的起始地址

offs_k[None, :] = [[0, 1, 2, 3]]

两者广播相加 =
[[32, 33, 34, 35],
 [40, 41, 42, 43],
 [48, 49, 50, 51],
 [56, 57, 58, 59]]
```

这正是 `A[4:8, 0:4]` 这个 `[BLOCK_M, BLOCK_K] = [4, 4]` tile 的 16 个地址。`[:, None]` 把行索引变成列向量，`[None, :]` 把 K 索引变成行向量；二者才能通过广播组成二维 tile。

B 的 tile 指针（形状 [BLOCK_K, BLOCK_N]）：

```python
b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
```

C 的 tile 指针（形状 [BLOCK_M, BLOCK_N]）：

```python
c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
```

**为什么用 `[:, None]` / `[None, :]`**：把一维索引扩成二维，才能生成 [BLOCK_M, BLOCK_K] 这样的二维 tile。`offs_m[:, None]` 是列向量（BLOCK_M 行 1 列），`offs_k[None, :]` 是行向量（1 行 BLOCK_K 列），相加广播成完整二维索引。

K 循环时指针怎么走：

```python
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)   # [BLOCK_M, BLOCK_K]
    b = tl.load(b_ptrs)   # [BLOCK_K, BLOCK_N]
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K            # A 往右走一个 BLOCK_K（列偏移 = BLOCK_K）
    b_ptrs += BLOCK_K * N        # B 往下走一个 BLOCK_K（行偏移 = BLOCK_K 行 × N 列）
```

第二个步进（`BLOCK_K * N`）是新手最常错的地方：B 是 [K, N]，跳过一个 BLOCK_K 行，需要跨过 `BLOCK_K × N` 个元素。

## 5.4 tl.dot：Tensor Core 入口

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
acc += tl.dot(a, b)   # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
```

`tl.dot` 的语义：

- 输入形状必须匹配：`[BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]`
- **累加器用 fp32**：即使 a/b 是 fp16，累加也在 fp32 做，精度更好
- 编译器看到 `tl.dot` 会生成 tensor core 指令（NVIDIA Ampere/Ada 上通常是 `mma.sync`），而不是普通乘加循环

dtype 决定能不能用 tensor core：

| dtype | NVIDIA Ampere/Ada 行为 | 备注 |
|---|---|---|
| fp16 / bf16 | 走 tensor core，最快 | 模型推理/训练的主流 |
| fp32 | 默认 TF32（`tl.dot` 的 `input_precision` 控制） | 精度比真 fp32 低，性能接近 fp16 |
| fp32 (ieee) | 普通 CUDA core 算 | 精度最高，最慢 |

这不是只改一个 Python 参数，而是在选择不同计算单元：

```text
tl.dot(..., input_precision="ieee")
→ FP32 CUDA pipeline
→ 严格 FP32 基线；当前 MatMul 的 16,706 GFLOPS 属于这条路径

tl.dot(..., input_precision="tf32")（或默认允许 TF32 的路径）
→ Ampere Tensor Core 的 TF32 MMA
→ FP32 tensor 不改存储格式，但乘法输入只保留 10-bit mantissa，FP32 累加
```

因此性能实验分两步，不能混写成绩：

1. **先固定 IEEE FP32**：PyTorch 同时关闭 TF32，只 sweep `BLOCK_M/N/K`、`num_warps`、`num_stages`。这回答“代码生成和 tile 是否更好”。
2. **再单独做 TF32 对照**：保持同一 shape 和 tile，记录相对 IEEE reference 的 `max_abs_error`、`max_rel_error`、耗时和 GFLOPS。这回答“可接受的精度损失能否换来 Tensor Core 吞吐”。

若 TF32 更快，首先归因于计算路径从 CUDA FP32 pipeline 切到 Tensor Core；不能把这部分收益误写成 tile 优化。PyTorch 当前推荐用 `torch.set_float32_matmul_precision("highest" | "high" | "medium")` 控制内部精度；`highest` 是严格 FP32，`high`/`medium` 在支持的 CUDA GPU 上允许 TF32。

BLOCK 尺寸注意：

- `tl.dot` 一般要求每个维度 ≥ 16（tensor core 的 mma 指令最小形状）
- BLOCK_K 太小（比如 8）会让 dot 退化成低效路径，还浪费 shared memory 带宽
- 常见组合：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32/64`

### A100 资源推导：先选一个可解释的初始 tile

LeetGPU #02 使用 `A[M, N] @ B[N, K] = C[M, K]`。本小节约定：`BLOCK_M` 为输出行 tile，`BLOCK_N` 为归约 tile，`BLOCK_K` 为输出列 tile。先从保守配置开始：

```python
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 64
num_warps = 4
num_stages = 3
```

`num_warps=4` 表示一个 Triton program（约等于一个 CUDA block/CTA）使用 4 个 warp，即 128 threads。A100 每个 SM 最多有 2048 threads、64 warps、65536 个 32-bit registers 和 164 KB shared memory；所以若只看 thread/warp 上限，一个 SM 最多能驻留 `2048 / 128 = 16` 个此类 block。实际数目还要受 registers 和 shared memory 限制。

每个 program 的单轮归约涉及三个逻辑 tile：

```text
A[64, 32]：2048 个 FP32 值，容量等价 8 KB
B[32, 64]：2048 个 FP32 值，容量等价 8 KB
C[64, 64]：4096 个 FP32 accumulator，容量等价 16 KB
```

不要把上述数字相加后当作某个硬件资源池的占用。C accumulator 是跨所有 N 维循环持续存活的线程私有状态，其**最低** register 需求是 4096 个 32-bit registers/block，或在 128 threads/block 下平均 32 registers/thread。A/B 是当前轮输入；它们会以 register fragment 形式进入 `tl.dot`，编译器也可能为其分配 shared-memory pipeline/staging，精确用量不能仅由源代码推得。

一份完整 A+B 输入 tile 的容量等价为 16 KB。若把 `num_stages=3` 粗略看作三份完整 A/B pipeline buffer，则 shared-memory 估算为约 48 KB/block，进而得到约 `floor(164 / 48) = 3 block/SM`、12 warp/SM、384 threads/SM。这只是帮助建立直觉的估算；Triton 的真实 shared-memory、register 与 resident-block 数必须以编译结果和 profiler 为准。

因此 `64×64` 的目的不是追求 16 block/SM，而是让 C accumulator、A/B fragment、地址和临时变量有足够余量，降低 register spill 风险。正确性通过后，再比较 `128×32×64` 与 `128×32×128`，用 A100 的 GFLOPS、registers per thread、shared memory per block 和 occupancy 选择最终配置。

## 5.5 LeetGPU：正确性与代码归档

**题目入口**：[#02 Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) · [题目规格](https://github.com/HaoyangPing0324/LeetGPU/blob/main/problems/02_Matrix_Multiplication.md)。题目要求 FP32、row-major 的 `A(M×N) × B(N×K) = C(M×K)`。

**当前代码入口**：

- LeetGPU 最终原始 `solve`/kernel 归档：[`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)，状态 `LEETGPU_PASS`；
- 历史 TF32 失败快照：[`solutions/triton/matmul_leetgpu_wip.py`](../solutions/triton/matmul_leetgpu_wip.py)，仅用于复盘，不能代表当前平台版本；
- 服务器验证版：[`solutions/triton/matmul.py`](../solutions/triton/matmul.py)，包含正确性测试和 GFLOPS benchmark；
- 参考实现：[`reference/triton/matmul/matmul.py`](../reference/triton/matmul/matmul.py)，只在自己提交后对照。

### 当前单元卡（2026-08-28）

| 项目 | 当前状态 |
|------|------|
| LeetGPU | `LEETGPU_PASS`：SuccessPublicTrace，A100-80GB，2026-08-28 22:23:16，24.54 ms，55.3th percentile |
| 服务器 | `GPU_VALIDATED`：`solutions/triton/matmul.py` 已在 RTX 3090 完成正确性与性能验证 |
| 单元总体 | `GPU_VALIDATED`，尚未 `COMPLETE` |
| P0-lite | Nsight Systems s3/s2 对照已完成；[详细分析](../notes/triton/matmul-nsys-p0-lite-2026-08-30.md) · [s3 raw](../notes/triton/logs/2026-08-29-matmul-k256-s3-nsys.txt) · [s2 raw](../notes/triton/logs/2026-08-30-matmul-k256-s2-nsys.txt) |
| 下一步 | `128×32×128, w8, s3`：缩小 accumulator，观察 Reg/Trd 与性能 |

下面是通过后的 LeetGPU 原始代码快照，来源为 [`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)，已与平台版本同步。历史 [`solutions/triton/matmul_leetgpu_wip.py`](../solutions/triton/matmul_leetgpu_wip.py) 保留默认 TF32 精度失败过程：4×4 case 最大绝对误差为 `0.1275177001953125`；指定 IEEE 输入精度后平台通过。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # A[M, N] @ B[N, K] = C[M, K]，N 是归约维度。
    for n in range(0, N, BLOCK_N):
        offset_n = n + tl.arange(0, BLOCK_N)

        ptr_a = a + offset_m[:, None] * N + offset_n[None, :]
        ptr_b = b + offset_n[:, None] * K + offset_k[None, :]

        mask_a = (offset_m[:, None] < M) & (offset_n[None, :] < N)
        mask_b = (offset_n[:, None] < N) & (offset_k[None, :] < K)

        tile_a = tl.load(ptr_a, mask=mask_a, other=0.0)
        tile_b = tl.load(ptr_b, mask=mask_b, other=0.0)
        acc += tl.dot(tile_a, tile_b, input_precision='ieee')

    mask_c = (offset_m[:, None] < M) & (offset_k[None, :] < K)
    ptr_c = c + offset_m[:, None] * K + offset_k[None, :]
    tl.store(ptr_c, acc, mask=mask_c)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
          M: int, N: int, K: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
```

这一单元的 LeetGPU 正确性门已完成，最终原始代码已保存为 [`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)。归档口径如下：

1. 选择 Triton，保留平台 `solve` 接口；
2. 先写单 tile，再加入 N 维归约循环；
3. 覆盖非整除边界，提交并通过；
4. 归档当次原始 `solve`/kernel 为 `solutions/triton/matmul_leetgpu.py`，只将 `tl.dot` 的输入精度指定为 IEEE；
5. 在本课索引、`PATH.md` 和 `solutions/triton/README.md` 填上题目、代码和通过状态。

历史 `solutions/triton/matmul_leetgpu_wip.py` 是默认 TF32 的失败快照，不覆盖、不替代最终归档；服务器结果仍与平台结果分开记录。

### 参考实现（只用于对照）

卡住后再参考 [`reference/triton/matmul/matmul.py`](../reference/triton/matmul/matmul.py)，重点对照 mask、K/N 维步进和输出写回，不直接复制成自己的完成记录。

## 5.6 服务器：真实性能

LeetGPU #02 已通过且原始代码已经同步到本地；服务器使用单独的适配版做真实性能验证。两条证据分别记账，MatMul 单元总体状态为 `GPU_VALIDATED`，尚未 `COMPLETE`。

1. 同步通过版本到 AutoDL；
2. 用 `torch.cuda.get_device_name(0)` 记录实际 GPU 型号；
3. 先与 `torch.matmul` 对齐正确性，再测固定 shape 的耗时和 GFLOPS；
4. 最后比较 BLOCK、`num_warps`、`num_stages`，再考虑 autotune；
5. 把配置、GPU、耗时、GFLOPS 和 PyTorch/cuBLAS 对照写入 README。

### 本次服务器结果（2026-08-26）

```text
GPU: NVIDIA GeForce RTX 3090 (AutoDL)
shape: M=8192, N=6144, K=4096
正确性: 4 组测试全部 OK，包括 M/N/K 非 tile 整除的 (257, 513, 129)
精度: FP32，tl.dot(input_precision="ieee")，torch.backends.cuda.matmul.allow_tf32=False
Triton: 24.681 ms，16,706.0 GFLOPS
torch.mm: 17.120 ms，24,083.3 GFLOPS
相对性能: Triton ≈ torch.mm 的 69.4%，耗时约慢 1.44x
服务器状态: GPU_VALIDATED（服务器适配版）
```

解释：当前版本先保证 tile、边界 mask 和 FP32 语义正确；初始配置与 cuBLAS 的差距来自 tile、`num_warps`、`num_stages` 和数据搬运仍未充分调优。

### 本次配置 sweep（2026-08-26）

固定 IEEE FP32、RTX 3090 和 `8192×6144×4096`，结果如下：

| 配置 | 结果 | 相对 `torch.mm` |
|---|---:|---:|
| `64×32×64, w4, s3` | 24.924 ms / 16,542.7 GFLOPS | 68.8% |
| `128×32×64, w4, s3` | 22.298 ms / 18,491.3 GFLOPS | 76.9% |
| `128×32×128, w4, s3` | 28.354 ms / 14,541.8 GFLOPS | 60.4% |
| `128×64×128, w4, s3` | 编译失败：shared memory 131,072 B > 101,376 B 上限 | — |
| `128×32×256, w8, s3` | **22.033 ms / 18,713.5 GFLOPS** | **77.8%** |

当前最佳：`BLOCK_M=128, BLOCK_N=32, BLOCK_K=256, num_warps=8, num_stages=3`。相对 `64×32×64` baseline，耗时下降约 11.6%，GFLOPS 提升约 13.1%。`BLOCK_N=64` 的失败是 shared memory 资源超限，不是正确性问题。

详细的硬件机制、profiler 观察项和后续优化顺序见：[MatMul 性能分析记录](../notes/triton/matmul-performance-analysis.md)。

服务器端的调优顺序固定，避免把不同精度下的数字混在一起：

1. 运行服务器验证版 `python solutions/triton/matmul.py`。它先以 IEEE FP32 跑边界正确性，再比较 `64×32×64`、`128×32×64`、`128×32×128`、`128×64×128`、`128×32×256`；每行都和同样关闭 TF32 的 `torch.mm` 比。
2. 只从 IEEE sweep 中选最快配置，再记录一次 profiler：`ncu --set roofline -o matmul_ieee python solutions/triton/matmul.py --config <最快配置名>`。脚本支持 `--config` 单独运行一组，避免 profile 把 sweep 中的所有 kernel 混进报告。重点看 roofline、registers/thread、shared memory/block、active warps、L1/L2/HBM traffic；occupancy 低本身不是失败，必须和吞吐、stall/traffic 一起解释。
3. IEEE 的最快配置归档后，复制为单独的 TF32 对照实验：保持 shape/tile 不变，只改 Triton 的 `input_precision` 和 PyTorch 的 MatMul precision，并额外报告相对 IEEE reference 的 `max_abs_error`、`max_rel_error`。TF32 对照是“精度换 Tensor Core 吞吐”，不替代 IEEE FP32 的正确性/性能记录。

**性能调优：autotune**

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
```

autotune 在第一次启动时对每个 config 各跑一遍 benchmark，选最快的，之后都用它。

- `configs` 里的字典 key 必须和 kernel 参数名一致（BLOCK_M/BLOCK_N/BLOCK_K）
- `num_warps` 和 `num_stages` 也是编译选项：warps 多 = 并行度大；stages 多 = 更深流水线（用更多 shared memory 换更少的访存等待）
- `key=['M','N','K']`：告诉 autotune 哪些运行时参数变化时需要重新调参（不同形状的最优 config 不一样）
- 用了 autotune 后，启动 kernel **不要再传** BLOCK_M/BLOCK_N/BLOCK_K 这些 constexpr（config 会填）

### GFLOPS 怎么算

> **记录口径**：LeetGPU #02 已通过并完成原始代码归档；下列性能数字明确属于服务器适配版 benchmark。性能形状为 M=8192、N=6144、K=4096。


```text
FLOPs = 2 × M × N × K   （每个输出元素做 K 次乘 + K 次加）

例子：M=N=K=4096，耗时 1.2 ms
FLOPs = 2 × 4096³ = 137.4 GFLOP
TFLOPS = 137.4e9 / 1.2e-3 ≈ 114 TFLOPS
```

RTX 4090 教材参考锚点（fp16；实际 AutoDL GPU 以当前型号为准）：

```text
理论峰值 ≈ 330 TFLOPS（fp16 dense）
torch.matmul 通常能到 250-300 TFLOPS（cuBLAS，大矩阵）
初版 Triton matmul 能到 50-150 TFLOPS 就算入门
```

先和 `torch.matmul` 比，别和自己比。如果差很多，先检查：dtype 是不是 fp16、BLOCK 是不是太小、autotune 有没有生效。

---

# Part 6：常见坑（每条都带原理）

| 坑 | 现象 | 原因 | 怎么避免 |
|---|---|---|---|
| `tl.arange` 大小不是 2 的幂 | 编译报错 | arange 长度必须是编译期确定的 2 的幂，编译器按此分配向量宽度 | BLOCK 一律取 2 的幂（128/256/512...） |
| 把 offsets 当成字节 | 结果错得离谱或越界 | `ptr + offsets` 是元素偏移，编译器按 dtype 换算字节 | 记住 `x_ptr + i` 是"第 i 个元素" |
| 忘 mask | 结果错 / 崩 | 最后一段读写了不存在的元素 | 非整除的 N 必须 mask，且测试 N=257 |
| 忘 `other=` | 结果时对时错 | mask=False 位置返回未定义值 | 所有 mask 的 load 都写 `other=0.0` |
| 解释器测性能 | 数字毫无意义 | 解释器是 Python 逐行模拟，不是编译后的 GPU 代码 | 性能只在真机测 |
| grid 没覆盖全部元素 | 结果尾部全 0 | 比如用 `N // BLOCK_SIZE` 而不是 cdiv，最后一段被丢掉 | 用 `triton.cdiv` |
| matmul 忘加 `[:, None]` | 形状错/广播错 | 一维索引无法生成二维 tile | 行索引 `[:, None]`，列索引 `[None, :]` |
| B 的 K 循环步进写错 | K>BLOCK_K 时结果错 | B 跳过一个 BLOCK_K 行要跨 `BLOCK_K * N` 个元素，不是 `BLOCK_K` | 用 5.3 的公式，或先跑 BLOCK_K=K 验证 |
| `tl.dot` dtype 不匹配 | 编译/运行报错 | a、b 必须是同 dtype | 先统一 fp16 或 fp32 |
| 一上来就 autotune | 不知道问题在哪 | autotune 只是找最快 config，不修 bug | 先正确，再手动试 BLOCK，最后 autotune |

---

# Part 7：调试手段

| 手段 | 怎么用 | 解决什么 |
|---|---|---|
| CPU 解释器 | `TRITON_INTERPRET=1 python xxx.py` | 逻辑错误、边界错误，不依赖 GPU |
| 边界测试 | N=1/256/257/1000 | mask 和 grid 错误 |
| `assert_close` | 和 `torch.add` / `torch.matmul` 对比 | 正确性 |
| 打印 IR | `print(add_kernel.asm['ttgir'])`（编译后） | 看编译器做了什么布局/同步 |
| 打印 PTX | `print(add_kernel.asm['ptx'])` | 确认 `tl.dot` 是否真的生成了 mma 指令 |
| autotune 日志 | `TRITON_PRINT_AUTOTUNING=1` | 看每个 config 的实测耗时，确认 autotune 生效 |
| `do_bench` | `triton.testing.do_bench(fn)` | 稳定计时（自动 warmup + 多次取均值） |

调试顺序固定：解释器跑对 → 边界全过 → 真机跑对 → 才谈性能。

---

# ✅ 本课验收清单

环境与基础：

- [ ] `import torch, triton` 能跑
- [x] vec_add 已在 LeetGPU 题目编辑器写完并通过，之后完成 CPU 解释器 + 真机验证，与 `torch.add` 对齐
- [x] N=1 / 256 / 257 / 1000 边界正确
- [x] LeetGPU Triton vec_add 通过（2026-08-20）
- [ ] LeetGPU vec_add 原始 `solve` 单独归档（当前 `vector_add.py` 是 wrapper）

理解：

- [ ] 能解释 `tl.program_id` / `tl.arange` / `tl.load` / `tl.store` 对应 CUDA 的什么
- [ ] 能解释 mask 和 other 的作用，以及为什么加法必须 `other=0.0`
- [ ] 能说清 vec_add 为什么是带宽瓶颈（算术强度 vs 交叉点）
- [ ] 能徒手算出一个 N、BLOCK_SIZE 下的 grid 大小和每段范围

性能（B1 完成标准）：

- [x] 真实 GPU 实测，带宽数字记入 `solutions/triton/README.md`（AutoDL RTX 3090：840.1 GB/s）

- [x] 能解释你的 GB/s 和 `torch.add` 差多少、为什么

进阶（B1 之后）：

- [ ] matmul 单 tile 跑通（BLOCK_K=K）
- [x] matmul K 循环跑通，记录 GFLOPS（RTX 3090 最佳：18,713.5 GFLOPS）
- [x] 至少一轮 tile/warp/stage sweep（最佳：128×32×256，w8，s3）
- [x] MatMul LeetGPU 原始 `solve`/kernel 已归档到 [`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)，状态 `LEETGPU_PASS`
- [x] MatMul 服务器适配版 [`solutions/triton/matmul.py`](../solutions/triton/matmul.py) 已在 RTX 3090 `GPU_VALIDATED`
- [x] Nsight Systems P0-lite：s3/s2 timeline + launch metadata + 完整日志归档
- [ ] NCU counters / PTX/SASS / accumulator-register 单变量实验（因此单元尚未 `COMPLETE`）
- [ ] autotune 至少给出一组调参结论（哪个 config 快、为什么）

---

# 资源

| 想深入 | 去看 |
|---|---|
| Triton 官方教程 01-vector-add | https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html |
| Triton 官方教程 03-matrix-multiplication | https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |
| CUDA-MODE L14（Triton 引导） | https://github.com/cuda-mode/lectures —— 写完 vec_add 再看 |
| CUDA-MODE L9（reductions） | https://github.com/cuda-mode/lectures —— 写 softmax（B2）前看 |
| Triton 中文教程（已归档） | https://github.com/dsl-learn/triton-tutorial —— vec_add/转置部分可用 |
| LeetGPU | https://github.com/dsl-learn/LeetGPU |
| CUDA-MODE 中文笔记 | https://github.com/BBuf/how-to-optim-algorithm-in-cuda |
| 下一课 | B2 Triton Fused Softmax（Lesson 按需生成；前置：online softmax 已掌握 + CUDA-MODE L9） |

# 知识库索引

| 想深入理解 | 去看 |
|---|---|
| Triton 所有 API（按场景查） | [triton-cheatsheet.md](../notes/triton/triton-cheatsheet.md) |
| Triton → CUDA 底层实现（layout/sync/mma） | [triton-under-the-hood.md](../notes/cuda/triton-under-the-hood.md) |
| Triton vs CUDA 编程模型 | [triton-vs-cuda.md](../notes/triton/triton-vs-cuda.md) |
| Triton MatMul 参考实现 | [reference/triton/matmul/matmul.py](../reference/triton/matmul/matmul.py) |
| Triton 调试方法 | [Lesson 07 — Triton Debugging](07-triton-debugging.md) |
| 接下来去哪 | [PATH.md](../PATH.md) — B2 Triton Fused Softmax |

---

*Lesson 06 · Triton 入门 · B1 当前主线 · 2026-08-13 全面重写*

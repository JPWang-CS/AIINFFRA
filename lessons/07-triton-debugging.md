# Lesson 07 — Triton Debugging：从“跑不通”到定位根因

> 定位：Lesson 06 Triton 算子实现的配套调试课，不另起学习主线。
> 当前案例：用 Triton MatMul 练习解释器、tile 打印、断言、内存检查和数值误差定位。
> 前置：[Lesson 06 — Triton 入门](06-triton-intro.md)。

## 本课单元卡

| 项目 | 内容 |
|---|---|
| 调试对象 | [solutions/triton/matmul.py](../solutions/triton/matmul.py) 与 [matmul_leetgpu_wip.py](../solutions/triton/matmul_leetgpu_wip.py) |
| 目标 | 区分编译错误、逻辑错误、越界错误、数值错误和性能问题 |
| 当前状态 | 配套教程；MatMul 服务器版 `GPU_VALIDATED`，LeetGPU 原始代码仍为 `WIP` |
| 验收 | 能用正确工具定位一次错误，并留下最小复现、原因和修复证据 |
| 下一步 | 回到 Lesson 06，继续 LeetGPU MatMul；通过后再做 BLOCK/warp/stage 调优 |

## 1. 先判断是哪一类问题

不要一上来就调 `BLOCK_M`。先把错误归类：

| 现象 | 优先怀疑 | 第一工具 |
|---|---|---|
| kernel 编译失败 | shape、dtype、constexpr、API 签名 | `tl.static_print`、`tl.static_assert` |
| 输出完全错 | pointer、stride、grid、mask、归约 | 小尺寸 + CPU interpreter |
| 只有边界尺寸错 | mask 或 `other` | 非整除 M/N/K case |
| 运行时报非法地址 | 越界 load/store、异步错误 | `CUDA_LAUNCH_BLOCKING=1`、`compute-sanitizer` |
| 大归约维度误差变大 | TF32、累加顺序、dtype | 固定 seed + IEEE/TF32 对照 |
| 正确但很慢 | tile、warps、stages、寄存器压力 | benchmark、Nsight Compute |

调试顺序固定为：

```text
最小输入
→ CPU interpreter
→ 打印 pid/offset/mask
→ 非整除边界测试
→ 数值误差定位
→ compute-sanitizer
→ 最后才做性能分析
```

## 2. 先做可复现的最小测试

每次调试先固定随机数，并只跑一个 case：

```python
torch.manual_seed(0)

cases = [
    (1, 1, 1),
    (64, 32, 64),       # 完全整除 tile
    (65, 33, 67),       # 三个维度都越界
    (257, 513, 129),    # 较大归约维度
]
```

不要一开始使用 `8192×6144×4096`。大输入只能告诉你“错了”，不能快速告诉你哪一个 tile 错了。

正确性测试必须在 kernel 后同步：

```python
solve(a, b, out, M, N, K)
torch.cuda.synchronize()
torch.testing.assert_close(out, torch.matmul(a, b), rtol=1e-2, atol=1e-2)
```

GPU kernel 是异步发射的；没有 `synchronize()`，报错位置可能落在后面的无关操作上。

## 3. CPU interpreter：先看 Triton 逻辑

Triton 官方提供解释模式：

```bash
TRITON_INTERPRET=1 python debug_matmul.py
```

解释器会跳过 GPU 编译，在 CPU 上逐个 program instance 模拟执行，适合查看 `offset`、`mask`、`tile` 和中间结果。官方调试说明见 [Triton Debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)。

当前 `solutions/triton/matmul.py` 是 GPU benchmark 脚本，内部固定使用 `device="cuda"`，所以不能直接当 interpreter harness。调试时单独写一个最小入口：

```python
import os

device = "cpu" if os.getenv("TRITON_INTERPRET") == "1" else "cuda"
a = torch.randn((5, 7), device=device, dtype=torch.float32)
b = torch.randn((7, 9), device=device, dtype=torch.float32)
out = torch.empty((5, 9), device=device, dtype=torch.float32)

solve(a, b, out, 5, 7, 9)
print(out)
print(torch.matmul(a, b))
```

解释器的限制：它不是性能工具，且官方文档列出了 `bfloat16` 和间接内存访问等限制；所以 interpreter 通过只说明控制流/索引逻辑基本正确，不代表真 GPU 性能正确。

## 4. 打印 tile：只打印一个 program

设备端打印很容易刷爆终端。只用极小输入，并限制到一个 program：

```python
# 先把输入缩到只有一个 program，再打印，避免多个 program 同时刷屏。
tl.device_print("pid_m", pid_m)
tl.device_print("pid_k", pid_k)
tl.device_print("offset_m", offset_m)
tl.device_print("offset_n", offset_n)
tl.device_print("offset_k", offset_k)
tl.device_print("mask_a", mask_a)
tl.device_print("mask_b", mask_b)
tl.device_print("tile_a", tile_a)
tl.device_print("tile_b", tile_b)
```

对当前 MatMul，重点核对：

```text
ptr_a[i,j] = A[offset_m[i], offset_n[j]]
ptr_b[i,j] = B[offset_n[i], offset_k[j]]
ptr_c[i,j] = C[offset_m[i], offset_k[j]]
```

`tl.device_print` 是运行时打印；`tl.static_print` 用于打印 `BLOCK_M`、`BLOCK_N` 等编译期常量。参数必须是简单的字面量/标量/tensor，不要使用 Python f-string 拼运行时 tensor。参考：[Triton language API](https://triton-lang.org/main/python-api/triton.language.html)。

## 5. 编译期和运行时断言

### 编译期断言

用于检查 tile 配置，不需要启动 GPU：

```python
tl.static_print("BLOCK_M", BLOCK_M)
tl.static_print("BLOCK_N", BLOCK_N)
tl.static_print("BLOCK_K", BLOCK_K)
tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be aligned")
```

### 运行时断言

用于检查真正的运行时 invariant：

```python
tl.device_assert(M > 0, "M must be positive")
tl.device_assert(N > 0, "N must be positive")
tl.device_assert(K > 0, "K must be positive")
```

运行：

```bash
TRITON_DEBUG=1 python debug_matmul.py
```

`device_assert` 默认不开启，必须设置 `TRITON_DEBUG=1`；参考 [device_assert API](https://triton-lang.org/main/python-api/generated/triton.language.device_assert.html)。

注意：边界 tile 中 `offset_m >= M` 或 `offset_k >= K` 是预期现象，不能简单断言所有 offset 都小于维度。应该断言 mask 覆盖正确，或者只断言非边界 lane。

## 6. 越界和非法访问：compute-sanitizer

当怀疑 `tl.load` / `tl.store` 越界时：

```bash
CUDA_LAUNCH_BLOCKING=1 \
compute-sanitizer --tool memcheck \
python solutions/triton/matmul.py
```

它适合查：

- 越界读写；
- 非法地址；
- data race；
- 异步 kernel 错误被延迟报告。

代价是很慢，所以只跑一个最小 case，不要直接跑完整 benchmark。Triton 官方也把 NVIDIA GPU 的 `compute-sanitizer` 列为内存/竞态调试工具。

## 7. 数值误差：不要把 mask 错误和 TF32 混在一起

先输出最大绝对误差和位置：

```python
diff = (out - expected).abs()
flat_idx = diff.argmax()

print("max_abs_error:", diff.flatten()[flat_idx].item())
print("out:", out.flatten()[flat_idx].item())
print("expected:", expected.flatten()[flat_idx].item())
```

相对误差在 reference 接近 0 时可能很大，所以要同时看 `abs error` 和 `relative error`。

对于 FP32 MatMul，先固定比较口径：

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

```python
acc += tl.dot(
    tile_a,
    tile_b,
    input_precision="ieee",
)
```

如果默认 `tl.dot` 使用 TF32，而 PyTorch reference 使用另一种 FP32/TF32 设置，大归约维度下就可能出现：小 case 通过，大 case 的最大误差超阈值。当前 MatMul 的 `N=513` case 就是这种调试路径：边界逻辑没错，问题在计算精度口径。

## 8. 当前 MatMul 的实际调试记录

本次调试得到的最小证据链：

```text
(1, 1, 1)       OK
(64, 32, 64)    OK
(65, 33, 67)    OK
(257, 513, 129) 初始 TF32 口径失败；切换 IEEE FP32 后 OK
```

服务器 benchmark：

```text
GPU: NVIDIA GeForce RTX 3090
shape: 8192 × 6144 × 4096
Triton: 24.681 ms，16,706.0 GFLOPS
torch.mm: 17.120 ms，24,083.3 GFLOPS
```

结论：当前 kernel 的 tile、pointer、边界 mask 和归约逻辑已经通过服务器正确性；性能差距属于后续调优问题，不要在还没有正确性证据时直接调参。

## 9. 调试完成后再恢复 benchmark

调试代码必须临时关闭或隔离：

- 删除 `device_print`；
- 关闭 `TRITON_DEBUG`；
- 不再使用 `compute-sanitizer`；
- 保留固定 seed 的 correctness tests；
- 再运行 warmup、GPU Event timing 和 GFLOPS benchmark。

不要把带大量打印、断言或 sanitizer 的耗时当成性能数字。

## 本课验收

- [ ] 能用最小 case 区分 pointer/mask/shape/precision 问题
- [ ] 能用 `TRITON_INTERPRET=1` 查看一个 program 的中间结果
- [ ] 能用 `tl.device_print` 打印 offset 和 mask
- [ ] 能用 `tl.static_assert` / `tl.device_assert`
- [ ] 能用 `compute-sanitizer` 排查 GPU 非法访问
- [x] 能解释当前 MatMul 为什么需要 IEEE FP32 对照
- [x] 能记录一次完整的正确性 + 性能调试证据

## 官方参考

- [Triton Debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)
- [static_print](https://triton-lang.org/main/python-api/generated/triton.language.static_print.html)
- [device_assert](https://triton-lang.org/main/python-api/generated/triton.language.device_assert.html)
- [NVIDIA compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

# softmax 本地测试（LeetGPU 不可用时在服务器跑）

LeetGPU 挂了 / 想看真实带宽时，在本地或服务器上验证 `solve()` 精度并测吞吐。

## 文件

| 文件 | 谁写的 | 作用 |
|------|:--:|------|
| `softmax_naive.cu` | **我** | LeetGPU `5_softmax` 提交：3-pass 跨 block 归约，`extern "C" solve`。**不动** |
| `main.cu` | harness（非学习目标） | 造数据 → 调 `solve` → 对比 CPU 双精度参考 → 报误差/吞吐/带宽。**不含 kernel** |
| `run.sh` | harness | 编译 + 运行 |
| `bench_softmax.cu` | （参考，不是我写的） | naive vs online 自带 benchmark，kernel inline，2D batched 题型，跟我的 1D 题型不同 |

## 跑

```bash
./run.sh                                # 默认测 N = 1024 / 65536 / 500000 / 1048576
./run.sh 500000                         # 只测 LeetGPU 题面 max
KERNEL=softmax_online.cu ./run.sh       # 以后测新版本（同签名 solve）
ARCH=-arch=sm_89 ./run.sh               # 指定架构（默认 -arch=native）
```

可调环境变量：`NVCC` / `ARCH` / `KERNEL` / `OUT`。

> 没执行权限就 `chmod +x run.sh`，或直接 `bash run.sh`。

## 怎么读结果

- **`max abs err < 1e-4 → [PASS]`**：`solve()` 输出对得上 CPU 双精度参考。float kernel 对 N~百万级期望 ~1e-6。
- **`output sum ≈ 1.0`**：概率分布归一化正确（softmax 定义要求）。
- **`effective BW`**：naive 3-pass 算法流量按 `3 read + 1 write = 4·N·4` 算。
  对比 HBM 峰值看离 memory-bound 上限多远（4090 ≈ 1008 GB/s，A100 ≈ 1550 GB/s）。
  注意 solve() 内部每次 `cudaMalloc` + 2 次 D2H round-trip，小 N 时这些开销会压低 BW——这本身就是要去掉的税。

## 加新版本（online / warp shuffle）

新建一个文件（如 `softmax_online.cu`），**保持** `extern "C" void solve(const float*, float*, int)` **签名不变**，
然后：

```bash
KERNEL=softmax_online.cu ./run.sh
```

harness 不用改。注意 online 版本算 HBM 流量时心里换一下：`2·N·4`（1 read + 1 write），别直接拿 naive 的 BW 比时间——要按"打满多少 GB/s"比。

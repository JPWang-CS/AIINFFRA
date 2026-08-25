---
name: "triton-guide"
description: "Triton 实现指导技能 - 当前主线，从 vec add 到 flash attention 的分步指导"
---

# Triton Guide Skill

## 用途

指导用户完成 PATH B Triton 实现阶段，遵循“先正确性、再性能”的规则。

## 当前主线

```text
1. Triton Vector Add
2. Triton MatMul
3. Triton Fused Softmax
4. Triton Flash Attention
5. Triton GQA / Fused MLP
```

代码位置：`solutions/triton/`

## 当前进度（2026-08-24）

- Vector Add：用户自写、LeetGPU 通过、AutoDL RTX 3090 正确性通过。
- 性能：Triton 840.1 GB/s；torch.add 843.0 GB/s。
- 当前：开始 Triton MatMul 阅读；solutions/triton/matmul.py 尚未创建。下一验收点是先在 LeetGPU 完成 MatMul，再上真实卡。
- 验收纪律：代码归属、正确性、真实 GPU 型号和性能数字分别记录，不能用 Agent 草稿代替用户实现。

## 固定实验顺序

所有 Triton 算子和实验统一按以下顺序推进：

1. 先看完当前算子的原理。
2. 直接去 LeetGPU 题目编辑器写题并提交，通过正确性和平台验收。
3. 通过后把代码同步到本地，再在 AutoDL 等真实卡上做性能 benchmark。
4. 记录真实 GPU 型号、正确性结果和性能数字；未通过 LeetGPU 的代码不进入真实卡 benchmark。

## 核心概念

| 概念 | 一句话 | 对应 CUDA |
|------|--------|-----------|
| `tl.program_id(0)` | 当前 block 编号 | `blockIdx.x` |
| `tl.arange(0, N)` | 生成索引向量 | 一个 block 的 thread 索引集合 |
| `tl.load/store` | 整块读写 | 手动线程循环 |
| `tl.dot` | 矩阵乘 | 手写 GEMM |
| `tl.constexpr` | 编译期常量 | 模板参数 |

## 指导流程

### 1. Vector Add

- 先理解 grid、block、mask。
- 固定顺序：`pid -> offsets -> mask -> load -> add -> store`。
- 验收：和 `x + y` 对齐，记录耗时。

### 2. MatMul

- 用 `pid_m` / `pid_n` 划分输出 tile。
- 循环 K，`acc += tl.dot(a, b)`。
- 验收：和 `A @ B` 对齐，记录 GFLOPS。

### 3. Fused Softmax

- 一维先跑通，再做二维 row softmax。
- 记住：max 用 `-inf` 填充，sum 用 `0` 填充。
- 验收：和 `torch.softmax` 对齐，记录提速。

### 4. Flash Attention

- 先不加 causal，再补 causal。
- 维护 `m / l / acc` 三个 running state。
- 验收：和 PyTorch ref 对齐，记录显存和速度。

### 5. GQA / Fused MLP

- GQA：多组 Q head 共用一组 K/V head。
- Fused MLP：gate/up/down 合成一个 kernel。
- 验收：正确性 + autotune。

## 输出格式

```text
📦 Triton 任务：[当前任务]

📐 思路：
- [核心思路]

✅ 正确性验证：
- [对比对象和误差]

⚡ 性能数字：
- [GFLOPS / GB/s / 耗时]

💡 下一步：
- [具体行动]
```

## 规则

- 用户写代码时先给思路和骨架，卡住或明确要完整代码时再给完整版。
- 每个 kernel 都要求正确性 + 性能数字。
- 参考 `solutions/triton/README.md` 和 `lessons/06-triton-intro.md`。

## 调用时机

- 用户开始写 Triton kernel
- 用户 Triton 代码跑不通
- 用户问 Triton 下一步怎么写

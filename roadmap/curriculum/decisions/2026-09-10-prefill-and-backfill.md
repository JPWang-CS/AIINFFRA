# 2026-09-10 Prefill Attention 与旧章补缺记录

## 已确认的范围

本次只补完整算子课程的 GEMM、Activation/Fusion 和 Prefill Attention，并把算子课程网站从第 10 页接到第 11 页。没有修改 `PATH.md`、`NOW.md`、`HISTORY.md`、`AGENTS.md`、旧 `solutions/` 或 `reference/`，没有 commit/push，也没有新增 GPU 性能结果。

## 旧章补缺

- GEMM 统一按 `A[M,N]B[N,K]=C[M,K]` 讲解；split-K 明确拆归约轴 N；CPU 模型覆盖 partial 合并后一次性 beta/bias/activation、persistent `tile_id=pid+k*num_programs` 和 ragged grouped prefix sums。
- `epilogue_once` 显式接收真实 `M/K/BM/BK`，单测使用 `M=3,N=5,K=7,BM=2,BK=4,parts=3`，覆盖多输出 tile、K 尾块、非零 beta、bias、非线性和 beta=0 不读取 NaN old-C。
- Activation/Fusion 增加明确的 head-dim RMSNorm、L2/F.normalize 与 LayerNorm 区分；RoPE 使用 split-half、外部 cos/sin 表、绝对 position、偶数 Drot prefix 和 cache/residual 顺序。CPU reference 将按 position 取表的 wrapper 与使用已取表行的低层函数分开；测试使用非零 position，验证真实正交表下的范数、relative position、布局往返和 per-dim gamma 反例。测试表是 test-only repeated-half 数值，不替代模型实际 checkpoint/input 表。

## Prefill baseline 合同

新增 `examples/prefill_fp32.py` 是教学 forward baseline：contiguous `[B,H,S,D]`、同 shape Q/K/V、FP32、`D∈{16,32,64,128}`、`S>0`，`BLOCK_Q=16`、`BLOCK_KV=32`，grid 为 `(ceil(S/BQ),B*H)`，base 和 row offset 用 int64。Q tile 只 load 一次，KV tile 扫描；QK/PV dot 均指定 `input_precision="ieee"`，scale 由 host 算成 constexpr 传入，launch 固定 `num_warps=4` 并在 `torch.cuda.device(q.device)` 中执行。

online 状态为未归一化 `m/l/U`，score key-tail 必须 mask 为 `-inf`；全 masked 虚拟 Q 行通过 `safe_m`、`alpha/p` 和 `denom` 分支避免 `-inf-(-inf)` NaN 并输出零。wrapper 默认不做 finite scan；`check_finite=True` 只用于 correctness preflight。Q/K/V 可以只读别名，输出必须是独立 contiguous storage，并使用连续 byte interval overlap 检查拒绝与任一输入重叠。benchmark 预分配 output，排除 JIT/allocation warmup；墙钟边界包含 wrapper 检查、launch 和 synchronize，不是纯 kernel 时间。

## 题面边界与官方材料

- 当前 [#6 Softmax Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/6_softmax_attention) 是矩形 `Q[M,d],K/V[N,d],out[M,d],solve(Q,K,V,out,M,N,d)`、FP32 noncausal，示例 `M=2,N=3,d=4`，`rtol=atol=1e-4`。本章 square `[B,H,S,D]` baseline 不能直接提交为 #6，也不沿旧 reference 的 `N,d` 索引签名。
- [#80 Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention) 是无 batch 的 `[Hq,S,D]`/`[Hkv,S,D]`，`kvhead=qhead//(Hq/Hkv)`；[#61 RoPE Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/61_rope_embedding) 是 split-half，cos/sin 表由输入提供。三者均 free，应从 [LeetGPU 平台总入口](https://leetgpu.com/challenges) 按准确题名搜索和原样归档；[GitHub 题库源](https://github.com/AlphaGPU/leetgpu-challenges) 用于核对公开题面，不造 slug。
- FA2 原论文 [§3.2/§3.3 PDF](https://tridao.me/publications/flash2/flash2.pdf) 的教学表述采用：初版 parallel batch/head；FA2 forward 额外把 Q-row tiles 分给 CTA，outer-Q/inner-KV；初版 split K/V 在 warp 间产生 partial output 归约，FA2 split-Q 共享 KV、减少 output fragment 跨 warp 合并。仍需 shared load、barrier 和其它必要同步，不能说 FA2 不需要 warp 同步。
- CUDA Programming Guide 13.3 的寄存器/local memory 对照为 PDF 67（印刷 51）页，barrier 等待未退出线程与 memory ordering 为 PDF 577（印刷 561）页。旧 CUDA tail-Q 例子的主要问题是退出 tid 不再承担 cooperative KV load，导致其它线程读未初始化数据；不泛化为任意提前 exit 必死锁。
- Triton 当前 fused-attention 教程在 PV 前处理概率 dtype；若使用 exp2 需乘 `log2(e)`。本 baseline 使用自然 exp，不混用两条精度路径。

## 验证边界

已运行并通过 stdlib CPU checks：GEMM 调度、RoPE/QK Norm、dense-vs-block online attention，覆盖 `S=1/3/5/33/65`、多 `block_kv`、causal、all-masked zero state、B/H offsets 和 GQA mapping。已写 GPU correctness harness：FP32 full reference、TF32 关闭、`S=1` 的 `D=32/64/128`、`S=33/65`、B/H、多 causal/noncausal、Q/K=0 且按 BH 填不同常数的尾块测试、#6 矩形独立均值期望；当前环境尚未运行 GPU，因此没有声称 Triton 编译、GPU correctness、LeetGPU 通过或服务器性能。

本次由一个 Luna worker 实施，主 agent 进行官方源核查和独立复核。后续题面提交和真实 GPU 验证仍是独立门槛。

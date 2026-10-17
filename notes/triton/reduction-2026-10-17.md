# Triton Reduction 单元卡（2026-10-17）

## 当前单元

- 题目入口：LeetGPU [Challenges](https://leetgpu.com/challenges)，搜索 **Reduction**；题库原题目录为 [Reduction #4](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/4_reduction)。以登录后的当前题面与函数签名为准。
- 用户对话代码快照：[单 block 版本](../../solutions/triton/reduction_single_block.py)；[两阶段多 block 版本](../../solutions/triton/reduction_multi_block.py)。除文件头声明外，代码按用户提供内容原样保存。
- 独立服务器校验入口：[reduction_check.py](../../solutions/triton/reduction_check.py)。`--help` 与 `--cpu-check` 不需要 GPU；默认/`--correctness-only` 才执行 GPU 正确性，不 benchmark。
- LeetGPU：`WIP`。快照文件标明尚无平台提交证据；尚未从题面提交或归档通过版。
- 服务器：correctness-only 已通过，NVIDIA L20X、PyTorch 2.13.0+cu126、Triton 3.7.1；单 block 36/36，多 block 40/40。逐例、环境和退出收据见[服务器日志](./logs/2026-10-17-reduction-nvidia-l20x.txt)。
- 下一步：按当前 LeetGPU #4 题面独立实现并归档平台版本；服务器结果不能替代平台通过记录。性能工作留待后续真实工作负载决定。

## 本轮已确认的讨论

- `BLOCK_SIZE` 表示 program 的逻辑元素数，不等于线程数。常规 NVIDIA Triton 配置 `num_warps=4` 对应 4 个 warp、128 个线程；`tl.sum` 是编译器生成的协作归约，不是单线程逐项累加。
- 单 block 版本以 `next_power_of_2(N)` 覆盖整个输入，mask 使越界逻辑位置读取加法单位元 0，最后写一个标量。它适合小归约；随着 N 增大，单个 program 的逻辑 tile 与资源需求也增大。
- 多 block 版本用固定 `BLOCK_SIZE=256` 产生 `partial[pid]`，每个 program 写入独立位置，再以第二个 kernel 合并 `num_blocks` 个 partial。`partial` 由 `torch.empty` 分配，第一阶段对所有有效 program 全量覆写，所以不需要预先清零；同一 CUDA stream 上的顺序提交保证第二阶段在第一阶段之后执行。
- 例如 `N=1025`：第一阶段启动 `ceil(1025/256)=5` 个 program，各写一个 partial；第二阶段有效长度为 5，tile 为 `next_power_of_2(5)=8`，末尾 3 个位置由 mask 作为 0 处理。
- 这里的全局数组、load 后的寄存器值和缓存命中属于不同层次：数组在全局地址空间；被 `tl.load` 的值进入 kernel 的寄存器/编译器分配的数据路径；L1/L2 命中只改变实际访问层级，不改变全局地址空间语义。
- 当前多 block 代码固定两阶段。若 N 极大，第二阶段 tile 仍可能过大，因此它不是任意 N 的多级归约实现；本轮保留快照，不把它替换成优化版。

## 验证合同

- 输入限正整数 N、连续 FP32、一维 CUDA input 和 shape `(1,)` 的 FP32 output。覆盖 `N=1,31,255,256,257,1023,1024,1025,4097`；多 block 另覆盖 `N=1048576`。
- 四种数据：全零、全一、正负交替、有界固定种子的随机数。对每个输出使用 FP64 求和参考；精确整数模式要求 `actual == reference`，随机模式固定为 `abs_error <= 2e-4 + 2e-5 * abs(reference)`，不按结果调宽容差。
- FP32 并行归约与 FP64 参考的求和顺序不同，随机模式的误差属于浮点累加差异。输出逐项打印 PASS、actual、FP64 reference 和绝对误差，并汇总每个实现/长度的最大误差及 GPU CC、CUDA runtime、PyTorch/Triton 版本与样例数。
- CPU 模式只运行独立的 tile 划分、partial 分组和尾部补零模型；它不验证 Triton 编译器、实际 `tl.sum` lowering、GPU 执行或平台正确性。指定服务器上的 76 例 GPU correctness 已另行通过，记录和机器收据见[日志](./logs/2026-10-17-reduction-nvidia-l20x.txt)。
- 用户明确授权本次先做服务器 correctness-only；服务器记录与 LeetGPU 状态分开维护，服务器通过不构成 `LEETGPU_PASS`。

从仓库根目录运行：

~~~bash
python solutions/triton/reduction_check.py --help
python solutions/triton/reduction_check.py --cpu-check
python solutions/triton/reduction_check.py --correctness-only
~~~

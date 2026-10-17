# Reduction 讨论记录（2026-10-17）

## 已确认

以下结论由用户讨论确认；代码执行状态与学习平台状态分开记录。

1. `BLOCK_SIZE` 是 program 处理的逻辑元素数，不是线程数。`num_warps=4` 在常规 NVIDIA Triton 模式下表示 4 个 warp、128 个线程。`tl.sum` 会编译成线程间协作归约，不等于单个线程顺序累加全部元素。
2. 单 block 版本选择 `BLOCK_SIZE=next_power_of_2(N)`，一个 program 覆盖输入，掩码尾部以 0 补齐并写出一个标量。随着 N 增大，单个 program 的 tile 增大，可能带来较高资源需求。
3. 多 block 版本固定第一阶段 tile 为 256，每个 program 写 `partial[pid]`，避免多个 program 写同一地址；第二个 kernel 以实际 partial 数为 N，并使用其二次幂 tile 完成归约。
4. `partial` 用 `torch.empty` 分配后，第一阶段覆盖所有有效 partial，因此无需清零。两次 kernel 在同一 stream 按提交顺序执行，第二阶段读取第一阶段结果。
5. `N=1025` 时，第一阶段有 5 个 program 和 5 个 partial；第二阶段有效长度为 5、tile 为 8，最后 3 个逻辑位置由 load mask 设为 0。它们是逻辑元素位置，不对应固定数量的硬件 lane。
6. 全局地址空间、load 后的寄存器数据及 L1/L2 命中是不同概念。缓存命中不改变数组位于全局地址空间这一事实。
7. 当前实现是固定两阶段；超大 N 时第二阶段仍可能需要较大的 tile，不宣称支持任意规模的多级归约。保留用户原始代码，当前 `BLOCK_SIZE=256` 不擅自调整；极致优化留待后续模型和多算子工作负载。
8. 正确性合同预先固定为正 N、连续 FP32 输入、`output.shape==(1,)`；全零、全一、正负交替样例要求逐项精确相等，有界随机样例使用固定 `atol=2e-4, rtol=2e-5`。FP32 与 FP64 的归约顺序不同，随机样例报告误差时说明顺序差异。CPU 模拟是独立分组模型，不覆盖编译器或实际 `tl.sum` 路径，也不代表 GPU 正确性。

## 提案

- 用户已授权先做服务器多 shape correctness，再按当前 LeetGPU #4 题面独立实现并归档。平台通过和服务器正确性分别记录；服务器结果不能替代 `LEETGPU_PASS`。
- 后续如进入性能工作，按真实设备、精度合同、输入规模和多算子边界选择最小 A/B；本轮不发起广泛参数扫描或 benchmark。

## 用户此前提出的问题与选择依据

以下按本轮对话中的问题主题整理，并非逐字引语；用户的学习偏好单独列出，不将其改写成 Agent 观点或用户感想。

- A100 的一个 CTA 最多 1024 个线程，是否因此一个 block 只能处理 1024 个元素？`num_warps=4` 和 `BLOCK_SIZE=1024` 如何换算线程与元素工作量？
- 一个普通 CTA 在一个 SM 上执行；grid 只有少数 CTA 时，即使 CTA 资源能放下，为什么整张 GPU 仍可能缺少并行度？应先选分解还是先看 occupancy？
- Triton `tl.sum`、CUDA 显式线程索引/shuffle/shared/barrier、Gluon 显式 layout、TileLang 的 thread binding 与 memory scope/sync 分别由开发者控制什么？换成 `T.reduce_sum` 是否自然更快？
- Python Tensor、设备指针、全局数组、load 后寄存器值、HBM 事务以及 L1/L2 命中各处于哪一层？partial 写回为什么需要 `tl.store`？
- 用户明确要求先讨论选择依据，除非主动询问，不主动给完整新实现；当前归档保持其原代码，极致优化放到后续模型/多算子实践。

## 整理后的技术结论

**分解与每线程工作量。** CTA 的线程上限约束线程数，不约束它处理的逻辑元素数。`num_warps=4` 对应 128 个线程；`BLOCK_SIZE=1024` 表示 1024 个逻辑元素，平均每线程 8 个元素，具体映射由编译器 layout 决定。普通 CTA 由一个 SM 执行；一个 CTA 即便能驻留，也只给一个 SM 提供该任务。若总 grid 很小，其他 SM 无任务可执行，因此先根据 N 和目标并行度确定 grid/分解，再估每线程工作和资源需求。

**资源与 occupancy 估算示例（假设，不是编译或实测结果）。** A100 每 SM 有 65,536 个 32-bit 寄存器、最多 2,048 个线程。假设一个 CTA 有 128 线程、每线程分配 64 个寄存器，则每 CTA 约占 `128×64=8192` 个寄存器；只按寄存器容量估算可同时容纳 `65536/8192=8` 个 CTA，即 32 个 warp。若该 SM 上限按 64 个 warp 计算，这相当于理论 50% warp 驻留。实际驻留还受寄存器分配粒度、shared memory、线程上限和 CTA 上限影响，必须读编译资源信息及 profiler；occupancy 用于解释资源与延迟隐藏，不是单独追求的目标。`BLOCK_SIZE=1024`、128 线程下“平均 8 元素/线程”也不等于已知每线程寄存器数。

**归约抽象与协作控制。** Triton 提供 `tl.sum`，编译器为目标布局和类型选择线程级归约路径（例如树形/蝶形协作）；源代码不直接固定每个元素落在哪条 lane。CUDA 可显式控制线程索引、shuffle、shared memory 与 barrier。Gluon 把 layout 作为显式编程对象；TileLang 可描述 thread binding、memory scope 与同步。使用 `T.reduce_sum` 只表达归约操作，不单独保证比 Triton `tl.sum` 更快；比较需要同一数值合同和目标设备上的编译/执行证据。

**地址与数据层次。** Python `torch.Tensor` 持有设备数据并提供元数据；kernel 参数中的 pointer 指向设备地址。`tl.load` 从全局地址读取，值通常由线程放在寄存器中参与计算，元素到线程/寄存器的具体映射由编译器布局决定。HBM/L2/L1 描述实际流量或命中层级，不改变 pointer 所指对象的全局地址空间。每个 CTA 的 partial 写到 `output_ptr + program_id`，因此第一阶段必须显式执行相应 `tl.store`；仅在 Python 中分配 Tensor 不会自动把归约结果写进去。

## 用户材料归属

- 用户原话/问题主题：上节以问题形式归纳；本轮给出的两份 Triton 源码保存在 `solutions/triton/reduction_single_block.py` 与 `reduction_multi_block.py`，只添加文件头状态说明。
- Agent 整理：本记录中的机制解释、数量级估算和验证合同；A100 驻留数字明确是基于假设的算术示例，不是编译报告或设备测量。
- 已验证状态：NVIDIA L20X 服务器 correctness 76/76 通过；实验边界与详细机器收据见 [`Reduction GPU 记录`](../../../notes/triton/logs/2026-10-17-reduction-nvidia-l20x.txt)。平台提交仍待进行。

## 待定

- 当前 LeetGPU 题面是否与归档快照接口完全一致：待进入平台核对并提交。
- 多算子/模型工作负载中是否需要改为递归多级归约：待对应工作负载出现后判断。

## 当前边界

本地归档与 CPU/offline 检查不升级学习状态。没有平台证据时为 `WIP`；服务器 correctness 通过也不等于 `LEETGPU_PASS`。性能优化属于后续阶段。

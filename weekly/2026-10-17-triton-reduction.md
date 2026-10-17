# 2026-10-17 — Triton Reduction 学习快照

本轮把对话中的单 block 和两阶段多 block Reduction 代码分别按原样保存为本地快照，并建立独立 correctness 入口。快照页注明“用户对话快照，尚无平台提交证据”。

讨论确认了逻辑 tile 与线程数的区别、`tl.sum` 的协作归约、partial 的唯一写入位置、`torch.empty` 全量覆写、同 stream 阶段顺序，以及 `N=1025` 时 `5 partials → tile 8` 的尾部处理。多 block 版本目前固定两阶段，大 N 的第二阶段容量仍需结合真实规模判断。

本轮准备了正 N 连续 FP32、shape `(1,)` 输出、FP64 reference 与固定误差界限的校验合同。指定 NVIDIA L20X 上单 block 36/36、多 block 40/40，共 76 例 GPU correctness 通过；整数样例精确相等，随机样例保持 `atol=2e-4, rtol=2e-5`，最大绝对误差为多 block `N=1048576` 的 `0.000199173651`。服务器收据见 [Reduction GPU 记录](../notes/triton/logs/2026-10-17-reduction-nvidia-l20x.txt)。LeetGPU 尚无新通过证据，服务器结果不替代平台记录；本轮不执行性能 benchmark。GPU 基础阅读仍在第 4 章，实践焦点转到 Reduction，论文线 MLA 保持不变。

记录入口：[Reduction 单元卡](../notes/triton/reduction-2026-10-17.md) · [讨论记录](../roadmap/curriculum/decisions/2026-10-17-reduction-discussion.md)。

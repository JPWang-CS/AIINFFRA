# 第 4 章第 10 节配图来源记录

生成日期：2026-10-16。制作方式：Codex 内置 ImageGen。以下内容是已确认的核心提示与构图要求摘要，不是逐字提示词转录；完整生成记录由主 Agent 保管。生成 PNG 原件留在用户生成目录，课程目录保存复制件。

## 执行路径图

- 原件：`C:/Users/w00939120/.codex/generated_images/01a01efb-ee29-7483-8cd0-436cd61b70d4/exec-2dc1b4d5-5711-4f13-9fe4-55fe06229ecf.png`
- 正式文件：`figures/graph-execution-paths.png`
- 图像：1536 × 1024，PNG
- SHA-256：`C76E55B35862F0E0F1BEBB2C7BA6EDA765FE90E3C0F968A3F68951900B8CC882`
- 提示摘要：同一计算 `y=ReLU(alpha*x+beta)` 的三行路径图；CPU 用紫色、GPU 用青色、准备阶段用琥珀色。Eager 表示 mul、add、ReLU 分别提交并执行；编译路径画出成功融合为一个 kernel 的示意，同时说明 `fullgraph` 不承诺单 kernel；CUDA Graph 捕获 Eager 后仍是三个 kernel、一次图提交。标明这是结构示意，不是实测时间线。
- 检查：主 Agent 已逐项核对三条路径、kernel 关系、颜色图例和“结构示意非实测时间线”标注。

## 固定 buffer 生命周期图

- 原件：`C:/Users/w00939120/.codex/generated_images/01a01efb-ee29-7483-8cd0-436cd61b70d4/exec-3d2de900-689b-4215-a6ff-7489a9f23a2a.png`
- 正式文件：`figures/graph-static-buffers.png`
- 图像：1536 × 1024，PNG
- SHA-256：`10D488719D689753AED4EEC53BC4C44425C9C85CE26447C4AD062DD24D8F4307`
- 提示摘要：说明地址保持不变而每轮数据更新；`copy_` 先把新值放入固定输入地址，同 stream 的 replay 随后由 GPU 从原输入地址读取并写入固定输出地址；下一轮 replay 覆盖输出结果。单独赋值只改变 Python 变量绑定，图仍引用原地址。图中强调输入和输出 buffer 在图及 replay 使用期间必须存活。
- 检查：主 Agent 已逐项核对输入/输出地址、复制和 replay 顺序、下一轮覆盖行为与赋值语义；正文结合此图说明输入/输出 buffer 引用必须保持存活。

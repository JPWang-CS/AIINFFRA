# 读者走读与求职覆盖复核

## 读者与检查范围

读者设定：有 NPU 算子开发经验，能阅读 C++/Python，已有部分 CUDA/Triton 实践，但不假定已经掌握 NVIDIA 执行模型或框架扩展接口。

先检查正式课程路由及九类算子、模型分析、系统与论文目录，再沿以下路径核对正文、公式、代码和运行命令。该复核不是逐字审完所有 PDF，也不代表执行了每个 CUDA 示例。

| 路径 | 读者要完成的动作 | 发现的问题 | 整改 |
|---|---|---|---|
| GPU 执行与访存 → Copy/Transpose | 从坐标算出实际地址 | `base` 未说明是 storage base 还是 tensor.data_ptr，切片 offset 可能重复相加 | 加入 `[4,6]` 切片到 `[2,2]` 的具体地址例，分别计算两种起点 |
| 同步 → Row-wise Reduction | 解释无效输入与参与线程 | 旧算子正文仍将提前 return 一概归因于 barrier 等待，与基础章更正矛盾 | 按单位元初始化、后续读取和具体同步原语说明风险 |
| 性能分析 → GEMM 优化报告 | 判断微小收益能否复现 | 只有一次平均计时，不足以组织波动与跨 shape 回归 | 加入独立轮次、交替 A/B、中位数/四分位范围、逐 shape 加速比及几何平均；数据表留空 |
| CUDA 程序 → C++/Linux 调试 | 区分编译、链接、装载与设备错误 | 错误定位资料分散，容易把同步点当出错点或混淆 device link 与 host library | 原执行章节内补完整分类及对应示例命令 |
| Norm → Mini Transformer | 将 kernel 接到框架与模型 | 原第6节只有替换 lambda，没有 schema/fake/注册/编译验证 | 增加注册 RMSNorm、CPU/CUDA 分支、FakeTensor、一阶梯度、opcheck/fullgraph 和模型三种输入方式的完整程序 |
| 正式课程 → 面试资料 | 用同一技术口径回答追问 | 旧面试资料存在固定性能倍数、错误硬件数字、过度概括和无来源招聘频率 | 重写两份复习资料，补适用条件、推导和个人证据要求；恢复 MLA 等重要主题 |
| 总路线 → 当前学习位置 | 找到唯一阅读断点 | 兼容路线复制了过时断点 | 移除重复进度，仍由 PATH/NOW 管理实际状态 |

## 岗位覆盖判断

仅以招聘方公开页面为能力样本，不据此推断招聘频率、个人资格或录用概率。核对了 Anthropic 的 GPU Performance Engineer、RadixArk 的跨硬件推理，以及 Jane Street 的 ML Performance Engineer；NVIDIA Workday 页只能读取搜索摘要，不当作全文已核验来源。

| 能力 | 正式教材位置 | 当前材料判断 | 需由实际经历补足的证据 |
|---|---|---|---|
| GPU 执行、存储、同步、数值 | GPU 与 CUDA 第1-4章 | 有正文、图解、代码与局部实验 | 对目标设备解释地址、指令、同步和资源的实际行为 |
| 完整算子与深入优化 | 九类算子，尤其 GEMM、Norm、Prefill/Decode、量化和 MoE | 已有数学、实现及优化内容；不重新拆成独立优化路线 | 至少三项有强基线、多轮优化、退化分析和跨 shape 结果的作品 |
| 编译产物与性能诊断 | 执行章、性能章、GEMM 的 PTX/SASS/CUTLASS 分析 | 有 source/汇编阅读与工具入口；补统计方法 | Nsys 时间线、NCU 指标、编译资源与实际代码改动互相对应 |
| PyTorch 算子接入 | Mini Transformer 第6节 | 本次补完整注册、FakeTensor、梯度、编译与模型替换实验 | 框架实际检查通过；CUDA 后端实际执行；版本兼容和性能回归 |
| 量化 | 量化算子与 INT8/FP8、相关模型论文 | 包含编码/打包、校准、算法与质量评估 | 同一模型、固定数据与低精度 kernel 的质量—显存—速度对照 |
| 模型执行分析 | Prefill/Decode 分析、Mini Transformer | 有形状、FLOPs、容量与状态对照 | 将局部算子收益落实到模型时间，而非只比较孤立 kernel |
| vLLM 与分布式 | 系统、多 GPU 与 MoE 章节 | 用于系统落地；不是取代算子主线 | 后端确认、真实请求与通信测量；训练/运维专项按岗位需要 |
| 论文能力 | 同一阅读器的独立论文线 | 公式、形状、算法状态与关键代码保持独立推进 | 本人解释推导、作者实现与优化边界，不要求每篇复现 |
| NPU → GPU 迁移 | 架构、算子与面试材料 | 保留方法类比，否定执行/同步机制一一等价 | 同一数学任务的两端实现、数值差异、工具和实际优化经验 |

总体判断：课程主题与既定岗位方向一致，主要不足是框架接入、性能证据组织及新旧资料口径。修订解决教材衔接问题，不把“内容已写”标成用户掌握或岗位达标。

## 新程序的验证边界

`registered_rmsnorm.py` 需要 PyTorch 2.4+；CUDA 分支另需匹配的 Triton、驱动和 GPU。首次复核时，本机尚无 PyTorch、Triton 与 nvcc。下方 2026-09-23 的续验记录更新了 CPU 路径状态。

- 两个新增 Python 文件通过 AST 语法检查。
- `--help` 实际退出0；CPU/CUDA 请求在缺少 PyTorch 时实际退出77，未执行注册或设备计算。
- NumPy 有限差分验证三种形状的一阶 RMSNorm 梯度公式，最大绝对误差约 `3.71e-9`；这不验证 PyTorch 注册、fake、autograd 或 Triton。
- 已复核并修正 FakeTensor 连续性对 size=1 的约束，以及 PyTorch FP32 对 NumPy FP64 比较的 dtype 误报。
- 此时 opcheck、fullgraph、模型三路径和性能结果均未运行。CPU aot_eager 只检查图组合，不作为 GPU 编译或性能证据。

交付检查：全站构建、content-only 与 editorial 检查通过；39 篇课程、102 个附属页面、171 处源码摘录逐字校验。测量汇总函数用独立算例验证了中位数、四分位数、几何平均、最差回归及非法输入拒绝，不生成或伪造性能样本。主阅读器与变更页面的本地目标/片段检查通过。浏览器中核对了 Mini Transformer 新小节、同页完整程序入口及面试页三列表格与数学显示。

原始 solutions/reference、PDF、已有性能日志和 PATH/NOW 不改，未提交或推送。

## 2026-09-23：CPU 框架路径续验

仓库根目录原先不存在 `.venv`。本轮使用 Python 3.11 建立 Git 忽略的隔离环境；官方 PyTorch CPU 源的首次请求因证书链无法建立而失败，使用系统证书后后台安装完成。当前环境版本为 PyTorch `2.7.1+cpu`、NumPy `2.4.6`。安装没有修改系统 Python。

从仓库根目录执行：

```powershell
& .venv\Scripts\python.exe -B -X utf8 roadmap/curriculum/model-analysis/examples/registered_rmsnorm.py --device cpu
```

首次运行在“非连续模型 Norm 输入应被拒绝”的测试处失败：`torch.randn((1,3)).t()` 有单例维度，转置后仍可被 PyTorch 视为连续。将该测试改为 `torch.randn((2,3)).t()` 后重跑退出码为 `0`，逐项通过：形状 `(1,1)`、`(3,65)`、`(8,128)`、`(0,65)`；非零 `storage_offset`；非连续模型输入、FP16、非连续算子输入、`D=0/4097`、非正数与非有限 epsilon 的拒绝；一阶梯度对照、`torch.library.opcheck`、`torch.compile(fullgraph=True, backend="aot_eager")`；Mini Transformer 整段、逐 token cache、分块 Prefill 的 logits 对照与同一权重的 NumPy 参考。

这个结果证明当前 PyTorch CPU 版本下的注册、FakeTensor、反向公式、图捕获和模型替换路径可运行。它没有编译或执行 Triton/CUDA；真实 GPU 与 benchmark 表继续留在教程现场填写。CPU `aot_eager` 的通过不构成内部融合或加速结论。

## 外部依据

- https://job-boards.greenhouse.io/anthropic/jobs/4926227008
- https://job-boards.greenhouse.io/radixark/jobs/4343666009
- https://job-boards.greenhouse.io/janestreet/jobs/7449252002
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.pytorch.org/docs/stable/library.html
- https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

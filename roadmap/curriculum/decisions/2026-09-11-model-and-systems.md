# 清理路线并补齐模型与系统正文

用户选择先写完正文，后续统一整改；不以实卡验收阻止教材继续编写。

## 已落地

- PATH/NOW 清理“第三篇优化课”“新极致性能主课”等旧表述，学习事实、数值与当前 Softmax/MLA 焦点不变。
- 量化旧目录改为主题索引，不增加必修循环。
- 课程站增加模型分析、Mini Transformer、vLLM、多 GPU 四章（16–19）；单独模型与系统分区，不计入九类算子编号。
- 最后一章移除自动回到开篇的循环链接。
- 原始 solutions/reference、PDF 与实验记录不变，没有 commit/push。

## 验证范围

CPU block ledger、NumPy 两层 GQA 模型的 full/token/chunk、因果性与 norm 替换检查通过。调度预算、缓存身份、TP 分片和 bias 位置的 CPU 检查通过。

PyTorch 模型对照和 NCCL probe 提供可执行脚本，但本机缺少相应依赖，返回 SKIP；未执行模型服务、GPU benchmark、collective 或远端安装。Tiny 模型是随机参数实验台，不是预训练模型，不具备语言质量意义。

vLLM 文档的 serve/bench 参数和公开源码方法经过只读核对，版本快照在 .codex/course-source-index.json；不代表用户环境已安装该版本。正文中引擎循环标为教学伪代码。

## 完成口径

课程主干第一版正文已从 GPU 基础连接到九类算子、模型分析与系统应用。它不等于全部高级优化、论文精读或真实硬件实验已完成；后续按用户安排统一复核内容深度、图示、执行环境与实测。

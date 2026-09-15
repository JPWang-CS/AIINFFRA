# AIINFFRA总执行计划（兼容入口）

> 旧M0–M5/W1–W12现行课表已由模块化课程替代。本文件保留原链接兼容，只负责导航。

## 正式课程入口

- [模块化课程总路由](curriculum/README.md)
- [第一篇 GPU架构与性能工程](curriculum/gpu/README.md)
- [第二篇 算子实现与优化](curriculum/operators/README.md)
- [第四篇 低精度与量化](curriculum/quantization/README.md)
- [Prefill/Decode与Mini Transformer](curriculum/model-analysis/README.md)
- [vLLM与多GPU/MoE](curriculum/systems/README.md)
- [独立论文线](../notes/algorithms/README.md)

## 当前动作

实践：从[第一篇第一章 GPU硬件总图与架构代际](curriculum/gpu/01-gpu-hardware-map-and-generations/README.md)重新学习；已掌握内容由用户自行跳过，既有状态与实测证据不清零、不降级，未新增完成事实。论文：继续MLA。准确状态只看[PATH](../PATH.md)与[NOW](../NOW.md)。

## 按需查阅

[性能实验检查表](curriculum/performance/README.md)用于整理报告，优化正文在各算子章节内，不作为额外学习步骤。

## 统一规则

有平台题面的算子按原理→LeetGPU原始代码→真实GPU→性能优化；核心锚点进入完整极致优化阶段。课程以通用架构和方法为正文，具体GPU只作为案例和实验身份。旧代码、性能记录和历史课程不删除，新课程通过路由复用。

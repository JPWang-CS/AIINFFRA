# 实操模块

这里承载需要在明确目标环境中逐步复现、记录和复盘的推理服务实操。当前先建立盘古92B模块，部署顺序为 PD 混部，再 PD 分离；框架不预先锁定 vLLM，也不替用户确认模型结构或支持矩阵。

- [盘古92B部署实操](./pangu-92b/README.md)：阶段目标、输入、记录和退出条件。
- [通用整网分析方法](./llm-analysis/README.md)：跨模型的环境、调用链、时间线、内存、通信和证据方法。
- [运行记录索引](./pangu-92b/records/README.md)：真实实验的 run 入口。
- [Serving 实验模板](./templates/serving-run.md) / [问题复盘模板](./templates/serving-issue.md)。

本地文件可读不证明目标远端主机或容器内同样可读。实际执行前，按用户设置→连接中的目标主机和保存的远端项目进入环境；远端与容器内重新核查 skill、memory 和源码引用。当前已完成授权范围内的只读容器盘点，尚未部署、安装模型或形成性能验证；框架产物不等于学习进度、部署完成或真实性能结论。

权威入口：当前本地工作区的 [fullnet-vllm-ascend skill](D:/Desktop/Code/CC4Ascend/.codex/skills/fullnet-vllm-ascend/SKILL.md)、[fullnet memory](D:/Desktop/Code/CC4Ascend/.codex/agent-memory/fullnet/MEMORY.md)；AIINFFRA 进度仍以 [PATH](../PATH.md)、[NOW](../NOW.md)、[HISTORY](../HISTORY.md) 为准。跨仓链接只适用于当前本地工作区，不代表远端 Linux 路径。

盘古整网分析立项见 [CC4Ascend 项目入口](../../../CC4Ascend/projects/pangu-92b整网分析/README.md)。

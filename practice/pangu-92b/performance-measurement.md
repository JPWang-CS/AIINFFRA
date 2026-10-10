# 盘古7B/92B性能测量口径

项目证据入口：[profiler 审计证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-profiler-tool-audit/README.md)。

### 15.1 测量层次与触发方式

正式模型性能采用 ACS 工具报告与昇腾 profiler 采集；自写 `time`、`perf_counter`、`monotonic` 只用于控制器超时、轮询或诊断，不进入性能表。历史 23 分钟是模型加载日志，不是推理性能。

ACS 1.4.1 的 `prof` 报告提供请求层 TTFT、TPOT、E2E、吞吐和失败率。`--trace` 生成 ACS 请求时间线，不能替代设备采集。请求层基线应先关闭设备 profiler，再以少量固定请求采集设备 trace。ACS `--profile` 只是请求服务端 `/start_profile` 和 `/stop_profile`；当前 7B 证据没有证明这条服务路由已打通；本轮 92B 四案例 HTTP 已通过，但 `/start_profile` 和 `/stop_profile` 到 worker 的采集路由仍待验证。

镜像 worker 的 `npu_worker.py:393-429` 显示 `VLLM_TORCH_PROFILER_DIR` 启用 `torch_npu.profiler.profile`，记录 CPU、NPU、Level1、PipeUtilization，并用 TensorBoard handler 输出。`worker.py:276-285` 负责 start/stop，`330-389` 按请求/token 条件触发。`envs.py:371` 的 `OMNI_PROFILE_TOKEN_THRESHOLD` 默认是 `None`；设为非 None 时，`profile()` 在 `worker.py:279-281` 直接返回，不能与自动 token 触发模式混用。自动停止条件是 `profile_step > stop_step`，不是固定采集五步。`VLLM_SERVICE_PROFILER_DIR` 不是本 worker 读取的启动变量。7B 需要先验证 API→engine→worker 的真实路由，再做设备采集。

### 15.2 msprof、帮助命令与查看工具

已核对的私有工具路径为：

```text
/data/docker/w00939120/pangu-deploy/cann/9.2.0/cann-9.2.0/tools/profiler/bin/msprof
```

本次实际执行的只读帮助命令如下；它只验证工具和参数，不启动采集：

```bash
R=/data/docker/w00939120/pangu-deploy
cd "$R"
scripts/with-cann-9.2.0.py /usr/bin/env HOME="$R/runs/profiler-audit-home" TMPDIR="$R/runs/profiler-audit-tmp" MSPROF_OUTPUT_PATH="$R/runs/profiler-audit-out" "$R/cann/9.2.0/cann-9.2.0/tools/profiler/bin/msprof" --help
```

所有目录都在个人运行根 `R` 下。帮助输出确认了采集、`--parse`、`--query`、`--export`、`--analyze`、`--output`、`--task-time`、`--aicpu` 和 `--runtime-api` 等选项；实际采集命令仍需按当前服务进程和帮助文本逐项确定。

设备采集完成后用 MindStudio Insight 查看时间线，用 `msprof-analyze` 辅助分析。官方 950 CCU 说明对 slow-rank、slow-link 和 communication 分析有适用限制；缺少这些数据只能记为未采集，不能解释为没有通信。`msServiceProfiler` 与 `acs-service-profiler` 是不同工具；当前官方产品表中 950 对 msServiceProfiler 为 No，本次不选该路线，ACS-service-profiler 是否适用于本环境尚未核对。

### 15.3 口径、顺序与证据

TTFT、TPOT、E2E 和吞吐必须记录测量端与统计范围。ACS 的 `AVG_TPOT(s)` 与从第二 token 计算的 `AVG_TPOT_SEC(s)` 分开记录。CPU API、NPU kernel 和通信事件可能重叠，不能把各 kernel duration 相加当请求时延；跨进程或跨主机 trace 保存 request id、rank、batch/step、时钟来源和偏移，时钟未对齐时不直接相减。

执行顺序为：基线（关闭 profiler）→少量固定请求与 `--trace`→torch_npu profiler 或 msprof 采集→Insight/`msprof-analyze` 查看→在相同输入、输出长度、并发、batch、缓存、EOS、卡组和资源条件下比较混部与分离。分离还要记录 KV connector、KV 传输字节、网络事件、P/D 独立 trace 和资源分配。当前证据覆盖 7B 请求层短输入基线、工具核对和 92B 四案例 HTTP 请求通过；正式 ACS 性能报告、NPU trace 和 PD 分离仍未采集。自写 Python 计时只用于控制器超时、轮询或诊断，不进入性能表。

工具与证据：[profiler 审计证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-profiler-tool-audit/README.md)、[CANN msprof 950采集说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0011.html)、[CANN torch profiler说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0033.html)、[msprof-analyze Advisor](https://raw.githubusercontent.com/Ascend/msprof-analyze/master/docs/zh/user_guide/advisor_instruct.md)。以上官方资料访问日期为 2026-10-09。

官方补充：[ACS 模型服务与 profiler 示例](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910029.html)；[msServiceProfiler 服务调优说明](https://raw.githubusercontent.com/Ascend/msserviceprofiler/master/docs/en/msserviceprofiler_serving_tuning_instruct.md)。

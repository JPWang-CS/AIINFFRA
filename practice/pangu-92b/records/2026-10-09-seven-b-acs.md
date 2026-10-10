# 2026-10-09 7B ACS 评测记录

两台机器分别完成 ACS prof 与质量试评。并发 1/2 各 32 条请求，四档均 32/32 成功；质量试评为 C-Eval 四科 val 前 8 条，各 32 条，两台均 22/32（68.75%）。分科为计算机网络 6/8、操作系统 6/8、初中数学 4/8、高中语文 6/8。

prof 指标（CSV 原始秒值换算）：11 号 C1 TTFT 24 ms、TPOT 16 ms、E2E 2.076 s、输出 61.65 token/s；C2 为 36 ms、18 ms、2.303 s、111.10 token/s。4 号 C1 为 25 ms、17 ms、2.136 s、59.91 token/s；C2 为 34 ms、17 ms、2.224 s、115.03 token/s。每档 warmup 2，输入平均 33.125 token，实际输出 128 token，EOS 未忽略但均达到输出上限；TPOT 使用 CSV 的 AVG_TPOT(s)，AVG_TPOT_SEC(s) 为从第二 token 计算的独立字段。

质量判分使用 ACS prof 采集回答和项目 `score_quality.py` 精确提取 `trunk_details.content`，未执行 OpenCompass eval。质量请求使用 `/no_think`、maxout32、warmup0；逐题回答、finish_reason、request_id/span_id、标准答案和数据源哈希均在项目证据中。该 32 题是 C-Eval 子集试评，不与模型卡完整分数比较。

证据入口：[项目评测记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-acs.md)、[项目证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-seven-b-acs/README.md)。两台服务均 exit0，owned PID 清空、18081 关闭、物理 1 号卡空闲。评测日志和缓存保存在各自远端个人目录，重跑需使用新日期/run 目录。

# GPU 与 CUDA 教材复核

范围为正式课程第一篇五章。此次修改是教材修订，不是学习进度或实验完成记录。

## 阅读顺序与实践

| 章 | 连续讲解 | 同页代码与练习 | 服务器记录 |
|---|---|---|---|
| 1 | CPU/GPU、Host/Device、SM、线程组织、完整程序、索引、架构和工具 | Vector Add 完整程序及平台接口，CUDA/Triton 索引对照，设备属性查询 | 设备、版本、正确性与测量空表；696 GB/s 旧案例保留其原有条件，不推断精确 N |
| 2 | 线性线程编号、warp、驻留/就绪/发射、分支、数值类型 | 4×2/16×4 图解；四个执行对照 kernel；完整 host/reference/计时程序 | 小输入与稳态运行结果留空；不同 FMA 工作量不能直接比较毫秒数 |
| 3 | 线程私有值、寄存器、地址空间、访问合并、bank、转置 | 原样 naive/tiled 转置代码、完整程序及 Matrix Transpose 题目；padding 对照 | 三种实现分别记录正确性、ms、GB/s 和可用的访存计数 |
| 4 | 共享数据交接、block 归约、warp 交换、stream/event、异步搬运 | block_sum 与完整程序；Reduction 题面；原子通知、双缓冲及高级 API 同页扩展 | 归约、stream、Graph、TMA 等示例按实际运行填写；CPU 合并 partial 不是平台全 GPU 答案 |
| 5 | 计时范围、occupancy、Roofline、精度、MatMul、性能工具 | 用户 MatMul 记录；GPU MODE 方法改写的逐元素平方正确性/计时/trace 实验 | 保留已有真实数据，新环境的复测表与实际 kernel 名称/次数留空 |

## 资料使用

- 本地 CUDA Programming Guide 13.3：核对系统角色、线程层次、warp 分组、地址空间和同步。具体页码与外部来源维护在 `course-source-index.json`，不在正文反复插入出处定位。
- NVIDIA CUDA Samples 固定为 `5443602d89ed99aede2e4b7bf329daddeadb320e`：deviceQuery、transpose、simpleStreams 的机制与本地代码对照。作者示例的线程数、整除前提与本地实现不同，未挪用性能结果。
- Triton 官方 Vector Addition 与 Fused Softmax：索引向量/指针向量、尾块 mask、编译资源与驻留估算。
- GPU MODE 第一讲：分别检查框架表达式、实际 kernel 和未开启 profiler 的时间；新实验为重新编写，不复制讲义中的未定义变量。
- AIInfraGuide：课程组织与主题覆盖参考，硬件和 API 结论由官方手册及代码核对。

## 检查与限制

- `node build.cjs`：39 篇课程、100 个附属 Markdown 页面、42 个资源；168 处原样源码摘录校验。
- `node check.cjs --content-only`、`node editorial-audit.cjs`：通过；公式、链接结构、重复锚点与发布内容检查。
- 与 HEAD 的网页对照：原有 211 个 GPU/CUDA 小节 ID 全部保留。合并段落使用兼容锚点，不改变原始实验文件。
- `node navigation.test.cjs`：通过；目录、折叠、直接 URL、前进后退、搜索和分类导航。新增深层正文折叠测试。测试保留所有导航结构及搜索数据，仅清空非导航的高亮代码和 KaTeX 内部节点，避免多 JSDOM 会话耗尽默认堆。
- `node link-audit.cjs`：通过；101 个 HTML 文件、5123 个本地目标、3956 个片段链接。它不验证外站题目可运行或平台提交状态。
- `node theme.test.cjs`、`git diff --check`：通过。内联 PyTorch profiling 实验通过 Python AST 语法检查，没有在 CPU 环境假装执行 CUDA。
- 原始 solutions/reference、下载 PDF、PATH/NOW 与 PDF 覆盖状态未改。PDF SHA256 仍由既有检查验证。
- 没有执行 nvcc 编译或 GPU benchmark。测量空表留给对应课程的现场实验，不据静态检查填写成功或性能成绩。

## 浏览器复核

已检查线程分组 SVG、存储范围 SVG、第二章实际正文、第一章设备查询，以及旧 `chapter=4#chapter-4-section-13` 进入折叠内容后的展开和定位。设备查询与折叠目标均在页顶约 94 px 的固定导航栏下正确定位，图片加载完整。静态站点更新后需刷新页面，旧文档缓存不能作为新稿的验收结果。

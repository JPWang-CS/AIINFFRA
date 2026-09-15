# 第八章 Benchmark、Profiling、正确性与调试

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章建立正确性、计时、Profiler和调试的统一证据链。命令只有在回答明确问题时才有意义；一条孤立耗时不能成为性能结论。

术语：Non-Uniform Memory Access（NUMA，非一致内存访问）；Peer-to-Peer（P2P，点对点传输）；Host-to-Device（H2D，主机到设备）；Device-to-Host（D2H，设备到主机）；Device-to-Device（D2D，设备内部传输）。

## 建立实验身份与软件栈

“某kernel耗时20 ms”不是性能结论，因为缺少回答“哪份代码、什么软件、哪张设备、什么资源配置”的信息。本部分建立一次实验的静态基线。下一部分再处理拓扑、时钟、温度和带宽等动态条件。

## 1. 从代码身份开始

最少保存以下信息：

```bash
date -Is
pwd
git rev-parse HEAD
git status --short
```

commit只能标识已提交内容。工作树有改动时，同一个commit可以对应多个实际程序，因此还要保存dirty状态和与本次实验相关的diff。benchmark入口和完整参数也必须记录，不能只写脚本名。

```text
date:
repository commit:
working tree clean/dirty:
entrypoint:
arguments:
input source or random seed:
```

## 2. Host、容器和设备是三层身份

```text
Host：CPU、socket、NUMA、OS、kernel、Host Driver
Container/venv：镜像、Python、用户态CUDA库、PyTorch、Triton
Device：UUID、PCI bus id、产品SKU、CC、可见逻辑编号
```

逻辑设备号由`CUDA_VISIBLE_DEVICES`和容器映射决定。容器里的`cuda:0`可能对应Host上的任意物理GPU；跨机器恢复时应依赖UUID和PCI bus id。

```bash
uname -a
cat /etc/os-release
lscpu
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,driver_version --format=csv
```

## 3. Driver、Runtime、Toolkit为什么不是一个版本

```text
PyTorch / Triton / CUDA应用
           ↓
CUDA Runtime与用户态库
           ↓
CUDA Driver API与Host Driver
           ↓
GPU
```

| 查询 | 看到的是什么 | 不能推出什么 |
|---|---|---|
| `nvidia-smi` Driver | Host驱动版本 | shell一定安装了`nvcc` |
| `nvidia-smi` CUDA Version | Driver支持的兼容级别 | 程序实际链接的Runtime |
| `nvcc --version` | PATH中的Toolkit编译器 | PyTorch wheel构建版本 |
| `torch.version.cuda` | PyTorch构建对应的CUDA | Host Toolkit版本 |
| `triton.__version__` | Triton软件版本 | 当前kernel已生成何种目标代码 |

容器通常使用Host驱动接口，同时携带自己的Runtime、Toolkit或框架库。因此版本数字不同不自动等于冲突，要通过最小分配和kernel launch验证有效路径。

```bash
which python
which nvcc || true
nvcc --version || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch build CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.ones(1, device="cuda")
    print("smoke:", (x + 1).item())
try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton unavailable:", repr(exc))
PY
```

版本表只证明组件可见；最小运算成功才证明当前Python进程完成了设备分配、launch和同步。

## 4. 静态资源卡要为每个设备单独建立

后续tile、occupancy和理论上界依赖的字段包括：

- Compute Capability和SM数量；
- warp size、每block/SM线程与warp上限；
- registers/SM、每thread或block限制与分配粒度；
- shared memory/SM、每block默认和opt-in上限；
- L2容量；
- 显存容量、介质、时钟与bus width；
- 支持的数值格式、矩阵与异步搬运能力；
- 官方理论带宽及计算口径。

运行时可查询值、官方产品规格和架构/CC上限要分列。它们可能名称相似，但回答的问题不同。

```bash
python - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"device[{i}]")
    print(p)
PY
```

再使用CUDA Samples中的`deviceQuery`或Runtime API补齐属性，并与官方Compute Capability资料交叉确认。

## 5. 资源表怎样变成kernel判断

资源卡不是资产登记表。每个字段都要能落回kernel：

| 资源 | 影响 | 不能这样推断 |
|---|---|---|
| SM数 | grid并行度、wave数量 | SM翻倍则任意kernel加速两倍 |
| Warp/线程上限 | 理论驻留上界 | block越大occupancy越高 |
| Register | 驻留CTA和spill | 源码变量少就一定register少 |
| Shared memory | tile、stage与CTA驻留 | 能编译就代表资源利用合理 |
| L2 | 跨CTA和重复访问缓存机会 | 数据集小于L2就一定全部命中 |
| 显存带宽 | memory-bound roof | 理论带宽等于实际有效带宽 |
| CC/目标 | 指令和dtype可用性 | 目标更新就自动使用专用路径 |

### 用tile算一次shared memory

设GEMM每个stage缓存：

$$
A_{tile}\in\mathbb{R}^{B_M\times B_K},\qquad
B_{tile}\in\mathbb{R}^{B_K\times B_N}
$$

元素宽度为$b$字节，stage数为$s$，忽略padding与额外buffer时：

$$
Shared\ Bytes\approx s\times(B_M\cdot B_K+B_K\cdot B_N)\times b
$$

例如把$B_M$或$B_N$翻倍，不仅增加单CTA shared，还可能扩大accumulator、提高register压力。最终驻留CTA数由线程、warp、register、shared和硬件block上限共同取最小值。

实际编译结果还可能包含padding和编译器生成的额外空间，所以手算用于预判，`ptxas -v`、Triton编译元数据和Profiler用于确认。

## 6. 本部分产物

```text
date / commit / dirty state:
host / OS / kernel / container:
CPU / socket / NUMA:
logical GPU index / UUID / PCI bus id:
product / CC / SM count:
driver / toolkit / runtime / framework:
register / shared / L2 / VRAM:
supported feature and source:
missing fields and reason:
```

最后选择一个已有kernel，用tile和dtype手算shared，用编译结果读取register，再说明该配置受到哪项资源约束。只抄规格表不算完成。

仓库提供只读采集入口：[collect-gpu-environment-baseline.sh](../../../../scripts/collect-gpu-environment-baseline.sh)。运行后必须审查输出，不能把脚本成功当成环境验收完成。

## 练习

1. 为什么commit相同仍可能不是同一份benchmark代码？
2. `nvidia-smi`中的CUDA Version为什么不等于`nvcc --version`？
3. 容器没有`nvcc`时，PyTorch为什么仍可能正常使用GPU？
4. 每SM shared与每block shared分别约束什么？
5. 根据上面的公式，计算一个两stage FP16 GEMM tile的最低shared需求。
6. 为什么资源卡必须按具体设备和板型分别建立？

官方资料：[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) · [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) · [CUDA Samples Utilities](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/1_Utilities/README.md)

## 建立可以相信的性能实验

上一部分建立了静态身份，但同一份代码在同一张设备上仍可能跑出不同结果。原因可能在GPU外部的拓扑，也可能在时钟、功耗、温度、其他进程或计时方法。本部分把这些变量逐一排除，并建立数据搬运基线。

## 1. 先画数据经过的路径

GPU程序至少涉及三类链路：

```text
H2D/D2H：CPU内存 ↔ PCIe或一致性互联 ↔ GPU显存
P2P：    GPU显存 ↔ PCIe/NVLink/NVSwitch ↔ 另一GPU显存
D2D：    同一GPU内部显存控制器完成的数据搬运
```

它们的瓶颈不在同一位置，不能把数字放进一列直接排名。

H2D/D2H受Host内存、NUMA、PCIe链路和pinned/pageable内存影响。P2P要看两张GPU间的实际拓扑、peer access和路由。D2D更接近设备内部显存路径，但copy结果仍不等于某个真实kernel的有效带宽。

## 2. NUMA和PCIe为什么会影响GPU实验

双路CPU服务器中，每张GPU通常更靠近一个CPU socket和NUMA node。若Host buffer分配在远端NUMA内存，数据可能先跨CPU互联，再进入GPU所在PCIe root complex。

```text
NUMA 0 memory → CPU socket 0 → 跨socket链路 → CPU socket 1 → PCIe → GPU
```

这会影响H2D/D2H，也可能影响由CPU持续驱动的小kernel或通信线程。单个纯GPU kernel的显存访问瓶颈则不能用PCIe或NVLink解释。

```bash
lscpu
numactl --hardware || true
nvidia-smi topo -m
nvidia-smi topo -p2p r || true
lspci -tv || true
```

`nvidia-smi topo -m`中的`PIX/PXB/PHB/NODE/SYS/NV#`描述设备间经过的拓扑层级。报告不能只粘贴矩阵，还要翻译成实际路径，例如“两张GPU经同一PCIe switch”或“GPU与当前CPU线程跨socket”。

多GPU时，拓扑标签只说明连接关系，不证明带宽已经达到上限。还需要P2P能力检查和实际copy/通信测试。

## 3. 运行状态怎样污染结果

GPU存在动态频率和功耗管理。代码完全不变，以下因素也会改变耗时：

| 因素 | 常见现象 | 检查方法 |
|---|---|---|
| JIT和冷启动 | 第一轮明显更慢 | 把编译、初始化与warmup放在计时外 |
| GPU Boost/P-state | 前后频率不同 | 记录P-state、SM/memory clock |
| 功耗限制 | 长时间运行后频率受限 | 记录power draw、limit和throttle原因 |
| 温度 | 持续测试逐步变慢 | 记录温度与时钟时间序列 |
| 其他进程 | 方差和显存占用增加 | 保存进程列表 |
| MPS/MIG/共享环境 | 可见资源不是完整设备 | 保存MPS、MIG和compute mode |
| CPU launch抖动 | kernel之间出现空洞 | 用Nsight Systems看时间线 |

`GPU-Util=100%`只表示采样窗口内GPU处于忙状态，不表示Tensor Core、CUDA Core或显存带宽达到高效率。

```bash
nvidia-smi
nvidia-smi -q
nvidia-smi --query-gpu=timestamp,index,name,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,memory.used,utilization.gpu,utilization.memory --format=csv
```

正式benchmark前后各采一次。无权限读取的字段记录为“未获取及原因”，不要为了完成表格擅自修改clock、power limit或服务器服务。

## 4. 一套可复现的计时协议

1. 固定shape、stride、dtype、输入分布和正确性容差；
2. 把JIT、内存分配和数据初始化移出计时区间；
3. warmup后同步；
4. 使用CUDA Event或经过验证的benchmark工具，不用未同步的Host墙钟；
5. 报告重复次数、中位数，并保留方差或分位数；
6. baseline和待测实现使用同一输入、同步与统计方法；
7. 前后核对运行状态，发现降频、后台任务或异常方差时作废重测。

PyTorch中可以使用CUDA Event理解基本计时：

```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

for _ in range(warmup):
    fn()
torch.cuda.synchronize()

starter.record()
for _ in range(repeat):
    fn()
ender.record()
torch.cuda.synchronize()

ms_per_call = starter.elapsed_time(ender) / repeat
```

正式课程代码还应处理多轮统计、输入复用和正确性检查。这里的关键是Event记录GPU时间线，并在读取结果前同步。

## 5. 理论带宽、有效带宽和真实流量

理论显存带宽通常根据memory clock、bus width和数据传输率计算。它是硬件上界，不是应用保证值。

有效带宽按算法认为必要的读写字节数计算：

$$
BW_{effective}=\frac{B_{read}+B_{write}}{t}
$$

设元素数为$N$、每元素$b$字节：

```text
Copy：       读 N*b + 写 N*b = 2*N*b
Vector Add：读 A/B 共 2*N*b + 写 C 的 N*b = 3*N*b
```

Softmax不能机械套“一读一写”。多pass实现可能反复读写中间结果，融合实现可能把partial保留在片上。应先按算法口径声明最小字节数，再用Profiler观察实际DRAM和cache流量。

因此：

- effective bandwidth适合比较同语义实现；
- Profiler bytes用于观察真实memory行为；
- 二者不同不一定是错误，可能来自cache、重复访问、write allocation或workspace；
- 厂商理论带宽只能作为reference roof，不能当实测。

## 6. H2D/D2H为什么区分pageable和pinned

pageable Host内存通常不能直接作为稳定DMA源，驱动可能先复制到pinned暂存区；pinned内存可直接参与DMA并支持异步传输，但占用不可分页系统内存，不应无限申请。

测试时要报告：

- buffer size；
- pageable或pinned；
- 同步或异步；
- 是否与计算重叠；
- warmup和repeat；
- CPU/NUMA绑定；
- H2D、D2H、D2D或P2P的方向。

小buffer常被API与launch固定成本主导，不能代表大块传输上限，应做size sweep。

## 7. 使用什么官方工具

CUDA Samples仍提供`deviceQuery`用于枚举设备属性，也提供`topologyQuery`查看多GPU拓扑。旧教程常引用`bandwidthTest`，但该sample从CUDA Samples 12.9起已移除。新版官方README建议使用[NVBandwidth](https://github.com/NVIDIA/nvbandwidth)进行带宽测量。

因此课程执行时先记录工具和版本：

```text
deviceQuery：记录PASS/FAIL和关键属性
topologyQuery：多GPU环境记录拓扑
NVBandwidth：记录版本、testcase、buffer size和方向
自建copy benchmark：保存源码、计时口径和正确性验证
```

若使用旧版`bandwidthTest`复现实验，必须写清CUDA Samples版本，不能把它描述成当前版本始终存在的工具。

## 8. 本章最终报告

```text
代码：date / commit / dirty state / entrypoint / arguments
Host：OS / kernel / CPU / socket / NUMA / container
Device：UUID / PCI bus id / SKU / CC / count
资源：SM / warp / register / shared / L2 / VRAM
软件：Driver / Runtime / Toolkit / Python / PyTorch / Triton
拓扑：CPU-NUMA-PCIe-GPU / P2P / NVLink或其他链路
状态：P-state / clock / power / temperature / other processes
计时：warmup / repeat / synchronization / statistic
带宽：H2D pageable+pinned / D2H pageable+pinned / D2D / optional P2P
缺失字段：原因
结论：是否适合继续算子性能实验
```

## 验收问题

1. H2D、P2P和D2D的瓶颈位置为什么不同？
2. 绑错NUMA为什么可能影响传输，却不能解释纯kernel的DRAM瓶颈？
3. `GPU-Util=100%`为什么不是高效率证据？
4. 为什么第一次Triton运行不能直接计入稳定benchmark？
5. effective bandwidth与Profiler观测流量分别回答什么问题？
6. pageable和pinned Host内存的数据路径有什么差别？
7. 为什么旧`bandwidthTest`命令必须带版本条件？
8. 哪些字段缺失时，一条性能结果应该直接判为无效？

完成全部采集并解释结果后，才能更新`PATH.md`和`HISTORY.md`。课程文件存在、脚本运行成功或只读完理论都不等于本章完成。

官方资料：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) · [CUDA Samples Utilities](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/1_Utilities/README.md) · [NVBandwidth](https://github.com/NVIDIA/nvbandwidth) · [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 章节导航

[上一章](../07-cuda-triton-compilation-ptx-sass-and-libraries/README.md) · [返回第一篇目录](../README.md) · [下一篇：完整GPU / LLM算子体系](../../operators/README.md)


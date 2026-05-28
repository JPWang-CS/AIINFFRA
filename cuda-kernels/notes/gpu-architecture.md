# GPU 架构：NVIDIA vs Da Vinci（昇腾）对比

*Phase 1 Week 1-2 学习笔记*
*状态：📖 在读*

---

## 1. 整体架构对比

### NVIDIA GPU (e.g., A100)
```
GPU
├── GPC (Graphics Processing Cluster) × 8
│   └── TPC (Texture Processing Cluster) × 2
│       └── SM (Streaming Multiprocessor) × 2
│           ├── CUDA Cores (FP32/INT32)
│           ├── Tensor Cores (FP16/BF16/FP8/FP64)
│           ├── Register File (64K × 32-bit per SM)
│           ├── L1 Cache / Shared Memory (192 KB, configurable)
│           ├── Warp Scheduler × 4
│           └── Load/Store Units, SFU
├── L2 Cache (40 MB)
└── HBM2e (80 GB, ~2 TB/s bandwidth)
```

### 昇腾 Da Vinci (e.g., Ascend 910B)
```
NPU
├── AiCore × 32 (compute unit)
│   ├── Cube Unit (矩阵乘，类似 Tensor Core)
│   ├── Vector Unit (向量计算，类似 CUDA Core)
│   ├── Scalar Unit
│   ├── L1 Buffer (1 MB)
│   ├── Unified Buffer
│   └── L0A/L0B/L0C (专用缓存)
├── L2 Cache
├── HBM (64 GB)
└── TS (Task Scheduler)
```

## 2. 关键概念映射

| 概念 | NVIDIA CUDA | Ascend CANN | 说明 |
|------|-------------|-------------|------|
| 计算单元 | SM (Streaming Multiprocessor) | AiCore | 基本计算单元 |
| 线程 | Thread | - | CUDA 更细粒度 |
| 线程组 | Warp (32 threads) | Block (多核协同) | Ascend 的 block 概念不同于 CUDA block |
| 线程块 | Block (≤1024 threads) | - | |
| 矩阵计算 | Tensor Core | Cube Unit | 都做 FP16/BF16/INT8 的 MMA |
| 片上内存 | Shared Memory / L1 | L1 Buffer + Unified Buffer | Ascend 的 buffer 层次更显式 |
| 设备内存 | HBM (Global Memory) | HBM (Global Memory) | 几乎一致 |
| 同步 | __syncthreads() | block_sync() | |
| 异步 | CUDA Stream | Ascend Stream | 概念完全一致 |
| 编译 | nvcc → PTX → SASS | Ascend C → TBE → 二进制 |

## 3. 编程模型差异

### CUDA: SIMT (Single Instruction, Multiple Threads)
```cuda
__global__ void kernel(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 每个 thread "看起来是" 独立执行的
    data[tid] = data[tid] * 2.0f;
}
```

### Ascend C: 显式数据搬运 + 计算流水
```cpp
// Ascend C 把搬运(tpipe)和计算(pipe)显式分离
// 通过 pipe 机制做 double/triple buffer 流水
class Kernel {
    void Process() {
        // CopyIn → Compute → CopyOut pipeline
    }
};
```

## 4. 内存层级对比

```
NVIDIA A100:
  Register (256 KB/SM) → L1/Shared (192 KB/SM) → L2 (40 MB) → HBM (80 GB)

Ascend 910B:
  L0 Buffer → L1 Buffer (1 MB/AiCore) → Unified Buffer → L2 → HBM (64 GB)
```

关键差异：
- NVIDIA 的 Shared Memory 是 programmer-managed cache，需要显式管理
- Ascend 的 L1/UB 也是 programmer-managed，理解成本类似
- 两者都要关注 bank conflict、alignment、coalescing

## 5. 学习要点

- [ ] 理解 warp 调度和 divergence 的代价
- [ ] 理解 shared memory bank conflict
- [ ] 理解 tensor core 的 tile 尺寸和精度
- [ ] 写出正确使用 shared memory 的 tiled GEMM
- [ ] 用 Nsight Compute 分析 occupancy 和 memory throughput

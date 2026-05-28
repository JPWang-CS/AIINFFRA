# PagedAttention: Efficient Memory Management for LLM Serving

**Authors**: Kwon, Li, Zhuang et al. (UC Berkeley)
**Venue**: SOSP 2023 | **优先级**: P0 | **状态**: ✅ | **日期**: 2026-05-28

---

## 解决了什么问题

LLM serving 中 KV Cache 的内存碎片化：
- 传统方式为每个请求预分配连续显存来存 KV Cache，请求长度不一导致严重碎片
- 无法在 beam search / parallel decoding 等场景共享 KV Cache
- 实际可用显存利用率只有 20-40%

## 怎么解决的

借鉴 OS 虚拟内存的 **分页管理**，把 KV Cache 切分为固定大小的 physical blocks：

1. **Block Table**: 每个请求维护 logical block → physical block 的映射表
2. **Block Manager**: 全局物理块分配器，支持 allocate / free / copy-on-write
3. **PagedAttention Kernel**: 修改后的 attention kernel 根据 block table 做间接寻址
4. **Copy-on-Write**: 物理块被多个请求引用时，写入前复制，实现 KV Cache 共享（beam search 场景）
5. **Eviction**: 物理块不够用时，按策略淘汰不常用的块（类似 OS 页面置换）

## 关键数据

| 指标 | 传统方案 | PagedAttention |
|------|:---:|:---:|
| KV Cache 显存利用率 | 20-40% | **~96%** |
| 最大 batch size | 受碎片限制 | **2-4×** 提升 |
| 吞吐量 | 1× | **2-4×** (更高 batch) |
| Beam search 内存 | N× copies | **~1×** (COW 共享) |

## 与我何干

> OS 几十年前就解决了虚拟内存分页的问题，搬到 GPU KV Cache 管理上照样有效。好的系统设计思想是跨领域的，不要被 "AI Infra" 的标签限制思路。vLLM 的核心创新就是这一层 memory management，剩下的都是工程。

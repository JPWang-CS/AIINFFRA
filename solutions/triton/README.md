# Triton 实现

> 位置：自己写的 Triton kernel，跑通 + 验证后才进这里。
> 计划入口：[roadmap/ai-infra-curriculum.md](../../roadmap/ai-infra-curriculum.md) M2

## 预期文件

| 文件 | 功能 | 验收 |
|------|------|------|
| `vector_add.py` | Triton Vector Add | 和 PyTorch 对齐，记录耗时/吞吐 |
| `matmul.py` | Triton tiled GEMM | 正确性 + GFLOPS |
| `fused_softmax.py` | Triton Fused Softmax | 正确性 + 提速 |
| `flash_attention.py` | Triton Flash Attention | 对比 PyTorch ref + 显存/速度 |
| `gqa.py` / `fused_mlp.py` | 模型结构组件 | 正确性 + autotune |

## 规则

1. 从空文件开始写，不要直接复制 `reference/triton/`。
2. 每个 kernel 先验证正确性，再记录性能。
3. 每个文件开头写一行说明：算子、版本、关键优化。
4. 跑不通、没有数字的文件不要标完成。

## 验证命令

```bash
python vector_add.py
python matmul.py
python fused_softmax.py
python flash_attention.py
```

无 GPU 时可用：

```bash
TRITON_INTERPRET=1 python xxx.py
```

## 参考

- [Triton 语法速查](../../notes/triton/triton-cheatsheet.md)
- [reference/triton](../../reference/triton/)
- [Lesson 06](../../lessons/06-triton-intro.md)

# Papers Inbox

> 自动抓取的临时收件箱。这里的条目尚未核验，不代表已学习，也不改变 `PATH.md`。

## 使用

```bash
python scripts/research_watch.py --topic kernels --days 7 --write-inbox
python scripts/research_watch.py --topic distributed --days 14 --write-inbox
python scripts/research_watch.py --topic inference --days 7 --write-inbox
```

主题：`kernels`、`inference`、`distributed`、`training`、`architecture`。

## 每周筛选

1. 删除标题相关但内容无关的误命中。
2. 到 arXiv、作者主页、官方 GitHub 核对版本与代码。
3. 标记 `P0 / P1 / P2 / skip`。
4. 只有 P0/P1 且能挂到当前/近期 PATH 节点，才进入正式索引。
5. 性能数字补全 GPU、dtype、shape、baseline、测量范围。

Inbox 文件可以按月归档或删除；正式论文入口在 [papers/README.md](../README.md)。

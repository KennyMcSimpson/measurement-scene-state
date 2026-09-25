# 协作规则

这个仓库同时保存已验证代码、失败实验和研究假设。提交代码前先读 `STATUS.md`、`EXPERIMENTS.md` 和对应协议。

## 每个实验必须留下

1. 一个不可事后修改的 protocol JSON 或 Markdown；
2. 数据来源、版本、split、标签开放时点和本地 hash；
3. 精确命令、环境、seed、测试退出码和代码 hash；
4. 原始逐样本记录与汇总脚本；
5. 可声称与不可声称的边界；
6. 若修复了旧结果，写清楚 supersedes、correction reason，并保留旧 artifact。

## 不要做的事

- 不要提交数据集、权重、缓存、checkpoint、`.venv` 或大日志。
- 不要把 development、oracle 或 geometry diagnostic 数字写成官方 benchmark。
- 不要在看过 validation 后继续调参，再把它叫独立测试。
- 不要把 RGB MSE 下降自动解释成语义对应提升。
- 不要删除失败记录；用 `STOPPED` 或 `SUPERSEDED` 标记。

## 提交前检查

```powershell
uv run pytest -q
uv run ruff check .
git diff --stat
git status --short
```

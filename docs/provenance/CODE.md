# 代码和运行 provenance

公开仓库第一次提交会建立新的 Git commit；在此之前 `<workspace-root>` 没有 Git 提交，不能把本地文件时间戳当作版本。

二维 probe 的运行 provenance 已记录在本地 corrected output：

- `code_provenance.json`：源码和脚本 SHA-256；
- `protocol.json`：事前 split、动作、反馈、评价和停止边界；
- `completion_verification.json`：2424 raw rows、32 validation pairs、完整覆盖、代码运行期间 hash 不变；
- `analysis.json` / `posthoc_diagnostic.json`：聚合结果和事后机制检查；
- `CORRECTION_REQUIRED.json`：原始运行的汇总 bug 及修复理由。

运行环境：Python 3.11–3.12，项目依赖由 `pyproject.toml` 和 `uv.lock` 声明；上一轮使用 CUDA-enabled PyTorch 和 RTX 5070。公开 bundle 不包含本机 cache、checkpoint 或 venv。

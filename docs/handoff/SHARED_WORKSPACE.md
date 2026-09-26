# 共享仓库交接：2026-09-27

新增代码、测试与研究报告一起提交。旧报告保留当时的结论和运行证据；当前汇总以 README、STATUS、EXPERIMENTS 最新条目为准。

## 从哪里开始

| 目标 | 入口 |
|---|---|
| 查看最新3D机制结论 | [已训练权重补测](../experiments/EXP-3D-20260927-trained-state-history-v1/README.md) |
| 从零准备数据、训练1000A+600B并复测 | [便携复现说明](3D_REPRODUCTION.md)，`bash scripts/reproduce_small_3d_pipeline.sh outputs/my-new-run cuda` |
| 查看权重规格与记录的SHA256 | [checkpoint索引](../experiments/EXP-3D-20260927-centered-training-1000a-600b-v1/checkpoint_index.json) |
| 二维V2及其阻塞原因 | [V2报告](../experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md) |
| 二维最后一次事后公平基线分析 | [follow-up](../experiments/EXP-2D-followup-20260927/README.md) |

3D最新权重可运行，但静态资格未建立；同场景留出帧的有益history flip为0/3。二维V2仍缺独立确认。请勿把工程测试或oracle正值描述为泛化成功。

## 报告与本地资源

七份较大的JSON/JSONL/CSV报告使用确定性gzip无损压缩，从约92 MB缩小到约9.4 MB。克隆后运行：

```bash
python scripts/restore_report_artifacts.py --check
python scripts/restore_report_artifacts.py
```

脚本按 `docs/report_artifacts.json` 同时校验压缩文件和原文SHA256，只恢复到原报告路径；已有文件内容不同会拒绝覆盖。解压文件被Git忽略。报告中的旧CHECKSUMS可能引用当时完整本地输出，因此不是整个Git checkout的文件清单；本次提交范围见 `docs/publication_manifest.json`。

数据、checkpoint、`.venv`、`outputs/` 不提交。checkpoint索引中的绝对路径是原机器的历史位置，不是下载链接。其他协作者按便携脚本重建；跨硬件重训不保证相同权重hash。如需确切历史权重，应由持有者另行传输并按索引校验。原来报告、锁、dirty.patch中的绝对路径及源码hash作为历史证据保留，不要批量改成新路径或把它们当新机器可直接执行的命令。

## 环境与检查

训练复现使用 `uv.lock`，Python3.11/3.12；3D脚本依赖受控范围的Hypersim网络下载。CPU代码测试不需要下载数据或权重：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
```

GitHub Actions运行CPU代码测试与报告归档校验；它不执行实验数据下载、GPU训练或正式研究评估。CPU CI从依赖声明安装；准确实验环境使用锁文件。Ruff排除third_party、backups、outputs，避免修改第三方或封存历史源码。缺失旧V5权重的兼容性测试仍明确skip，不能用dynamic权重替代。

## 多人修改约定

1. 开工先 `git fetch origin`，从最新main建立个人分支；合并通过PR，避免多人直接改同一工作目录。
2. 新实验用新目录，固定配置、数据范围、评价规则和checkpoint后再运行；不要覆盖旧raw、失败日志或封存集。
3. 代码改动带相关测试，更新README/STATUS入口；未经验证的结果标清scope。
4. 不提交数据、权重、环境、秘密凭据；新增较大报告采用同样可校验归档策略。
5. 历史协议中的工具路径可能属于旧机器；以新的便携入口为实际执行起点。

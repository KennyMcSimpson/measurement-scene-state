# 给合作者和 AI 的仓库地图

## 先建立正确的模型

这个项目不是一个已经赢得 benchmark 的方法包，而是一个带有历史、负结果和当前机制筛查的研究档案。`STATUS.md` 的证据等级高于任何单个 JSON 里的 `status: complete`。

## 目录

| 目录 | 内容 | 读者用途 |
|---|---|---|
| `src/mcss/` | 原来的 Measurement Scene State 代码、dynamic 分支、data/evaluation 模块 | 复现和修改工程代码 |
| `src/mcss/vision_probe/` | 当前二维 probe 的特征、几何评价、写入和实验运行器 | 理解逆向 JEPA/动态 TTT 的最小机制接口 |
| `scripts/` | 下载、准备、训练、验证、二维分析和诊断脚本 | 找精确命令 |
| `tests/` | 合同测试、runner 测试、二维 probe 回归测试 | 先确认接口和隔离 |
| `configs/` | 三维训练和 smoke 配置 | 只在对应协议允许时运行 |
| `docs/` | 原协议、设计、失败封存、handoff 和 provenance | 研究边界与证据解释 |
| `research/ideas/` | 原始方案、合作者讨论、二维切口和首轮报告 | 了解为什么这样设计 |
| `research/source/` | ICLR 初稿 PDF | 原始论文材料；不等于已发表 |
| `docs/experiments/` | 可公开的小型实验 bundle | 不下载本地大数据也能读懂结果 |
| `outputs/` | 本机完整运行产物；默认不进 Git | 追溯本地运行，不作为远程依赖 |
| `data/` | 本机数据和权重；默认不进 Git | 按 provenance 下载 |

## 阅读顺序

1. `README.md` → `STATUS.md` → `EXPERIMENTS.md`；
2. `docs/provenance/CLAIMS.md` 和 `docs/provenance/DATA.md`；
3. `research/ideas/CV_逆向JEPA与动态TTT_合作者完整讨论方案_2026-09-17.md`；
4. `docs/experiments/EXP-2D-20260921-corrected/README.md`；
5. 最后才读 `src/mcss/vision_probe/` 和运行器。

## 当前主问题

真实观测能不能指导“有用的状态修正”，而不是只让 RGB 重建更容易？首轮结果支持反馈通道存在，但 RGB 代理和对象对应目标失配。下一步是高分辨率评价 + OFF-aware task-grounded gate，不是直接加大训练。

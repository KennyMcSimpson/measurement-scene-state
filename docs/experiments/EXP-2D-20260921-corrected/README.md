# EXP-2D-20260921-corrected

状态：`EXPLORATORY`。这是原始二维机制 probe 的修正版，不能当完整方法或官方 DAVIS benchmark。

## 问题

真实可见观测的 residual 能否指导一个小型私有 fast-weight 写入，使冻结 DINOv2 特征更适合对象对应？动态 selector 是否比固定动作更好？

## 固定协议

- DAVIS train 60 按序列名 SHA-256：fit 24 / discovery 12 / validation 16 / reserve 8；official val 30 sealed。
- 每序列首帧作为 source，`n//3` 和 `2*n//3` 作为两个 target。
- DINOv2-S/14 frozen，224 square，16×16×384；fit-only PCA whitening 64 维；fit-only RGB ridge。
- 每个 target reset 两份 64×64 私有矩阵：A 是 token channel residual，B 是 3×3 neighborhood residual。
- 两步动作 `OFF/A/B/ALL`；selector 只用可见 RGB probe，oracle 看答案只作诊断。
- 指标为 16×16 token-grid cosine mask transfer mean object IoU，不是 DAVIS J&F。

## 结果

| 方法 | mean object IoU | RGB MSE |
|---|---:|---:|
| OFF/OFF | 0.443378 | 0.030219 |
| fixed OFF/ALL selected on discovery | 0.450837 | 0.017701 |
| RGB residual selector | 0.440269 | 0.012502 |
| finite trajectory oracle (diagnostic) | 0.455144 | 0.024065 |

RGB selector 的重建 MSE 下降约 58.6%，但对象 IoU 下降约 0.31 个百分点。答案 oracle 的额外机会主要集中在两个 image pairs，不能作为稳定动态控制证据。

## 文件

- `protocol.json` / `splits.json`：事前协议和身份划分；
- `summary.json` / `analysis.json`：机器可读汇总；
- `raw.jsonl`：逐动作、逐 target 的原始记录；
- `posthoc_diagnostic.json`：特征改变率和 tiny-object 敏感性；
- `completion_verification.json`：覆盖和 hash 核验；
- `code_provenance.json`：运行时源码 hash。

`readout.pt`、DINO 权重、feature cache 和原始 DAVIS 不在公开 bundle 中。按根目录 [docs/provenance/DATA.md](../../provenance/DATA.md) 下载后可重跑。

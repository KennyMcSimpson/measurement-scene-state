# EXP-3D-READOUT-METHOD-V1：可靠几何读出（Reliable Geometry Readout）

2026-10-15，hit，只用 GPU 4。分支 `research/reliable-geometry-readout-v1`（本地，未推送）。决策报告（Claude Docs）：可靠几何读出研究决策报告。

## 问题

在流式 RGB-D（3 个上下文帧 + 4 个流式帧，ARKitScenes）与无图像的查询相机下，回答三个问题：

1. V2 的大幅收益能否被简单标定解释（Gate A）？
2. 有强读出时，100 步 Gaussian TTT 是否仍必要（Gate B）？
3. 有没有值得作为 CVPR 方法的新机制（Gate C）？

## 设计

- **留一队列**：SYSTEM3、VAL、CAJU、SYSTEM4 依次作为评价折，合并 350 个场所、1,400 个查询；FIT_SYSTEM、FIT_SYSTEM2 只作训练。
- **全部评价折都已用于以前的决策**，结论是开发级证据（DATA_EXPOSURE_REGISTRY.md）。
- **Phase 1**：B0（Core B）、B1（+64 箱表）、B2A–B5（只用训练集拟合的简单标定）、B6/B6W（逐像素 MLP）、B7（V2 配方，每折重训）。
- **Phase 2**：SYS_A（只用几何）、SYS_B（+前馈渲染）、SYS_C（+TTT 渲染），三者用同一种读出网络，另做交换检验。
- **判定规则**：在看任何评价折之前锁定（PREREGISTRATION.json，2026-10-15 00:46:10）。
- **三个事后修正案**：都在各自运行前锁定，结论只作探索性证据，不改变预注册判定。
  - 修正案 1：KF 表 + V2 结构中的 TTT 对照。
  - 修正案 2：候选 B 的神谕上界。
  - 修正案 3：候选 A 的自由空间下界。

## 结果（合并 350 个场所，场所平均 AbsRel）

| 判定 | 结果 |
|---|---|
| Gate A | `V2_METHOD_NECESSITY = NOT_ESTABLISHED`：B5 − B7 = +0.0092 [+0.0029, +0.0147]，但 SYSTEM4 的未覆盖像素上 B5 更好 |
| Gate B（预注册） | `TTT_ADDED_VALUE = NOT_ESTABLISHED`：SYS_B − SYS_C = +0.0010 [−0.0026, +0.0038] |
| 修正案 1（事后） | `TTT_ADDED_VALUE_IN_KF_V2_STRUCTURE = SUPPORTED`：B7_FF − B7 = +0.0064 [+0.0036, +0.0094]，94% 在已覆盖像素 |
| 修正案 2（事后，神谕） | `B_PREMISE_ACCURACY = NOT_SUPPORTED`：共享修正上界 +0.0008 [+0.0003, +0.0015] |
| 修正案 3（事后，可部署） | `A_PREMISE_FREE_SPACE = NOT_SUPPORTED`：自由空间下界 +0.0003 [+0.0002, +0.0005] |
| Gate C | `CVPR_METHOD_NOVELTY = INSUFFICIENT` |

- **最强可复现配置**：B7 = R7 + 100 步 TTT + 冻结 KF 表 + V2Net，0.1032（Core B 为 0.1395），每场景 16.6 s。
- **不做 TTT**：B7_FF 为 0.1096，每场景 0.37 s。
- **Core B → V2 的 0.036 如何分解**：
  - 64 箱表 8%；
  - 训练集零命中常数 31%（R7 原来对零命中查询填 Hypersim 常数 2.37 m）；
  - 288 箱简单标定 36%；
  - V2 特有的空间修正 25%。
- **测量与真值同源**：测量、TTT 监督与评价真值都是 iPad LiDAR（`lowres_depth`）。

## 结论

- **Go / No-Go：No-Go。** 不建议继续把这条线作为 CVPR 方法论文投入；保留实用系统与负结果。
- 剩余误差的 56% 在没有任何到达帧观测到的区域。跨视角共享、自由空间约束和跨视角挑选，在现有输入上的增益上界都低于 0.005；瓶颈是信息，不是读出机制。
- 若仍要继续，只推荐"证据分型可靠读出"（NEXT_METHOD_PROPOSAL.md），第一步只用 CPU。

## 文件

| 文件 | 内容 |
|---|---|
| CODE_AUDIT.md | Core B、R7、64 箱表、V1/V2、真值访问与传感器来源 |
| DATA_EXPOSURE_REGISTRY.md、MODEL_LINEAGE.md | 数据暴露与模型血缘 |
| PROTOCOL.md、PREREGISTRATION.json | 锁定的协议与判定 |
| BASELINE_MATRIX.csv、CALIBRATION_CONTROLS.csv | Phase 1 |
| TTT_NECESSITY_RESULTS.csv | Phase 2 |
| REGIONAL_ERROR_BREAKDOWN.csv、BOOTSTRAP_RESULTS.json | 分区误差与全部对比 |
| RUNTIME_PROFILE.json | 分阶段耗时（每场景 / 每查询） |
| AMENDMENT_1–3.md、.lock.json、_RESULTS.json | 三个事后修正案 |
| NOVELTY_AND_RELATED_WORK.md、NEXT_METHOD_PROPOSAL.md | 相关工作与下一项机制 |
| INTEGRITY_REPORT.json、commands.sh | 完整性检查与运行命令 |
| code/ | 全部脚本 |

缓存、逐查询预测（`results/*/preds_*.npz`）和网络权重不进 git，只保存在 hit 的 `outputs/EXP-3D-READOUT-METHOD-V1/` 下。

# STATUS：EXP-3D-READOUT-METHOD-V1

**状态：COMPLETE（2026-10-15）。结论：No-Go。**

## 已完成

| 步骤 | 结果 |
|---|---|
| Phase 0 审计 | 实现与描述一致，没有真值泄漏；发现两个问题：R7 的零命中常数来自 Hypersim；测量与真值是同一个 iPad LiDAR |
| Phase 1 | `V2_METHOD_NECESSITY = NOT_ESTABLISHED` |
| Phase 2（预注册） | `TTT_ADDED_VALUE = NOT_ESTABLISHED` |
| 运行时间剖析 | TTT 占 B7 每场景 16.6 s 中的 97.8%；B7_FF 每场景 0.37 s |
| 修正案 1（事后） | `TTT_ADDED_VALUE_IN_KF_V2_STRUCTURE = SUPPORTED`（探索性） |
| 修正案 2（事后） | `B_PREMISE_ACCURACY = NOT_SUPPORTED` |
| 修正案 3（事后） | `A_PREMISE_FREE_SPACE = NOT_SUPPORTED` |
| Phase 3 | `CVPR_METHOD_NOVELTY = INSUFFICIENT`；只推荐一个方向（ETR），且只在继续时才做 |

## NOT_RUN

- 少步 TTT（10 步）在匹配读出下的精度：约 1 GPU 小时。
- 零命中常数换成训练集标定值后的 B7：需要重训，约 15 分钟 GPU。
- ETR 第 0 步（CPU）与第 1 步（约 2 GPU 小时）。
- 全新数据确认（61 个未准备的扫描）、第二个真实数据源、独立真值（Faro 激光扫描）。

## 需要用户确认

- 是否用 61 个未准备的扫描做一次性全新确认（最后一份同源全新数据）。
- 是否下载新数据：Faro 激光扫描，或第二个 RGB-D 数据源。
- 是否推送分支并开 PR。

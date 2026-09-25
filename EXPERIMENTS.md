# 实验注册表

这里是唯一的人类可读实验索引。机器可读的细节放在各 artifact bundle 和本地 `outputs/` 中。

## EXP-2D-20260921-corrected

- **状态 / 证据**：`EXPLORATORY`；修正版，替代 `EXP-2D-20260921-original`。
- **问题**：可见观测残差产生的私有 fast-weight 写入，能否改善后续对象对应？RGB probe 能否选到有益动作？
- **数据**：DAVIS-2017 train 60 段；fit/discovery/validation/reserve = 24/12/16/8；官方 val 30 sealed。
- **模型**：冻结 DINOv2-S/14，224 输入、16×16×384 token；fit-only PCA whitening 到 64 维；冻结 RGB ridge readout；每个目标图像两份私有 64×64 状态矩阵。
- **动作**：两步 `OFF/A/B/ALL`，共 16 条路径；selector 每步以可见 probe RGB MSE 选四个候选。selector 的候选计算额外计账，oracle 看答案且仅作诊断。
- **主指标**：16×16 token-grid mask transfer 的 mean object IoU；不是 DAVIS J&F。
- **验证结果**：OFF/OFF 44.34%；发现集选定固定 OFF/ALL 45.08%；RGB selector 44.03%；有限答案 oracle 45.51%。selector RGB MSE 从 0.03022 降到 0.01250，但 IoU 下降 0.31 个百分点。
- **不确定性**：按序列的描述性 bootstrap 区间宽，未做多重比较校正；tiny-object token 分辨率是明显限制。
- **可声称**：当前启发式反馈通道能改变特征并降低重建误差；RGB 重建和对象对应存在失配。
- **不可声称**：完整逆向 JEPA、自然 streaming TTT、动态策略成功、官方 DAVIS 排名或 CVPR 级结果。
- **artifact**：[curated bundle](docs/experiments/EXP-2D-20260921-corrected/README.md)。

## EXP-2D-20260921-original

- **状态**：`SUPERSEDED`。
- **原因**：`bestgloballyfixed` 汇总时每段序列只复制了第一个 target slot；原始 raw records 保留在本机 `outputs/vision_2d_probe_20260921/`，并有 `CORRECTION_REQUIRED.json`。
- **修复规则**：同一协议、同一缓存、同一参数选择；只修复记录覆盖并重跑。修正版的未受影响指标逐条精确复现。

## 既有三维记录

`BASELINE-V5-4500`、`EXP-3D-20260919-engineering`、`EXP-3D-20260920-method-dev`、`EXP-GEOMETRY-CERT-V1` 和 `EXP-MAPANYTHING-V1` 的状态与证据边界见 [STATUS.md](STATUS.md)。历史输出没有删除；公开仓库只提交代码、说明和小型 provenance，避免误把数 GB 的 checkpoint/log 当成可复核结果。

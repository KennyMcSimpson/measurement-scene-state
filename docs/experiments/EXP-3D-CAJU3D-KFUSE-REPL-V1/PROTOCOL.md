# EXP-3D-CAJU3D-KFUSE-REPL-V1 预注册协议（冻结的卡尔曼式读出融合在另一批场所上的复现）

## 0. 目的

- **要复现的结论**：EXP-3D-CAJU3D-KFUSE-V1 的次要结论是，非 JEPA 的按像素读出融合（KF_BASE）在 SYSTEM3 和 VAL 上都改善了 Core B（均为 SUPPORTED）。
- **为什么要复现**：这两个确认队列以前也用作过确认集。
- **做法**：把冻结的融合表原样用到另一批从未用于融合选择的场所上，不跑 GPU，不重新拟合任何东西。

## 1. 数据

- **场所**：EXP-3D-CAJU-HEADROOM-V1（CAJU-V1 Gate 0）的 CAJU_PILOT、CAJU_SEL、CAJU_EVAL，共 58 个 ARKit 训练折场所、116 个片段、232 个查询（第 14、15 帧）。
  - 它们与 KFUSE-V1 的拟合集和确认集都不重叠。
  - 真值在 CAJU-V1 中读过（用于效用标签），但从未用于选择融合方式。
- **预测**：
  - TTT：CAJU-V1 的 FULL 回放，即发布版权重、7 视角 CACHE、100 步原始 TTT，与 Core B 的 TTT 相同。CAJU-V1 的修订 1 已核对它与封存的 Core B 分布一致。
  - 几何：CAJU-V1 的几何封存（R7、near7、cosine）。
  - 每个文件核对哈希。
- **H-JEPA 不参与**：这些场所在 H-JEPA 的训练集里，所以不评估 KF_JEPA 和 KF_FULL。

## 2. 臂与检验（锁定）

- **臂**：都用 KFUSE-V1 冻结的 fusion_tables.json（sha256 `15100550…`），不做任何修改：
  - CORE_B（w = 0.5）；
  - GEO_U（w = 0.95）；
  - KF_DIS；
  - KF_BASE。
- **主检验**：P = CORE_B − KF_BASE，在 58 个场所上合并计算，用 V1 的配对场所 bootstrap 和 V2 判定。
  - SUPPORTED → REPLICATED；
  - PARTIAL → PARTIALLY_REPLICATED；
  - HARMFUL → REVERSED；
  - 其余 → NOT_REPLICATED。
- **次要**：
  - CORE_B − GEO_U；
  - GEO_U − KF_BASE；
  - 按三个来源队列分别报告均值（描述性）。
- **完整性**：
  - 真值只经 TeacherAccess 读取；
  - 分析运行两次，结果必须逐字节一致；
  - 锁定前不运行打分代码。

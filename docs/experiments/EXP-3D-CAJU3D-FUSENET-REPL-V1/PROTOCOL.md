# EXP-3D-CAJU3D-FUSENET-REPL-V1 预注册协议（冻结的学习式读出在另一批场所上的复现）

## 0. 目的

- **要复现的结论**：EXP-3D-CAJU3D-FUSENET-V1 的次要结论是，非 JEPA 的学习式读出（FUSE_BASE）大幅改善了 Core B：
  - SYSTEM3：0.1569 → 0.1143，PARTIAL；
  - VAL：0.1161 → 0.0984，SUPPORTED。
  - 改善主要来自未覆盖区域对 R7 填充的修正，逐场所的效果偏重尾。
- **做法**：把三个冻结的 FUSE_BASE 网络原样用到另一批从未参与训练和选择的场所上，不跑 GPU，不重新训练。

## 1. 数据

- **场所**：CAJU-V1 的 CAJU_PILOT、CAJU_SEL、CAJU_EVAL，共 58 个 ARKit 训练折场所、232 个查询。与 FUSENET-V1 的训练、验证和确认场所都不重叠。
- **输入**：与 FUSENET-V1 的基础图像相同（fn_data.base_image），来自：
  - CAJU-V1 的 FULL 回放（发布版权重、7 视角 CACHE、100 步 TTT 的 depth_z 和 alpha）；
  - CAJU-V1 的几何封存（R7、near7、cosine）。
  - 每个文件核对哈希。
- **H-JEPA 不参与**：这些场所在 H-JEPA 的训练集里，所以不评估 FUSE_JEPA。

## 2. 检验（锁定）

- **冻结的模型**：FUSENET-V1 的 BASE_s20261101、BASE_s20261102、BASE_s20261103，按 FUSENET-V1 的 audit/fit_log.txt 核对哈希；读出取三者 log 预测的平均。
- **主检验**：P = CORE_B − FUSE_BASE，在 58 个场所上合并计算，用 V1 的配对场所 bootstrap 和 V2 判定。
  - SUPPORTED → REPLICATED；
  - PARTIAL → PARTIALLY_REPLICATED；
  - HARMFUL → REVERSED；
  - 其余 → NOT_REPLICATED。
- **次要**：
  - KF_BASE − FUSE_BASE；
  - FUSE_BASE_W − FUSE_BASE（关掉未覆盖修正）。
- **描述性**：
  - 已覆盖与未覆盖像素的 AbsRel；
  - 逐场所改善的分位数；
  - 变差的场所数。
- **完整性**：
  - 真值只经 TeacherAccess 读取；
  - 分析运行两次，结果必须逐字节一致；
  - 锁定前不运行打分代码。

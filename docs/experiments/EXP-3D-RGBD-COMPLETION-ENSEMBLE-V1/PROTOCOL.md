# EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1：把学到的补全与调和插值取平均，能否稳定地胜过调和插值？

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 来源与问题

- **项目一直的结论：** 学到的补全（旧载体 G1、DepthSplat 上的逆向 JEPA）在 FAR 区域平均胜过调和插值，但只在约 60% 的场景里更好，V2 gate 判为 PARTIAL。EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1 的 A1 也是如此（+0.069，45/32）。
- **新假设的来源：** V1 结束之后，在 FRESH78 上做了探索性分析（`gate_explore.py`，样本内，不是检验）。把学到的补全与调和插值逐像素取平均，场景一致性大幅提高：
    - HG = (调和 + G1) / 2：FAR 0.300（调和 0.444），68/76 个场景变好；
    - HJG = (调和 + J + G1) / 3：FAR 0.285，66/76；
    - 只平均两个学到的补全 JG = (J + G1) / 2：0.287，只有 54/76；
    - 一致性门控（两者一致才用）几乎没有帮助。
- **本实验要回答的问题：**
    1. 这种"与调和插值取平均"的集成，能否在另一个队列上稳定胜过调和插值（V2 gate，包括 ≥ 75% 的场景一致性）？
    2. 逆向 JEPA 训练出的 DepthSplat 状态（J）能否在旧载体之上为集成增加价值？

## 2. 队列与输入

- **EVAL34 = EVAL-V3 + EVAL-V4：** 83 个场景、166 个 episode、332 个查询，即 DepthSplat 探针的合并清单。
    - 以前被很多实验用过（V17–V32 的 G1 类载体、DepthSplat 探针的零样本、测量规划审计）。
    - 但从来没有评估过 J、JG、HG 或 HJG，对这个在 FRESH78 上提出的新假设是样本外的。
    - 与 TRAIN72、SEG-V1、DEV、保护集在场景和物理场景上都不相交（脚本断言）。
- **episode：** 记录 × 角色 A/B；上下文 = 角色的 3 帧 RGB-D；查询 = primary_query；没有流帧。
- **J：** V1 训练好的三个逆向 JEPA 检查点（冻结，哈希见锁定文件），用 V1 锁定的 `ds_eval.py` 只跑 CTX 臂，尺度上限 3.0，注入测得深度。
- **G1：** V21 的 G1 载体（V17 L 配方，T24000），3 个种子，由 arms 评估器封存。REPROJ_HARMONIC 与 NEAR 掩码由同一封存给出。

## 3. 读出（只在 FAR 上定义；ALL 版本的 NEAR 一律用调和插值）

- **H3：** REPROJ_HARMONIC。
- **G1：** 三个种子深度的平均。
- **J：** 在至少 2 个种子覆盖（α ≥ 0.5）的像素上，取覆盖种子的平均射线深度 (depth_z / α) / cos；其余像素用 H3。
- **JG：** 覆盖处为 (J + G1) / 2，其余为 G1。
- **HG：** (H3 + G1) / 2。
- **HJ：** 覆盖处为 (H3 + J) / 2，其余为 H3。
- **HJG：** 覆盖处为 (H3 + J + G1) / 3，其余为 (H3 + G1) / 2。
- **HG_GEO、HJG_GEO：** 与上面相同，但用几何平均。

指标为 AbsRel。场景值 = 各角色查询平均后再对角色平均；没有 FAR 像素的查询跳过。

## 4. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| E1 | H3 − HJG | FAR | ENSEMBLE_CONSISTENT（SUPPORTED）/ ENSEMBLE_PARTIAL / ENSEMBLE_NOT_ESTABLISHED / ENSEMBLE_HARMFUL |
| E2 | HG − HJG | FAR | JEPA_ADDS（SUPPORTED）/ JEPA_ADDS_PARTIAL / JEPA_NO_ADD / JEPA_HARMFUL |

## 5. 次要分析（只给标签）

- **相对调和插值（FAR）：** HG、JG、G1、J、HJ 分别对 H3。
- **集成之间（FAR）：**
    - JG 相对 G1；
    - HG 相对 G1，即"向调和插值收缩"对 G1 的作用；
    - HJG 相对 JG。
- **几何平均版本（FAR）：** HJG_GEO、HG_GEO 相对 H3。
- **ALL 区域：** HJG、HG 相对 H3。

## 6. 完整性与局限

- **完整性：**
    - 读真值之前，两种封存（arms、J 的 DepthSplat 封存）必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
    - GPU 推理不可逐位复现，但 J 的权重是冻结的。
- **局限：**
    - 这是一个在 FRESH78 上事后形成的假设。EVAL34 对它是样本外的，但这个队列本身以前被多次使用过。
    - 平均的权重（等权）事先固定，不做任何调节。

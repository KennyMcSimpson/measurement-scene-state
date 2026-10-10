# EXP-3D-RGBD-ENSEMBLES-ARKIT-V1：两个集成在真实传感器数据（ARKitScenes）上的检验

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

用户要求（2026-10-08）："在真实传感器数据（ARKitScenes）上检验这两个集成"。两个集成都已在 Hypersim 的 EVAL34 上通过预注册检验，但所有已知结论都只来自合成数据：

1. **补全集成（EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1）：** 学到的补全与调和插值逐像素平均，在 FAR 上稳定胜过调和插值。
    - (调和 + J + G1) / 3：+0.131，60/80；
    - (调和 + G1) / 2：+0.133，65/15；
    - 逆向 JEPA 没有增加价值。
2. **TTT 集成（EXP-3D-RGBD-TTT-ENSEMBLE-V1）：** TTT 后的 DepthSplat 状态与 7 帧无模型重投影取几何平均，稳定胜过后者（+0.029，71/12）。TTT 是必要的（80/3）。

本实验在真实 iPad RGB 与 LiDAR 深度上复现这两组检验，检验和判定规则与 Hypersim 上的完全相同。

## 2. 数据

- **ARKitScenes 3dod：** Validation 折，每个场馆取一个扫描（视频编号最小者），场馆按种子 20261005 的固定顺序（EXP-3D-RGBD-ARKIT-VAL-DATA-V1）。
    - 帧：低分辨率 RGB 与 LiDAR 深度（uint16 / 1000 = 平面深度，换算为射线距离），裁剪到 Hypersim 的视场，再缩放到 128×160。
    - 帧选择与有效性规则在读取任何图像或深度之前锁定：
        - 每个扫描 16 帧，帧间旋转匹配 Hypersim；
        - 帧质量、相机正交性；
        - 源几何一致性：相邻帧重投影的中位相对误差 ≤ 0.01；
        - 角色规则沿用 V2。
    - 前 70 个场馆中 41 个合格。按同一顺序继续往后取，直到凑满 80 个合格扫描：后面 112 个场馆中，处理到第 66 个时达到 80 个合格，其中 27 个不合格。
    - 准备过程不运行任何模型，也不计算任何分数。
- **队列 ARKIT_VAL：** 80 个扫描（80 个不同场馆）、160 个 episode、320 个查询。
    - episode：记录 × 角色；上下文 = 角色的 3 帧；流 = Core B 帧号规则（4 帧）；查询 = primary_query。
    - 所有模型（G1、DepthSplat 发布权重与 J）都从没见过 ARKitScenes。
- **真值说明：** 查询处的"真值"就是同一 LiDAR 传感器在查询帧的测量。它本身有噪声，量程有限，弱纹理处有空洞。
- **许可：** 只用于非商业研究，媒体文件放在 git 之外，不再分发。

## 3. 状态与读出（与 Hypersim 上的两个实验完全相同）

- **补全集成（3 帧上下文；FAR = 有效像素中，距上下文重投影 8 像素以外的部分）：**
    - H3 = REPROJ_HARMONIC；G1 = 三个种子平均；
    - J = V1 的三个冻结检查点，用 V1 锁定的 `ds_eval.py` 只跑 CTX，尺度上限 3.0。至少 2 个种子覆盖（α ≥ 0.5）的像素取覆盖种子的平均射线深度，其余用 H3。
    - JG、HG、HJ、HJG：等权平均，定义同 COMPLETION-ENSEMBLE-V1。ALL 版本在 NEAR 中一律用 H3。
- **TTT 集成（3 帧上下文加 4 帧流）：**
    - R7 = REPROJ7_HARMONIC；
    - ZS = 发布版 DepthSplat，尺度上限 0.1：CACHE，以及 TTT（100 步，m = 1）；
    - CACHE、TTT = α ≥ 0.5 处的射线深度，其余用 R7；
    - R7_TTT_GEO = √(R7 × TTT)，R7_CACHE_GEO = √(R7 × CACHE)，R7_TTT = (R7 + TTT) / 2。
- **指标：** AbsRel；场景值 = 各角色查询平均后再对角色平均。

## 4. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| A1 | H3 − HJG | FAR | ENSEMBLE_CONSISTENT / _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| A2 | HG − HJG | FAR | JEPA_ADDS / JEPA_ADDS_PARTIAL / JEPA_NO_ADD / JEPA_HARMFUL |
| B1 | R7 − √(R7 × TTT) | ALL | TTT_ENSEMBLE_CONSISTENT / _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| B2 | √(R7 × CACHE) − √(R7 × TTT) | ALL | TTT_NEEDED / TTT_NEEDED_PARTIAL / TTT_NOT_NEEDED / TTT_HARMFUL |

## 5. 次要分析（只给标签）

- **补全集成：**
    - H3 分别对 HG、G1、J、JG（FAR）；
    - G1 − HG（FAR，向调和收缩）；
    - H3 分别对 HJG、HG（ALL）。
- **TTT 集成：**
    - R7 − TTT；CACHE − TTT；
    - R7 − R7_TTT；R7 − R7_CACHE_GEO；R7 − CACHE；
    - B1 在 NEAR7 与 FAR7 上的版本。
- 各读出的均值；J 的 FAR 覆盖率。

## 6. 完整性

- 读真值之前，arms 封存、几何封存（复核通过）以及 J1–J3、ZS 的 DepthSplat 封存都必须完整，哈希不变。
- 分析跑两次，结果文件必须逐字节相同。
- 保护集不涉及，不推送。

## 7. 已知局限（锁定前写明）

- **G1 迁移性差：** G1 只在 Hypersim 上训练过。在 Replica 上它比调和插值差（描述性检查）。ARKit 的真实噪声与视场裁剪也可能让它失效。
- **真值有噪声：** LiDAR 的真值本身有噪声且分辨率低，会压低所有方法之间的差异。
- **数据规则：** 数据的源几何规则（1% 阈值）来自合成的 Replica，对真实数据偏严（70 个中 29 个不合格）。我按锁定的规则沿顺序扩展，没有放宽阈值。

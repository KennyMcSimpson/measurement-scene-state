# EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1：在真实传感器数据上训练逆向 JEPA

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

用户要求（2026-10-08）："用 ARKit 训练折在真实数据上训练逆向 JEPA"。

到目前为止，DepthSplat 上的逆向 JEPA 只用合成数据（Hypersim）训练过：

- 在 Hypersim 的 FRESH78 上，它的 FAR 补全平均胜过调和插值，但不稳定（A_PARTIAL）。
- 在真实的 ARKit 数据上，它不比调和插值好：FAR 覆盖率只有 38%。

本实验用完全相同的训练配方，只把训练数据换成 ARKitScenes 的真实 iPad RGB-D，回答两个问题：

1. 在真实数据上训练的逆向 JEPA（JA），对上下文没看到的区域（FAR）的预测，能否稳定胜过调和插值？
2. 与合成数据训练的 J 相比，真实数据训练是否更好？

## 2. 数据

- **训练：** 试点为训练保留的 ARKitScenes Training 折场馆，即种子顺序 20261005 的前 360 位（EXP-3D-RGBD-ARKIT-DATA-V1）。
    - 每个场馆取一个扫描，帧与有效性规则在读取媒体前锁定，规则同评估队列。
    - 前 8 个合格扫描作为 DEV，只用于训练过程的记录；其余作为 TRAIN。数量记录在 `preregistration.json` 的 `training_counts`。
- **评估：** ARKIT160，即两个已有的 ARKit 评估队列合并，共 160 个扫描、320 个 episode、640 个查询：
    - ARKIT_VAL80（Validation 折）；
    - ARKIT_CONFIRM（Training 顺序第 360–519 位）。
    - 与训练场馆不相交，在建目录和每次加载训练数据时都会断言。
    - 这两个队列今天已用于评估 G1、J 与 TTT 集成，但从没评估过 JA。
- **episode：** 记录 × 角色；上下文 = 角色的 3 帧；查询 = primary_query。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型

- **JA1–JA3（本实验训练）：** V1 的训练脚本，只替换训练记录（`code/train_jepa_arkit.py`，与 V1 的差异只在记录来源）。
    - 种子 20261008–10；
    - 4000 步，学习率 5e-6（ViT 5e-7），预热 200 步后余弦衰减，AdamW（weight decay 1e-4），梯度裁剪 1.0；
    - 每步 2 个目标帧，损失 = 原始渲染深度相对 L1 + 0.5 × RGB L1；
    - 尺度上限 3.0，注入测得深度；使用最后一步。
- **对照：**
    - J1–J3：V1，在 Hypersim 上训练；
    - ZS3：发布权重，尺度上限 3.0，与训练同设置；
    - G1：V21 的旧载体，3 个种子，用归一化、会弃权的读出。

## 4. 读出（FAR = 有效像素中，距上下文重投影 8 像素以外的部分）

- **H3：** REPROJ_HARMONIC。
- **JA、J：** 在至少 2 个种子覆盖（α ≥ 0.5）的像素上，取覆盖种子的平均射线深度 (depth_z / α) / cos，其余用 H3。ZS3 只有一个模型，α ≥ 0.5 处取其射线深度，其余用 H3。
- **G1N：** 每个 G1 种子在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余用 H3，再对种子平均。HGN = (H3 + G1N) / 2。
- **集成：**
    - HJA：覆盖处为 (H3 + JA) / 2，其余为 H3；HJ 同理。
    - HJAGN：覆盖处为 (H3 + JA + G1N) / 3，其余为 (H3 + G1N) / 2。
- **指标：** AbsRel，区域为 FAR、NEAR，以及 ALL（ALL 中 NEAR 用 H3）。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| R1 | H3 − JA | FAR | REAL_JEPA_BEATS_HARMONIC（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| R2 | J − JA | FAR | REAL_DATA_TRAINING_HELPS（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

- **训练与基线：**
    - ZS3 − JA（FAR），同一尺度上限下的训练收益；
    - H3 − J（FAR）。
- **集成：**
    - H3 − HJA（FAR，也看 ALL）；
    - HGN − HJAGN（FAR），JA 能否为归一化 G1 的集成增加价值；
    - H3 − HJAGN；H3 − HGN（参照）；
    - HJ − HJA（FAR）。
- **其他：** H3 − JA（NEAR）；各构建器的 FAR 覆盖率；按来源队列分别给出的均值。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，arms 封存与 JA1–3、J1–3、ZS3 的 DepthSplat 封存都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
    - 训练中任何一个种子失败，整个运行停止并如实报告。
- **局限：**
    - 训练数据只有约 200 个扫描，每个扫描 16 帧；Hypersim 上用了 312 条记录。
    - 真值是 iPad LiDAR 的测量。
    - 调和插值在 ARKit 上偏远（可能与补洞常数有关），这会影响所有"相对 H3"的对比，但对每个方法的影响相同。

## 修订 1（2026-10-09 00:03:50 +0800 记录；此时本实验尚未读取任何 ARKIT160 查询深度）

- **起因：** 先前在 ARKit 上发现调和插值在 FAR 中偏远。修订前在训练折的 TRAIN 记录上查明了原因（episode 规则同 ARKIT160，只读训练数据）：
    - 728 个 TRAIN 查询中有 367 个（50%）完全没有上下文命中。这时调和插值整张图都是 Hypersim 的补洞常数 2.37 m，FAR AbsRel 为 1.05；有命中的查询只有 0.38。
    - 有命中的查询上，H3 反而略偏近（中位 log(H3 / 真值) = −0.32），但从 AbsRel 看，不缩放（k = 1）已经最优。
- **H3CAL（训练折拟合的校准调和插值）：**
    - 无命中的查询：整张图取常数 c = 1.05 m。c 取自 0.50–3.00 m 的网格（步长 0.05），使 TRAIN 无命中查询的场景平均 FAR AbsRel 最小。
    - 有命中的查询：NEAR 内取 H3，NEAR 外取 k × H3，k = 1.00。k 取自 0.40–1.20 的网格，平局时取更接近 1 的值。
    - 也就是说，H3CAL 只把补洞常数换成 ARKit 训练折拟合的 1.05 m。TRAIN 上 FAR AbsRel 从 0.742 降到 0.379。
    - 第一版只拟合了一个全局 k（= 0.55）。它主要在修正常数，同时把有命中的查询压得过近。这一版保留在 `amendment_1/first_run_single_k/`，不使用。
- **JA_CAL：** JA 覆盖处取 JA（至少 2 个种子覆盖，沿用锁定的合并规则），其余取 H3CAL。
- **新增对比**（FAR，V2 标签；不改变 R1、R2、S1–S9，也不改动锁定代码与封存）：
    - X1 H3CAL − JA_CAL：JA 在校准基线之上是否带来信息；
    - X2 H3 − H3CAL：只换常数的效果；
    - X3 H3CAL − JA：锁定的 JA 读出对比校准基线；
    - X4 H3CAL − HGN：归一化 G1 集成对比校准基线；
    - X5：只在 JA 覆盖的 FAR 像素上比较 H3CAL 与 JA（描述性）；
    - X6 H3 − JA：只看有上下文命中的查询（描述性）。
- **执行：** `amendment_1/amend1_score.py` 在锁定的 score 阶段之后运行，重复同样的封存检查。分析跑两次，结果必须逐字节相同。

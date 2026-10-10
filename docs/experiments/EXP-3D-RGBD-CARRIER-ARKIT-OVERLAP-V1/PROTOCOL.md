# EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1：用 ARKit 真实数据做重叠感知训练的旧基座补全载体

本文件即预注册。锁定时间与哈希见 `preregistration.json`。本设计在 EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 的任何结果出来之前写定。

## 1. 问题

旧基座的补全载体 G1 在 Hypersim TRAIN72 上训练，采用 V17 L 配方，约 6.4 万参数。

- **它在 ARKit 上已有迁移：** 在 EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1 的事后拆分中，它在有上下文重叠的 ARKit 查询上带来了小而一致的增益（HGN +0.051，67/0/31）。
- **DepthSplat 上的逆向 JEPA 则没有学到补全：** 改用 ARKit 真实数据训练后，它只学到了"室内大概多深"。

本实验回答：用同一冻结配方，只把训练集换成 ARKit 训练折、训练目标只用与上下文有重叠的帧，G1 架构的载体（GA）在有重叠的新查询上：

1. 能否稳定胜过调和插值？
2. 能否胜过 Hypersim 训练的 G1？

## 2. 数据

- **训练集：** ARKit TRAIN，即 EXP-3D-RGBD-ARKIT-DATA-V1 的 182 个扫描（训练折中为训练保留的场馆）。
- **训练查询：** 每条记录的训练查询是同时满足两个条件的帧：不在两组上下文中；两个角色的上下文命中比例都 ≥ 0.05。命中表就是 OVERLAP-V1 的 `training/hit_shares_TRAIN.json`，只用上下文深度与目标相机计算，不读目标深度。
    - 合格帧超过 4 个时，按在排序列表中的位置均匀取 4 个（含首尾），保证冻结的额外视角规则总能找到 4 个空闲帧。
    - 合格帧不足 2 个的记录剔除，数量记在 `freeze.json` 里。
    - 额外视角按冻结规则（`stream_frame_ids`，4 个平均分布的空闲帧）在修改后的记录上重新计算。
    - **第一次冻结（未训练）被替换：**
        - 它没有"最多 4 帧"这一限制，结果冻结规则因为空闲帧不足，剔除了 40 个重叠最多的记录（相机移动最慢的扫描），训练集会系统性偏向快速移动的扫描。
        - 那次冻结整体移到 `first_freeze_without_cap/`，没有进行任何训练。
        - 替换发生在任何训练和任何评估结果之前，`freeze.json` 中有记录。
- **评估队列：** ARKIT160_OVERLAP，与 OVERLAP-V1 的 manifest 和 frame-role lock 逐字节相同。共 160 个扫描、320 个 episode，查询为第 6、7 帧（个别记录顺延）。
- **许可：** 数据只用于非商业研究，媒体文件在 git 之外。

## 3. 模型

- **GA：** 用冻结的 V17 L 训练代码（`scripts/train_rgbd_long_train_carrier.py`）训练，在 V17 风格的根目录 `GA/EXP-3D-RGBD-LONG-TRAIN-V17` 中进行。
    - **配方不变：** 宽度 32、32³ 网格、CONTEXT_DEPTH bounds、冻结的 V2 C1 损失；Adam 1e-3，无学习率调度；每步 1 个场景、1024 条光线；可变 3–7 个视角；梯度裁剪 1.0；24000 步。训练深度先验与 V17 相同。
    - **种子：** 20260928–30，在 CPU 上训练（AVX2，单线程）。
    - **唯一改动：** 训练集与训练查询帧。
    - **检查点：** 用第 24000 步，不做 DEV 选择。冻结代码在训练中对 Hypersim DEV 的评估只作记录。
- **G1：** V21 的 G1（Hypersim TRAIN72，第 24000 步），与之前所有 ARKit 实验相同。
- **评估：** 两个载体都用 arms 评估器，在同一个封存里对 ARKIT160_OVERLAP 做预测。arms 合约在 GA 训练完成后写入，只指向第 24000 步的检查点及其哈希。

## 4. 读出与区域

- **H3：** REPROJ_HARMONIC。
- **H3CAL：** 无上下文命中的查询整张图取 1.05 m，其余为 H3。这是 JEPA-ARKIT-V1 修订 1 的训练折校准。
- **G1N、GAN：** 每个种子在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余取 H3，再对 3 个种子平均。
- **集成：** HGN = (H3 + G1N) / 2；HGAN = (H3 + GAN) / 2。
- **GAN_CAL：** 与 GAN 相同，只是回退值改为 H3CAL。
- **区域：**
    - FARH：至少有一个上下文命中的查询中，NEAR 之外的有效像素。这是主区域。
    - FAR：所有查询中，NEAR 之外的有效像素。
    - NEAR。
- **指标：** AbsRel。场景值 = 各角色查询平均后再对角色平均，空区域跳过。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| C1 | H3 − GAN | FARH | REAL_CARRIER_BEATS_HARMONIC（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| C2 | G1N − GAN | FARH | REAL_DATA_HELPS_CARRIER（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 |
| --- | --- | --- |
| S1 | H3 − HGAN | FARH |
| S2 | HGN − HGAN | FARH |
| S3 | H3 − G1N | FARH |
| S4 | H3 − HGN | FARH |
| S5 | H3CAL − GAN_CAL | FAR（全部查询） |
| S6 | H3 − GAN | NEAR |

另外报告：两个载体在 FARH 上的弃权比例（不透明度 < 0.5 的比例），以及按来源队列（ARKIT_VAL、ARKIT_CONFIRM）分别给出的均值。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，arms 封存必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
    - 任何一个种子训练失败，整个运行停止并如实报告。
- **局限：**
    - 评估扫描与 JEPA-ARKIT-V1 和 OVERLAP-V1 相同，只有查询帧是全新的。
    - 训练集只有约 180 个扫描。
    - 真值是 iPad LiDAR 的测量。
    - 对 GA 来说，冻结代码中"训练日志前 6000 行与 V11 C1 一致"的内置对照不适用，因为训练数据不同。

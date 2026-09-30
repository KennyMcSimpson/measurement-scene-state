# EXP-3D-RGBD-FRESH-QUALIFICATION-V1 协议

起草：Claude（经用户授权补充数据并持续推进），2026-09-29。本协议在对任何 FRESH 场景做模型前向之前冻结。计算全部在 CPU（8–23 号核）上完成，不使用 GPU。

## 0. 背景

- V5（RGB-D 轨道）在 8 个 DEV 场景上首次同时通过 DEPTH、SCENE_SPECIFICITY、STATIC 三个冻结 gate：DEPTH_GAIN +0.114，CI [+0.065, +0.166]。
- 这 8 个 DEV 场景在 V2–V5 中反复使用，所以 DEV 结果不构成资格证明。按项目规则，要在独立数据上复现才算通过。

## 1. 独立队列 FRESH-V1

- **来源：** 官方 Hypersim val split，不在历史上那 365 个训练场景之内；分区表标记为从未被观察过；不在任何保护或排除名单中。
- **独立性约束：** 源资产与 V2–V5 的全部 32 个 TRAIN/DEV 场景及受保护场景都不重复；volume 与这 32 个场景不重叠；每个 volume 只取一个场景。
- **选择过程：** 名单在读取任何媒体之前按规则冻结（`EXP-3D-RGBD-FRESH-SCALE-DATA-V1/candidate_lock.json`），共 9 个候选。
- **有效性检查：** 与模型无关，沿用 V2 的规则。4 个候选没有通过源数据几何校验。修订 1 允许失效的 volume 改用同 volume 的下一个合格场景，但结果没有找到替代。最终 FRESH-V1 为 5 个场景：ai_015_004、ai_022_010、ai_041_003、ai_051_004、ai_052_003。
- **样本量：** 低于数据锁预设的最低 6 个，统计功效偏低。
- **不打开的数据：** 受保护的官方 test split final holdout（26 个）不打开。

## 2. 被验证的对象

- V5 已封存的全部选择：C0、C1、C2 各自 3 个 seed 的 DEV 选择 checkpoint，逐个核对哈希，不重新训练，也不重新选择。
- 状态构建、bounds、renderer、对照（anchor、wrong-scene、spatial shuffle、zero）与 V5 的 DEV 评价完全相同。
- 数据边界：上下文深度只经只含上下文帧的 RGBD loader 进入；所有 FRESH 状态封存之后才读取 query GT。

## 3. 判定（冻结 V2 估计器，与 V5 相同）

- DEPTH_STATUS_FRESH：C0−C1 的冻结 V2 主对比 gate。
- SCENE_SPECIFICITY_STATUS_FRESH：C1 的 wrong-scene 与 spatial-shuffle 损伤，CI 下界都 >0。
- STATIC_STATUS_FRESH：C1 的冻结 V2 静态 gate。
- **FRESH_QUALIFICATION_STATUS：**
  - 三者都 SUPPORTED 时为 QUALIFIED，表示 Core A 的静态 carrier 在 RGB-D 轨道上通过资格验证，按项目规则 Core B 的下一阶段可以开启；
  - 否则为 NOT_QUALIFIED。
- 描述性结果（不作 gate）：与 TRAIN 拟合常数的比较（RAW、OPACITY_NORMALIZED、MEDIAN_SCALED，并附排除 NO_HIT 场景的敏感性）；OBS 区域；逐场景表。

## 4. 禁止项

- 重新训练，或重新选择 checkpoint；
- 根据 FRESH 结果修改任何规则、场景或阈值；
- 打开受保护的 final holdout；
- 使用 GPU。

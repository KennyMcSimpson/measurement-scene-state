# EXP-MEASUREMENT-PLANNING-ORACLE-V1 协议（MEASUREMENT_PLANNING_TASK_V1）

**目的：** 这是 BUDGETED-MEASUREMENT-PLANNING 研究线的第一阶段，即 oracle 余量审计。设定是 RGB 与相机免费、深度测量有预算。要检验的是："在哪些帧上请求深度"这个决策，是否存在简单启发式没有吃掉的余量。本阶段不训练任何模型。

判定规则见 [preregistration.md](preregistration.md)，机器可读的契约在 `task_contract.json`、`information_contract.json`、`baseline_contract.json`、`scene_manifest.json`、`frame_role_lock.json`。

## 1. 任务

- **episode：** 一条记录加一个上下文角色（A 或 B）。
- **初始上下文：** 该角色的 3 个 RGB-D 帧（`context_a` 或 `context_b`，含共享的 anchor 帧），在任何决策之前融合。
- **候选池：**
    - 记录中不属于任何角色的帧，按帧序排列：不是两个角色的上下文、不是查询，也不是 stream、ignored、far_query；
    - EVAL-V3/V4、EVAL-FAR、Replica、DEV 的每条记录都是 9 帧；
    - 两个角色共用同一个池。
- **动作：** 对候选帧请求深度（REQUEST_DEPTH），共 K 次。后端对所选帧取并集，所以请求顺序不影响最终重建。
- **预算：** 主判定 K ∈ {1, 2, 4}；预算曲线 K = 0 到池的大小。
- **查询：** 主分析用 `primary_query` 的 2 帧；EVAL-FAR 的次要分析用 `far_query` 的 2 帧。
- **信息设定（离线）：**
    - 轨迹已知，全部候选帧的 RGB 与相机在决策前都可见；
    - 逐帧到达、看不到后续帧的在线设定留到以后；
    - 离线设定给启发式更多信息，所以对"存在余量"这一结论是保守的。
- **可部署策略能看到：** 初始上下文的 RGB、深度与相机；全部候选帧的 RGB 与相机；已请求候选帧的深度。
- **可部署策略看不到：** 未请求候选帧的深度；查询帧的 RGB、深度与相机；另一个角色的上下文帧；任何 oracle 输出。
- **强制方式：** `MeasurementEnv` 只持有一份净化过的记录，其中只有本角色的上下文帧和候选帧，查询帧根本不在其中。`request_depth` 检查候选身份、不重复和预算，每一次读取都按用途写入访问记录，并在计算阶段逐一审计。

## 2. 重建后端（冻结、无模型）

- **重建：** 所选候选帧与初始上下文的全部测量点，用冻结的 1 像素 z-buffer 重投影到查询相机，再用冻结的调和补洞（REPROJ_HARMONIC）填补空洞。
- **实现：** 用到的函数全部从冻结脚本导入，不改写：`run_rgbd_geometric_baselines.py` 的 `context_points`、`reproject`，`run_rgbd_inpaint_baselines.py` 的 `harmonic_fill`、`ray_cosine`、`near_mask`。
- **计算方式：** z-buffer 逐像素取最小值，所以每帧先单独重投影一次，任意子集取这些图的逐像素最小值。合成场景上的测试确认，这与把多帧拼在一起重投影逐位相同。
- **K=0 复核：** 结果必须与 `~/cv_work/depthsplat/probe_eval34`、`probe_dev` 中封存的 REPROJ_HARMONIC 一致：最大绝对差不超过 1e-9 m，NEAR 掩码完全相同。DEV 彩排的实测最大差为 7.1e-15 m。
- **区域：** NEAR 与 FAR 由初始上下文的命中像素按冻结的 8 像素规则确定，对所有子集保持不变。
- **覆盖率：**
    - `coverage_all`：有效查询像素中，离任何已测命中点不超过 8 像素的比例；
    - `coverage_far`：初始 FAR 像素中满足同一条件的比例。
- **完全没有命中时的常数：** TRAIN 常数 2.3736 m，读自 V17 训练契约。

## 3. 指标与聚合

- **AbsRel：** 按 arms 评估器的方法，在 GT 有效像素上计算 ALL、NEAR、FAR 三个区域。
- **聚合：** episode 的值是其查询的平均，场景的值是两个角色的平均，统计单位是场景。
- **主指标：** ALL AbsRel。
- **bootstrap：** 10,000 次，种子 20260928，使用项目的 `describe` 与 `paired`。报告均值、中位数、95% 区间、变好/持平/变差的场景数、LOSO，以及排名前 1、前 3 的场景贡献占比。

## 4. 方法

**可部署的选择规则（10 个），只通过 `MeasurementEnv` 行动，打平时取较早的帧：**

| 名称 | 定义 |
| --- | --- |
| UNIFORM | 候选序上的居中分层：取第 ⌊(i + ½)·n/k⌋ 个 |
| TEMPORAL_EVEN | 在候选帧序跨度上等距取 k 个目标位置，各取最近的候选帧 |
| LATEST | 最新的 k 个候选帧 |
| RANDOM_SEEDED | 每个 episode 和预算用固定种子（20261006、场景、角色、k）抽一次 |
| POSE_FARTHEST | 贪心：离已测或已选相机最远；距离 = 平移（m）+ 1 m/rad × 旋转角 |
| MAX_UNCOVERED | 按初始上下文重投影到候选视角后的未覆盖比例（8 像素规则）取前 k 个 |
| MAX_NOVELTY3D | 按候选视角的预测表面中，离所有已测点超过深度 2% 的比例取前 k 个。预测表面由已测点 z-buffer 的冻结调和补洞得到 |
| GREEDY_COVERAGE | 贪心未覆盖比例；每请求一帧，就把它的深度加入已测几何再重新打分 |
| GREEDY_FILL_DISTANCE | 贪心：候选像素到最近已测命中点的平均距离（上限 32 像素），作为不确定性的代理 |
| GREEDY_PHOTOMETRIC | 贪心光度残差：把已测 RGB-D 重投影到候选视角后与候选 RGB 的平均差，没有命中的像素记为 1 |

**诊断量（只走教师路径，不可部署）：**

| 名称 | 定义 |
| --- | --- |
| ORACLE | 每个 episode 中，大小为 K、使 episode 平均 ALL AbsRel 最小的子集（样本内） |
| XORACLE | 交叉查询 oracle：用 episode 中另一个查询选子集，在这一个查询上打分 |
| RANDOM_EXPECTED | 大小为 K 的全部子集的精确平均，即均匀随机策略的期望 |
| WORST | 最差子集，只用来看范围 |
| QUERY_NEAREST | 离任一查询相机最近的 K 个候选帧（用到查询相机） |

**最佳启发式：** 每个 K 下，在同一数据上取场景平均 ALL AbsRel 最低的可部署规则。这样选出的启发式偏强，所以对余量判定是保守的。

## 5. 队列

| 名称 | 场景 | 用途 |
| --- | --- | --- |
| EVAL34 | EVAL-V3 ∪ EVAL-V4，83 个 | 主判定 |
| EVAL34_GAP3 | 同一批场景，只保留帧序距每个查询都 ≥ 3 的候选帧（每条记录 4–9 个） | 次要：排除相邻帧效应 |
| EVAL_FAR | EVAL-FAR，79 个 | 次要：远处查询，与所有候选帧都相距很远 |
| REPLICA | 8 个 | 描述：跨数据集 |
| DEV | 8 个 | 只用于彩排，不进入结果 |

**次要的学习式后端：** 已测区域（距任何已测命中点不超过 8 像素）用调和补洞，其余用 G1。G1 的预测来自封存的探针：EVAL34 用 1 个 seed，DEV 用 3 个 seed。只作描述，字段为 `g1_*`。

## 6. 第一阶段问题与对应的量

| 问题 | 量 |
| --- | --- |
| Q1 NO_DEPTH 到 best-1 的改善 | 调和插值 − ORACLE_1 |
| Q2 best-2/best-4 是否收益递减 | ORACLE 的逐 K 增量，以及按历史大小分组的边际效用 |
| Q3 最强启发式离 oracle 多远 | BEST_H_K − ORACLE_K、BEST_H_K − XORACLE_K、余量比例 |
| Q4 最优帧是否因场景而异 | best-1 在池中位置的分布，以及众数位置的占比（低于 0.5 记为因场景而异） |
| Q5 位姿或覆盖规则是否已接近 oracle | 各规则的余量比例 |
| Q6 效用是否有可预测的结构 | 单步效用与每个可部署特征的 episode 内 Spearman，及其 top-1 一致率 |
| Q7–Q9 覆盖与效用的错位 | 未覆盖比例高但效用低的候选帧数，以及反过来的情况 |
| Q10–Q11 是否值得训练效用预测器 | 由判定决定 |
| Q12 与 NBV / active mapping 的边界 | 见 `docs/measurement_planning_related_work.md` |

可预测性按单个特征的平均 |Spearman| 标记：大于 0.6 为 STRONG，0.3–0.6 为 MODERATE，小于 0.3 为 WEAK。这只是描述，不用于判定。

## 7. 完整性

- K=0 复核：误差不超过 1e-9 m，NEAR 掩码相同。
- 读取审计：每个可部署规则的读取全部合法，且正好有 K 次候选深度读取。
- 教师路径的读取用途只有三种：`ORACLE_TEACHER_ONLY`、`QUERY_CAMERA_TEACHER_ONLY`、`SEALED_QUERY_EVALUATION`。
- 分析在同一份原始数据上运行两次，结果必须逐字节相同。
- 原始表格按 sha256 登记。
- `tests/test_measurement_planning_oracle_v1.py`（41 项）覆盖以下检查：
    - 候选深度在请求前隐藏；
    - 查询帧与另一个角色的帧不可达；
    - 预算计数准确；
    - 每个规则正好请求 k 次且读取合法；
    - 选择规则拒绝教师对象；
    - 未请求的候选深度不影响选择；
    - 随机种子固定；
    - 子集并集与冻结多帧路径逐位相同；
    - 教师不把查询帧融合进状态；
    - 分析结果确定；
    - 受保护场景与重复场景会被拒绝。

## 8. 不做的事

- 不训练任何选择器或预测器；
- 不打开受保护的保留集；
- 不下载新数据；
- 不推送 GitHub；
- 不删除、不覆盖任何旧实验的输出。

# EXP-3D-RGBD-DYNAMIC-WRITE-V1 协议（Core B 第一轮）

起草：Claude（经用户授权持续推进），2026-09-29。本协议在任何写入规则的正式训练之前冻结。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 背景

- **Core B 的前提已满足：** V7 的 RGB-D carrier 在 FRESH-V2 上 `QUALIFIED`，按项目规则可以开启 Core B 的下一阶段（动态 TTT 写入）。
- **以往结果：** 2026-09-27 的写入实验都建立在一个未合格、只用 RGB 的小 carrier 上：
  - 写入收益约为 0，有益的历史动作翻转为 0/3；
  - 写入训练阶段相对静态阶段反而退步。
- **探索性试点（只用 TRAIN）：** 冻结 V7 C1，在 TRAIN24 上训练项目原设计的写入规则 1000 步，在 48 个 TRAIN-X 场景上评价。结果存于 `audit/exploratory_coreb_pilot_train_only/`。

  | seed | 只看上下文 | OFF | 写入（训练后） | 写入相对 OFF |
  |---|---|---|---|---|
  | 20260928 | 0.278 | 0.306 | 0.267 | +0.039 |
  | 20260929 | 0.281 | 0.432 | 0.274 | +0.158 |

  - OFF 反而比只看上下文更差。原因是提升统计量里含有原始的有效视角计数，carrier 训练时最多只见过 3 个视角。
  - 写入相对 OFF 的收益，大部分可能是在修补这一偏移；写入相对只看上下文只好约 0.01。
  - 所以本轮额外设置两个对照：只看上下文，以及不需要学习的计数截断。
- **试点 2（只用 TRAIN，采用本协议的 stream 规则与代码，训练 1500 步）：** 结果存于 `audit/exploratory_coreb_pilot2_train_only/`。

  | seed | NO_STREAM | OFF | OFF_CLAMP3 | ALL | ALL_WRONG_SCENE |
  |---|---|---|---|---|---|
  | 20260928 | 0.278 | 0.298 | 0.260 | 0.254 | 0.257 |
  | 20260929 | 0.281 | 0.403 | 0.280 | 0.263 | 0.269 |

  - 写入相对 OFF：+0.044 / +0.140，CI >0。
  - 写入相对计数截断：+0.006 / +0.017，CI 跨零。
  - 写入相对别的场景的快速权重：+0.003 / +0.006，CI 跨零。
  - 试点预示本轮很可能落在分支 B。本协议照常执行，作为正式记录。
- **本轮问题：** 在冻结且已合格的静态 carrier 上，项目原设计的学习写入能否让流式到达的 RGB-D 观测带来真实、场景专属的收益？这个收益要超出三样东西：只缓存观测、静态状态、非学习的计数修正。

## 1. 数据与 episode

- **场景：** 与 V7 相同，训练用 TRAIN72，评价用 8 个 DEV 场景。FRESH-V1、FRESH-V2 不使用；受保护的 final holdout 不打开。
- **episode 的构成：**
  - warmup：一个冻结的 V2 上下文角色（A 或 B，各 3 个 RGB-D 视角）；
  - stream：随后 4 个 stream RGB-D 帧。
- **stream 帧规则：** 只看帧号。
  - 从场景的空闲帧（不属于任何上下文或 query 角色）中按帧号均匀取 4 个，A、B 两个角色共用。
  - 空闲帧不足 4 个的 episode 不合格，不缩短。
  - 每个 DEV 场景都有 9 个空闲帧，所以 16 个 DEV episode 全部合格；TRAIN 中合格的 episode 用于训练。
  - stream 帧按帧号从小到大依次到达。缓存里的帧号改用到达序号，因为 carrier 只用到先后次序；原始帧号记入元数据。
  - 改用这条规则的原因：试点采用的"只取 warmup 最后一帧之后"的规则，只能让 11/16 个 DEV episode 合格。这一修改发生在冻结之前，没有查看任何 DEV 结果。
- **数据边界：**
  - stream 帧的 RGB、测得深度与相机是测试时输入，与上下文同等对待；
  - 所有 DEV 状态封存之前，深度只按 CONTEXT_ONLY_RGBD（上下文帧）或 STREAM_RGBD（stream 帧）读取；
  - query 帧不可达。
- **bounds：** V7 C1 规则，只用 warmup 上下文计算。

## 2. 模型与训练

- **慢 carrier：** V7 C1 每个 seed 已封存的 DEV 选择 checkpoint（Core A），全程冻结，从不更新。
- **写入规则：** 项目原设计的 `DirectWriteRule`，使用 WriteConfig 默认值（学习率 0.01，单次更新范数上限 0.05），约 500 个参数。
  - 初始化：`torch.manual_seed(seed)`。
  - 初始规则单独保存，作为 ALL_UNTRAINED 参与评价。
- **训练：**
  - 每个 seed 在合格的 TRAIN72 episode 上训练 3000 步；
  - 每个 stream 到达都执行 ALL；
  - loss 是冻结的 V2 C1 loss，在 stream 之后，于一个 TRAIN primary query 的 1024 条随机射线上计算；
  - Adam，学习率 1e-3，梯度裁剪 1.0。
- **不做 checkpoint 选择：** 评价最后一步，训练期间不对 DEV 做任何前向。

## 3. 策略（每个 episode 固定）

| 策略 | 含义 |
|---|---|
| NO_STREAM | 只用 warmup 的状态（静态 Core A 状态） |
| OFF | 缓存全部 stream 帧后重建，不写入 |
| OFF_CLAMP3 | 同 OFF，但把提升统计量里的视角计数截断为 3（非学习对照） |
| FUSE / COMPLETE | 同 OFF，并把每次到达的写入提议提交到 fuse 或 complete 矩阵 |
| ALL | 同 OFF，并提交到两个矩阵（**PRIMARY_METHOD**） |
| ALL_UNTRAINED | 同 ALL，写入规则未训练 |
| ALL_WRONG_SCENE（对照） | 本场景的缓存，配上另一个 DEV 场景 ALL episode 的快速权重 |

## 4. 判定

- **聚合方式：** 每个 seed 内，先对 query 取平均，再对角色取平均；场景内各 seed 等权。
- **统计：** 10,000 次 scene paired bootstrap（seed 20260928），与冻结的 V2 估计器相同。
- **gate：** 采用冻结的 V2 主对比检查：
  - 均值 >0、CI 下界 >0、≥75% 的场景不差；
  - 每个 LOSO 均值 >0；
  - 正向贡献最大的单个场景占比 ≤0.5；
  - 封存状态完整。
  - 状态分为 HARMFUL / SUPPORTED / PARTIAL / NOT_ESTABLISHED。
- **各项判定：**
  - WRITE_STATUS（主对比）：WRITE_GAIN = AbsRel(OFF) − AbsRel(ALL)。
  - STREAM_STATUS：STREAM_WRITE_GAIN = AbsRel(NO_STREAM) − AbsRel(ALL)。
  - BEYOND_CLAMP_STATUS：BEYOND_CLAMP_GAIN = AbsRel(OFF_CLAMP3) − AbsRel(ALL)。
  - WRITE_SPECIFICITY_STATUS：AbsRel(ALL_WRONG_SCENE) − AbsRel(ALL) 的 CI 下界 >0。
- **描述性结果：**
  - 缓存本身的效果（NO_STREAM − OFF）；
  - 计数截断的效果；
  - FUSE、COMPLETE 各自的收益；
  - 训练的作用（ALL_UNTRAINED − ALL）；
  - 各 seed 的结果；
  - 快速权重范数、delta1、训练曲线。
- **分支（预先写定）：**
  - A：WRITE、STREAM、BEYOND_CLAMP 都为 SUPPORTED，且 WRITE_SPECIFICITY 为 SUPPORTED。学到的写入带来场景专属的、超出缓存、静态状态和计数修正的收益。下一步需要一个新的独立队列做 Core B 资格验证，并学习何时写入。
  - B：WRITE_STATUS 为 SUPPORTED，但不满足 A。写入主要在修补视角数量偏移。下一步用可变视角数重新训练静态 carrier（需要重新做静态资格验证），再检验写入。
  - C：其余情况。冻结的写入设计在合格的 carrier 上没有稳定收益。下一步重新设计写入，例如利用 stream 帧测得深度的写入统计，或基于梯度的测试时训练。

## 5. 禁止项

- 更新慢 carrier；
- 用 DEV 选择写入规则 checkpoint；
- 根据 DEV 结果修改策略、规则、步数或 stream 长度；
- 使用 GPU；使用 FRESH-V1 或 FRESH-V2；打开 final holdout。

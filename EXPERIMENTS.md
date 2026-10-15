# 实验注册表

## EXP-3D-READOUT-METHOD-V1（2026-10-15，hit GPU 4 + CPU，可靠几何读出：V2 必要性、TTT 必要性与新机制的前提检验）

- [README](docs/experiments/EXP-3D-READOUT-METHOD-V1/README.md)、[协议](docs/experiments/EXP-3D-READOUT-METHOD-V1/PROTOCOL.md)。2026-10-15 00:46:10 锁定；三个事后修正案各自在运行前锁定，只作探索性证据。
- **设计：** 四个留一队列折（SYSTEM3、VAL、CAJU、SYSTEM4，合计 350 个场所，都是开发数据）。
  - Phase 1：Core B、64 箱表、5 种只用训练集拟合的简单标定、逐像素 MLP 与 V2 配方。
  - Phase 2：只用几何、加前馈渲染、加 TTT 渲染三个同读出系统，另做交换检验与分阶段计时。
- **结果：**
  - Gate A：`V2_METHOD_NECESSITY = NOT_ESTABLISHED`。B5 − B7 = +0.0092 [+0.0029, +0.0147]，但 SYSTEM4 的未覆盖像素上 B5 更好。Core B → V2 的 0.036 中，零命中常数占 31%，简单标定占 36%，V2 特有的占 25%。
  - Gate B：`TTT_ADDED_VALUE = NOT_ESTABLISHED`（+0.0010 [−0.0026, +0.0038]）。修正案 1 在 KF 表 + V2 结构中为 +0.0064 [+0.0036, +0.0094]，SUPPORTED（探索性），几乎全在已覆盖像素；TTT 占每场景 16.6 s 中的 97.8%。
  - 修正案 2、3：候选 B 的神谕上界 +0.0008，候选 A 的自由空间下界 +0.0003，都是 NOT_SUPPORTED。`CVPR_METHOD_NOVELTY = INSUFFICIENT`。
- **状态：** 结束；结论 No-Go。决策报告见 Claude Docs 文档《可靠几何读出研究决策报告》。

## EXP-3D-CAJU3D-READOUT-LOCO-V1（2026-10-14，hit GPU 4，学到的读出的留一队列稳健性分析）

- [README](docs/experiments/EXP-3D-CAJU3D-READOUT-LOCO-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-READOUT-LOCO-V1/PROTOCOL.md)。2026-10-14 23:31:35 锁定；描述性分析，结果一律报告。
- **设计：** 四个留出折（CONF_SYSTEM3、CONF_VAL、CAJU、ARKIT_SYSTEM4），每折按 FUSE_V2 的配方用其余全部队列训练 3 个种子；FIT 两个队列只作训练。
- **结果：**
  - 合计 350 个场所：0.1395 → 0.1033，+0.0362 [+0.0230, +0.0524]，278/0/72，SUPPORTED。
  - 各折中，SYSTEM3、VAL、CAJU 为 SUPPORTED，SYSTEM4 为 PARTIAL（头部占比 0.52）。
- **状态：** 结束。

## EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1（2026-10-14，hit GPU 4 + CPU，全新保留队列上的读出检验）

- [README](docs/experiments/EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1/PROTOCOL.md)。
  - 计划于 2026-10-14 22:39:36 锁定（sha256 `65dbce23…`），早于任何 SYSTEM4 数据准备；
  - V2 代码于 22:43:21 锁定；评估代码于 22:45:40 锁定。
- **数据：** ARKIT_SYSTEM4（52 个场所、104 个片段、208 个查询），按 FULL-SYSTEM-V3 的流程跑 Core B。
- **读出：**
  - CORE_B；
  - KF_BASE（冻结表）；
  - FUSE_V1（FUSENET-V1 冻结的 BASE 网络）；
  - FUSE_V2：已覆盖像素用 KF 表，只学未覆盖修正，主干膨胀率到 16，训练数据为除 SYSTEM4 外的 458 个场所，3 个种子。
- **结果：**
  - P1 = CORE_B − FUSE_V2：+0.0539 [+0.0186, +0.1151]，47/0/5，PARTIAL（头部占比 0.52）。
  - P2 = CORE_B − FUSE_V1：+0.0502 [+0.0157, +0.1110]，46/0/6，PARTIAL（头部占比 0.56）。
  - CORE_B − KF_BASE：+0.0035，SUPPORTED。
  - 分析两次逐字节相同。
- **状态：** 结束（LEARNED_READOUT_PARTIAL）。

## EXP-3D-RGBD-ARKIT-SYSTEM4-DATA-V1（2026-10-14，hit，ARKitScenes 第四个完整系统队列）

- **来源：** `scripts/prepare_rgbd_arkit_system4_cohort.py`。EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1 的批次（训练折种子顺序第 1000–1359 位）在第 160 个合格扫描（批内第 268 位）处停止准备。本队列按批内顺序准备其余 91 个扫描（第 269–359 位）：它们已下载、相机元数据已锁定，但从未读取过任何图像或深度。
- **过程：** 用不变的基础规则和 SYSTEM3 的候选锁（其源文件哈希已核对）准备全部 91 个，52 个合格、39 个不合格。不下载数据，不运行模型，不计算分数。计划在 EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1 中先行锁定。
- **许可与存放：** 只用于非商业研究，媒体在 `zonghan_o_mirror/_data_arkitscenes`，不进 git。

## EXP-3D-CAJU3D-FUSENET-REPL-V1（2026-10-14，hit CPU，冻结的学习式读出的复现）

- [README](docs/experiments/EXP-3D-CAJU3D-FUSENET-REPL-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-FUSENET-REPL-V1/PROTOCOL.md)。2026-10-14 22:34:28 锁定，在任何打分之前。
- **数据：** CAJU-V1 的 58 个场所（100 步 TTT 回放与几何封存）；FUSENET-V1 的三个 BASE 网络按哈希核对，不重新训练。这些场所在 H-JEPA 的训练集里，所以不评估 JEPA 臂。
- **结果：**
  - P = CORE_B − FUSE_BASE：+0.01973 [+0.00716, +0.03795]，48/0/10，SUPPORTED（REPLICATED）。
  - 未覆盖像素的 AbsRel 从 0.288 降到 0.228。
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-CAJU3D-FUSENET-V1（2026-10-14，hit GPU 4 + CPU，CAJU-3D 第四步：学到的按像素读出）

- [README](docs/experiments/EXP-3D-CAJU3D-FUSENET-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-FUSENET-V1/PROTOCOL.md)。2026-10-14 22:20:46 锁定，sha256 `fda3cf90…`。
- **数据：** 与 KFUSE-V1 相同。SYSTEM + SYSTEM2 分为 128 个训练场所和 32 个验证场所；确认集为 SYSTEM3（160）和 VAL（80）。
- **模型：** FuseNet 为 5 层膨胀卷积，已覆盖像素输出卡尔曼式权重，未覆盖像素输出 R7 的有界修正。BASE 只用 8 通道基础图像，JEPA 另加 H-JEPA 查询格的 ẑ、log P 和命中。每个臂 3 个种子，30 个 epoch，按验证集选 epoch。
- **结果：**
  - P1 = FUSE_BASE − FUSE_JEPA：SYSTEM3 −0.00098（80/0/80），VAL +0.00186（34/0/46），都不成立。
  - CORE_B − FUSE_BASE：SYSTEM3 +0.0426（PARTIAL），VAL +0.0176（SUPPORTED）。
  - 未覆盖修正贡献了几乎全部收益。
  - 8 个测试通过；分析两次逐字节相同。
- **状态：** 结束（JEPA_FUSION_NOT_ESTABLISHED）。

## EXP-3D-CAJU3D-KFUSE-REPL-V1（2026-10-14，hit CPU，冻结的卡尔曼式读出融合的复现）

- [README](docs/experiments/EXP-3D-CAJU3D-KFUSE-REPL-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-KFUSE-REPL-V1/PROTOCOL.md)。2026-10-14 22:12:05 锁定，在任何打分之前。
- **数据：** CAJU-V1 的 CAJU_PILOT、CAJU_SEL、CAJU_EVAL（58 个场所、232 个查询），使用其 100 步 TTT 回放与几何封存；KFUSE-V1 的融合表（`15100550…`）不变。这些场所在 H-JEPA 的训练集里，所以不评估 JEPA 臂。
- **结果：**
  - P = CORE_B − KF_BASE：+0.00272 [+0.00145, +0.00417]，47/0/11，SUPPORTED（REPLICATED）。
  - GEO_U − KF_BASE：+0.00046，40/0/18，PARTIAL。
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-CAJU3D-KFUSE-V1（2026-10-14，hit GPU 4 + CPU，CAJU-3D 第三步：卡尔曼式读出融合）

- [README](docs/experiments/EXP-3D-CAJU3D-KFUSE-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-KFUSE-V1/PROTOCOL.md)。2026-10-14 21:56:35 锁定，sha256 `b9e08b44…`。设计前在第一、二步的开发集上做过探索（`exploration/`）。
- **数据：** 拟合集为 ARKIT_SYSTEM + ARKIT_SYSTEM2 的全部片段（160 个场所）；确认集为 SYSTEM3（160 个场所，P1）和 VAL（80 个场所，P2）。四个队列互不重叠，也不与 H-JEPA 训练数据重叠。复用 Core B 的 TTT 封存和几何封存；H-JEPA 特征只用到达帧和查询相机。
- **臂：** CORE_B、GEO_U（统一权重）、KF_DIS、KF_BASE（分歧 × α × near7）、KF_JEPA（分歧 × JEPA log P × near7）、KF_FULL。每箱一个权重，在拟合集上按场所加权 AbsRel 选出后冻结（`15100550…`）。
- **结果：**
  - P1 = KF_BASE − KF_JEPA：SYSTEM3 −0.00005 [−0.00013, +0.00003]，61/0/99；P2（VAL）−0.00007，33/0/47。都不成立。
  - Core B − KF_BASE：SYSTEM3 +0.00308（SUPPORTED），VAL +0.00267（SUPPORTED）。
  - GEO_U − KF_BASE：+0.00055 / +0.00066（PARTIAL）。
  - 8 个测试通过；分析两次逐字节相同。
- **状态：** 结束（JEPA_KALMAN_NOT_ESTABLISHED）。

## EXP-3D-CAJU3D-JEPAGATE-V1（2026-10-14，hit GPU 4 + CPU，CAJU-3D 第二步：JEPA 潜空间创新能否挑出更值得排除的测量）

- [README](docs/experiments/EXP-3D-CAJU3D-JEPAGATE-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-JEPAGATE-V1/PROTOCOL.md)。2026-10-14 20:37:12 锁定，sha256 `82d86a50…`。H-JEPA 在锁定前训练并冻结。
- **H-JEPA：** 源视角潜特征按测量深度几何对齐到目标视角（z-buffer），预测 ẑ 与 log P，用高斯 NLL 训练。数据为 ARKit TRAIN 182 个场所，3 万步；监控集上 log 误差与 log P 的相关为 0.78。
- **臂：** 在 Core B 的 TTT 深度项中剔除每个片段 10% 的有效测量像素，挑法为 JEPA（留一归一化潜空间创新）、DEPTH（跨视角深度不一致）、INC（入射角/跳变）、RANDOM。开发集与第一步相同（24 个场所、48 个片段）。
- **GO 规则：** GATE_JEPA 的开发集均值同时低于 CORE_B 和 GATE_DEPTH。
- **结果：**
  - Core B 0.09902，RANDOM 0.09904，DEPTH 0.09910，INC 0.09925，JEPA 0.09929，所以停止。
  - JEPA 对 Core B：−0.00027 [−0.00040, −0.00016]，3/0/21。
  - 排练中 NONE 与封存的 Core B 一致；13 个测试通过；剔除集合的复现差值为 0。
- **状态：** 结束（STOP_AT_DEV），确认队列没有读取。

## EXP-3D-CAJU3D-ENGINE-V1（2026-10-14，hit GPU 4 + CPU，CAJU-3D 第一步：不含学习模块的 TTT 引擎）

- [README](docs/experiments/EXP-3D-CAJU3D-ENGINE-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU3D-ENGINE-V1/PROTOCOL.md)。2026-10-14 18:09:32 锁定，sha256 `e92c77ea…`。
- **数据：** 开发集从 ARKIT_SYSTEM 和 ARKIT_SYSTEM2 各按种子选 12 个扫描，共 24 个场所、48 个片段。确认队列 CONF_SYSTEM3（160 个场所）和 CONF_VAL（80 个场所）没有使用。
- **臂：** 两个底座 ZS（发布版权重）和 JN1，各配 RAW / NORM / NORM_DEC / NORM_DEC_INC，都从同一个 CACHE 出发跑 100 步，用 Core B 读出。复用已封存的 Core B、CACHE 和 R7。
- **结果：**
  - 最佳候选 ZS_NORM 对 Core B：−0.0011 [−0.0021, −0.0003]，7/0/17，所以停止。
  - JN1_NORM 对 JN1_RAW：+0.0016 [−0.0013, +0.0061]，10/0/14。
  - 新算的 CACHE 与封存逐位一致（96 个查询）；10 个测试通过。
- **状态：** 结束（STOP_AT_DEV）。

## EXP-3D-CAJU-HEADROOM-V1（2026-10-14，hit GPU 4 + CPU，CAJU-V1 Gate 0：Core B 更新动作的反事实 headroom）

- [README](docs/experiments/EXP-3D-CAJU-HEADROOM-V1/README.md)、[协议](docs/experiments/EXP-3D-CAJU-HEADROOM-V1/PROTOCOL.md)。2026-10-14 01:07:17 锁定，sha256 `f82179bf…`。修订 1（01:31:37，正式回放之前）只改了引擎检查的放行依据：用分布比较代替单像素最大差。引擎和判定都没变。
- **数据：** ARKit 训练折中从未跑过 Core B 的场所。SEL 16 个场所用来选固定动作，EVAL 34 个场所用来测 headroom，DEV 8 个场所做工程试点，其余 132 个场所不动。
- **动作：** 都从同一个 CACHE 状态出发：OFF、SHORT（同一条 TTT 轨迹的第 10 步）、FULL（第 100 步）。三者都用冻结的 Core B 读出。
- **主检验：** 在 EVAL 场所上，交叉查询 oracle 减最佳固定动作。要求均值 ≥ 0.001、CI 下界 > 0、LOSO 全部 > 0、top-1 ≤ 0.5。
- **结果：**
  - 固定动作为 FULL。EVAL 的 AbsRel：OFF 0.1112，SHORT 0.1103，FULL 0.1082。
  - 主检验 +0.00076 [+0.00005, +0.00191]，9/20/5，top-1 0.57：NOT_ESTABLISHED。
  - 事后逐片段 oracle +0.00123；交叉角色 +0.00115；最省充分策略省 33%，低于 50%。
  - 测试 17 个通过；分析两次逐字节相同；完整性 PASS_WITH_AMENDMENT_1。
- **状态：** 结束（`NO_USEFUL_DYNAMIC_ACTION_SPACE`，`GATE1_ALLOWED=false`）。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1（2026-10-10，hit GPU + CPU，更多真实训练数据能否改进逆向 JEPA 的补全）

- [协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1/PROTOCOL.md)。2026-10-10 01:58:03 锁定，sha256 `35f85731…`；设计受 NS 结果启发，锁定早于本实验的任何训练与评估。
- **训练：** 只改一个因素，JO 配方（重叠感知、上限 3.0）的训练数据从 182 个扫描增加到 502 个（与 NS 相同）；JOL1–JOL3。
- **评估：** ARKIT160_OVERLAP 的 VAL80 部分（第 6、7 帧），CTX 臂；复用 OVERLAP-V1 的 arms 与 JO 封存。
- **主检验（FARH）：** P1 = JO − JOL（更多数据是否改进补全）；P2 = H3 − JOL。
- **结果：**
  - P1 = JO − JOL（FARH，VAL80）：−0.0044，42/0/33，NOT_ESTABLISHED。
  - P2 = H3 − JOL：+0.1125，42/0/33，PARTIAL。
  - S2（对比 G1N）n.s.；复现差值 0；分析两次逐字节相同。
- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1/README.md)。
- **状态：** 结束。

## EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1（2026-10-10，hit CPU，Core B 中 TTT 与 R7 的集成权重）

- [README](docs/experiments/EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1/PROTOCOL.md)。2026-10-10 01:51:05 锁定，sha256 `178a022a…`；权重选择只读开发集真值，记录哈希后才读确认集。
- **方法：** GEO_w = R7^(1−w) × ZS_TTT_F7^w，w ∈ {0.5, …, 1.0}，在 ARKIT_SYSTEM + ARKIT_SYSTEM2 上选择，在 ARKIT_SYSTEM3 与 ARKIT_VAL 上确认；不运行模型。
- **结果：**
  - w* = 0.9。
  - P1 = CORE_B − GEO_0.9（SYSTEM3）：+0.0025，106/0/54，PARTIAL。
  - P2（VAL）：+0.0019，52/0/28，PARTIAL。
  - 对 R7：GEO_0.9 130/0/30，TTT 单独 126/0/34。
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1（2026-10-10，hit GPU + CPU，用逆向 JEPA 学到的有选择弃权改进 Core B）

- [README](docs/experiments/EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1/PROTOCOL.md)。2026-10-10 01:37:36 锁定，sha256 `091f0b96…`；候选选择只读开发集真值，01:40:08 记录哈希后才读确认集。
- **方法：** 三种候选（MASK、TRIPLE、PAIR），只组合已有状态。在 ARKIT_SYSTEM + ARKIT_SYSTEM2 上选择（JN1 的前馈由本实验评估），在 ARKIT_SYSTEM3 与 ARKIT_VAL 上确认。
- **结果：**
  - 选出 PAIR。
  - P1 = CORE_B − PAIR（SYSTEM3）：+0.0029，102/0/58，PARTIAL。
  - P2（VAL）：+0.0019，55/0/25，PARTIAL。
  - S6（对比 TTT 单独）：+0.0004，76/0/84，n.s.。
  - 复现差值 0；分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1（2026-10-10，hit CPU，只调高读出阈值能否得到逆向 JEPA 的前馈改进）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1/PROTOCOL.md)。2026-10-10 01:28:14 锁定，sha256 `644bad0d…`；阈值选择只读开发集真值，01:28:28 记录哈希后才读确认集。
- **方法：** 不运行模型，复用已有封存。阈值网格 0.5–0.99，在开发集（ARKIT_SYSTEM + ARKIT_SYSTEM2）上分别为前馈与 Core B 选最优阈值，在 ARKIT_SYSTEM3 与 ARKIT_VAL 上确认。
- **结果：**
  - 开发集上读出随阈值升高单调变差，τ_C* = τ_T* = 0.5。
  - P1 = ZS_CACHE(τ_C*) − JNS_CACHE（SYSTEM3）：+0.0041，110/0/50，PARTIAL；VAL 上 +0.0041，61/0/19，SUPPORTED。
  - P2（Core B 换阈值）：τ_T* = 0.5，NOT_ESTABLISHED。
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1（2026-10-09，hit GPU + CPU，更多真实训练数据能否放大逆向 JEPA 的前馈改进）

- [协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1/PROTOCOL.md)。2026-10-09 23:18:31 锁定，sha256 `d9ed9d50…`；设计受 NF 结果启发，锁定早于本实验的任何训练与评估。
- **训练：** 只改一个因素，训练数据从 182 个扫描增加到 502 个（加入实验已结束的 CONFIRM、SELECT、SYSTEM、SYSTEM2 四个队列，无需新下载）；JL1–JL3，其余设置与 JN 相同。
- **评估：** ARKIT_SYSTEM3 与 ARKIT_VAL，只跑 CACHE 臂；复用 V3、NF、ENSEMBLES-ARKIT、OBJECTIVE-TTT 的封存。
- **主检验（SYSTEM3）：** P1 = JNS_CACHE − JLS_CACHE（更多数据是否有帮助）；P2 = ZS_CACHE − JLS_CACHE。
- **结果：**
  - P1 = JNS_CACHE − JLS_CACHE（SYSTEM3）：−0.0000，82/0/78，NOT_ESTABLISHED；VAL 上 −0.0002，44/0/36。
  - P2 = ZS_CACHE − JLS_CACHE（SYSTEM3）：+0.0041，111/0/49，PARTIAL；VAL 上 +0.0039，58/0/22，PARTIAL。
  - S4（对比 Core B）+0.0002，51/0/109，n.s.；三个种子各自 PARTIAL。
  - 12 项复现检查差值为 0；分析两次逐字节相同。
- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1/README.md)。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1（2026-10-09，hit GPU + CPU，只奖励精度的逆向 JEPA 能否改进前馈状态）

- [协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/PROTOCOL.md)。2026-10-09 19:23:38 锁定，sha256 `7ff1827a…`；设计受 OBJECTIVE-TTT 的 S3 启发，锁定时 V3 尚未读取 ARKIT_SYSTEM3 的任何真值。
- **数据：** ARKIT_SYSTEM3（V3 的队列，160 个扫描、640 个查询），复用 V3 的几何封存。
- **模型：** ZS（发布权重）；JN1、JK1（OBJECTIVE-TTT 的检查点）；JN2、JN3（新训练，种子 20261009、20261010）。只评估 CACHE 臂，上限 0.1。
- **主检验：** P1 = ZS_CACHE − JN1_CACHE（ALL）；P2 = ZS_CACHE − JNS_CACHE（ALL，三个种子平均）。
- **结果：**
  - P1 = ZS_CACHE − JN1_CACHE：+0.0036，109/0/51，PARTIAL。
  - P2 = ZS_CACHE − JNS_CACHE：+0.0041，110/0/50，PARTIAL。
  - S4（NEAR7）+0.0046，110/0/50；S1（对 R7）+0.0058，94/0/66；S3（发布权重对 R7）+0.0022，79/0/81，n.s.；S8、S9（JN2、JN3）+0.0045、+0.0042；JK1 n.s.。
  - 分析两次逐字节相同。事后探索见 `exploratory_posthoc/`（与 Core B 均值打平，Core B 在 113/160 个场景更好）。
- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/README.md)。
- **状态：** 结束。

## EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3（2026-10-09，hit CPU + GPU，Hypersim 小载体 G1 的选择性集成能否改进 Core B）

- [协议](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/PROTOCOL.md)。计划锁写于 2026-10-09 14:50:58，早于任何新数据下载；正式锁定 15:08:10，sha256 `0a33ebdf…`。
- **来源：** FULL-SYSTEM-V2 的次要分析 S5（G1 接法 +0.0042，21/43/16）；同一接法在 V1 上不成立。
- **方法：** 在被选中的 episode（S ≥ 1.2）中，把 Core B 在 FAR7 的基底换成 sqrt(R7 × 7 帧 G1)。
- **数据：** 全新队列 ARKIT_SYSTEM3，160 个扫描（为提高功效，比以往多一倍）；320 个 episode 中 125 个被选中。
- **主检验：** P1 = CORE_B − SYS_G1（ALL）；P2 = R7 − SYS_G1（ALL）。
- **结果：**
  - P1 = CORE_B − SYS_G1：+0.0010，38/99/23，NOT_ESTABLISHED。
  - P2 = R7 − SYS_G1：+0.0071，139/0/21，SUPPORTED。
  - S1（FAR7）+0.0062，36/96/25，SUPPORTED；S2（Core B 复现）+0.0061，149/0/11，SUPPORTED；S3（TTT 增益）+0.0064，130/0/30；S5（GA 接法）+0.0003，n.s.。
  - 分析两次逐字节相同。
- [README](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/README.md)。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1（2026-10-09，hit GPU + CPU，只奖励精度的逆向 JEPA 能否改进 TTT）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/PROTOCOL.md)。2026-10-09 14:41:45 锁定，sha256 `a826b21c…`；设计受 JBT 与 JC 结果启发，锁定早于本实验的任何训练与评估。
- **训练：** 只奖励已覆盖像素精度的目标，各一个种子，上限 0.1。
  - JN1：OVERLAP-V1 的 episode。
  - JK1：上下文 + 4 个流式帧 → 第 14、15 帧。
- **评估：** ARKIT_VAL，CACHE 与 TTT；复用 ENSEMBLES-ARKIT、JBT、JC 的封存，复现差值为 0。
- **结果：**
  - P1 = ZS_TTT − JN_TTT：−0.0008，30/0/50，NOT_ESTABLISHED。
  - P2 = ZS_TTT − JK_TTT：−0.0038，24/0/56，HARMFUL。
  - S3 = ZS_CACHE − JN_CACHE：+0.0036，60/0/20，SUPPORTED（次要）。
  - 前馈覆盖率从 0.73 降到 0.55（JN）和 0.26（JK）。
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1（2026-10-09，hit，ARKitScenes 第三个完整系统队列）

- **来源：** `scripts/prepare_rgbd_arkit_system3_cohort.py`，训练折种子顺序第 1000–1359 位场馆（360 个）。与此前所有队列以及 VAL 都不相交。
- **过程：** 360 个扫描全部下载（33.8 GB）并锁定相机元数据（1 个失败），按顺序准备到 160 个合格（109 个不合格）；不运行模型，不计算分数。
- **许可与存放：** 只用于非商业研究，媒体在 `zonghan_o_mirror/_data_arkitscenes`，不进 git。

## EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2（2026-10-09，hit CPU + GPU，TTT 集成加上 7 帧小载体的选择性集成）

- [README](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2/README.md)、[协议](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2/PROTOCOL.md)。计划锁写于 2026-10-09 13:08:03，早于任何新数据下载；正式锁定 13:16:03，sha256 `b36f2657…`。
- **来源：** FULL-SYSTEM-V1 的事后探索（样本内，从 9 种变体中挑出）。
- **方法：** 在被选中的 episode（S ≥ 1.2）中，把 Core B 在 FAR7 的基底换成 sqrt(R7 × 7 帧 GA)。
- **数据：** 全新队列 ARKIT_SYSTEM2，取训练折种子顺序第 840–999 位。
- **主检验：** P1 = CORE_B − SYS（ALL）；P2 = R7 − SYS（ALL）。
- **结果：** P1 = CORE_B − SYS：−0.0022，16/42/22，NOT_ESTABLISHED；P2：NOT_ESTABLISHED。S2（Core B 复现）+0.0049，70/0/10，SUPPORTED；S5（G1 接法）+0.0042，21/43/16，SUPPORTED（次要）；S6（7 帧 GA 对比 R7，FAR7）HARMFUL。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1（2026-10-09，hit CPU + GPU，TTT 集成加上按深度范围选择的补全）

- [README](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1/PROTOCOL.md)。计划锁 2026-10-09 11:28:13，写在下载任何新数据之前；正式锁定 11:35:44，sha256 `8bfe2c18…`。
- **数据：** 全新队列 ARKIT_SYSTEM，80 个扫描（见下一条）。episode 为上下文 3 帧 + 流式帧 4 帧，查询第 14、15 帧；160 个 episode 中有 69 个被选中（S ≥ 1.2）。
- **主检验：**
  - P1 = CORE_B − FULL（ALL）：把选择后的补全接进 Core B 的 FAR7 区域，是否改进 Core B；
  - P2 = R7 − FULL（ALL）。
- **结果：** P1 = CORE_B − FULL：−0.0031，15/49/16，NOT_ESTABLISHED；P2 = R7 − FULL：+0.0015，NOT_ESTABLISHED。S2（Core B 复现）= R7 − CORE_B：+0.0047，71/0/9，SUPPORTED；S3（TTT 增益）+0.0071，61/0/19，SUPPORTED。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-ARKIT-SYSTEM-DATA-V1（2026-10-09，hit，ARKitScenes 完整系统队列）

- **来源：** `scripts/prepare_rgbd_arkit_system_cohort.py`，训练折种子顺序第 680–839 位场馆。与试点（0–359）、CONFIRM（360–519）、SELECT（520–679）以及 VAL 都不相交。
- **过程：** 160 个扫描全部下载并锁定相机元数据，按顺序准备到 80 个合格（75 个不合格）；不运行模型，不计算分数。
- **许可与存放：** 只用于非商业研究，媒体在 `zonghan_o_mirror/_data_arkitscenes`，不进 git。

## EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1（2026-10-09，hit CPU + GPU，深度范围选择规则能否推广到远处视角）

- [README](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1/PROTOCOL.md)。2026-10-09 11:10:24 锁定，sha256 `62fcff3c…`。
- **设计：** 规则和阈值不变（S ≥ 1.2）。选择器封存复制自 SELECTOR-V1（上下文相同）。查询改为同一批扫描的第 14、15 帧，这两帧此前从未被评估。
- **模型：** G1、GA（arms 封存，CPU）；JO、JA（GPU，只跑 CTX）。
- **主检验：** P1 = H3 − SEL_GAN（FARH）。
- **结果：** P1 = H3 − SEL_GAN（FARH）：+0.0577 [+0.0058, +0.1083]，24/17/12，SUPPORTED（FAR_SELECTOR_CONSISTENT）；SEL_G1N +0.0579（28/17/8）SUPPORTED；SEL_JO、SEL_JA 不成立；不选择时四种模型都不成立。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1（2026-10-09，hit CPU + GPU，用上下文深度范围决定何时使用学到的补全）

- [README](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1/PROTOCOL.md)。
    - 计划锁于 2026-10-09 10:42:42 写定，早于任何新数据下载。
    - 正式锁定于 10:51:00，sha256 `c6dc93e1…`。
- **来源：** EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 的事后探索（样本内）：上下文深度范围 S 预测学到的补全是否有用。
- **规则：** S ≥ 1.2 时用学到的补全，否则用调和插值。
- **数据：** 全新队列 ARKIT_SELECT（80 个扫描，见下条），查询为第 6、7 帧；160 个 episode 中有 70 个被选中。
- **主检验：** P1 = H3 − SEL_GAN（FARH）。
- **结果：** P1 = H3 − SEL_GAN（FARH）：+0.1115 [+0.0600, +0.1675]，32/31/9，SUPPORTED（SELECTOR_CONSISTENT）。SEL_G1N +0.1138（31/31/10）、SEL_JO +0.0731（29/31/12）也都是 SUPPORTED；不选择时三者都是 PARTIAL。修订 1（11:05:48）只修复打分代码中空 FAR 时的 NaN，修复前没有产生任何结果。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-ARKIT-SELECT-DATA-V1（2026-10-09，hit，ARKitScenes 选择器确认队列）

- **来源：** `scripts/prepare_rgbd_arkit_select_cohort.py`，取训练折种子顺序第 520–679 位场馆。
    - 第 0–359 位留给试点训练，第 360–519 位是 CONFIRM 队列；本队列与它们和 VAL 都不相交。
- **过程：** 160 个扫描全部下载并锁定相机元数据，按顺序准备到 80 个合格（62 个不合格）。不运行模型，不计算分数。
- **许可与存放：** 只用于非商业研究，媒体在 `zonghan_o_mirror/_data_arkitscenes`，不进 git。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1（2026-10-09，hit GPU + CPU，在 TTT 的尺度上限下训练逆向 JEPA）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1/PROTOCOL.md)。2026-10-09 06:03:48 锁定，sha256 `1b00184e…`；设计受 JBT 与 OVERLAP-V1 结果启发，锁定早于本实验的任何训练与评估。
- **训练：** OVERLAP-V1 的配方，尺度上限 0.1（JC1，一个种子）。
- **评估：** ARKIT_VAL，CACHE 与 TTT；复用发布权重（上限 0.1）的封存以及 JBT 中 JA、ZS3 的封存，复现差值为 0。
- **结果：** P1 = ZS_TTT − JC_TTT：−0.0070，24/0/56，HARMFUL；P2 = ZS_CACHE − JC_CACHE：−0.0517，5/0/75，HARMFUL；S4 = JA_TTT − JC_TTT：+0.0375，55/0/25；集成后与发布权重打平（0.116）。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1（2026-10-09，hit CPU，用 ARKit 做重叠感知训练的旧基座补全载体）

- [README](docs/experiments/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1/PROTOCOL.md)。冻结记录见 `freeze.json`（2026-10-09 02:24:02，sha256 `2cef537d…`），早于 OVERLAP-V1 的任何结果。第一次冻结有缺陷，未训练就被替换，保留在 `first_freeze_without_cap/`。
- **训练：** 冻结的 V17 L 代码与配方，在 CPU 上训练 3 个种子、24000 步。训练集是 151 个 ARKit TRAIN 扫描，训练查询为两个角色的上下文命中比例都 ≥ 0.05 的帧（GA）。
- **评估：** ARKIT160_OVERLAP，查询为第 6、7 帧。G1（Hypersim）与 GA 在同一个 arms 封存中评估。
- **主检验：** C1 = H3 − GAN（FARH）；C2 = G1N − GAN（FARH）。
- **结果：** C1 = H3 − GAN（FARH）：+0.1215，79/0/68，PARTIAL；C2 = G1N − GAN：+0.0138，NOT_ESTABLISHED；S1 = H3 − HGAN：+0.0923，100/0/47。FARH：H3 0.452，GAN 0.330，G1N 0.344。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1（2026-10-09，hit GPU + CPU，重叠感知的真实数据逆向 JEPA）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/PROTOCOL.md)。2026-10-09 01:19:13 锁定，sha256 `27d25dd4…`；锁定时 JEPA-ARKIT-V1 尚无任何结果。
- **训练：** JEPA-ARKIT-V1 配方，目标只取上下文命中比例 ≥ 0.05 的帧（JO1–JO3）。
- **评估：** ARKIT160_OVERLAP，查询为第 6、7 帧（从未评估或输入过）；147 个场景有 FARH。
- **结果：** O1 = H3 − JO（FARH）：+0.1101，80/0/67，PARTIAL（中位数 +0.005）；O2 = JA − JO：−0.0047，NOT_ESTABLISHED。FARH：H3 0.452，JO 0.341，JA 0.337，G1N 0.344，J 0.407。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1（2026-10-09，hit GPU + CPU，以真实数据 JEPA 为起点的 TTT）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1/PROTOCOL.md)。2026-10-09 01:19:01 锁定，sha256 `6441e982…`；修订 1（01:33:33，只改执行方式：为留出显存停掉一个分片后单独重跑）。
- **数据：** ARKIT_VAL（80 个扫描），episode 与 ENSEMBLES-ARKIT 相同；复用其几何封存和发布权重的 CACHE / TTT 封存（复现差值为 0）。
- **主检验：** Q1 = ZS_TTT − JA_TTT（ALL）；Q2 = ZS3_TTT − JA_TTT（ALL）。JA 只用事先指定的 JA1。
- **结果：** Q1：−0.0445，15/0/65，HARMFUL；Q2：−0.0226，30/0/50，HARMFUL。S4（JA 上的 TTT 增益）+0.1070，70/0/10；S6（上限 0.1 优于 3.0）+0.0219，70/0/10。分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1（2026-10-08/09，hit GPU + CPU，在 ARKit 真实数据上训练逆向 JEPA）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/PROTOCOL.md)。2026-10-08 23:42:00 锁定，sha256 `de61109d…`。修订 1 于 2026-10-09 00:03:50 记录，早于读取评估真值。
- **训练：** ARKit 训练折 TRAIN 182 个扫描，3 个种子，V1 配方（JA1–JA3）。
- **评估：** ARKIT160 = ARKIT_VAL80 + ARKIT_CONFIRM80。
- **结果：**
  - R1 = H3 − JA（FAR）：+0.2101，90/14/55，PARTIAL；R2 = J − JA：+0.2307，104/13/42，PARTIAL。
  - 修订 1：
    - X2 只换补洞常数：+0.3749，SUPPORTED；
    - X1 在校准基线之上加 JA：−0.0289，HARMFUL；
    - X6 只看有重叠的查询：+0.0269，NOT_ESTABLISHED。
  - 主分析与修订分析都运行两次，结果逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1（2026-10-08，hit CPU，归一化 G1 补全集成的真实数据确认）

- [README](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1/PROTOCOL.md)。2026-10-08 13:46:12 锁定，sha256 `b5b24308…`。
- **数据：** `outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1`，80 个扫描、160 个 episode。
- **主检验：** C1 = H3 − (H3 + G1N) / 2（FAR，V2 gate）。G1N 在不透明度 ≥ 0.5 处取 深度 / 不透明度，否则用 H3。
- **结果：** C1：+0.0642，48/19/12，SUPPORTED；分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1（2026-10-08，hit，ARKitScenes 确认队列）

- **来源：** `scripts/prepare_rgbd_arkit_confirm_cohort.py`。训练折种子顺序第 360–519 位场馆；前 360 位由试点保留。
- **过程：** 160 个扫描全部下载并锁定相机元数据，按顺序处理到 80 个合格（67 个不合格）；不运行模型，不计算分数。

## EXP-3D-RGBD-ENSEMBLES-ARKIT-V1（2026-10-08，hit GPU + CPU，两个集成在 ARKitScenes 上的检验）

- [README](docs/experiments/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1/PROTOCOL.md)。2026-10-08 12:08:56 锁定，sha256 `6947f14f…`。
- 数据见 `outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1`：80 个扫描、160 个 episode。
- **主检验（V2 gate）：**
  - A1 = H3 − HJG（FAR）；A2 = HG − HJG（FAR）；
  - B1 = R7 − √(R7 × TTT)（ALL）；B2 = √(R7 × CACHE) − √(R7 × TTT)（ALL）。
- **结果：**
  - A1：+0.4216，56/24，PARTIAL；A2：−0.0086，NOT_ESTABLISHED；
  - B1：+0.0049，71/9，SUPPORTED；B2：+0.0031，60/20，SUPPORTED；
  - 分析两次逐字节相同；另有事后偏差检查（`exploratory_posthoc/`）。
- **状态：** 结束。

## EXP-3D-RGBD-ARKIT-VAL-DATA-V1（2026-10-08，hit，ARKitScenes Validation 评估队列）

- **来源：** `scripts/prepare_rgbd_arkit_data.py --folds Validation`，加上 `scripts/prepare_rgbd_arkit_val_cohort.py`（按同一种子顺序扩展）。
- **过程：**
  - 前 70 个场馆 41 个合格；扩展部分处理到第 66 个场馆时凑满 80 个合格（27 个不合格）；
  - 帧与规则在读取媒体之前锁定（`candidate_lock*.json`）；
  - 不运行模型，不计算分数。
- **许可与存放：** 只用于非商业研究，媒体在 `zonghan_o_mirror/_data_arkitscenes`，不进 git。

## EXP-3D-RGBD-TTT-ENSEMBLE-V1（2026-10-08，hit GPU + CPU，TTT 状态与无模型重投影取平均）

- [README](docs/experiments/EXP-3D-RGBD-TTT-ENSEMBLE-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-TTT-ENSEMBLE-V1/PROTOCOL.md)。2026-10-08 06:31:24 锁定，sha256 `5154f900…`。假设来源：`exploratory_source/`（FRESH78 上的探索）。
- **设置：** 发布版 DepthSplat（ZS），CACHE 与 TTT（100 步，m = 1）；R7 = REPROJ7_HARMONIC；队列为 EVAL34，流帧同 Core B V8。
- **主检验（V2 gate，ALL）：**
  - T1 = R7 − √(R7 × TTT)；
  - T2 = √(R7 × CACHE) − √(R7 × TTT)。
- **结果：**
  - T1：+0.0290，71/12，SUPPORTED；
  - T2：+0.0185，80/3，SUPPORTED；
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1（2026-10-08，hit GPU + CPU，TTT 是否依赖学到的基座）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1/PROTOCOL.md)。2026-10-08 06:25:19 锁定，sha256 `cebe6e23…`。
- **NV（无网络高斯）：** 每个有效测量像素一个各向同性高斯，σ = c × 像素足迹，不透明度 0.95，颜色取像素 RGB。DEV 上选 c = 0.0625、m = 1。TTT 与 V1 的 ZS 完全相同。
- **主检验（V2 gate，ALL）：** P1 = NV_TTT − ZS_TTT，P2 = NV_CACHE − ZS_CACHE。
- **结果：**
  - P1：+0.0237，61/17，SUPPORTED；
  - P2：+0.0062，47/31，NOT_ESTABLISHED；
  - 分支 LEARNED_BASE_MATTERS；分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1（2026-10-08，hit CPU + GPU 推理，学到的补全与调和插值取平均）

- [README](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1/PROTOCOL.md)。2026-10-08 05:41:37 锁定，sha256 `d6248ece…`。假设来源：`exploratory_source/`（FRESH78 上的探索）。
- **读出（FAR）：**
  - H3：调和；
  - G1：3 种子平均；
  - J：V1 的逆向 JEPA，3 个冻结检查点，至少 2 个种子覆盖处取覆盖种子的平均，其余为 H3；
  - 组合：JG、HG、HJ、HJG（等权平均），以及几何平均版本。
- **主检验（V2 gate）：** E1 = H3 − HJG，E2 = HG − HJG（FAR）。
- **结果：**
  - E1：+0.1310，60/20，SUPPORTED；
  - E2：−0.0023，34/46，NOT_ESTABLISHED；
  - S1 = H3 − HG：+0.1332，65/15，SUPPORTED；S12 = H3 − HG（ALL）：SUPPORTED；
  - 分析两次逐字节相同。
- **状态：** 结束。

## EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1（2026-10-08，hit GPU + CPU，以 DepthSplat 为基座检验逆向 JEPA 与 TTT）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/README.md)、[协议与修订](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/PROTOCOL.md)。2026-10-08 01:16:18 锁定，sha256 `fa87f117…`。
- **构建器：**
  - ZS：发布权重，尺度上限 0.1；
  - ZS3：发布权重，上限 3.0；
  - J1–J3：逆向 JEPA。TRAIN72 + SEG-V1，4000 步，lr 5e-6，渲染 2 个其他帧的深度相对 L1 + 0.5 × RGB L1，上限 3.0。
  - 所有构建器都注入测得深度。
- **臂：** CTX（3 帧）、CACHE（3 帧上下文加 4 帧流帧）、TTT（只用于 ZS，100 步，m = 1，在 DEV 上选出）；无模型的 REPROJ_HARMONIC、REPROJ7_HARMONIC 与 G1。
- **主检验（V2 gate）：**
  - A1 = FAR：调和补全 − J；
  - B1 = ALL：CACHE − TTT；
  - B2 = ALL：REPROJ7 − TTT。
- **结果：**
  - A1：+0.0687，45/32，PARTIAL；
  - B1：+0.0298，73/5，SUPPORTED；
  - B2：+0.0567，54/24，PARTIAL；
  - 分支为 A_PARTIAL、B_MECHANISM_ONLY；分析两次逐字节相同。
- **状态：** 结束。修订 1（显存不足后重跑）、修订 2（先读 TTT 的真值）都只涉及执行。

## EXP-MEASUREMENT-PLANNING-ORACLE-V1（2026-10-06，hit CPU，预算化深度采集的 oracle 余量审计）

- [README](outputs/EXP-MEASUREMENT-PLANNING-ORACLE-V1/README.md)、[协议](outputs/EXP-MEASUREMENT-PLANNING-ORACLE-V1/PROTOCOL.md)、[预注册](outputs/EXP-MEASUREMENT-PLANNING-ORACLE-V1/preregistration.md)、[交接审计](docs/experiments/EXP-MEASUREMENT-PLANNING-ORACLE-V1/HANDOFF_AUDIT.md)、[相关工作](docs/measurement_planning_related_work.md)。
- **任务：**
  - episode = 记录 × 上下文角色；
  - 初始为 3 个 RGB-D 帧；
  - 候选是 9 个不属于任何角色的帧，RGB 与相机可见，深度只在请求后读取；
  - 预算 K ∈ {1, 2, 4}，查询是 primary_query。
- **判定：** 交叉查询 oracle 相对最强可部署规则通过 V2 gate，且余量比例 ≥ 0.20。
- **结果：** NOT_ESTABLISHED（K = 1、2、4 都不成立），POLICY_TRAINING_ALLOWED = false；最强规则是 LATEST；完整性 PASS。
- **状态：** 结束。方向停止，见 `negative_result.md`。

## EXP-3D-RGBD-DEPTHSPLAT-PROBE-V1（2026-10-05 至 10-06，hit GPU + CPU，以 DepthSplat 为基座的探索性检查）

- [README](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-PROBE-V1/README.md)。探索性，未预注册。使用 DEV（8 个场景）和 EVAL-V3/V4（83 个场景，以前已被使用）。
- **设置：**
  - 模型：DepthSplat base（re10k+dl3dv）@2dad25a7；
  - 输入：3 个 RGB-D 上下文，可注入实测深度；可再加每条记录 9 个未使用的帧作为只有 RGB 的视角；
  - 渲染深度按累积不透明度归一化；
  - 组合方式：FAR 中已覆盖的像素用 DepthSplat，其余用调和插值或 G1；
  - 另做了 TRAIN 数据上的深度微调，以及不经渲染的深度融合。
- **结果：** 见 STATUS 的 2026-10-06 条目。
- **状态：** 结束。不做预注册确认，因为排除相邻帧之后没有可以确认的效应。

## EXP-3D-RGBD-ARMS-CONFIRM-V33（2026-10-04，hit CPU，在新队列上确认三维上下文分支）

- [计划](outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33/PLAN_BEFORE_V33.md)（05:22 写定，sha256 `f4b5b846…`，此时 V32 尚无结果）。
- **设计：** 组 G1、U、P（第 24000 步，3 个 seed），队列 EVAL-V5、TRAIN-EXT、Replica；P1 = FAR(G1) − FAR(U)（EVAL-V5 ∪ TRAIN-EXT），P2 = 调和插值 − AHFILL8(U) 的标签；分支 A/B/C；只在 V32 的 P1 均值大于 0 时运行。
- **状态：** 05:30 冻结（预注册 sha256 `9ebd3e79…`）；06:36 按预先写定的条件未运行（V32 的 P1 均值 −0.0020）。

## EXP-3D-RGBD-UNET-CONTEXT-V32（2026-10-04，hit GPU 训练 + CPU 评价，覆盖整个体积的三维上下文）

- [计划](outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/PLAN_BEFORE_V32.md)（01:52 写定，sha256 `4bbdacfa…`）、[修订 1](outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/PLAN_AMENDMENT_1.md)（01:56，sha256 `960bf302…`，冒烟测试的设备与时机）。
- **动机：** carrier 的空间感受野只有 5³ 个体素，可能是补全不一致的结构性原因；加宽（V19）不改变感受野。
- **设计：** U = V17 L 加三维 U-Net 残差分支（32³→4³→32³，每级 32 通道，带全局项，输出层初值为零，新增 361,856 个参数），对照 V21 的 G1；P1 = FAR(G1) − FAR(U)（第 24000 步，EVAL-V3/V4），P2 = 调和插值 − AHFILL8(U) 的标签；分支 A/B/C。
- **状态：** 02:12 冻结（freeze.json sha256 `8df99840…`），03:21–05:16 在 GPU 上先做冒烟测试再训练，05:17–06:34 CPU 评价。
- **结果（分支 C）：** P1 FAR(G1) − FAR(U) −0.0020 [−0.0200, +0.0206]（36/0/44），NOT_ESTABLISHED；P2 +0.0329（45/3/35），NOT_DISTINGUISHABLE；第 6000 步 U 好 +0.0178（57/23），第 24000 步 EVAL-FAR、FRESH-V2 上 U 更差。

## EXP-3D-RGBD-PRETRAINED-FEATURES-V31（2026-10-04，hit GPU 训练 + CPU 评价，冻结的预训练视觉特征）

- [计划](outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/PLAN_BEFORE_V31.md)（01:08 写定，sha256 `82ff870d…`）、[冒烟测试](outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/audit/smoke_v31.md)。
- **动机：** V17–V30 在现有结构里的改动，都没有让 carrier 相对无模型基线有稳定的提升；瓶颈可能是只有 8 个从零学出的图像通道。
- **设计：** P = V17 L 加上冻结的 DINOv2 ViT-S/14 特征（384→24 投影，经初值为零的 48→64 旁路进入融合），对照 V21 的 G1；P1 = FAR(G1) − FAR(P)（第 24000 步，EVAL-V3/V4），P2 = 调和插值 − AHFILL8(P) 的标签；分支 A/B/C。
- **状态：** 01:45 冻结（freeze.json sha256 `a49ca180…`），01:45–03:20 GPU 训练，03:21–04:35 CPU 评价。
- **结果（分支 C）：** P1 FAR(G1) − FAR(P) +0.0041 [−0.0139, +0.0235]（36/0/44），NOT_ESTABLISHED；P2 +0.0347（47/3/33），NOT_DISTINGUISHABLE；EVAL-FAR、FRESH-V2 上 P 略差。

## EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30（2026-10-03，hit GPU + CPU，只有 RGB 的新帧只作监督）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30/PLAN_BEFORE_V30.md)（17:55 写定，sha256 `f3a0783c…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30/PROTOCOL.md)。
- **动机：** V29 中缓存没有深度的新帧有害（carrier 只在带深度的视图上训练过）；本轮不缓存它们，只用它们的 RGB 监督写入。
- **设计：** EVAL-V3/V4；P1 = NO_STREAM − RGB_SUP_100，P2 = CTX_TTT_100 − CTX_RGB_SUP_100；分支 A/B/C/D。
- **状态：** 18:23 冻结（预注册 sha256 `5b51493e…`），18:55–20:01 运行。
- **结果（分支 D，完整性 PASS）：** P1 NO_STREAM − RGB_SUP_100 −0.0796（20/0/63），HARMFUL；P2 CTX_TTT_100 − CTX_RGB_SUP_100 +0.0013（35/0/48），NOT_ESTABLISHED；封存前没有读取任何流帧深度。

## EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29（2026-10-03，hit GPU + CPU，只有 RGB 的新帧）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29/PLAN_BEFORE_V29.md)（14:09 写定，sha256 `d82283ee…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29/PROTOCOL.md)。
- **动机：** V24、V25、V27 说明新帧带深度时，无模型重投影就胜过 carrier；没有深度时，无模型方法用不上新帧，carrier 的写入可能有实际价值。
- **设计：** EVAL-V3/V4；P1 = NO_STREAM − TTT_RGB_100，P2 = REPROJ3_HARMONIC − HFILL8_3(TTT_RGB_100)；分支 A/B/C/D。
- **状态：** 14:28 冻结（预注册 sha256 `f3a1c9bb…`），16:51–17:52 运行。
- **结果（分支 D，完整性 PASS）：** P1 NO_STREAM − TTT_RGB_100 −0.0261（27/0/56），HARMFUL；P2 +0.0205（43/3/37），PARTIAL；只缓存没有深度的帧 −0.0386；封存前没有读取任何流帧深度。

## EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28（2026-10-03，hit GPU + CPU，跨数据集的流帧写入）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/PLAN_BEFORE_V28.md)（13:46 写定，sha256 `8b154c92…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/PROTOCOL.md)。
- **设计：** Replica 8 个场景，V17 L；P1 = OFF − TTT_100，P2 = REPROJ7_HARMONIC − HFILL8_7(TTT_100)；分支 A/B/C/D。
- **状态：** 14:06 冻结（预注册 sha256 `a1019f23…`），18:12–18:44 运行。
- **结果（分支 B，完整性 PASS）：** P1 OFF − TTT_100 +0.0250（7/0/1），SUPPORTED；P2 REPROJ7_HARMONIC − HFILL8_7(TTT_100) −0.1468（1/0/7），HARMFUL；carrier 0.445 与常数预测 0.411 相当，REPROJ7_HARMONIC 0.062。

## EXP-3D-RGBD-HYBRID-CONFIRM-V27（2026-10-03，hit CPU，两个新队列上的混合方法与 MAXFILL 规则）

- [计划](outputs/EXP-3D-RGBD-HYBRID-CONFIRM-V27/PLAN_BEFORE_V27.md)（13:25 写定，sha256 `d991381f…`）、[协议](outputs/EXP-3D-RGBD-HYBRID-CONFIRM-V27/PROTOCOL.md)。13:40 冻结（预注册 sha256 `f1535aa5…`），13:43 完成。
- **动机：** V24、V25 的混合方法只有 PARTIAL；探索性分析在同样的数据上找到 MAXFILL 规则，需要在新数据上检验。
- **设计：** TRAIN-EXT（V9）与 EVAL-V5（V7）合并 78 个场景；P1 = REPROJ7_HARMONIC − MAXFILL8_7(TTT_100)，P2 = REPROJ7_HARMONIC − HFILL8_7(TTT_100)。
- **结果（分支 D，完整性 PASS）：** P1 +0.0003（29/31/18），P2 +0.0013（36/19/23），都 NOT_ESTABLISHED；写入在 FAR7 上 +0.028（CI 不含 0）。

## EXP-3D-RGBD-FAR-FOCUS-V26（2026-10-03，hit GPU，训练光线集中在测量覆盖不到的像素上）

- [计划](outputs/EXP-3D-RGBD-FAR-FOCUS-V26/PLAN_BEFORE_V26.md)（13:04 写定，sha256 `0cce2455…`）、[修订 1](outputs/EXP-3D-RGBD-FAR-FOCUS-V26/PLAN_AMENDMENT_1.md)（13:13，sha256 `08e6e904…`，FAR 光线不放回抽取）。
- **动机：** V24、V25 说明 carrier 在实际可用的混合方法里只负责测量覆盖不到的像素，而那里的补全在场景之间不一致。
- **设计：** GF（V17 L 配方 + FAR 集中采样，GPU）对照 V21 的 G1；P1 = FAR(G1) − FAR(GF)，P2 = 调和插值 − AHFILL8(GF) 的标签。
- **状态：** 13:23 冻结（freeze.json sha256 `790b7e6e…`），16:50 GPU 训练完成（三个 seed），18:01 CPU 评价与分析完成。
- **结果（分支 C）：** P1 FAR(G1) − FAR(GF) +0.0198 [−0.0045, +0.0511]（42/0/38），NOT_ESTABLISHED；P2 调和插值 − AHFILL8(GF) +0.0348（49/3/31），NOT_DISTINGUISHABLE；EVAL_FAR 与 FRESH_V2 上方向相反。

## EXP-3D-RGBD-STREAM-FAR-V25（2026-10-03，hit CPU，远处 query 上的 Core B 状态与无模型基线）

- [计划](outputs/EXP-3D-RGBD-STREAM-FAR-V25/PLAN_BEFORE_V25.md)（11:52 写定，sha256 `f525b031…`）、[协议](outputs/EXP-3D-RGBD-STREAM-FAR-V25/PROTOCOL.md)。12:11 冻结（预注册 sha256 `06437de3…`），12:55 完成。
- **设计：** Core B V8 封存的状态在 EVAL-FAR 远处 query 上渲染（5,688 个视图）；与 V24 相同的基线与混合；P1 = REPROJ7_HARMONIC − HFILL8_7(TTT_100)，P2 = HFILL8_7(OFF) − HFILL8_7(TTT_100)。
- **结果（分支 D，完整性 PASS）：**
  - 7 帧 NEAR 占 87.1%；P1 +0.0145 [+0.0020, +0.0301]，35/14/30，PARTIAL；P2 +0.0004，NOT_ESTABLISHED；
  - carrier 内部写入 +0.0102（58/21）；归一化读出的混合相对无模型基线 +0.0202（40/14/25）。

## EXP-3D-RGBD-STREAM-BASELINES-V24（2026-10-03，hit CPU，Core B 与使用同样帧的无模型基线）

- [计划](outputs/EXP-3D-RGBD-STREAM-BASELINES-V24/PLAN_BEFORE_V24.md)（11:33 写定，sha256 `ff8ea7e0…`）、[协议](outputs/EXP-3D-RGBD-STREAM-BASELINES-V24/PROTOCOL.md)。11:46 冻结（预注册 sha256 `10ad8715…`），11:49 完成。
- **动机：** Core B 的写入一直只与 carrier 自己的其他策略比较；流帧也是测得的 RGB-D，同样可以重投影。
- **设计：** 重新计分 Core B V8 的封存预测；REPROJ7_HARMONIC 与混合 HFILL8_7(p)；P1 = HFILL8_7(OFF) − HFILL8_7(TTT_100)，P2 = REPROJ7_HARMONIC − HFILL8_7(TTT_100)。
- **准备：** 预演指向 Core B V8 自己的 CPU 预演目录；第一次预演在冻结阶段因取错常数的字段位置（在 V17 合同的 `eval` 之下）而停止，未计算任何东西，改正后通过；完整测试 1125 个通过。
- **结果（分支 D，完整性 PASS）：**
  - 7 帧 NEAR 占 93.8% 的有效像素；REPROJ7_HARMONIC 0.1746，优于 carrier TTT_100（0.1858，19/83 个场景 carrier 更好）；
  - P1 +0.0001，NOT_ESTABLISHED；P2 +0.0081 [+0.0000, +0.0189]，31/19/33，PARTIAL；
  - FAR7 像素上 carrier 平均优于调和插值（+0.060），但不一致。

## EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11（2026-10-03，hit GPU，Core B 第十一轮：逐帧在线写入）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11/PLAN_BEFORE_COREB_V11.md)（11:08 写定，sha256 `94a461e2…`，在 Core B V9、V10 冻结之后、得到其结果之前）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11/PROTOCOL.md)。
- **动机：** 到 Core B V8 为止的写入都在 4 个流帧全部到达之后一次性进行。
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景）；P1 = OFF − ONLINE_100，P2 = ½(OFF + TTT_100) − ONLINE_100，P3 = ONLINE_NEW_100 − ONLINE_100；分支 A/B/C。
- **状态：** 11:32 冻结（预注册 sha256 `b23f1cb9…`），20:13–22:03 运行。
- **结果（分支 B，完整性 PASS）：** P1 OFF − ONLINE_100 +0.0145（68/0/15），SUPPORTED；P2 +0.0038（52/0/31），NOT_ESTABLISHED；P3 −0.0027（41/0/42），NOT_ESTABLISHED；在线保住一次性写入增益的约 2/3。

## EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10（2026-10-03，hit GPU，Core B 第十轮：写入的增益是否依赖 carrier）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10/PLAN_BEFORE_COREB_V10.md)（10:46 写定，sha256 `e8081221…`，在 Core B V9 冻结之后、得到其结果之前）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10/PROTOCOL.md)。
- **动机：** Core B V4–V9 都只用 V17 L 一个 carrier。
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景）；V17 L、V18 L、V19 L、V20 L（宽度 × 训练数据的 2×2）各做 NO_STREAM、OFF、TTT_100 与 TTT_100_WRONG_SCENE；P1–P3 = 三个新 carrier 上的 OFF − TTT_100；分支 A（三者都 SUPPORTED）/B/C。
- **状态：** 11:08 冻结（预注册 sha256 `aba2a5f1…`），12:41–15:03 运行。
- **结果（分支 A，完整性 PASS）：**
  - OFF − TTT_100：V18 L +0.0244（77/6）、V19 L +0.0286（70/13）、V20 L +0.0327（75/8），都 SUPPORTED；V17 L +0.0213；
  - 宽度对增益的作用 +0.0078（CI 不含 0），训练数据与交互作用的 CI 跨 0；
  - 15,936 个指标重新计算一致；V17 L 的 NO_STREAM、OFF 与 Core B V8 逐位相同。

## EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9（2026-10-03，hit GPU，Core B 第九轮：300 步流帧写入与监督来源的重现）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9/PLAN_BEFORE_COREB_V9.md)（10:21 写定，sha256 `8956b200…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9/PROTOCOL.md)。
- **动机：** 流帧写入从 10 步加到 100 步，增益约翻倍（Core B V7、V8）；V8 中新帧监督相对上下文监督 +0.0114，但只有 69% 的场景不变差（PARTIAL）。
- **设计：** TRAIN-EXT（55 个场景）；P1 = TTT_100 − TTT_300，P2 = OFF_CTX_TTT_100 − TTT_100；分支 A/B/C/D。
- **状态：** 10:45 冻结（预注册 sha256 `25644e43…`），12:30 完成。
- **结果（分支 D，完整性 PASS）：**
  - P1 TTT_100 − TTT_300 +0.0041 [−0.0033, +0.0096]，44/0/11，NOT_ESTABLISHED；P2 OFF_CTX_TTT_100 − TTT_100 +0.0162 [+0.0038, +0.0296]，37/0/18，PARTIAL；
  - OFF − TTT_100 +0.0355（51/4），OFF − TTT_300 +0.0396；特异性 +0.0651；
  - 3,960 个指标重新计算一致；NO_STREAM、OFF 与 Core B V6 逐位相同。

## EXP-3D-RGBD-DEPTH-READOUT-V22（2026-10-03，hit CPU，深度读出与长训练增益）

- [计划](outputs/EXP-3D-RGBD-DEPTH-READOUT-V22/PLAN_BEFORE_V22.md)（05:17 写定，sha256 `022fe661…`），05:24 冻结。
- **动机：** 事后诊断（探索性，只用 EVAL-V3/V4，[记录](outputs/EXP-3D-RGBD-POSTHOC-DIAG-V1/README.md)）发现渲染器的深度未除以不透明度。改用归一化读出后，V17–V20 的长训练 FAR 增益几乎消失。
- **设计：**
  - 只在 CPU 上重新计分 V17–V20 已封存的 EVAL-FAR 预测，不运行任何模型；
  - 读出 RAW 与 NORM = 深度 / max(不透明度, 10⁻³)，区域、聚合、bootstrap 与 V2 gate 都与 V11/V17 相同；
  - 主判定一：长训练增益在 NORM 下的残差，若 CI 上界小于 RAW 增益的一半，则为 EXPLAINED；
  - 主判定二：第 6000 步的归一化增益，用 V2 gate；只有 V17 用于判定。
- **结果（分支 C，完整性 PASS）：**
  - V17：RAW +0.058 [+0.036, +0.079] → NORM +0.021 [+0.003, +0.039]，NOT_EXPLAINED；
  - 第 6000 步的归一化增益 +0.050，PARTIAL（55/0/23）；
  - V18–V20 的残差为 +0.009 到 +0.026；
  - 长训练增益约 2/3 来自读出偏差，远处 query 上仍有约 +0.02 的真实补全改进。

## EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8（2026-10-03，hit GPU，Core B 第八轮：新帧监督与上下文监督）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/PLAN_BEFORE_COREB_V8.md)（08:49 写定，sha256 `2f04035b…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/PROTOCOL.md)。
- **动机：** V6、V7 中，新帧监督相对上下文监督的点估计都为正（+0.0073、+0.0107），但 CI 跨 0；V7 中 FAR 区域 +0.0239。
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景）；P1 = OFF − TTT_100，P2 = OFF_CTX_TTT_100 − TTT_100；分支 A/B/C/D。
- **状态：** 09:03 冻结（预注册 sha256 `e4848ca4…`），10:14 完成。
- **结果（分支 B，完整性 PASS）：**
  - P1 OFF − TTT_100 +0.0213 [+0.0141, +0.0297]，73/0/10，SUPPORTED；P2 OFF_CTX_TTT_100 − TTT_100 +0.0114 [+0.0053, +0.0180]，57/0/26，PARTIAL；
  - 上下文监督本身 +0.0099，FAR 上为 0；FAR 上 TTT_100 相对上下文监督 +0.0190；
  - 5,976 个指标重新计算一致，NEAR 掩码 332/332 与 V17 相同。

## EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7（2026-10-03，hit GPU，Core B 第七轮：全新队列上的更高剂量）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/PLAN_BEFORE_COREB_V7.md)（07:56 写定，sha256 `d48a9b28…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/PROTOCOL.md)。
- **动机：** V23 显示 10 步的上下文 TTT 在 EVAL-V3/V4 上只是 PARTIAL，而 100 步的增益约为 3 倍。
- **设计：** EVAL-V5（23 个场景）；V17 L 第 24000 步；8 种策略与 4 个对照；P1 = NO_STREAM − CTX_TTT_100，P2 = OFF − TTT_100；分支 A/B/C/D。
- **状态：** 08:11 冻结（预注册 sha256 `e7ede177…`），08:46 完成。
- **结果（分支 B，完整性 PASS）：**
  - P1 NO_STREAM − CTX_TTT_100 +0.0059 [−0.0003, +0.0132]，NOT_ESTABLISHED；P2 OFF − TTT_100 +0.0150 [+0.0069, +0.0243]，19/0/4，SUPPORTED；
  - 特异性 TTT_100 +0.0228（22/1）；监督来源整体 +0.0107（CI 跨 0），FAR +0.0239；
  - TTT_100 在 FAR 上 +0.0171；3,312 个指标重新计算一致。

## EXP-3D-RGBD-EVAL-V5-DATA（2026-10-03，hit，放宽规则的最后一批未观察过的场景）

- [计划](outputs/EXP-3D-RGBD-EVAL-V5-DATA/PLAN_BEFORE_EVAL_V5_DATA.md)、[记录](outputs/EXP-3D-RGBD-EVAL-V5-DATA/README.md)。
- **动机：** 按 EVAL-V4 的规则已无候选。用户决定允许 val 划分、允许与 DEV/FRESH 共享 volume。
- **结果：** 34 个候选，23 个有效，7 个 volume（都只与 DEV 共享）；下载 332 MB；PASS。

## EXP-3D-RGBD-STATIC-TTT-EVAL-V23（2026-10-03，hit GPU，只用上下文的测试时精修）

- [计划](outputs/EXP-3D-RGBD-STATIC-TTT-EVAL-V23/PLAN_BEFORE_V23.md)（07:02 写定，sha256 `8e0f54e2…`）、[协议](outputs/EXP-3D-RGBD-STATIC-TTT-EVAL-V23/PROTOCOL.md)。
- **动机：** Core B V6 中，只用上下文的 TTT 改进了静态状态（+0.0078，45/0/10）。
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景）；V17 L 第 24000 步；NO_STREAM、CTX_TTT（10 步）、CTX_TTT_30、CTX_TTT_100 与对照；P1 = NO_STREAM − CTX_TTT，P2 = CTX_TTT_WRONG_SCENE − CTX_TTT；描述性：FAR/NEAR、调和插值、AHFILL8。
- **状态：** 07:15 冻结（预注册 sha256 `e629b9b9…`），07:54 完成。
- **结果（分支 C，完整性 PASS）：**
  - P1（10 步）+0.0054 [+0.0019, +0.0101]，59/0/24，PARTIAL；P2 特异性 +0.0062，PARTIAL；
  - 剂量（描述性）：30 步 +0.0112，100 步 +0.0176 [+0.0079, +0.0321]（66/0/17）；100 步时 FAR +0.0124；
  - 10 步的设置对没见过的场景太少；V7 把 100 步作为主判定的设置。

## EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6（2026-10-03，hit GPU，Core B 第六轮：测试时写入需要新帧吗）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/PLAN_BEFORE_COREB_V6.md)（06:22 写定，sha256 `658025fd…`）、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/PROTOCOL.md)。
- **动机：** Core B V4/V5 的 TTT 增益集中在已观测区域，这引出两个问题：它能否在第三个独立队列上重现？它是否其实不需要新帧，只是对状态的一般性精修？
- **设计：** TRAIN-EXT（55 个场景）；策略 NO_STREAM、CTX_TTT、OFF、OFF_CTX_TTT、TTT 与两个对照；Core B V4 的设置；P1 = OFF − TTT，P2 = NO_STREAM − CTX_TTT。
- **状态：** 06:42:46 冻结（预注册 sha256 `136d777a…`），06:59 完成。
- **结果（分支 A，完整性 PASS）：**
  - P1 OFF − TTT +0.0148 [+0.0054, +0.0245]，49/0/6，SUPPORTED；P2 NO_STREAM − CTX_TTT +0.0078 [+0.0026, +0.0133]，45/0/10，SUPPORTED；
  - 监督来源 OFF_CTX_TTT − TTT +0.0073，CI 跨 0；两种 TTT 都有场景特异性；
  - 增益以 NEAR 为主；4,620 个指标重新计算一致，NEAR 掩码 220/220 与 GPU 试点相同。

## EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5（2026-10-03，hit GPU，Core B 第五轮：独立场景上的确认）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/PLAN_BEFORE_COREB_V5.md)、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/PROTOCOL.md)。
- **动机：** Core B V4 为分支 A，但 DEV 只有 8 个反复使用的场景，TRAIN 选择集上也没有增益。
- **设计：**
  - 队列：EVAL-V3（57）∪ EVAL-V4（26），共 83 个场景。它们从未用于 Core B，也从未用于 carrier 的训练或选择；
  - carrier、TTT 设置（10 步，学习率 1e-3，不重新选择）、episode 规则、策略与对照都与 Core B V4 相同；
  - 主判定 TTT_BEYOND_BASELINE 与 TTT_SPECIFICITY，用冻结的 V2 gate；分支 A/B/C 同计划；
  - 描述性：每个队列、每个 seed，以及归一化的深度读出。
- **准备：**
  - 新模块 `rgbd_stream_ttt_eval_v5.py` 为 EVAL 记录读取上下文。测试证明，同一批合成场景上它与 Core B V4 的 DEV 路径得到逐位相同的缓存状态；
  - CPU 预演（4 个 TRAIN 场景充当两个 EVAL 队列）全流程通过；完整测试 1081 个通过、3 个跳过。
- **状态：**
  - 05:15 第一次冻结（`a0183207…`）。冻结的封存屏障每次读取都重算全部状态的哈希（平方级），05:57 在写出任何指标之前停止，产物保留在 `audit/attempt1_slow_barrier/`（`audit/incident_attempt1_slow_seal_barrier.md`）；
  - 加入 `CachedSealBarrier`（封存关闭时与最后一个 query 之后各完整校验一次）后，06:06 重新冻结（`f2a6f0e9…`），06:18 完成。
- **结果（分支 A，完整性 PASS）：**
  - OFF 0.207，TTT 0.199，TTT_WRONG_SCENE 0.206，NO_STREAM 0.269；
  - TTT_BEYOND_BASELINE +0.0079 [+0.0030, +0.0135]，65/0/18，SUPPORTED；TTT_SPECIFICITY +0.0068 [+0.0015, +0.0122]，67/0/16，SUPPORTED；
  - 3,984 个指标重新计算一致，58,174 个旧文件未变；
  - 事后（探索性）：增益集中在 NEAR 区域，FAR 区域的改善几乎都来自新帧本身的观测。

## EXP-3D-RGBD-DYNAMIC-TTT-V4（2026-10-03，hit GPU，Core B 第四轮）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4/PLAN_BEFORE_COREB_V4.md)、[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4/PROTOCOL.md)。
- **动机：** Core B V1–V3 中，学到的写入规则都没有胜过只缓存新帧。写入规则要从一次前向的激活里猜出快速权重的增量；本轮改为直接用新帧测得的 RGB-D，对快速权重做梯度下降（测试时训练）。
- **设计：**
  - carrier、episode 与 DEV 队列都与 Core B V3 相同（V17 L 第 24000 步，冻结）；
  - TTT：缓存流帧之后，fuse 与 complete 两个增量从 0 开始，用冻结的 V2 C1 损失对 4 个流帧自己的 RGB 与测得深度做 K 步 Adam，每步每帧 256 条光线；
  - 超参数网格 K ∈ {10, 50} × 学习率 ∈ {1e-3, 1e-2}，只在 32 个 TRAIN episode 上按 OFF − TTT 选择；
  - 对照 TTT_WRONG_SCENE：把下一个 DEV 场景同一角色的 TTT 增量用到本场景的缓存上；
  - 主判定 TTT_BEYOND_BASELINE = OFF − TTT 与 TTT_SPECIFICITY = TTT_WRONG_SCENE − TTT，用冻结的 V2 gate；分支 A/B/C 同计划。
- **准备：**
  - 冻结前检查代码时发现，场景数据类初始化时会读取实验目录里的 `train_depth_prior.json`。冻结流程改为从 V7 复制它，并核对与 Core B V3 的副本相同；
  - CPU 预演（2 个 TRAIN 场景充当 DEV，极小网格）全流程通过；完整测试 1077 个通过、3 个跳过。
- **状态：** 2026-10-03 02:26 冻结（预注册 sha256 `4f750d9c…`）。V21 训练释放 4 号卡之后，04:43 开始 GPU 预演，04:45 预演通过，正式运行 04:56 完成。
- **结果（分支 A，完整性 PASS）：**
  - TRAIN 选择：4 组的平均增益都不为正，选 10 步、学习率 1e-3（−0.0001）；
  - DEV AbsRel：NO_STREAM 0.330，OFF 0.236，TTT 0.224，TTT_WRONG_SCENE 0.239；
  - TTT_BEYOND_BASELINE +0.0115 [+0.0050, +0.0187]，8/0/0，SUPPORTED；TTT_SPECIFICITY +0.0146 [+0.0072, +0.0224]，8/0/0，SUPPORTED；
  - 384 个指标重新计算一致，57,724 个旧文件未变；
  - 事后（探索性）：归一化深度读出下 TTT 的优势不变（+0.0120），不透明度不变，所以不是读出偏差。
  - 增益小、DEV 只有 8 个场景，需要在独立数据上确认。

## EXP-3D-RGBD-DYNAMIC-WRITE-V3（2026-10-03，hit GPU，Core B 第三轮）

- [计划](outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V3/PLAN_BEFORE_COREB_V3.md)、[协议](outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V3/PROTOCOL.md)。
- **动机：** Core B V2 中学到的写入不胜过只缓存新帧；检验是不是因为当时的 carrier（V9 C1，宽度 8、6000 步）太弱。
- **设计：**
  - 与 Core B V2 相同，只把冻结的 carrier 换成 V17 L 第 24000 步的 checkpoint；
  - 写入规则的训练与 DEV 评价都在 4 号卡上（关闭 TF32）；
  - 主判定 BEYOND_BASELINE = OFF − ALL 与 SPECIFICITY = ALL_WRONG_SCENE − ALL。
- **彩排：** 第一次彩排时，GPU 渲染的 RGB 因舍入略出 [0, 1]，触发了指标函数的范围检查。V3 的评价改为先截断再计分，并记录截掉的量（最多 2.4×10⁻⁷）；第二次彩排全流程通过。
- **结果（2026-10-03 01:59，分支 C，完整性 PASS）：**
  - BEYOND_BASELINE −0.0015 [−0.0079, +0.0031]，NOT_ESTABLISHED；SPECIFICITY −0.0011，NOT_ESTABLISHED；
  - 缓存新帧的作用 +0.094（Core B V2 为 +0.052）；训练写入规则的作用 −0.0015；
  - 学到的写入在更强的 carrier 上同样不胜过只缓存新帧。

## EXP-3D-RGBD-SEGMENT-DIVERSITY-V21（2026-10-03，hit GPU，同一批场景里的内容多样性）

- [计划](outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/PLAN_BEFORE_V21_DATA.md)，冻结记录 `outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/freeze.json`。
- **动机：** Hypersim 只剩 41 个未用场景，而且都在已用过的 volume 里；但每个训练场景只用了相机轨迹开头的 16 帧（轨迹中位数 100 帧）。
- **数据 SEG-V1（`outputs/EXP-3D-RGBD-SEGMENT-DATA-V1`）：**
  - TRAIN72 每个场景的轨迹第 16–79 帧位置切成最多 4 个片段，沿用原记录的冻结有效性规则与相机角色规则。
  - 下载在锁定顺序的第 64 个场景中途碰到 4 GB 预算，读取器报错，没有写出清单。按冻结规则（读到预算为止）不用这个场景及其后的 8 个场景；由 `scripts/recover_rgbd_segment_data.py` 在不联网的情况下重建清单，每个片段的角色支持都与原运行一致。
  - 结果：240 个有效片段，来自 62 个场景。
- **设计：**
  - 两组都在 GPU 上用冻结的 V17 代码训练 24000 步，各 3 个 seed：G1 用原片段（72 个），G5 用原片段加 240 个新片段（312 个）；
  - P1：DIVERSITY_GAIN = FAR(G1) − FAR(G5)，第 24000 步，V2 gate；P2：调和插值 − AHFILL8(G5)，INPAINT 标签。
- **状态：** 2026-10-03 01:31 冻结。第一次启动因缺 `checkpoints/` 目录在第 0 步失败（`audit/incident_launch_missing_checkpoints.md`），02:03 以同一冻结命令重启；训练 04:42 结束，CPU 上的评价 05:55 结束。
- **结果（分支 C）：**
  - P1 DIVERSITY_GAIN −0.023 [−0.063, +0.014]，32/0/48，NOT_ESTABLISHED；P2 调和插值 − AHFILL8(G5) +0.031，NOT_DISTINGUISHABLE；
  - 每个检查点上 G5 都比 G1 差；EVAL-FAR 上 −0.034 [−0.060, −0.006]；
  - 设备检查：V17 L（CPU）− G1（GPU）= −0.005 [−0.016, +0.005]；
  - 同一批场景里的内容多样性不能改善补全，也不能解决不一致。

## EXP-3D-RGBD-GPU-LENGTH-PILOT-V1（2026-10-02，hit GPU，探索性）

- [计划](outputs/EXP-3D-RGBD-GPU-LENGTH-PILOT-V1/PLAN_BEFORE_PILOT.md)、[结果](outputs/EXP-3D-RGBD-GPU-LENGTH-PILOT-V1/README.md)。
- 宽度 32、64 训练到 96000 步，在 TRAIN-EXT 的 55 个场景（TRAIN72 训练从未见过）上评价。
- **结果：**
  - FAR 在 12000–24000 步之后不再下降（宽度 32：第 24000 步 0.316，第 96000 步 0.325；宽度 64：0.312、0.321）；
  - 宽度 64 在任何步数都没有可区分的优势；
  - 相对调和插值（FAR 0.418）平均好约 0.10，但只有约 33/54 个场景变好。

## EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20（2026-10-02，hit，宽度 64 × 127 个训练场景 × 长训练）

- [计划](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20/PLAN_BEFORE_V20_TRAINING.md)、[协议](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20/PROTOCOL.md)。
- **动机：** 检验只有同时放大容量、数据与训练长度时补全才改善的假设；(宽度 64, TRAIN127, 24000 步) 是前几轮都没有覆盖的一角。
- **设计：**
  - V15 C1 配方（宽度 64，127 个场景）训练 24000 步；前 6000 步必须与 V15 C1 逐位相同。
  - 与 V18 每一步的数据相同，只有宽度不同。
  - P1：LENGTH_GAIN；P2：WIDTH_GAIN_24K = FAR(V18 第 24000 步) − FAR(V20 第 24000 步)，都用冻结的 V2 gate。
  - 四个实验都完成后，另报宽度 × 数据 × 步数的 2×2×2（描述性）。
- **结果（2026-10-03 01:38，分支 C）：** 内置对照 PASS；P1 LENGTH_GAIN +0.058 [+0.026, +0.092]（58/22，PARTIAL）；P2 WIDTH_GAIN_24K −0.002 [−0.015, +0.012]（41/39，NOT_ESTABLISHED）；FAR 第 6000 → 24000 步 0.418 → 0.360（宽度 32 为 0.420 → 0.358）。2×2×2 见上方 STATUS。原状态：2026-10-02 06:16 冻结。

## EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19（2026-10-02，hit，长训练下的宽度 64）

- [计划](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19/PLAN_BEFORE_V19_TRAINING.md)、[协议](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19/PROTOCOL.md)。
- **动机：** V12、V15 中宽度 64 在 6000 步时没有改善补全；检验更宽的载体是否需要更长的训练才显出优势。
- **设计：**
  - V12 C1 配方（宽度 64，TRAIN72）训练 24000 步；前 6000 步必须与 V12 C1 逐位相同。
  - 与 V17 每一步的数据相同，只有宽度不同。
  - P1：LENGTH_GAIN；P2：WIDTH_GAIN_24K = FAR(V17 第 24000 步) − FAR(V19 第 24000 步)，都用冻结的 V2 gate。
- **结果（2026-10-03 00:49，分支 C）：** 内置对照 PASS；P1 LENGTH_GAIN +0.042 [+0.015, +0.072]（53/27，PARTIAL）；P2 WIDTH_GAIN_24K +0.004 [−0.014, +0.030]（41/39，NOT_ESTABLISHED）；FAR 第 6000 → 24000 步 0.386 → 0.344（宽度 32 为 0.392 → 0.348）。原状态：2026-10-02 05:47 冻结。第一次完整测试时漏设 `CUDA_VISIBLE_DEVICES=""`，两个 CUDA 小测试在默认 GPU 上跑了几秒；已在隐藏 CUDA 的条件下重跑，见该实验的 `audit/incident_first_suite_cuda_visible.md`。

## EXP-3D-RGBD-LONG-TRAIN-DATA-V18（2026-10-02，hit，127 个训练场景上的长训练）

- [计划](outputs/EXP-3D-RGBD-LONG-TRAIN-DATA-V18/PLAN_BEFORE_V18_TRAINING.md)、[协议](outputs/EXP-3D-RGBD-LONG-TRAIN-DATA-V18/PROTOCOL.md)。
- **动机：** V14 中更多训练场景让补全变差；事后分析显示，两种训练集的补全都随训练步数持续改善。检验更长的训练能否扭转数据的作用。
- **设计：**
  - V14 C1 配方（TRAIN72 + TRAIN-EXT，127 个场景）训练 24000 步；前 6000 步必须与 V14 C1 逐位相同。
  - 主判定与 V17 相同；与 V17 合成训练集 × 步数的 2×2（描述性）。
- **结果（2026-10-02 12:41，分支 C）：**
  - 内置对照 PASS。
  - P1 LENGTH_GAIN +0.062 [+0.030, +0.095]，55/25 个场景，PARTIAL。
  - P2 调和插值 − AHFILL8(第 24000 步) +0.032 [+0.012, +0.055]，46/3/34，NOT_DISTINGUISHABLE。
  - FAR：0.420（第 6000 步）→ 0.390（第 12000 步）→ 0.354（第 16000 步）→ 0.358（第 24000 步）。
  - DEV 选中第 5000、2000、4000 步。
  - 与 V17 的 2×2（描述性，`audit/two_by_two_v17_v18.json`）：更多数据的效果在第 6000 步为 −0.027 [−0.049, −0.007]，在第 24000 步为 −0.009 [−0.031, +0.014]，交互 −0.018 [−0.049, +0.012]。长训练缩小了劣势，但没有反转。

## EXP-3D-RGBD-LONG-TRAIN-V17（2026-10-02，hit，更长的训练能否改善补全）

- [计划](outputs/EXP-3D-RGBD-LONG-TRAIN-V17/PLAN_BEFORE_V17_TRAINING.md)、[协议](outputs/EXP-3D-RGBD-LONG-TRAIN-V17/PROTOCOL.md)。
- **动机：** 据 V14 事后分析，V11 C1 配方的补全误差随步数单调下降，到第 6000 步仍在下降；V13 的弃权提高了组合方法的稳定性。
- **设计：**
  - V11 C1 配方训练 24000 步，检查点在 6000 步后追加 8000–24000 步；前 6000 步必须与 V11 C1 逐位相同（内置对照）。
  - P1：LENGTH_GAIN = FAR(第 6000 步) − FAR(第 24000 步)，冻结的 V2 gate。
  - P2：调和插值 − AHFILL8(第 24000 步)，INPAINT-V1 的标签规则。
  - 主队列 EVAL-V3 ∪ EVAL-V4（83 个场景）；EVAL-FAR 与 FRESH-V2 只作描述。
- **结果（2026-10-02 12:29，分支 C）：**
  - 内置对照 PASS。
  - P1 LENGTH_GAIN +0.044 [+0.018, +0.069]，54/26 个场景，PARTIAL（只差场景一致性：67.5% < 75%）。
  - P2 调和插值 − AHFILL8(第 24000 步) +0.035 [+0.015, +0.057]，52/3/28，NOT_DISTINGUISHABLE。
  - FAR：0.392（第 6000 步）→ 0.365（第 10000 步）→ 0.356（第 16000 步）→ 0.348（第 24000 步）。
  - DEV 选中第 750、1500、4000 步，与 V11 C1 相同；SEL 的 FAR 0.421，比第 6000 步差 0.029。
  - 描述性：EVAL-FAR 上 LENGTH_GAIN +0.058，SUPPORTED；FRESH-V2 上 +0.006，看不出。

## 迁移到 hit（2026-10-02）

- **内容：** 代码与文档从 GitHub main `249becb` 克隆；数据、记录与模型打成 5 个包，经本地中转后逐文件核对哈希（CV1–CV3 共 22,412 个文件，CV4 1,338 个，CV5 109 个）。
- **环境：** 按 `uv.lock` 同步，Python 3.12.7、PyTorch 2.11.0+cu128、NumPy 2.5.2，与 zonghan_o 相同；完整测试 1031 个通过、3 个跳过。
- **路径：** 记录里写的是 zonghan_o 的 `/home/zonghan/…` 路径，在 hit 上用启动包装 `cv_host_run.py` 在运行时映射，文件本身不改。
- **复现（[HIT-REPRO-V1](outputs/EXP-3D-HIT-REPRO-V1/README.md)）：** 在 hit 上重训 V15 C0（seed 20260928）。设置 `ATEN_CPU_CAPABILITY=avx2`、`MKL_CBWR=AVX2`、`DNNL_MAX_CPU_ISA=AVX2` 并用单线程 BLAS 时，与 zonghan_o 逐位一致（6000 步训练记录、第 5000/6000 步参数、12 个 DEV 检查点）。不设置时第 1 步就不同，DEV 最多差 0.047，选出的 checkpoint 也不同。GPU 上（4 号卡）训练无法逐位复现：强制确定性时 3D `grid_sample` 的反向没有确定性实现；默认设置下两次运行从第 2 步起不同，选出的 checkpoint 也不同；每步比 CPU 快 7–20 倍。

## EXP-3D-RGBD-VOLUME-DISJOINT-V16（2026-09-30，陌生 volume 上的补全优势）

- [结果](outputs/EXP-3D-RGBD-VOLUME-DISJOINT-V16/README.md)、[协议](outputs/EXP-3D-RGBD-VOLUME-DISJOINT-V16/PROTOCOL.md)。
- **动机：** V15 的事后分析提示，补全优势部分来自对训练中同一 volume 的熟悉。V16 在与训练集 volume 完全不相交的 FRESH-V2 上直接检验。
- **设计：**
  - 不训练；主模型为只用 TRAIN72 训练的 V11 C1，次要模型为 V12 C1，代码路径与 V13 相同。
  - 这是 FRESH-V2 的第三次使用，只作机制检验。
  - 对照为 V13 已评分的 EVAL-V3/V4 结果（volume 共享）。
  - 08:27:42 锁定；封存前 4 个 carrier 家族在 EVAL-V3 上的 96 个数组与各自封存预测完全一致。
- **FRESH-V2（17 个场景，其中 14 个有 FAR 像素）AbsRel（ALL / FAR）：** 调和插值补洞 0.307 / 0.508，V11 C1 0.251 / 0.323，HFILL8 0.278。
- **判定：**
  - P1（补全优势 A = 调和插值 FAR − V11 C1 FAR）+0.185，CI [−0.061, +0.590]，8 好 6 差，`NOT_DISTINGUISHABLE`。
  - P2（熟悉度差 Δ = A(EVAL) − A(FRESH)）−0.114，CI [−0.550, +0.177]，没有差距。
  - 预注册分支 **D**：结论不确定。
- **描述性：**
  - 组合方法相对调和插值 +0.029，CI [−0.021, +0.108]。
  - 按 FRESH 场景所在 volume 是否有 TRAIN-EXT 场景分组，所有模型（包括从未见过这些 volume 的 V11 C1）在前一组的优势都约为 0，所以分组差异来自场景本身。
- **结论：** 在陌生 volume 上，carrier 并没有简单地失去补全优势（平均 +0.185，比共享 volume 时还大），但优势在场景之间极不一致，样本也小。V15 的"熟悉度"解释只能算未证实的假设。

## EXP-3D-RGBD-WIDTH-DATA-V15（2026-09-30，更多数据时的宽度）

- [完整报告](docs/experiments/EXP-3D-RGBD-WIDTH-DATA-V15/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-WIDTH-DATA-V15/STATUS.md)、[计划](outputs/EXP-3D-RGBD-WIDTH-DATA-V15/PLAN_BEFORE_V15_TRAINING.md)、[协议](outputs/EXP-3D-RGBD-WIDTH-DATA-V15/PROTOCOL.md)。
- **动机：** V14 的事后分析显示，容量固定时更多数据改善重建、损害补全。V15 检验这是否是容量不足：在 127 个训练场景上把宽度从 32 加到 64。
- **完整性：** C0（宽度 32）的全部训练记录、DEV 曲线与 648 个评价预测都与 V14 C1 逐位相同；宽度 64 的 C1 全程与 C0 共享数据流；finalize 的完整性检查为 PASS。
- **2×2（FAR AbsRel，EVAL-V3 + EVAL-V4 合并的 83 个场景）：**
  - 宽度 32：72 个场景 0.421，127 个场景 0.443；
  - 宽度 64：72 个场景 0.415，127 个场景 0.446。
- **判定：**
  - WIDTH_AT_SCALE_GAIN（127 个场景上宽度 32 − 64）−0.003，CI [−0.014, +0.007]，`NOT_ESTABLISHED`。
  - HARMONIC_HYBRID_GAIN（调和插值 − HFILL8 C1）−0.002，CI [−0.023, +0.024]，`NOT_DISTINGUISHABLE`。
  - HFILL8_WIDTH_GAIN −0.002，CI [−0.004, −0.0002]，`HARMFUL`（幅度很小）。
  - 预注册分支 **C**。
- **描述性：**
  - 更多数据在两个宽度下都让补全变差（宽度 64 时 −0.031，CI 上界 +0.0003）。
  - 交互作用 +0.009，CI [−0.006, +0.025]：数据多了之后宽度更不起作用，与容量不足的假设方向相反。
  - 宽度 64 在被看到的区域更好（NEAR 0.239 对 0.246）；DEV 0.262 对 0.270。
  - 第 6000 步：宽度的作用 +0.002（CI 跨零），调和插值 − HFILL8 C1 +0.010（CI 跨零）。
- **结论：** V14 中数据带来的补全下降不是容量不足造成的。在 CPU 可及的宽度（8/32/64）与数据量（72/127）范围内，这个架构的补全没有随规模改善。
- **事后分析（探索性，[说明](outputs/EXP-3D-RGBD-WIDTH-DATA-V15/audit/exploratory_posthoc_volume_familiarity/README.md)）：** EVAL-V3/V4 与 TRAIN72 共享 Hypersim volume。按评价场景所在 volume 在训练集中的占比变化分组：占比下降的 22 个 volume，更多数据让补全变差（宽度 32 时 −0.032）；占比上升或不变的 10 个 volume 反而改善（+0.023）。两组之差 +0.055，CI [+0.013, +0.099]（按 volume bootstrap；宽度 64 时 +0.054，CI [−0.002, +0.111]）。这提示补全可能部分来自对同一 volume 设计风格的熟悉；但 V16 在 volume 不相交的 FRESH-V2 上没有证实这一点（见 V16），所以只能作为假设。

## EXP-3D-RGBD-DATA-SCALE-V14（2026-09-30，只增加训练场景）

- [完整报告](docs/experiments/EXP-3D-RGBD-DATA-SCALE-V14/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-DATA-SCALE-V14/STATUS.md)、[计划](outputs/EXP-3D-RGBD-DATA-SCALE-V14/PLAN_BEFORE_TRAIN_EXT_DATA.md)、[协议](outputs/EXP-3D-RGBD-DATA-SCALE-V14/PROTOCOL.md)、[新训练数据](outputs/EXP-3D-RGBD-TRAIN-EXT-DATA/README.md)。
- **设计：** V11 C1 的配方不变，只增加训练场景。
  - C0：TRAIN72；C1：TRAIN72 + TRAIN-EXT，共 127 个场景（新增 55 个有效场景，来自 29 个 volume）。
  - 主评价：EVAL-V3 + EVAL-V4 合并的 83 个场景；EVAL-FAR 只作描述。
- **代码：** 由 V12 派生，有两处实质改动。
  - V12 借用的 V8 `train_records` 查的是 V8 的训练集表，照抄会让 C1 仍只用 TRAIN72；V14 自带实现，并有测试专门检查。
  - DEV 曲线不再保存状态文件，checkpoints 目录只有 208 MB。
- **完整性：**
  - C0 三个 seed 的全部 6000 步训练记录、DEV 曲线、选中的 checkpoint（750/1500/4000 步）都与 V11 C1 逐位相同；648 个 EVAL 预测的最大差为 0。
  - C1 用遍了全部 127 个场景；finalize 的完整性检查为 PASS。
- **合并 83 个场景的 AbsRel（ALL / NEAR / FAR）：**
  - C0 0.319 / 0.288 / 0.421；C1 0.295 / 0.246 / 0.443。
  - 调和插值补洞 0.289；HFILL8：C0 0.278、C1 0.289；AHFILL8：C0 0.271、C1 0.274。
- **判定：**
  - DATA_GAIN（FAR C0 − FAR C1）−0.022，CI [−0.048, +0.001]，34 好 46 差，`NOT_ESTABLISHED`。
  - HARMONIC_HYBRID_GAIN（调和插值 − HFILL8 C1）+0.000，CI [−0.021, +0.026]，`NOT_DISTINGUISHABLE`。
  - HFILL8_DATA_GAIN −0.011，CI [−0.018, −0.005]，`HARMFUL`。
  - 预注册分支 **C**。
- **描述性：**
  - DEV：C0 0.265，C1 0.270，差值 CI 跨零。
  - 训练后期 C1 在 DEV 上过拟合更轻，最后一步约低 0.03，但选中的 checkpoint 没有体现出来。
  - EVAL-V3、EVAL-V4、EVAL-FAR 各自的方向与合并结果相同。
- **结论：** 训练场景增加到 1.76 倍，改善的是被看到区域的精度，而补全略变差。组合方法只在 FAR 用 carrier，因此显著变差，与调和插值补洞打平。在 CPU 可及的规模内，数据量不是补全的杠杆。
- **事后分析（探索性，[说明](outputs/EXP-3D-RGBD-DATA-SCALE-V14/audit/exploratory_posthoc_step_matched/README.md)）：** 在 9 个相同步数（500–6000）上比较 C0 与 C1，每一个步数都是 C1 的补全更差（FAR 差 0.02–0.04，8 个步数的 CI 不含 0），被看到的区域都更好。所以补全变差不是选中步数不同造成的：容量固定时，更多数据把模型推向重建被看到的几何。两个变体的补全都随训练持续改善，最后一步最好，按 DEV 总误差选中的 checkpoint 常常偏早。

## EXP-3D-RGBD-VOLUME-V13（2026-09-30，状态体积之外的弃权回退）

- [结果](outputs/EXP-3D-RGBD-VOLUME-V13/README.md)、[协议](outputs/EXP-3D-RGBD-VOLUME-V13/PROTOCOL.md)、[探索性分析](outputs/EXP-3D-RGBD-VOLUME-V13/audit/exploratory_opacity_fallback/)、[锁定前试运行](outputs/EXP-3D-RGBD-VOLUME-V13/audit/pre_lock_dry_run/)、[EVAL-FAR 数据](outputs/EXP-3D-RGBD-EVAL-FAR-DATA/README.md)。
- **设计：** 不训练，使用冻结的 V11 C1（3 个 seed）。
  - AHFILL8：NEAR 像素用调和插值补洞；FAR 像素中渲染不透明度 < 0.5 的也用调和插值补洞，其余用 carrier。
  - 阈值 0.5 来自已用队列（EVAL-V3/V4、Replica）上的探索性分析，所以主检验放在新的 EVAL-FAR 队列上。
- **EVAL-FAR 数据：**
  - 79 个场景，来自 EVAL-V3/V4，分布在 32 个 volume；每个场景 2 个远处 query（cam_00 轨迹第 32、48 个位置之后），此前从未被模型读取。
  - 下载 169 MB，完整性 PASS；上下文与原记录逐帧一致。
- **流程：** 01:22:23 锁定；三个队列分别封存，每次先重算 V11 的封存预测，最大差 0.0；648 个预测都在读取任何 query 深度之前封存。
- **EVAL-FAR AbsRel（ALL）：** 常数 0.499，REPROJ_NN 0.271，调和插值补洞 0.261，C1 0.304，HFILL8 0.256，AHFILL8 0.246。
- **判定：**
  - P1 ABSTAIN_GAIN（HFILL8 − AHFILL8）+0.011，CI [+0.005, +0.017]，场景 43 好 / 31 平 / 5 差，`SUPPORTED`。
  - P2（调和插值补洞 − AHFILL8）+0.016，CI [−0.003, +0.039]，33 个场景更好、45 个更差，`NOT_DISTINGUISHABLE`。
  - 预注册分支 **B**。
- **描述性：**
  - 覆盖率：远处 query 的 GT 表面仍有 93.1% 在体积内（EVAL-V3 近处 94.3%，Replica 44%），所以这个队列没有复现 Replica 的低覆盖。
  - 不透明度标记占 FAR 像素的 8.4%，对体积外像素的精度 0.56、召回 0.32。
  - 在远处 query 上，carrier 在 FAR 区域差于调和插值补洞（0.396 对 0.376）；在近处 query（EVAL-V3）上它领先 0.063。
  - 近处 query：EVAL-V3 上 AHFILL8 0.224（ABSTAIN_GAIN `SUPPORTED`，但阈值正是在这里选定的）；EVAL-V4 上 0.375（+0.003，CI 跨零）。
  - 阈值 0.7 的数值略好（EVAL-FAR 0.243），按协议不采用。
- **结论：** 弃权回退是稳定的小改进，但组合方法仍不能确立优于经典几何补洞。Hypersim 轨迹上的远处 query 基本没有离开状态的体积，所以 Replica 式的失败没有被这个队列检验到。

## EXP-3D-RGBD-REPLICA-TRANSFER-V1（2026-09-30，跨数据集迁移到 Replica）

- [结果](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/README.md)、[使用计划](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PLAN_BEFORE_REPLICA_USE.md)、[协议](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PROTOCOL.md)、[锁定前的代码修正](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/audit/CODE_FIX_BEFORE_LOCK.md)、[事后诊断](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/audit/exploratory_posthoc_bounds_coverage/README.md)。
- **设计：** 不训练，Replica 只使用这一次。方法按规则定为 V11 C1（V12 不是分支 A），放在 HFILL8 组合中。
- **流程复现：** 在 DEV 上重算 36 个数组，与原封存预测完全一致。
- **Replica AbsRel（ALL，8 个场景，FAR 像素占 76%）：**
  - 常数 0.411，REPROJ_NN 0.323，调和插值补洞 0.331；
  - V7 C1 0.611，V11 C0 0.645，V11 C1 0.635；
  - HFILL8_V11C1 0.556。
- **判定：**
  - 主对比（调和插值补洞 − HFILL8）−0.225，CI [−0.346, −0.116]，8 个场景中 7 个更差，`BELOW`，分支 **C**。
  - carrier 也显著差于常数：V11 C1 −0.224，V7 C1 −0.200。
- **事后诊断（描述性）：** 失败有两个来源。
  - 覆盖：Replica 的 query 表面只有 44% 落在状态的体积内（Hypersim 为 94%）。冻结的角色规则把 query 放在离上下文 500 帧以外的位置，这一点我在设计时没有预料到。
  - 跨数据集差距：即使在体积内，carrier 也比调和插值补洞差 0.17（Hypersim 上差 0.03）。
- **结论：** 只在 72 个 Hypersim 场景上训练的状态不能迁移到另一个数据集；几何基线不受影响。

## EXP-3D-RGBD-WIDTH-V12（2026-09-30，宽度 32→64）

- [完整报告](docs/experiments/EXP-3D-RGBD-WIDTH-V12/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-WIDTH-V12/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-WIDTH-V12/PROTOCOL.md)。
- **设计：** 按 V11 之前写定的计划运行（V11 的 COMPLETION_GAIN 点估计 +0.005 > 0）。
  - C1 为 hidden 64，250,542 个参数，在 EVAL-V4 上评价，对照为 V11 C1 已选中的 checkpoint。
  - 宽度 32 的重训与 V11 C1 逐位相同：训练记录、DEV 曲线与 EVAL-V4 预测的最大差都是 0。
- **EVAL-V4 AbsRel（ALL / FAR）：** V11 C1 0.382 / 0.406，宽度 64 0.381 / 0.402；HFILL8 分别为 0.3777 与 0.3775。
- **判定：**
  - WIDTH64_COMPLETION_GAIN +0.004，CI [−0.015, +0.022]，`NOT_ESTABLISHED`；
  - 相对调和插值补洞 +0.010，CI 跨零，`NOT_DISTINGUISHABLE`；
  - 预注册分支 **C**。
- **结论：** 在 TRAIN72 上，宽度 8→32→64 都没有改善补全。

## EXP-3D-RGBD-REPLICATION-EVAL-V4（2026-09-29，在 26 个新场景上复现 V11 与 INPAINT-V1）

- [结果](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/README.md)、[协议](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PROTOCOL.md)、[V11 之前写定的计划](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PLAN_BEFORE_V11_RESULTS.md)。
- **设计：** 不训练，冻结的 V11 C0/C1 用于 EVAL-V4（26 个从未读取过的场景）。
- **流程复现：** 在 EVAL-V3 前 2 个场景上重算 72 个数组，与原封存结果偏差为 0。
- **EVAL-V4 AbsRel（ALL）：**
  - REPROJ_NN 0.399，REPROJ_HARMONIC 0.388，C0 0.375，C1 0.382；
  - HFILL8_C1 **0.378**，FILL8_C1 0.387。
  - 这个队列整体更难：重投影在 NEAR 上也有 0.342（EVAL-V3 为 0.185）。
- **判定：**
  - 主对比（调和插值补洞 − HFILL8_C1）+0.010，CI [−0.026, +0.051]，场景 12 好 14 差，`NOT_DISTINGUISHABLE`，复现分支 **B**；
  - V11 的两个 gate 同样都未成立（COMPLETION +0.005，HYBRID +0.012，CI 均跨零）。
- **结论：** 与 EVAL-V3 一致。组合方法数值上领先经典几何，但场景之间方向不一致，证据达不到预注册标准。

## EXP-3D-RGB-PRIOR-BOUNDS-V10（2026-09-29，RGB-only 的 bounds 先验）

- [完整报告](docs/experiments/EXP-3D-RGB-PRIOR-BOUNDS-V10/README.md)、[终端摘要](docs/experiments/EXP-3D-RGB-PRIOR-BOUNDS-V10/STATUS.md)。
- **设计：** RGB-only（测试时只有 RGB 与相机）。C0 用冻结的 V2 先验（far 5.67 m，32³），C1 用 TRAIN72 的 q01/q99 先验（0.40/15.9 m，32³），C2 同 C1 但为 16³。
- **DEV AbsRel：** C0 0.603，C1 0.461，C2 0.467。
- **判定：**
  - PRIOR_BOUNDS_GAIN（C0−C1）+0.142，CI [+0.022, +0.261]，8 个场景中 7 个变好；排除 NO_HIT 场景 ai_009_001 后约 +0.10。
  - 状态为 `PARTIAL`，因为场景专属性未建立：wrong-scene +0.012，CI 跨零；打乱 +0.048，CI [+0.001, +0.098]。
  - C1 与无几何常数无法区分，预注册分支 **B**。
  - 不做 checkpoint 选择时仍是 C1 更好（0.543 对 0.619）。
- **结论：**
  - bounds 截断确实是 V2–V6 RGB-only 失败的原因之一。下午的只用 TRAIN 的小试验指向相反方向，已被这个正式检验更正。
  - 但放宽 bounds 后，RGB-only 状态仍不优于常数，也几乎不带场景信息。

## EXP-3D-RGBD-COMPLETION-V11（2026-09-29，carrier 宽度对补全的作用）

- [完整报告](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/PROTOCOL.md)、[修正记录](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/AMENDMENTS.md)。
- **设计：** V9 C1 配方（TRAIN72、32³、可变 3–7 视角），C0 为 hidden 8（5,150 参数），C1 为 hidden 32（64,238 参数）。
  - 主评价在 EVAL-V3（57 个场景）上进行，全部预测在读取 query 深度之前封存。
  - FAR 为离上下文深度重投影命中点超过 8 像素的像素，占 20.3%。
- **EVAL-V3 AbsRel（ALL / NEAR / FAR）：**
  - REPROJ_NN 0.258 / 0.185 / 0.537；
  - C0 0.287 / 0.247 / 0.434；
  - C1 0.290 / 0.248 / 0.428；
  - FILL8_C1（NEAR 用重投影，FAR 用 C1）0.241。
- **判定：**
  - COMPLETION_GAIN（FAR：C0−C1）+0.005，CI [−0.006, +0.017]，`NOT_ESTABLISHED`；
  - HYBRID_GAIN（REPROJ_NN−FILL8_C1）+0.017，CI [−0.012, +0.057]，`NOT_ESTABLISHED`；
  - 预注册分支 **C**。
  - DEV（次要）：C0 0.2575、C1 0.2652。C0 与 V9 C1 的 DEV 曲线逐位相同。
- **结论：**
  - 容量试验中加宽带来的补全改善（TRAIN24、16³ 上 +0.059）在 TRAIN72、32³ 上没有复现。
  - 学到的补全在 FAR 上明显优于最近邻补洞（0.43 对 0.54），但组合方法相对重投影的整体优势仍未确立。
- **Amendment 1：**
  - 问题：C1 的 seed 28 与 seed 30 在 DEV 检查点评价时，渲染颜色为 1.0000001（1 个 float32 最小单位），触发冻结的 [0, 1] 检查。
  - 修正：新增的评价模块只把 1×10⁻⁶ 以内的超界截回 [0, 1]。在看任何结果之前写定，并从头重跑这两个训练；训练记录与失败的那次逐位相同（5000/5000 与 6000/6000 步）。
- **按 V11 之前写定的计划：** COMPLETION_GAIN 点估计 > 0，所以 V12（hidden 64）运行。

## EXP-3D-RGBD-INPAINT-BASELINES-V1（2026-09-29，经典几何补洞对照）

- [结果](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/README.md)、[协议](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/PROTOCOL.md)。在 V11 出结果之前锁定，不训练，只用 V11 已封存的预测；V11 各行复现的偏差为 0。
- **EVAL-V3 AbsRel（ALL）：**
  - REPROJ_NN 0.258；
  - REPROJ_HARMONIC（逆深度调和插值补洞）0.244；
  - REPROJ_TELEA 0.252；
  - HFILL8_C1（NEAR 用调和插值，FAR 用 C1）**0.2325**，是目前所有方法中最低的。
- **判定：** AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8_C1) = +0.012，CI [−0.013, +0.044]，`NOT_DISTINGUISHABLE`，分支 **B**。
  - 只看 FAR：调和插值与 C1 的差 +0.063，CI [−0.051, +0.197]，场景分布重尾。
  - 调和插值相对最近邻 +0.014，CI [+0.003, +0.029]。
  - DEV（描述）：HFILL8_C1 0.175，调和插值 0.206。
- **结论：** 学到的补全平均而言优于经典几何补洞，但在 57 个场景上还达不到预注册的证据标准。按计划，这条主张目前不建议动用 final holdout。

## 2026-09-29 晚间：对照、数据与修正（均在 V11 出结果之前写定）

- **探索性补全试验**（只用 TRAIN，[说明](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/audit/exploratory_completion_pilot_train_only/README.md)）：
  - 调和插值补洞只略好于最近邻（ALL +0.006），Telea 更差；
  - 多尺度 3D 细化与"多采样看不到的像素"都没有改善补全，只有加宽有效。
- **[INPAINT-V1](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/PROTOCOL.md)**（18:55:58 锁定）：在 EVAL-V3 上比较"调和插值补洞 + V11 宽 carrier 补 FAR"与纯调和插值补洞，检验学到的补全能否胜过经典几何补洞。不训练，只用已封存的 V11 预测。
- **[EVAL-V4 数据](outputs/EXP-3D-RGBD-EVAL-V4-DATA/README.md)**：从 EVAL-V3 从未读取过的候选中得到 26 个有效场景（16 个 volume，309 MB），不运行模型。
- **[EVAL-V4 复现](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PROTOCOL.md)**（19:06:07 锁定）：由 [V11 出结果前写定的计划](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PLAN_BEFORE_V11_RESULTS.md)规定，在新场景上原样复现 V11 与 INPAINT-V1 的主对比。
- **[V12 协议](outputs/EXP-3D-RGBD-WIDTH-V12/PROTOCOL.md)**：hidden 64。只有 V11 的 COMPLETION_GAIN 点估计 > 0 时才运行；主评价在 EVAL-V4 上，对照为 V11 C1 已选中的 checkpoint。
- **[V11 Amendment 1](outputs/EXP-3D-RGBD-COMPLETION-V11/AMENDMENTS.md)**：
  - 问题：C1 的 seed 28 与 seed 30 在 DEV 检查点评价时，渲染颜色为 1.0000001（1 个 float32 最小单位），触发冻结的 [0, 1] 检查，runner 按规则停止。
  - 修正：新增的评价模块只把 1×10⁻⁶ 以内的超界截回 [0, 1]；这两个训练从头重跑，训练记录必须与失败的那次逐位相同。
- **[Replica 数据](outputs/EXP-3D-RGBD-REPLICA-DATA-V1/README.md)**（用户同意下载）：
  - 下载：只分段读取所需帧，共 105 MB。
  - 处理：裁剪到与训练数据相同的视场。
  - 检查：8 个场景全部通过几何检查，重投影误差的中位数为 0.03%–0.05%。
  - 用途：[使用计划](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PLAN_BEFORE_REPLICA_USE.md)规定它只被跨数据集迁移实验使用一次，而且排在 EVAL-V4 复现之后。

## EXP-3D-RGBD-DYNAMIC-WRITE-V2（2026-09-29，Core B 第二轮）

- [完整报告](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/PROTOCOL.md)、[V9 之前写定的计划](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/PLAN_BEFORE_V9_RESULTS.md)。
- **设计：** 冻结的 V9 C1 carrier（可变 3–7 视角训练，32³）加学习写入规则（DirectWriteRule，528 个参数，TRAIN72 上训练 3000 步）。
  - 按 V9 之前写定的计划，carrier 为 V9 C1（VIEWCOUNT_GAIN 点估计 +0.002 >0），非学习基线为 OFF。
- **DEV query AbsRel：** NO_STREAM 0.2575、OFF 0.2047、OFF_CLAMP3 0.2079、ALL 0.2060、FUSE 0.2054、COMPLETE 0.2047、ALL_UNTRAINED 0.2045、ALL_WRONG_SCENE 0.2052。
- **判定：**
  - BEYOND_BASELINE（OFF−ALL）−0.0013，CI [−0.0046, +0.0020]，`NOT_ESTABLISHED`。
  - SPECIFICITY（WRONG_SCENE−ALL）−0.0007，CI [−0.0014, −0.00003]，`HARMFUL`。
  - 预注册分支 **C**。
- **结论：**
  - 可变视角训练的价值在这里兑现：直接缓存 4 个新帧就让误差从 0.258 降到 0.205（+0.053，CI [+0.014, +0.101]）；V1 的固定 3 视角 carrier 缓存新帧反而变差。
  - 学到的快速权重写入没有额外作用；未训练的写入同样好，也不带场景信息。状态更新靠重新融合更多视角就足够。
  - 0.205 用了 7 个视角，不能直接与 3 视角重投影（0.220）比较。
- **事故说明：** 派生运行脚本时沿用了 V1 的日志路径，第一次合成预演覆盖了 V1 留在 /tmp 的收尾标准输出日志；V1 仓库内的记录完整，路径已修正。

## EXP-3D-RGBD-VIEWCOUNT-V9（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/PROTOCOL.md)、[V8 之前写定的计划](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/PLAN_BEFORE_V8_RESULTS.md)。
- **设计：** 单因素检验训练时的上下文视角数，32³，CONTEXT_DEPTH bounds，全程 CPU。
  - C0 固定 3 视角（V8 C1 配方）、C1 可变 3–7 视角（主方法）、C2 固定 7 视角（次要）。
  - DEV 评价与 checkpoint 选择都用标准 3 视角。
- **DEV query AbsRel（选中 checkpoint）：** C0 0.259、C1 0.257、C2 0.275；C0 与 V8 C1 逐位相同。
  - VIEWCOUNT_GAIN +0.002，CI [−0.008, +0.013]，`VIEWCOUNT_STATUS=NOT_ESTABLISHED`，预注册分支 B（C1 在全部场景上优于常数）。
  - FIXED7_GAP +0.018，CI [+0.001, +0.035]：总是用 7 视角训练，在 3 视角评价时更差。
  - 场景专属性（wrong-scene +0.366、shuffle +0.532）与多视角融合（+0.109）都成立。
- **不做选择的最后一步：** 0.269 / 0.277 / 0.288。
- **补充分析 2（几何基线，[结果](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/ADDENDUM2_RESULTS.md)）：** 三个变体都与 REPROJ_NN 无法区分；OBS2PLUS 仍约 0.374。
- **下一步：** 按 V9 之前写定的计划（VIEWCOUNT_GAIN 点估计 >0），Core B 第二轮与 V11 都用 V9 C1 的配方，Core B 第二轮的非学习基线取 OFF。

## EXP-3D-RGBD-GEOMETRIC-BASELINES-V1（2026-09-29）

- [报告](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/README.md)、[协议](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/PROTOCOL.md)、[结果](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/baseline_results.json)。
- **问题：** 已合格的 V7 C1 能否超过不需要学习的几何基线？
  - 基线只用与 carrier 相同的测试时输入：3 个上下文帧的测得深度与相机，以及 query 相机。
  - 不训练；所有基线预测在读取 GT 之前封存；carrier 指标重算的偏差为 0。
- **整体 AbsRel：**

  | 方法 | DEV | FRESH-V2 |
  |---|---|---|
  | 上下文深度重投影 + 最近邻填洞（主要基线） | 0.220 | 0.313 |
  | 重投影 + 常数填洞 | 0.288 | 0.384 |
  | 同网格同 bounds 的非学习融合 | 0.402 | 0.383 |
  | 混合（有重投影处用重投影，其余用 carrier） | 0.216 | 0.279 |
  | 学习式 carrier V7 C1 | 0.280 | 0.290 |

- **主对比：** DEV 上 carrier 相对重投影为 −0.061，CI [−0.130, +0.013]，NOT_DISTINGUISHABLE，预注册分支 **B**。
  - FRESH-V2 上为 +0.023，CI 跨零，但重投影在 13/17 个场景上更好。
  - carrier 显著优于同表示的非学习融合：DEV +0.122，FRESH +0.093，均为 ABOVE。
- **分区域：**
  - 在被上下文看到的区域，重投影远比 carrier 准：OBS2PLUS 为 0.105 对 0.348（DEV）、0.049 对 0.218（FRESH）。
  - 在完全没看到的区域，carrier 更好：OBS0 为 0.305 对 0.334、0.348 对 0.501。
- **结论：**
  - 学到的状态有价值（优于同表示的非学习融合），但整体只与"直接重投影"相当。
  - 它的优势在补全未观测区域；劣势是 16³ 表示丢失了观测区域的精确几何。
  - 逆向 JEPA 要在 RGB-D 设定下明确胜过简单几何，需要让状态在保留观测几何精度的同时，保住对未观测区域的补全能力。
- **后续分析（探索性）：**
  - [上限分析](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/audit/exploratory_ceiling/README.md)：被看到的区域用重投影、其余用 carrier 的理想组合为 0.193，对重投影 0.220 的差 CI 跨零。保留测量不是主要杠杆，瓶颈在补全质量。
  - [容量试验](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/audit/exploratory_capacity_pilot/README.md)（TRAIN24 训练、48 个未见场景）：hidden 32 相对 hidden 8，在离测量较远的像素上好 0.059 [+0.021, +0.098]；空洞卷积加大感受野反而更差。

## EXP-3D-RGBD-DYNAMIC-WRITE-V1（2026-09-29，Core B 第一轮）

- [完整报告](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V1/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V1/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V1/PROTOCOL.md)。
- **设计：**
  - 慢 carrier：已合格的 V7 C1，冻结，从不更新。
  - 学习对象：只训练项目原设计的写入规则 `DirectWriteRule`（528 个参数），在 TRAIN72 上训练 3000 步，不用 DEV 选择 checkpoint。
  - episode：一个上下文角色（3 个 RGB-D 视角），加上 4 个按帧号规则选出的 stream 帧。
  - 全程 CPU。
- **DEV query AbsRel：**

  | 策略 | 说明 | AbsRel |
  |---|---|---|
  | NO_STREAM | 只看上下文（静态 Core A 状态） | 0.280 |
  | OFF | 缓存 stream，不写入 | 0.322 |
  | OFF_CLAMP3 | 缓存 stream，视角计数截断为 3（非学习） | 0.254 |
  | ALL | 写入两个矩阵（主方法） | 0.260 |
  | FUSE | 只写 fuse 矩阵 | 0.258 |
  | COMPLETE | 只写 complete 矩阵 | 0.318 |
  | ALL_UNTRAINED | 写入规则未训练 | 0.322 |
  | ALL_WRONG_SCENE | 别的场景的快速权重 | 0.256 |

- **判定：**
  - WRITE_GAIN（OFF−ALL）+0.061，CI [+0.035, +0.085]，`WRITE_STATUS=SUPPORTED`。
  - 相对静态状态：+0.020，CI [−0.021, +0.063]，`NOT_ESTABLISHED`。
  - 相对计数截断：−0.006，CI [−0.022, +0.008]，`NOT_ESTABLISHED`。
  - 场景专属性：别的场景的快速权重只差 −0.004，CI [−0.021, +0.008]，`NOT_ESTABLISHED`。
  - 预注册分支 **B**。
- **结论：**
  - 学到的写入是一个通用修正：它修补 carrier 只见过 3 个视角造成的视角计数偏移，效果等同于非学习的计数截断，并没有写入场景专属的信息。
  - Core B 的核心主张在这一轮没有得到支持。
  - 缓存更多帧本身会让冻结的 carrier 变差（−0.041）。
- **下一步：** 按分支 B，用可变视角数训练静态 carrier（V9），再检验写入（Core B 第二轮）。
- **探索性试点（只用 TRAIN，存于 audit/）：** 试点 1、2 预判了这一结果。视角数试点显示，可变视角训练在留出场景上把 3 视角 AbsRel 从 0.366 降到 0.339。

## EXP-3D-RGBD-RESOLUTION-V8（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/PROTOCOL.md)、[V7 之前写定的计划](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/PLAN_BEFORE_V7_RESULTS.md)。
- **设计：** 单因素检验体素网格，全程 CPU。
  - 三个变体都是 V7 C1 配方（测得深度 bounds），只有网格不同：C0 16³、C1 32³、C2 24³（次要）。
  - 三种网格的 5150 个共享参数与初始化完全相同。
  - C0 与 V7 C1 逐位相同，DEV AbsRel 0.280380。
- **DEV query AbsRel（选中 checkpoint）：** C0 0.280、C1 0.259、C2 **0.244**。
  - RESOLUTION_GAIN +0.021，CI [−0.017, +0.052]，`RESOLUTION_STATUS=NOT_ESTABLISHED`，预注册分支 B。
  - C1 在全部场景上仍优于常数；场景专属性与多视角融合都成立。
- **选择偏差：** 不做 checkpoint 选择的最后一步为 0.330 / 0.269 / 0.279。
  - 16³ 在第 2000 步之后过拟合，DEV 选择让它显得更好；32³ 选中的步数更晚（3000–5000），偏差更小。
  - 不做选择时，32³ 比 16³ 好 0.061。
- **探索性诊断（只用 TRAIN）：** 非学习融合在测得深度 bounds 下，32³ 比 16³ 低 0.054；在冻结 bounds 下几乎没有差别。
- **下一步：** 按 V8 之前写定的计划，V9（训练视角数）使用 32³。

## EXP-3D-RGBD-FRESH-QUALIFICATION-V2（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V2/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V2/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V2/PROTOCOL.md)。
- **结果：`FRESH_QUALIFICATION_STATUS=QUALIFIED`。** V7 已封存的 RGB-D carrier（不重新训练、不重新选择）在 FRESH-V2 的 17 个独立场景（10 个 volume）上通过全部四个冻结 gate。
- **C1 query AbsRel：** 0.290，DEV 上为 0.280。C0（冻结 bounds）0.435；C2（裁 1%）0.279。
- **四个 gate：**
  - BOUNDS_GAIN +0.145，CI [+0.046, +0.240]，14/17 个场景改善。
  - C1 相对 TRAIN 拟合的常数：+0.300，CI [+0.192, +0.396]，15/17 个场景更好。这一项是本轮新增的 gate。
  - 场景专属性：wrong-scene +0.261，shuffle +0.490。
  - 多视角融合：+0.095，CI [+0.008, +0.179]，下界接近 0。
- **按 volume 聚类的 bootstrap（描述性）：** BOUNDS_GAIN +0.166（9/10 个 volume 为正）；C1 相对常数 +0.329（10/10 个 volume 为正）。
- **仍差于常数的两个场景：** ai_041_009（0.727 vs 0.669）与 ai_044_010（1.148 vs 0.935）。
- **统计：** 使用 V7 修订 1 的估计器。
- **意义：** Core A 的静态 carrier 在 RGB-D 轨道上首次通过资格验证，同时满足机制复现与绝对精度优于平凡基线。按项目规则，Core B（动态 TTT 写入）的下一阶段可以开启。
- **边界：**
  - FRESH-V2 来自官方 train 划分中从未被观察过的场景，不是官方 val/test；
  - 受保护的官方 final holdout 仍未打开；
  - 结论限于 RGB-D 轨道（测试时有上下文深度）；
  - RGB-only 轨道仍未超过常数。

## EXP-3D-RGBD-DEPTH-BOUNDS-V7（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/PROTOCOL.md)、[修订记录](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/AMENDMENTS.md)。
- **设计：** 单因素检验 bounds 规则，全程 CPU。
  - 三个变体都是 V6 的 C1 配方（V5 C1 模型与 loss、TRAIN72、6000 步），训练和评价都用各自的规则。
  - C0：冻结的相机射线先验规则。
  - C1：测得上下文深度与相机的外包盒。
  - C2：同 C1，但裁掉 1% 分位尾部（次要）。
- **DEV query AbsRel：** C0 0.459（复现 V6 C1，前几步 loss 逐位相同）、C1 **0.280**、C2 0.265。
  - BOUNDS_GAIN +0.179，CI [+0.084, +0.279]，7/8 个场景改善，LOSO 范围 [0.145, 0.205]，`BOUNDS_STATUS=SUPPORTED`，预注册分支 **A**。
  - 与常数深度相比：C1 在全部 8 个场景上都更好，平均好 +0.227，CI [+0.113, +0.342]，RAW AbsRel 标签为 ABOVE。这是项目第一次满足这一条。
  - 此前的 NO_HIT 场景 ai_009_001：在测得深度 bounds 下全部射线都命中，单场景收益 +0.418。
  - 场景专属性：wrong-scene +0.364，shuffle +0.467。
  - 多视角融合：+0.171。三项机制检查都 SUPPORTED。
  - 裁 1% 与不裁剪差别不显著：TRIM_GAIN −0.015，CI [−0.039, +0.009]。
- **选择偏差：** C1 三个 seed 选中的都是第 2000 步；不做选择的最后一步为 0.330，比选中值高约 0.05。C0、C2 的偏差较小（0.462、0.285）。所以必须在独立数据上验证。
- **修订 1：**
  - 冻结的 V2 估计器有一条断言，要求各变体的几何射线命中比例相同。这是 V2–V6 共用 bounds 的前提；V7 的因素恰恰是 bounds，这条断言在分析阶段报错。
  - 修订只删去这一条断言，写于查看任何 V7 指标之前，其余一字不改。
  - 有测试证明：命中比例相同时，修订前后结果逐位一致。
- **下一步（按 V7 出结果前写定的计划）：** 在 FRESH-V2（17 个独立场景）上对 V7 C1 做一次资格验证。V8（32³ 分辨率）推迟到资格验证之后。仍不开 Dynamic TTT。

## EXP-3D-RGBD-FRESH-V2-DATA（2026-09-29）

- [说明](outputs/EXP-3D-RGBD-FRESH-V2-DATA/README.md)、[数据锁](outputs/EXP-3D-RGBD-FRESH-V2-DATA/candidate_lock.json)、[清单](outputs/EXP-3D-RGBD-FRESH-V2-DATA/manifest_fresh_v2.json)。
- **内容：** 第二个独立资格队列，共 17 个场景，分布在 10 个 volume。
  - 来源：官方 train 划分中从未被观察过、也从未被任何实验使用过的场景，volume 与 TRAIN72 和 DEV 都不重叠。
  - 选择：名单在读取媒体之前按哈希顺序锁定；每个 volume 取前 2 个通过与模型无关的有效性检查的场景。
  - 下载 553 MB，没有运行任何模型。
- **原因：** 官方 val 划分已经没有可用的独立场景；FRESH-V1 只有 5 个场景，且已用过一次。
- **使用规则：** 只给第一个在 DEV 上达到分支 A 的方法做一次资格验证。资格 gate 比 FRESH-V1 多一项：C1 在全部 FRESH-V2 场景上显著优于常数。

## EXP-3D-RGBD-TRAIN-SCALE-V6（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-TRAIN-SCALE-V6/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-TRAIN-SCALE-V6/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-TRAIN-SCALE-V6/PROTOCOL.md)。
- **设计：** 单因素检验训练规模，全程 CPU。
  - C0、C1 都是冻结的 V5 C1 配方，每个 run 6000 步。唯一差别是训练场景 24 个对 72 个，新增的 48 个 TRAIN-X 与 DEV、FRESH 的 volume 都不重叠。
  - C2 是只用 RGB 的配方，在 72 个场景上训练。
- **DEV query AbsRel：** C0 0.492、C1 0.459（至今最好）、C2 0.602。
  - SCALE_GAIN +0.033，CI [−0.008, +0.084]，三个 seed 方向一致，`SCALE_STATUS=NOT_ESTABLISHED`，预注册分支 C。
  - 在 72 个场景下，测得深度相对只用 RGB 的收益为 +0.143，CI [+0.068, +0.210]。
  - 不做 checkpoint 选择的最后一步结果为 0.495 / 0.462 / 0.611，选择偏差很小。
- **训练集拟合与状态检查：**
  - 训练集拟合：C0 0.195，已明显过拟合；C1 0.301。
  - 场景专属性：wrong-scene +0.198，shuffle +0.295。
  - 多视角融合：成立。
- **与常数深度相比：**
  - 排除 NO_HIT 场景后，C1 为 0.382，常数为 0.493，好 +0.111，CI [+0.011, +0.222]。
  - 但只有 5/7 个场景不差于常数，未满足 75% 规则，所以仍为 NOT_DISTINGUISHABLE。
  - 拖后腿的是 NO_HIT 场景 ai_009_001，以及 ai_014_003、ai_021_002。
- **结论与下一步：**
  - 数据规模有稳定但温和的作用；剩下的主要差距来自 bounds（NO_HIT 与截断）。
  - 下一步 V7 单因素检验由测得深度确定的 bounds。
- **归档说明：** 运行期间有几个与 V6 无关的新文件进入了仓库（非学习融合、深度 bounds 模块），V6 的源码快照因此也收录了它们。这不影响 V6 的锁与结果。

## EXP-3D-RGBD-NONLEARNED-FUSION-V1（2026-09-29）

- [报告](docs/experiments/EXP-3D-RGBD-NONLEARNED-FUSION-V1/README.md)、[预注册](docs/experiments/EXP-3D-RGBD-NONLEARNED-FUSION-V1/preregistration.json)。
- **设计：** 不经学习的 RGB-D 融合对照，全程 CPU。
  - 做法：上下文深度直接体素化进同一个 16³ 网格和同一 bounds，用同一个固定 renderer 读出。
  - 参数：只有 4 个标量，在 TRAIN24 的 query 上做网格搜索拟合，选中 a=16、b=4、c=−2、k=1。
  - 评价：DEV 与 FRESH-V1 都先封存全部状态，再读取 GT。
- **平均 AbsRel：**
  - DEV：非学习融合 0.596，学到的 C1 0.503，C0 0.617，常数 0.507。
  - FRESH-V1：非学习融合 0.528，C1 0.472，C0 0.479，常数 0.493。
- **结论：**
  - 把测得深度直接体素化，效果比常数深度还差。
  - 学到的 C1 在 DEV 上显著优于非学习融合（+0.093，CI [+0.006, +0.182]），在 FRESH 上方向相同但不显著；在 FRESH 上，只用 RGB 的 C0 显著优于非学习融合（+0.049，5/5 个场景）。
  - 学习本身有价值，但“16³ 网格 + 当前 renderer”这一环限制了新视角的深度质量。
  - 旁证：该 renderer 的深度读出未归一化，不透明度为 0.82 时深度按比例偏近约 18%。

## EXP-3D-RGBD-FRESH-QUALIFICATION-V1（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V1/README.md)、[协议](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V1/PROTOCOL.md)、[数据锁](outputs/EXP-3D-RGBD-FRESH-SCALE-DATA-V1/candidate_lock.json)。
- **第一次独立资格验证**：对 V5 已封存的 checkpoint 做评价，不重新训练、不重新选择；全程 CPU。
- **独立队列 FRESH-V1**：
  - 来源：官方 Hypersim val split，不在历史上 365 个训练场景之内，从未被观察过。
  - 独立性：与 V2–V5 的 volume 和源资产都不重叠，每个 volume 一个场景；名单在读取媒体之前冻结。
  - 规模：9 个候选中 4 个未通过与模型无关的源数据几何校验；修订 1 没有找到替代。最终为 5 个场景，低于预设的最低 6 个。
  - 受保护的 final holdout 未打开。
- **结果 `NOT_QUALIFIED`**：
  - 深度收益没有复现：FRESH 上 C0 0.479、C1 0.472，DEPTH_GAIN +0.006，CI [−0.112, +0.116]；DEV 上为 +0.114。逐场景看，2 个场景 C1 明显更好，2 个明显更差。
  - 场景专属性复现：wrong-scene 损伤 +0.131，shuffle +0.197，CI 均 >0。
  - 多视角融合复现：3 个视角比 1 个好 +0.200，CI 均 >0。
  - 仍未显著优于 TRAIN 拟合的常数：+0.021，CI 跨零。
- **解读**：
  - 按 DEV 挑选 checkpoint，加上 DEV 被反复使用，使 V5 的 DEV 收益偏乐观。
  - 测得深度能让状态针对具体场景并融合多视角，但在新场景上没有带来稳定的精度优势，24 个训练场景学到的东西泛化有限。
  - Core A 未通过资格验证，Core B 保持关闭。
- **新数据**（EXP-3D-RGBD-FRESH-SCALE-DATA-V1）：同时准备了 48 个通过校验的 TRAIN-X 场景（与 DEV、FRESH 的 volume 都不重叠），供 V6 训练规模实验使用。每个场景约 7.8 MB，只按 Range 取所需文件。

## EXP-3D-RGBD-EVIDENCE-CARRIER-V5（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/README.md)、[终端摘要](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/STATUS.md)、[协议](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/PROTOCOL.md)、[修订记录](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/AMENDMENTS.md)、[复现命令](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/commands.sh)。
- RGB-D 轨道的单因素机制诊断，Claude 起草、训练前冻结。
  - C0：V4 C0 的稠密 16³ carrier，只读 RGB+相机。
  - C1：C0 加一条零初始化的测得深度旁路，共 25 个参数。每个上下文视角的测距给出体素的表面与自由空间似然；旁路放在 ≥2 视角支持门之后。
  - C2：C1 去掉表面监督。
  - 数据边界：上下文深度只经只含上下文帧的 RGBD loader 进入状态构建；query 深度只在全部状态 seal 后评价，finalize 独立复核。
- DEV query AbsRel：C0 0.6174、C1 0.5030、C2 0.5258。
  - DEPTH_GAIN +0.1144，95% CI [+0.0649, +0.1664]；场景改善/持平/变差为 7/1/0（持平的是 NO_HIT 场景）；LOSO [0.096, 0.131]；三个 seed 分别 +0.111/+0.119/+0.114。`DEPTH_STATUS=SUPPORTED`。
- 首次同时通过三项冻结 gate：
  - 深度收益，见上。
  - 场景专属性：C1 wrong-scene 损伤 +0.169（CI [+0.047, +0.302]），shuffle +0.216（CI [+0.107, +0.339]），`SCENE_SPECIFICITY_STATUS=SUPPORTED`。
  - 多视角融合：3 个 RGB-D 视角相对 1 个（anchor）+0.152（CI [+0.074, +0.238]），`STATIC_DEV_STATUS=SUPPORTED`。
- 描述性结果：
  - 区域收益：OBS0 +0.103、OBS1 +0.104、OBS2PLUS +0.148，CI 均 >0。
  - 训练集深度 AbsRel：C0 0.370，C1 0.268。
  - 学到的深度宽度 σ≈0.76–0.82 个体素边长。
  - 表面监督的额外收益：+0.023，CI 跨零。
- 与无几何常数相比（预注册描述，同时是分支条件）：
  - 全部 8 个场景：C1 0.503，常数约 0.507（+0.004，CI [−0.130, +0.108]）。
  - 排除 NO_HIT 场景 ai_009_001：C1 0.432，常数约 0.493（+0.061，CI [−0.0035, +0.134]）。
  - 两种口径都是 NOT_DISTINGUISHABLE。
- 预注册分支：冻结规则与修订 1 规则下都是 **B**。测得深度让 carrier 形成了场景专属、多视角融合的状态，但绝对精度还没有显著超过平凡基线。
- 9 个训练无故障一次完成；1440 条预测逐行重算一致，统计字节级复现，30997 个旧文件未变，887 个测试通过。`FINAL_STATIC_STATUS=NOT_ESTABLISHED`（DEV 已曝光），`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`。

## EXP-3D-READOUT-REFERENCE-REANALYSIS-V1（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-READOUT-REFERENCE-REANALYSIS-V1/README.md)、[修订记录](docs/experiments/EXP-3D-READOUT-REFERENCE-REANALYSIS-V1/AMENDMENTS.md)、[预注册](docs/experiments/EXP-3D-READOUT-REFERENCE-REANALYSIS-V1/preregistration.json)。
- 预注册事后二次分析，Claude 起草，在读取任何 DEV GT 之前冻结。不训练、不加载模型，只读 V2–V4 全部 9 个 carrier × 3 seeds 的已封存 DEV 预测（1728 个，逐个核对哈希）。
  - 对照：两个只由 TRAIN primary-query GT 拟合的无几何常数。AbsRel 最优常数 2.37 m（1/t 加权中位数），以及中位数 4.20 m。
  - 读出方式：RAW、OPACITY_NORMALIZED、MEDIAN_SCALED（给定 GT 尺度的形状诊断）。
- 结果为 `NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE`：
  - AbsRel：常数 2.37 m 的 DEV query AbsRel 约 0.507，9 个 carrier 为 0.61–0.65，RAW AbsRel 全部显著更差（BELOW）。
  - delta1：carrier 为 0.09–0.14，常数为 0.26。
  - 形状：给定 GT 尺度后，carrier 的深度形状也不如平面常数（MEDIAN_SCALED AbsRel 全部 BELOW）。
  - opacity 归一化对任何 carrier 都没有显著作用。
  - RAW 重算与封存指标逐个一致。
- Amendment 1，在任何指标计算之前作出：DEV 场景 ai_009_001 的 query 射线全部落在 context bounds 之外（NO_HIT），216 个视角渲染全零，MEDIAN_SCALED 无定义，改为不缩放。排除该场景的次要敏感性分析中，V3 C1 与 V4 三个变体变为不可区分，但仍没有 carrier 优于常数。
- 结论：RGB-only learned carrier 在 DEV 上没有携带可迁移的几何。V2–V4 的主对比都是在低于平凡基线的模型之间比较。

## EXP-3D-PLANE-SWEEP-CARRIER-V4（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/README.md)、[终端摘要](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/STATUS.md)、[协议](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/PROTOCOL.md)、[复现命令](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/commands.sh)（其中第 3 步“V3 protocol text”应为 V4，已封存未改）。
- 单因素机制诊断，Claude 起草、训练前冻结。
  - C0：V3 的稠密 16³ carrier。
  - C1：C0 加一条零初始化的 plane-sweep 旁路，共 49 个参数。旁路使用不需要学习匹配的光度代价：32×40、32 个逆深度平面、跨视角总体方差；经深度 softmax 得到每个体素的表面与自由空间似然。
  - C2：C1 去掉表面监督。
  - 训练 2000 步、前期加密 checkpoint，其余与 V3 相同。
- DEV query AbsRel：C0 0.6174、C1 0.6102、C2 0.6184。
  - 主对比：SWEEP_GAIN +0.0072，95% CI [−0.0041, +0.0168]，三个 seed 方向一致（+0.0096/+0.0070/+0.0051），`SWEEP_STATUS=NOT_ESTABLISHED`。
  - 场景专属性：C1 wrong-scene 损伤 +0.029、shuffle +0.027，CI 跨零，`SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED`。
  - `STATIC_DEV_STATUS=NOT_ESTABLISHED`；预注册分支 C。
  - 学到的温度 τ≈0.007（初值 0.01）。
- 区域诊断（seal 后，描述性）：OBS1 +0.0169（CI [+0.0005, +0.0344]），OBS2PLUS +0.0403（CI [+0.0038, +0.0871]），OBS0 −0.0007。几何线索正好在多视角可观测区域起作用，全图收益被不可观测区域稀释。这是 V2–V4 中第一个正向的机制信号。
- 事后审查与 TRAIN-only 探索诊断，均不属于冻结协议，也未打开 DEV：
  - 独立审查没有发现正确性 bug；u/v 方向查找与 CUDA 路径另行验证正确。
  - 原始光度 plane-sweep 在真实 3 视角上下文上几乎不含深度信息：argmax AbsRel 0.518，常数为 0.557。更强的 128×160 窗口 ZNCC 也只到 0.49。视差不是原因（中位基线 1.26 m）。
  - 冻结的深度先验 far=5.67 m 只由 3 个场景得到，而 24 个 TRAIN 场景的 q99 为 14 m；截断代价 ≤0.06 AbsRel。
  - TRAIN 场景 ai_003_001 的 anchor 帧整幅在 3.8 cm 处，是退化帧。
- 9 个训练无故障一次完成；1440 条预测逐行重算一致，统计字节级复现，27888 个旧文件未变，测试通过。`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`。

## EXP-3D-DENSE-EVIDENCE-CARRIER-V3（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/README.md)、[终端摘要](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/STATUS.md)、[协议](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/PROTOCOL.md)、[复现命令](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/commands.sh)。
- 单因素机制诊断（Claude起草、训练前冻结）：同一16³ learned carrier，多视角证据候选从128个（V2配方，16³下只覆盖3.1%体素）加密到全部4096个；参数量5125、同seed初始参数、数据与预算全部相同。C2为稠密且无surface的次要变体。沿用V2锁定的24 TRAIN / 8 DEV（DEV已在V2评价过，本轮不是资格轮）。
- DEV query AbsRel：C0稀疏0.6301、C1稠密0.6200、C2稠密无surface 0.6348。DENSITY_GAIN +0.0101，95% CI [−0.0142, +0.0394]，`DENSITY_STATUS=NOT_ESTABLISHED`；C1 wrong-scene损伤+0.0214（CI [−0.061, +0.119]），`SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED`；`STATIC_DEV_STATUS=NOT_ESTABLISHED`。预注册解读分支C：排除候选稀疏为主因。
- 训练集深度AbsRel（末100步）稀疏0.490→稠密0.339，但DEV只改善约1.6%；三个seed的稠密模型均选中第一个checkpoint（500步），之后DEV query与DEV context误差都上升：稠密证据通路在训练场景上被利用，学到的却是不能迁移的场景记忆。
- 区域诊断：稠密版在OBS0显著改善（+0.0157，CI [+0.0025, +0.0285]），在OBS2PLUS略差（−0.0186）。稠密下表面监督收益C2−C1 = +0.0147，CI [−0.0030, +0.0358]。
- 探索性几何诊断（仅相机与bounds规则）：稀疏128个候选中平均只有37（TRAIN）/47（DEV）个获得≥2视角支持，稠密版约1200–1440个体素获得支持（约30倍）；bounds盒内约51–53%体素没有任何上下文视角可见。
- 9个训练无故障一次完成；1440条预测逐行重算一致，统计字节级复现，25087个旧文件未变；860测试通过、1跳过。`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`。

## EXP-3D-GEOMETRY-AWARE-CARRIER-V2（2026-09-29）

- [完整报告](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/README.md)、[终端摘要](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/STATUS.md)、[接手修订](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/TAKEOVER_AMENDMENTS.md)、[复现命令](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/commands.sh)。
- 首次训练16³ learned carrier：C0 BASELINE16 vs C1 SURFACE16（仅加0.1×surface loss），C2 FREE_SURFACE16为预注册次要消融；24 TRAIN / 8 DEV（均为历史曝光场景），3 seeds × 3000步，按同一DEV规则独立选checkpoint；测试输入仅RGB+相机。
- DEV query AbsRel：C0 0.6495、C1 0.6300、C2 0.6302。SURFACE_GAIN +0.0195，95% CI [−0.0253, +0.0768]，场景改善/持平/变差3/1/4，单场景贡献68%；3个seed方向均为正但CI均跨零。`SURFACE_TRAINING_STATUS=NOT_ESTABLISHED`。C1−C2 = −0.0002，free-space无额外价值。
- Full context相对anchor仅+0.0143（CI跨零）；wrong-scene与spatial shuffle不造成损伤（C1：−0.0094 / +0.0016），只有清空状态造成+0.37，状态近似场景无关的先验。`STATIC_DEV_STATUS=NOT_ESTABLISHED`。OBS2PLUS上C1略差（−0.0152），未复现direct-state的surface机制特征。
- 训练集深度AbsRel约0.48、DEV约0.63（同类场景direct-state为0.31–0.36）；C0的DEV误差随训练变差。carrier沿用128个证据候选点，在16³只覆盖约3%体素且分布不均，多视角证据只经稀疏候选进入状态，是下一轮首要检验的瓶颈。
- 协议CASE A：fresh未打开（独立性未解决），`FINAL_STATIC_STATUS=NOT_ESTABLISHED`，`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`。853测试通过、1跳过；1440条预测逐行重算一致，统计字节级复现，13989个旧文件未变。
- 封存前诊断：本机0–7号P核会引发numpy原生段错误（不绑核12次3次崩溃，绑定8–23号核0次），此后所有计算绑定8–23号核。

## EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1（2026-09-28）

- [完整报告](docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1/README.md)、[终端摘要](docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1/STATUS.md)、[复现命令](docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1/commands.sh)。
- 17个已曝光开发场景，16³ / 64 samples / GT-free bounds / 10k步direct-state；S0基线复现通过（query AbsRel 0.361556，旧值0.361480）。
- S1 free-space 0.357400、S2 surface 0.313772、S3 free+surface（预定主配置）0.312183；S0−S3 = 0.049372，95% CI [0.023642, 0.076404]，缩小基线与query-oracle（0.148045）差距约23.1%。`GEOMETRY_SUPERVISION_STATUS=SURFACE_DOMINANT`，S2→S3额外收益CI跨零。
- OBS0/OBS1/OBS2PLUS像素占比0.297/0.275/0.428，`OBSERVABILITY_STATUS=MIXED`；surface监督在OBS2PLUS也有改善。
- 仅为direct-state、已曝光场景上的归因证据，不是learned carrier或独立泛化结果；未训练新carrier，未开holdout，`DYNAMIC_TTT_RUN=false`。

## EXP-3D-16G-OPTIMIZATION-BOUNDS-V1（2026-09-28）

- [完整报告](docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/README.md)、[终端摘要](docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/STATUS.md)、[复现命令](docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/commands.sh)。
- 16³ / 64 samples RGB+D direct-state，1k/3k/10k预算与CURRENT/GT-free两种bounds，153条轨迹、782个状态。
- GT-free 10k context-only query AbsRel 0.361480；1k→10k收益0.002255，CI [−0.0144, 0.0202]跨零；两种bounds差异CI跨零。`OPTIMIZATION_STATUS=INCONCLUSIVE`，`BOUNDS_STATUS=NO_CLEAR_DIFFERENCE`。
- GT-free query-oracle 0.148045，context→oracle差距0.213；RGB-only 0.615155。`CONTEXT_INFERENCE_BOTTLENECK=true`，`CARRIER_TRAINING_READINESS=PROMISING_BUT_NOT_READY`。
- 17个已曝光场景上的归因，不是独立确认；未训练新carrier，`DYNAMIC_TTT_RUN=false`。

## EXP-3D-DIRECT-STATE-CAPACITY-V1（2026-09-27）

- [完整报告](docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1/README.md)、[终端摘要](docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1/STATUS.md)、[复现命令](docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1/commands.sh)。
- 17个已曝光开发场景、425个状态优化：冻结carrier 0.679826；RGB+D direct 8³ 0.460804、16³ 0.402736、32³ 0.389382；RGB-only 8³ 0.609175；query-oracle 8³ 0.289314。
- 8³→16³收益0.058068，CI [0.0287, 0.0918]；16³→32³收益0.013353不稳定；128 samples无收益。`STATE_CAPACITY_STATUS=RESOLUTION_LIMITED`。
- 全部状态的最佳checkpoint落在预算边界，表示能力瓶颈未由有限优化确立；未训练新carrier，`DYNAMIC_TTT_RUN=false`。

## EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1（2026-09-28）

- [完整报告](docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/README.md)、[终端摘要](docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/STATUS.md)、[复现命令](docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/commands.sh)。
- 旧987文件封存且统计精确复算；旧7场景标为REDESIGN_DEV_EXPOSED。新锁12DEV＋12HOLDOUT，几何合格分别10/8，不替换；holdout历史曝光无法排除，独立性未解决。
- 固定8³/128/旧B-final完成R0、oracle-volume、oracle-support、联合四组，17个开发场景、680行。
- Volume-only full-context gain +0.115349 [0.014210,0.217646]；绝对full-context改善仅0.034169且CI跨零，83.8%相对增幅是anchor变差的代数贡献。
- 联合gain +0.089622 [−0.026782,0.216702]；query surface support66.22%未达75%充分性门槛。SUPPORT_BOTTLENECK_STATUS=INCONCLUSIVE，不能声称理想支持都无效或network是唯一原因。
- 按冻结规则停止：GT-free实景实验、matched retraining、final holdout均未运行；STATIC_STATE_STATUS=NOT_ESTABLISHED；DYNAMIC_TTT_RUN=false。
- 669测试通过，1可选历史checkpoint测试跳过；680预测文件重算全部主指标完全一致；报告、raw、锁、图、命令及完整性证据齐全。

## EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1（2026-09-27）

- [完整报告](docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1/README.md)、[状态](docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1/STATUS.md)、[复现命令](docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1/commands.sh)。
- 固定1000A+600B B-final；仅static，旧90条精确复现，ai003保留。
- 锁定8个独立源资产候选，7个满足先验质量/几何规则，全部未参与此checkpoint训练；未打开保护集。供应商跨资产复用无法完全排除。
- 未见场景full-context相对anchor的AbsRel收益−0.189671，scene-bootstrap 95% CI [−0.291047, −0.091291]；wrong-scene damage CI跨0。
- Residual 308参数，30,000步训练，训练侧验证选择7,500步；平均A/B未见AbsRel由0.797229变为0.852789，无通用改善证据。
- 已发现source-black、观察支持与查询体积覆盖限制，不能直接归因为过拟合或固定decoder。`STATIC_STATE_STATUS=NOT_ESTABLISHED`；`DYNAMIC_TTT_RUN=false`。
- 610测试通过，1历史可选checkpoint测试跳过。原始记录、锁定记录、GT日志、审计、曲线和artifact hashes保留；报告raw可离线精确重算统计。

## EXP-2D-followup-20260927

- **状态**：POST_HOC_DISCOVERY_ONLY；保留V2全部冻结产物。
- **新增诊断**：仅用其余11序列选择全局16轨迹，12折都选ALL/B；CycleGate−该折外全局基线为−0.208623个百分点。f_cycle改善的168条候选中33条任务收益为负。
- **边界**：不改V2主要比较，不构成独立确认，不据此断言所有可见信号都无效。
- **验证**：2项新增测试通过；63份V2文件哈希无变化；重生成逐字节一致。
- **报告**：[结果与资源审计](docs/experiments/EXP-2D-followup-20260927/README.md)。

## EXP-2D-opportunity-selector-v2-20260926T015728+0800

- **状态**：`BLOCKED_NO_INDEPENDENT_DATA`；已完成实现、旧 raw 后处理、discovery nested LOSO、全套测试与审计。
- **旧 raw 诊断**：hindsight-global16=A/ALL；dynamic−hindsight-global16=+0.205456 个百分点，重采样/删一均重新求全局winner。
- **Discovery**：CycleGate 外层 OOF ΔJ=+0.124429 个百分点，GateOnly=+0.115612；Cycle−Gate 95%区间跨0。只属 discovery 过程评估。
- **独立数据**：VOST片段缺原视频映射；FBMS原视频身份及mask编码不合格。未打开旧保护集，独立序列0，未生成确认结果。
- **验证**：500 passed、1 skipped（重试）；原生崩溃日志保留，674项V1产物未变。
- **产物**：[完整报告](docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md)；协议/锁/raw/统计/复现命令均保存在独立目录。

这里是唯一的人类可读实验索引。机器可读的细节放在各 artifact bundle 和本地 `outputs/` 中。

## EXP-2D-20260926-opportunity-selector-v1

- **状态**：已完成锁定验证与独立审计；按预注册门槛 H1=STRONG、H2=NOT_ESTABLISHED。
- **主评价**：448 输入、真实32×32 DINO patch、原始像素空间对象 J；16个历史内部validation序列、32 pairs。不是新holdout或官方DAVIS J&F。
- **结果**：OFF J=55.2677%；dynamic oracle=55.5129%（+0.2451个百分点，sequence-bootstrap95%CI[+0.1173,+0.3934]）；visible selector=55.2207%（−0.0470个百分点）。
- **解释**：小幅答案可见机会分布于14个序列；非恒定轨迹机会比constant oracle多0.0750个百分点（不证明顺序因果机制）。部署选择器尚未获得净收益，29次写入有16次有害。
- **验证**：458 tests passed、1项缺失历史checkpoint明确skip；1792 raw rows完整、输入hash和封存边界通过审计，全部分析可逐字节重建。reserve及official val保持封存。
- **artifact**：[实验报告](docs/experiments/EXP-2D-20260926-opportunity-selector-v1/README.md)。

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

## 2026-09-27 三维小规模工程训练

已完成3个train场景上的100步静态+30步写入训练，生成新的 `mcss.dynamic.v1` 权重。严格重载与写入重放通过；全套测试561 passed、1 skipped。一个场景因固定体积缺乏多视图候选支持而全部写入梯度为零，保留负例；当前仅工程可用，未取得独立静态/泛化资格。

详见 [训练报告](docs/experiments/EXP-3D-20260927-small-training-v1/README.md) 和 [权重索引](docs/experiments/EXP-3D-20260927-small-training-v1/checkpoint_index.json)。

## EXP-3D-20260927-centered-support-v1

第三场景零写入支持已通过统一锚点中心体积修复：支持点 0→4，B 阶段非零 writer 梯度 0/10→10/10。100+30 步同种子重训、checkpoint 审计和 573 passed/1 skipped 全套测试通过；最终 TRAIN OFF 场景均值 AbsRel 0.919→1.229，未获得质量提升。旧实验和权重保留，新模式显式 opt-in，不构成独立泛化或写入收益证据。

[报告与复现](docs/experiments/EXP-3D-20260927-centered-support-v1/README.md)。

## EXP-3D-20260927-centered-training-1000a-600b-v1

2026-09-27 已完成 [1000A＋600B 三维训练](docs/experiments/EXP-3D-20260927-centered-training-1000a-600b-v1/README.md)：耗时112秒，TRAIN OFF AbsRel 为 A结束0.287、B结束0.332（短程1.229）；B相对A回退，未验证写入收益或泛化。 两阶段权重均保留，三个场景各200/200步非零writer梯度，15项回归测试及独立审计通过。

## EXP-3D-20260927-trained-state-history-v1

2026-09-27 [已训练权重机制补测](docs/experiments/EXP-3D-20260927-trained-state-history-v1/README.md)：同训练场景留出帧上，写入oracle AbsRel收益约0.000756，但有益历史动作翻转0/3、history额外空间0；静态context A AbsRel2.214，资格仍未建立。原训练query的1次翻转单列，不当独立证据。 581 passed、1 skipped；90条static和144条dynamic raw完整，R-Residual缺已训练head，自然历史及P0/P1/P2按原资格门槛跳过。

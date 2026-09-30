# 当前状态（2026-09-30）

2026-09-30 08:3x：[V16](outputs/EXP-3D-RGBD-VOLUME-DISJOINT-V16/README.md)（陌生 volume 上的补全优势，分支 D）。
- **结果：** 在与训练集 volume 不相交的 FRESH-V2 上，V11 C1 相对调和插值的补全优势平均 +0.185，但 8 好 6 差、CI 跨零；与共享 volume 的 EVAL-V3/V4 之间没有可检测的差距（−0.114，CI [−0.550, +0.177]）。
- **意义：** V15 事后分析的"熟悉度"解释没有得到证实，只能作为假设；补全优势在场景之间极不一致。

2026-09-30 08:1x：[V15](docs/experiments/EXP-3D-RGBD-WIDTH-DATA-V15/README.md)（127 个训练场景上宽度 32→64，分支 C）。
- **2×2（FAR）：** 宽度 32：72 个场景 0.421，127 个场景 0.443；宽度 64：0.415、0.446。127 个场景上加宽没有改善补全（−0.003，CI 跨零），交互作用 +0.009（方向与容量不足假设相反）。
- **完整性：** C0 与 V14 C1 逐位相同；C1 与 C0 共享数据流。
- **意义：** 宽度、数据量以及两者的组合都没有改善补全；CPU 上的规模杠杆已全部检验。
- **事后分析：** 数据带来的补全变化与评价场景所在 volume 的训练曝光变化相关（按 volume，曝光上升组比下降组好 +0.055，CI [+0.013, +0.099]），提示补全可能部分来自对同一 volume 的熟悉；V16 未能证实。

2026-09-30 03:5x：[V14](docs/experiments/EXP-3D-RGBD-DATA-SCALE-V14/README.md)（只增加训练场景，72→127，分支 C）。
- **结果：** C1 在被看到的区域明显更好（NEAR 0.288→0.246），但补全略差（FAR 0.421→0.443，DATA_GAIN −0.022，CI 上界 +0.001）；组合方法因此显著变差（−0.011，`HARMFUL`），与调和插值补洞打平（0.289 对 0.289）。
- **完整性：** C0 与 V11 C1 逐位相同（训练记录、DEV 曲线、选中的 checkpoint、648 个 EVAL 预测）。
- **意义：** 在 CPU 可及的规模内，宽度（V11/V12）与数据量（V14）都不是补全的杠杆；学到的状态仍与经典几何补洞持平。
- **事后分析：** 在相同训练步数下，更多数据的模型在每一个步数上补全都更差、被看到的区域都更好，说明容量固定时数据把模型推向重建而非补全。

2026-09-30 01:2x：[V13](outputs/EXP-3D-RGBD-VOLUME-V13/README.md)（状态体积之外的弃权回退，分支 B）。
- **结果：** 在新数据 [EVAL-FAR](outputs/EXP-3D-RGBD-EVAL-FAR-DATA/README.md)（79 个场景的远处 query）上，把低不透明度像素改用调和插值补洞，组合方法从 0.256 降到 0.246（+0.011，CI 下界 +0.005，43/31/5 个场景，`SUPPORTED`）。
- **但是：** 相对调和插值补洞本身仍只是数值领先（+0.016，CI 跨零，33 好 45 差）。
- **局限：** 远处 query 的表面仍有 93% 在体积内，Replica 式的低覆盖没有被检验到。
- **下一步：** 统计可用于扩大训练集的 Hypersim 场景，写定数据规模实验的计划（CPU）。

2026-09-30 凌晨：[V12](docs/experiments/EXP-3D-RGBD-WIDTH-V12/README.md)（宽度 64，分支 C）与 [Replica 跨数据集迁移](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/README.md)（分支 C）。
- **V12：** 宽度 32→64 在 EVAL-V4 上对补全只有 +0.004，CI 跨零。CPU 上可做的容量扩展到此为止，都没有见效。
- **Replica：** 冻结的方法在 Replica 上显著差于经典几何补洞（−0.225），carrier 甚至差于常数深度。
  - 事后诊断：一半以上的 query 表面落在状态的体积之外；即使在体积内，精度也明显下降。
  - 学到的状态没有跨数据集迁移。
- **本阶段总结：** Core A 在 Hypersim 上通过了资格验证，并与几何基线大体持平；组合方法在 Hypersim 上数值领先但未确立，在 Replica 上失败；Core B 两轮都未支持。受保护的 final holdout 未打开，按计划也不建议为补全这条主张打开。

2026-09-29 夜间：[V10](docs/experiments/EXP-3D-RGB-PRIOR-BOUNDS-V10/README.md)（RGB-only，分支 B）与 [EVAL-V4 复现](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/README.md)（分支 B）。
- **V10：** 改用 TRAIN72 先验确定 bounds，RGB-only 的 DEV 误差从 0.603 降到 0.461（+0.142，CI 下界 +0.022，7/8 个场景），但 RGB-only 状态仍与常数无法区分，也不带场景信息。
- **EVAL-V4 复现：** 在 26 个新场景上，组合方法（0.378）相对调和插值补洞（0.388）仍只是数值领先（+0.010，CI 跨零，12 好 14 差），结论与 EVAL-V3 一致。
- **进行中：**
  - [V12](outputs/EXP-3D-RGBD-WIDTH-V12/PROTOCOL.md)（hidden 64）。宽度 32 的对照与 V11 C1 逐位相同。
  - 之后是 [Replica 跨数据集迁移](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PLAN_BEFORE_REPLICA_USE.md)。

2026-09-29 晚间：[V11](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/README.md)（分支 C）与 [经典补洞对照 INPAINT-V1](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/README.md)（分支 B），全程 CPU。
- **V11：** 加宽 carrier（8→32）没有稳定改善看不到区域的补全（EVAL-V3 +0.005，CI 跨零）。V11 因浮点舍入导致的颜色越界做了 Amendment 1，重跑与失败的那次逐位相同。
- **INPAINT-V1：** "调和插值补洞 + 宽 carrier 补看不到的区域"在 EVAL-V3 上 AbsRel 0.2325，是目前最低的；但相对纯调和插值（0.244）的优势 +0.012，CI 跨零，未达到预注册标准。
- **进行中：**
  - [V10](outputs/EXP-3D-RGB-PRIOR-BOUNDS-V10/PROTOCOL.md)（RGB-only）与 [V12](outputs/EXP-3D-RGBD-WIDTH-V12/PROTOCOL.md)（hidden 64，按计划因 V11 补全增益点估计 > 0 而运行）并行训练；
  - 之后依次是 [EVAL-V4 复现](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PROTOCOL.md) 与 [Replica 跨数据集迁移](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/PLAN_BEFORE_REPLICA_USE.md)（用户已同意下载 Replica，[数据](outputs/EXP-3D-RGBD-REPLICA-DATA-V1/README.md)已准备好）。

2026-09-29：[Core B 第二轮](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/README.md)（V9 C1 carrier 上的学习写入），全程 CPU，预注册分支 C。
- 结果：直接缓存新帧让 DEV AbsRel 从 0.258 降到 0.205；学习写入没有任何额外收益（−0.001，CI 跨零），也不带场景信息。
- 结论：在对视角数稳健的 carrier 上，Core B 原设计的快速权重写入是多余的，动态更新靠重新融合更多视角即可。Core B 的核心主张未得到支持。
- 进行中：[V11](outputs/EXP-3D-RGBD-COMPLETION-V11/PROTOCOL.md)（加宽 carrier + 保留测量，EVAL-V3 57 个场景）。启动时遗漏了 checkpoints 目录，6 个训练在开始前失败，已按运行器规则存档（[说明](outputs/EXP-3D-RGBD-COMPLETION-V11/audit/failed_attempts/LAUNCH_INCIDENT.md)）并重新启动；窄版前 300 步与 V9 C1 逐位相同。

2026-09-29：[训练视角数 V9](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/README.md)，全程 CPU，预注册分支 B。
- 结果：DEV AbsRel 固定 3 视角 0.259、可变 3–7 视角 0.257、固定 7 视角 0.275；VIEWCOUNT_GAIN +0.002，CI 跨零。可变视角训练在标准 3 视角上没有收益，它的价值要在 7 视角的 Core B 场景中检验。
- 与几何基线（补充分析 2）：仍与直接重投影无法区分。
- 下一步：Core B 第二轮（V9 C1 carrier，非学习基线 OFF），随后 V11（加宽 carrier + 保留测量，EVAL-V3 评价）。

2026-09-29：三项探索性诊断（不训练新模型，或只用 TRAIN；不产生新的 DEV 评价）。
- [bounds 截断](outputs/EXP-3D-RGB-PRIOR-BOUNDS-DIAG/README.md)：V2–V6 的 RGB-only 实验所用的冻结先验（3 个场景拟合，far 5.67 m）把 TRAIN72 上 29.7% 的 query 表面留在盒外；TRAIN72 q99 先验只截掉 2.2%。
- [RGB-only 先验试验](outputs/EXP-3D-RGB-PRIOR-BOUNDS-V10/audit/exploratory_pilot_train_only/README.md)：TRAIN24 训练、48 个未见 TRAIN 场景评价。放大盒子后远处表面变好，但近处被整体推远，AbsRel 从 0.568 变差到 0.732。RGB-only 的失败不是截断造成的，瓶颈在几何证据本身。
- [保留测量的收益上限](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/audit/exploratory_ceiling/README.md)：被看到的区域用重投影、其余用 carrier 的理想组合只有 0.193，对重投影 0.220 的差区间跨零。学到的补全只比最近邻外推略好；补全质量才是瓶颈。
- 已就绪、待 V9 收尾后冻结：[Core B 第二轮](outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V2/PROTOCOL.md)（carrier 按 V9 出结果前写定的计划选择）与 [V10](outputs/EXP-3D-RGB-PRIOR-BOUNDS-V10/PROTOCOL.md)（RGB-only 先验的正式检验，排在 Core B 第二轮之后）。

2026-09-29：[强几何基线对照](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/README.md)，不训练，预注册分支 B。
- 结果：已合格的 V7 C1 与"上下文深度直接重投影 + 最近邻填洞"在整体 AbsRel 上无法区分（DEV 0.280 对 0.220，FRESH-V2 0.290 对 0.313）。它显著优于同表示的非学习融合。
- 分区域：在被看到的区域，重投影准得多；在未被看到的区域，carrier 更好。
- 结论：逆向 JEPA 学到的状态有价值，但目前还没有超过简单几何；瓶颈是观测区域的表示精度。

2026-09-29：[Core B 第一轮](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V1/README.md)（冻结的合格 carrier + 学习写入），全程 CPU，预注册分支 B。
- 结果：写入相对"只缓存新帧"好 +0.061（CI >0），但没有优于静态状态，没有优于非学习的视角计数截断，也不是场景专属。
- 结论：写入只是在修补视角计数偏移；Core B 的核心主张尚未得到支持。
- 下一步：V9（训练时视角数可变，32³），然后做 Core B 第二轮。

2026-09-29：[分辨率 V8](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/README.md)，全程 CPU，预注册分支 B。
- 结果：DEV AbsRel 16³ 0.280 → 32³ 0.259（24³ 为 0.244），RESOLUTION_GAIN +0.021，CI 跨零。
- 选择偏差：不做 checkpoint 选择时，32³ 比 16³ 好 0.061，因为 16³ 后期过拟合。
- 下一步：先运行 Core B 第一轮（冻结的 V7 C1 上的学习写入），再运行 V9（训练视角数，32³）。

2026-09-29：[第二次独立资格验证](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V2/README.md)，结果 **`QUALIFIED`**。
- 在 17 个独立场景上，V7 的 RGB-D carrier 通过全部四个冻结 gate：
  - BOUNDS_GAIN +0.145（CI [+0.046, +0.240]）；
  - 相对常数 +0.300（CI [+0.192, +0.396]）；
  - 场景专属性；
  - 多视角融合。
- C1 AbsRel 0.290，与 DEV 的 0.280 一致。
- Core A 静态 carrier 在 RGB-D 轨道上通过资格验证；按项目规则，Core B 下一阶段可以开启。受保护的 final holdout 仍未打开。
- 进行中：V8（32³ 分辨率），全程 CPU。其设计在 V7 出结果之前写定，冻结于资格验证出结果之前。

2026-09-29：[bounds 规则 V7](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/README.md)，全程 CPU，预注册分支 **A**。
- 结果：改用测得上下文深度确定 bounds 后，DEV AbsRel 从 0.459 降到 0.280（BOUNDS_GAIN +0.179，CI [+0.084, +0.279]）。
- 与常数相比：C1 首次在全部 8 个 DEV 场景上显著优于 TRAIN 拟合的常数深度，平均好 +0.227，CI [+0.113, +0.342]。
- 机制：场景专属性与多视角融合同时成立。
- 注意：C1 的 DEV checkpoint 选择有约 0.05 的乐观偏差（最后一步为 0.330）。
- 修订 1：删去冻结估计器中一条与本实验设计不相容的断言（各变体的命中比例必须相同），写于查看任何指标之前。
- 下一步：在新准备的 FRESH-V2（17 个独立场景）上做一次资格验证；Core B 仍关闭。

2026-09-29：[训练规模 V6](docs/experiments/EXP-3D-RGBD-TRAIN-SCALE-V6/README.md)，全程 CPU。
- 结果：训练场景 24→72，DEV AbsRel 0.492→0.459，至今最好；SCALE_GAIN +0.033，CI 跨零，分支 C。
- 与常数相比：排除 NO_HIT 场景后，C1 比常数好 +0.111（CI [+0.011, +0.222]），但只有 5/7 个场景不差于常数，未达冻结规则。
- 剩余差距：主要来自 bounds（NO_HIT 与截断）。下一步 V7 单因素检验由测得深度确定的 bounds（72 个场景，CPU）。

2026-09-29：[第一次独立资格验证](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V1/README.md)，结果 `NOT_QUALIFIED`。
- 深度收益：V5 在 5 个独立场景上的深度收益为 +0.006，CI [−0.112, +0.116]，没有复现 DEV 上的 +0.114。
- 机制：场景专属性（+0.131 / +0.197）与多视角融合（+0.200）都复现。
- 结论：机制站得住，精度优势站不住；DEV 选择偏差使收益偏乐观。Core A 未通过资格验证，Core B 保持关闭。
- 数据：补充了 5 个独立场景和 48 个训练场景；受保护的 final holdout 未打开。

2026-09-29：[RGB-D 证据 carrier V5](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/README.md)。
- 上下文深度经零初始化旁路进入状态后，DEV query AbsRel 从 0.617 降到 0.503（DEPTH_GAIN +0.114，CI [+0.065, +0.166]，三个 seed 一致）。
- 首次同时通过三项冻结 gate：深度收益；场景专属性（wrong-scene 损伤 +0.169、shuffle +0.216，CI 均 >0）；多视角优于单视角（+0.152）。
- 但 C1 仍未显著优于只由 TRAIN 拟合的常数深度：排除 NO_HIT 场景后 0.432 vs 约 0.493，CI 下界 −0.0035。冻结规则与修订 1 规则下，预注册分支都是 B。
- 下一步：按分支 B 检验 carrier 容量或训练规模。TRAIN-only 诊断显示，由上下文深度导出的 bounds 可把截断从 26% 降到 3.4%，但体素变粗，宜与分辨率一起考虑。训练规模与独立资格队列都需要新数据，等待用户决定。

2026-09-29：[无几何参考再分析](docs/experiments/EXP-3D-READOUT-REFERENCE-REANALYSIS-V1/README.md) 与 [plane-sweep carrier V4](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/README.md)。
- V4：光度 plane-sweep 旁路未建立全图收益（+0.0072，CI [−0.004, +0.017]，分支 C），只在多视角可观测区域有描述性改善（OBS2PLUS +0.040）。
- 再分析（预注册）：V2–V4 的 9 个 RGB-only carrier 在 DEV 上无一优于 TRAIN 拟合的常数深度。常数 AbsRel≈0.507，carrier 为 0.61–0.65；给定 GT 尺度后，形状也不如平面常数。
- 判断：RGB-only 轨道已到信息上限，与 direct-state 的 RGB-only 上限 0.617 一致。

2026-09-29：[稠密证据carrier V3](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/README.md)。加密证据候选（128→4096）未建立DEV收益（+0.0101，CI [−0.014, +0.039]），状态场景专属性仍未建立；训练集拟合大幅改善（0.490→0.339）而DEV停滞，表现为过拟合。候选稀疏不是主因，下一轮按预注册分支C研究几何推理式融合（多视角光度一致性等），并考虑训练规模。

2026-09-29：[16³ geometry-aware learned carrier V2](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/README.md)。表面监督在learned carrier上未建立收益（C0−C1 +0.0195，CI [−0.025, +0.077]）；状态近似场景无关（wrong-scene/空间打乱无损伤），full context未胜anchor；`STATIC_DEV_STATUS=NOT_ESTABLISHED`，`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`。direct-state归因链：16³有益（0.461→0.403）；更多优化与bounds无稳定差异；query-oracle 0.148 vs context-only 0.361；surface监督降低0.049（CI [0.024, 0.076]）。下一步检验carrier证据通路（128个候选点在16³只覆盖约3%体素）。本机0–7号CPU核会引发numpy原生段错误，所有计算须绑定8–23号核。

2026-09-27 后续探索：[折外全局轨迹与 cycle 诊断](docs/experiments/EXP-2D-followup-20260927/README.md)。12个discovery外层折都选择ALL/B；CycleGate仍低约0.209个百分点。仅属事后discovery分析，独立确认仍受阻。

V2 当前状态：`BLOCKED_NO_INDEPENDENT_DATA`。旧 raw 真正逐样本空间约 +0.2055 个百分点；CycleGate discovery 外层 OOF 约 +0.1244 个百分点，但尚无独立测试结果。完整测试重试通过（500 passed、1 skipped），首轮原生崩溃已留档。[V2 报告](docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md)。

最新机制实验已完成：448像素J下有小幅、跨序列的oracle机会（H1=STRONG），但部署可见选择器净收益未建立（H2=NOT_ESTABLISHED）。详见 [新实验报告](docs/experiments/EXP-2D-20260926-opportunity-selector-v1/README.md)。以下保留历史证据状态。

| ID | 状态 | 证据等级 | 已验证 | 当前可以说什么 | 当前不能说什么 |
|---|---|---|---|---|---|
| `BASELINE-V5-4500` | `FROZEN_REFERENCE` | `DEVELOPMENT` | V5 typed appearance 与旧工程记录保留 | 不能当 validation/test 或 CVPR 结果 |
| `EXP-3D-20260919-engineering` | `COMPLETE_ENGINEERING` | `ENGINEERING` | runner、CPU contract、CUDA smoke 和保存恢复检查 | 不能当三维方法胜出证据 |
| `EXP-3D-20260920-method-dev` | `COMPLETE_DEVELOPMENT` | `DEVELOPMENT` | carrier/policy 开发记录和失败修复被保留 | 不能当最终方法或外部 benchmark |
| `EXP-2D-20260921-original` | `SUPERSEDED` | `ENGINEERING` | 原始运行保留；已记录 pair 汇总 bug | 不能引用旧汇总数字 |
| `EXP-2D-20260921-corrected` | `EXPLORATORY` | `EXPLORATORY` | 2424 raw rows、32 validation pairs、10 focused tests、completion verification | 不能叫完整逆向 JEPA、streaming TTT 或 DAVIS J&F |
| `EXP-GEOMETRY-CERT-V1` | `STOPPED` | `DIAGNOSTIC` | certificate frontend 失败门槛已封存 | 不能继续把该 frontend 接入主模型 |
| `EXP-MAPANYTHING-V1` | `STOPPED` | `DIAGNOSTIC` | supplier qualification 失败记录保留 | 不能当外部监督或方法贡献 |
| `OFFICIAL-DAVIS-VAL` | `SEALED` | `UNRUN` | 30 段官方 val 未读取 | 不能暗示已经完成官方 benchmark |
| `DL3DV/Hypersim final holdout` | `SEALED` | `UNRUN` | 旧本地数据已按授权清理，结果和来源清单保留 | 不能声称有 final holdout 结果 |

## 当前研究判断

二维 probe 显示观测 RGB 反馈可以显著降低 RGB 重建误差，但当前 selector 没有把它转化成对象对应收益。下一步应先测试更可靠的高分辨率对应评价和允许 `OFF` 的 task-grounded gate，再决定是否扩大模型。不要直接扩大 RGB 重建训练规模。

## 存储和协作边界

旧三维数据删除记录和二维资产下载记录仍在 `outputs/` 本地。公开仓库只提交小型 provenance 和结果 bundle；数据、权重、缓存和 checkpoint 按 `.gitignore` 排除。项目记忆的持久状态写入中央 CV 分支，不以仓库文件替代。

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

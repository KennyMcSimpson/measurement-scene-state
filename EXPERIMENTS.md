# 实验注册表

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

# 当前状态（2026-10-10）

2026-10-10：**DepthSplat 基座上的逆向 JEPA 与 TTT 研究线结束，最终报告见 Claude Docs 文档[逆向 JEPA 与 TTT 最终报告](https://claude.ai/artifact/XKGJfbogpYoLH4LyNj8xU9)。**
- **结论：** Core B（TTT 与 7 帧重投影集成）是唯一稳健的实用结果，五个独立队列上复现，真实数据上误差降低约 3–4%；逆向 JEPA 不能改进 TTT，只在不做 TTT 时小幅改进前馈状态（约 2.5%）；学到的补全只有按深度范围选择时才稳定；真实训练数据从 182 增加到 502 个扫描没有作用。
- **推荐读出：** 要最一致用 Core B；要误差最低用 sqrt(TTT × JN 前馈)。

2026-10-10 03:49（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[更多真实训练数据能否改进逆向 JEPA 的补全](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OVERLAP-SCALE-ARKIT-V1/README.md)（预注册，01:58 锁定，sha256 `35f85731…`）。**MORE_DATA_COMPLETION_NOT_ESTABLISHED，SCALED_COMPLETION_PARTIAL。**
- **问题：** 重叠感知的逆向 JEPA（JO 配方）的补全停在"平均有效、逐场景不稳"。训练数据从 182 个扫描增加到 502 个（步数不变），能否改进？
- **结果：**
  - P1：VAL80 的 FARH 上 JOL 0.3585 对 JO 0.3541，−0.0044，42/0/33，不成立。
  - P2：对比调和插值 +0.1125，42/0/33，PARTIAL，与原来的 JO 相同。
- **意义：**
  - 与 NS 合起来：在这个范围内，真实数据量既不是逆向 JEPA 前馈精度的瓶颈，也不是补全的瓶颈。
  - 要明显改进，需要更强的先验或不同的方法，而不是多一些同类数据。

2026-10-10 01:51（hit CPU）：[Core B 中 TTT 与 R7 的集成权重](docs/experiments/EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1/README.md)（预注册，01:51 锁定，sha256 `178a022a…`）。**WEIGHT_IMPROVES_CORE_B_PARTIAL（两个确认队列都是 PARTIAL）。**
- **问题：** TTT 单独的均值一直比 Core B 低、但一致性较差。把 R7^(1−w) × TTT^w 的权重调到 0.5 以上，能否得到更好的折中？
- **结果：**
  - 开发集选出 w* = 0.9。确认集上均值稳定地降低：SYSTEM3 0.1569 → 0.1544（+0.0025，106/0/54），VAL 0.1161 → 0.1142（52/0/28），都是 PARTIAL。
  - 对 R7 的一致性从 149/160 降到 130/160；GEO_0.9 与 TTT 单独几乎相同。
- **意义：**
  - 均值与一致性之间是一条平滑的取舍，没有同时优于两端的权重。
  - 推荐的实用读出：要最一致用 Core B；要最低均值用 PAIR = sqrt(TTT × JN 前馈)（0.1540，对 R7 138/160），它比 TTT 单独更一致。

2026-10-10 01:46（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[用逆向 JEPA 学到的有选择弃权改进 Core B](docs/experiments/EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1/README.md)（预注册，01:37 锁定，sha256 `091f0b96…`）。**JEPA_ABSTAIN_IMPROVES_CORE_B_PARTIAL（两个确认队列都是 PARTIAL）。**
- **问题：** 把 JN 学到的有选择弃权用到 Core B 里，能否改进完整系统？三种接法在开发集（SYSTEM + SYSTEM2）上选择，在 SYSTEM3 与 VAL 上确认。
- **结果：**
  - 选出 PAIR = sqrt(TTT × JN 前馈)：SYSTEM3 0.1569 → 0.1540（+0.0029，102/0/58），VAL 0.1161 → 0.1142（+0.0019，55/0/25），都是 PARTIAL；换用 JN2、JN3 结果一致。
  - 但 PAIR 与 TTT 单独打平（0.1540 对 0.1544，76/0/84）；对无模型方法 138/0/22。
  - MASK（TTT 也在 JN 弃权处退回 R7）反而变差。
- **意义：**
  - JN 的前馈可以替代 R7 做 TTT 的集成伙伴，让系统均值回到 TTT 单独的水平，同时保留大部分一致性。
  - 相对 TTT 单独没有增益：逆向 JEPA 对完整系统的贡献是边际性的。

2026-10-10 01:32（hit CPU）：[只调高读出阈值，能否得到逆向 JEPA 的前馈改进](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1/README.md)（预注册，01:28 锁定，sha256 `644bad0d…`）。**JEPA_BEYOND_THRESHOLD_PARTIAL（VAL 上 SUPPORTED）；Core B 的阈值不需要改变。**
- **问题：** 逆向 JEPA 的前馈改进伴随覆盖率下降。不训练、只把发布权重的读出阈值调高（α ≥ τ），能否得到同样的效果？能否改进 Core B？
- **结果：**
  - 开发集（SYSTEM + SYSTEM2）上，发布权重的前馈、TTT、Core B 都随 τ 升高单调变差（前馈 0.1549 → 0.1569），最优阈值就是原来的 0.5。
  - 覆盖率相近时，发布权重（τ = 0.9，覆盖率 0.54）变差，而 JN（覆盖率约 0.53）变好。
  - 确认集上 JNS 仍胜过最优阈值的发布权重：SYSTEM3 +0.0041，110/0/50，PARTIAL；VAL +0.0041，61/0/19，SUPPORTED。
- **意义：**
  - 逆向 JEPA 学到的是有选择的弃权（哪些像素不可靠），全局阈值做不到。它的前馈改进小（约 2.5%）但真实。
  - Core B 的读出阈值不需要调整。

2026-10-10 01:21（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[更多真实训练数据能否放大逆向 JEPA 的前馈改进](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-SCALE-ARKIT-V1/README.md)（预注册，2026-10-09 23:18 锁定，sha256 `d9ed9d50…`）。**MORE_DATA_NOT_ESTABLISHED，SCALED_RECIPE_FEEDFORWARD_PARTIAL。**
- **问题：** 只奖励精度的逆向 JEPA 对前馈状态的改进只有约 2.5%；训练数据从 182 个扫描增加到 502 个（步数不变），能否放大它？
- **结果：**
  - P1：完全没有变化。SYSTEM3 上 JLS 与 JNS 都是 0.1567，82/0/78；VAL 上 0.1169 对 0.1167。
  - P2：扩数据后的配方对比发布权重 +0.0041，111/0/49，PARTIAL，与 JN 相同。
  - 四次测量的改进都在 +0.0036 到 +0.0041 之间，一致性 68–73%。
- **意义：**
  - 在这个范围内，真实训练数据量不是瓶颈。改进来自训练目标（学会弃权），不随数据量增长，不需要为这条线下载更多数据。
  - 下一个关键对照：不训练、只调高发布权重的读出阈值，能否得到同样的效果。

2026-10-09 23:04（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[只奖励精度的逆向 JEPA 能否改进前馈状态（预注册复现）](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/README.md)（19:23 锁定，sha256 `7ff1827a…`）。**JN1_FEEDFORWARD_PARTIAL，RECIPE_FEEDFORWARD_PARTIAL。**
- **问题：** OBJECTIVE-TTT 的次要发现（只奖励精度的逆向 JEPA 让不做 TTT 的前馈状态变好）能否在 160 个全新扫描、3 个种子上复现？
- **结果：**
  - P1（JN1）：+0.0036 [+0.0004, +0.0076]，109/0/51，PARTIAL；均值与 ARKIT_VAL 上相同。
  - P2（3 个种子平均）：+0.0041 [+0.0010, +0.0080]，110/0/50，PARTIAL；每个种子的 CI 都不含 0。
  - 好处都在 NEAR7（0.0763 → 0.0717）；JN 的前馈胜过 R7（94/0/66），发布权重的前馈与 R7 打平。
  - 事后探索：均值上与 Core B 打平（0.1567 对 0.1569），但 Core B 在 71% 的场景更好。
- **意义：**
  - 这是逆向 JEPA 在真实数据上第一个可复现的正面效应：前馈状态约 2.5% 的稳定小幅改进，逐场景约 2/3 变好。
  - 它不能改进 TTT；有时间做 TTT 时，Core B 仍是更好的系统。

2026-10-09 21:39（hit CPU + GPU 4）：[Hypersim 小载体 G1 的选择性集成能否改进 Core B（独立检验）](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/README.md)（预注册：计划锁 14:50，正式锁 15:08，sha256 `0a33ebdf…`）。**G1 接法没有确立（P1 不成立）；完整系统胜过无模型方法（P2）；Core B 第五次复现（149/0/11）。**
- **问题：** V2 次要分析中成立的 G1 接法（选中的 episode 中，FAR7 基底改为 sqrt(R7 × 7 帧 G1)），能否在 160 个全新扫描上改进 Core B？
- **结果：**
  - P1：+0.0010 [−0.0003, +0.0025]，38/99/23，不成立。只看 FAR7 时有小而稳定的改进（+0.0062，36/96/25，次要）。
  - P2：完整系统对比 R7 为 +0.0071，139/0/21，SUPPORTED（主要来自 Core B）。
  - Core B 对比 R7：+0.0061，149/0/11，五次复现中最一致的一次。
- **意义：**
  - G1 接法三次检验分别为 +0.0002（V1 事后）、+0.0042（V2 次要）、+0.0010（V3 主检验）。即使存在，效应也只有约 0.5%，没有确立。
  - 完整系统的实用结论仍是 Core B：五个独立队列上都成立，真实数据上的绝对增益约 4%。

2026-10-09 19:16（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[只奖励精度的逆向 JEPA 能否改进 TTT](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1/README.md)（预注册，14:41 锁定，sha256 `a826b21c…`）。**JN_TTT_NOT_ESTABLISHED，JK_HARMS_TTT。**
- **问题：** JC 的损害来自训练目标奖励覆盖。改成只奖励已覆盖像素的精度（JN1），或再按 TTT 的 7 帧布局训练（JK1），能否让 TTT 比发布权重更好？
- **结果：**
  - P1：JN 的 TTT 结果与发布权重打平（0.1149 对 0.1141，30/0/50）。JC 的损害消失了。
  - P2：JK 更差（0.1178，24/0/56，HARMFUL）。
  - 新现象：只奖励精度的目标让模型学会弃权，前馈覆盖率从 0.73 降到 0.55（JN）和 0.26（JK）。不做 TTT 时，JN 的前馈状态比发布权重好（0.1172 对 0.1208，60/0/20，次要分析 S3）；TTT 把差距抹平。
- **意义：**
  - 四种逆向 JEPA 设置都没有给 TTT 更好的起点，逆向 JEPA → TTT 这条线结束；Core B 的实用配置不变。
  - 进行中：S3 在全新扫描上用 3 个种子做预注册复现（[NF](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-NORM-FEEDFORWARD-ARKIT-V1/PROTOCOL.md)，19:23 锁定）；V2 的 G1 接法在 160 个全新扫描上做预注册检验（[FULL-SYSTEM-V3](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/PROTOCOL.md)，15:08 锁定）。

2026-10-09 14:29（hit CPU + GPU 4）：[TTT 集成加上 7 帧小载体的选择性集成](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2/README.md)（预注册：计划锁 13:08，正式锁 13:16，sha256 `b36f2657…`）。**V1 样本内挑出的接法没有复现（P1、P2 都不成立）；Core B 第四次复现（70/0/10）。**
- **问题：** 在 V1 样本内挑出的接法（选中的 episode 中，FAR7 基底改为 sqrt(R7 × 7 帧 GA)）能否在 80 个全新扫描上改进 Core B？
- **结果：**
  - P1：−0.0022，16/42/22，不成立。原因是这批扫描上 7 帧 GA 在 FAR7 中明显差于 R7（0.358 对 0.281）。
  - 次要的 G1 接法：+0.0042，21/43/16，SUPPORTED。但它在 V1 上不成立，只能算线索。
  - Core B 对比 R7：+0.0049，70/0/10，SUPPORTED。
- **意义：**
  - Core B 已在四个独立队列上复现：EVAL34 71/12，VAL80 71/9，SYSTEM80 71/9，SYSTEM2 70/10。
  - 小载体接入 Core B 的效果随扫描批次变化，没有可靠结论。

2026-10-09 12:48（hit CPU + GPU 4）：[TTT 集成加上按深度范围选择的补全](docs/experiments/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1/README.md)（预注册：计划锁 11:28，正式锁 11:35，sha256 `8bfe2c18…`）。**完整系统没有改进（P1、P2 都不成立）；Core B 在全新扫描上第三次复现（71/0/9，SUPPORTED）。**
- **问题：** 把按深度范围选择的补全接进 Core B（TTT 状态与 7 帧重投影的几何平均）的 FAR7 区域，能否改进完整系统？数据为 80 个全新的 ARKit 扫描。
- **结果：**
  - P1 Core B − 完整系统：−0.0031，15/49/16，不成立。
  - FAR7 中，只看 3 个上下文帧的补全不如 7 帧调和插值（0.291 对 0.270）。
  - Core B 对比 R7：+0.0047，71/0/9，SUPPORTED。TTT 的增益：61/0/19。
- **意义：**
  - Core B 稳健：EVAL34 71/12，VAL80 71/9，SYSTEM80 71/9。但真实数据上的绝对增益很小（0.152 → 0.148）。
  - 要整合两条线，补全模型必须吃进全部 7 帧。

2026-10-09 11:24（hit CPU + GPU 4）：[深度范围选择规则能否推广到远处视角](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-FAR-ARKIT-V1/README.md)（预注册，11:10 锁定，sha256 `62fcff3c…`）。**FAR_SELECTOR_CONSISTENT。**
- **问题：** 同一条规则、同一个阈值（S ≥ 1.2），在同一批 80 个全新扫描的远处视角（第 14、15 帧，此前从未评估）上是否仍然成立？
- **结果：**
  - P1（GA）：FARH 从 0.452 降到 0.394，+0.058，24/17/12（77% 不变差），SUPPORTED。
  - Hypersim 小载体也成立：+0.058，28/17/8。
  - DepthSplat-JEPA 即使经过选择也不显著：JO +0.022，JA +0.042。
  - 不选择时，四种模型都只与调和插值打平。
- **意义：**
  - 选择规则在真实数据的近处和远处视角上都通过了 V2 gate。
  - 远处视角上增益较小（−13%），而且只对小载体成立。

2026-10-09 11:06（hit CPU + GPU 4）：[用上下文深度范围决定何时使用学到的补全](docs/experiments/EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1/README.md)（预注册：计划锁 10:42，正式锁 10:51，sha256 `c6dc93e1…`；修订 1 只修复打分错误）。**SELECTOR_CONSISTENT：这条研究线上第一个在真实数据上通过 V2 gate 的补全结果。**
- **问题：** 事后探索发现，上下文深度范围 S = (p90 − p10) / 中位数能预测学到的补全是否有用。规则"S ≥ 1.2 时用学到的补全，否则用调和插值"能否在全新扫描上稳定胜过调和插值？
- **数据：** 80 个全新的 ARKit 扫描（训练折种子顺序第 520–679 位），查询为第 6、7 帧。
- **结果：**
  - P1（GA）：FARH 从 0.523 降到 0.411，+0.112，32/31/9（88% 不变差），SUPPORTED。
  - 同一规则用于 Hypersim 小载体：+0.114，31/31/10；用于 DepthSplat 逆向 JEPA：+0.073，29/31/12，都是 SUPPORTED。
  - 不做选择时三者都只是 PARTIAL；选择几乎不损失均值。
  - 被选中的查询上，调和插值为 0.694，GA 为 0.447，GA 在 81% 的查询上更好；未选中的查询上分别为 0.292 和 0.305。
- **意义：**
  - 真实数据上学到的补全的"不一致"是可预测的：在深度层次丰富的场景里调和插值失败，学到的补全有用；在近似平面的场景里调和插值已经很好。
  - 规则与模型无关，只需上下文深度即可判断。

2026-10-09 09:28（hit CPU）：[用 ARKit 真实数据做重叠感知训练的旧基座补全载体](docs/experiments/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1/README.md)（预注册；冻结于 02:24，sha256 `2cef537d…`）。**REAL_CARRIER_PARTIAL，REAL_DATA_CARRIER_NOT_ESTABLISHED。**
- **问题：** 旧基座的补全载体（G1 架构）用冻结的 V17 配方，在 ARKit 训练折上、只用有重叠的目标重新训练（GA）。它在有重叠的新查询（第 6、7 帧）上能否稳定胜过调和插值，并胜过 Hypersim 训练的 G1？
- **结果：**
  - C1：FARH 从 0.452 降到 0.330（+0.122），但只有 79/147 个场景变好，PARTIAL。
  - C2：与 G1N 的 0.344 相比没有可确认的差别。
  - 与调和插值取平均（HGAN）有 100/0/47 个场景变好（68%），是本轮最一致的组合。
- **意义：** 在真实数据的近处视角上，四种学到的补全（G1N、GA、JA、JO）给出几乎相同的结果：平均误差降约 1/4，但只有一半多的场景变好。模型类型、训练数据和训练目标都不是瓶颈。

2026-10-09 08:24（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[在 TTT 的尺度上限下训练逆向 JEPA](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-CAP01-TTT-ARKIT-V1/README.md)（预注册，06:03 锁定，sha256 `1b00184e…`）。**JEPA_CAP01_HARMS_TTT，JEPA_CAP01_HARMS_CACHE。**
- **问题：** JBT 中逆向 JEPA 基座让 TTT 变差，是不是因为尺度上限 3.0？在上限 0.1 下训练（JC1）后，能否让 TTT 比发布权重更好？
- **结果：**
  - P1：TTT 后为 0.121，发布权重为 0.114，−0.0070，24/0/56，HARMFUL（差距小）。
  - P2：前馈状态 0.173 对 0.121，5/0/75，HARMFUL。
  - 改用上限 0.1 消除了大部分损害：JA（上限 3.0）TTT 后为 0.159，JC 为 0.121。与 7 帧重投影集成后，两者打平（都是 0.116）。
- **意义：**
  - 剩下的差距来自训练目标：它对未覆盖像素按全深度惩罚，奖励覆盖而不是精度，在两个上限下都把前馈覆盖率推高、精度拉低。
  - 在当前目标下，"逆向 JEPA 给 TTT 更好的起点"不成立。Core B 的实用配置不变。

2026-10-09 05:58（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[重叠感知的真实数据逆向 JEPA](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/README.md)（预注册，01:19 锁定，sha256 `27d25dd4…`）。**OVERLAP_JEPA_PARTIAL，OVERLAP_TRAINING_NOT_ESTABLISHED。**
- **问题：** 只在与上下文有重叠的目标上训练逆向 JEPA（JO），在有重叠的新查询（第 6、7 帧，从未用过）上能否稳定胜过调和插值？是否比原配方好？
- **结果：**
  - O1：FARH 上从 0.452 降到 0.341（+0.110），但只有 80/147 个场景变好，中位数只有 +0.005，PARTIAL。
  - O2：与原配方的 JA 打平（0.341 对 0.337）。
  - Hypersim 训练的 J 为 0.407；旧基座 G1N 为 0.344，与 JO 打平。HJO 集成 95/0/52。
- **意义：**
  - 学到的补全在真实数据上"平均有效、逐场景不稳"：近处视角平均降约 1/4 的误差，但过不了一致性门槛。
  - 真实数据训练比合成数据训练好，但只追平了旧的小载体；只用有重叠的目标训练没有额外收益。
- **进行中：** [EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1](docs/experiments/EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1/PROTOCOL.md)：在 ARKit 上做重叠感知训练的旧基座载体（GA），CPU 训练中，约 09:00 完成。

2026-10-09 04:10（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[从真实数据训练的逆向 JEPA 基座出发做 TTT](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1/README.md)（预注册，01:19 锁定，sha256 `6441e982…`；修订 1 只改执行方式）。**JEPA_BASE_HARMFUL，JEPA_TRAINING_HARMS_TTT。**
- **问题：** 逆向 JEPA 能否为 TTT 提供更好的起点？在 ARKIT_VAL（80 个扫描）上，以 JA1 为起点做 TTT，与以发布权重为起点比较。
- **结果：**
  - Q1：以 JA 为起点不如以发布权重（上限 0.1）为起点，ALL 0.159 对 0.114，−0.0445，15/0/65。
  - Q2：同为上限 3.0，以 JA 为起点也更差，0.159 对 0.136，−0.0226，30/0/50。
  - TTT 本身在 JA 基座上依然有效（0.266 → 0.159，70/0/10），但修不回起点的差距。
  - 尺度上限 0.1 比 3.0 好（+0.022，70/0/10）。
- **意义：** 这个逆向 JEPA 配方没有为 TTT 提供更好的起点。Core B 最好的配置仍是"发布权重 + TTT + 与 7 帧重投影取几何平均"。

2026-10-09 01:26（hit：GPU 4 与另一用户的常驻进程共用 + CPU）：[在真实传感器数据上训练逆向 JEPA](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-V1/README.md)（预注册，10/8 23:42 锁定，sha256 `de61109d…`；修订 1 于 10/9 00:03 记录）。**R1 = REAL_JEPA_PARTIAL，R2 = REAL_DATA_TRAINING_PARTIAL；但增益来自替换错误的补洞常数，不是补全。**
- **问题：** 用户 10/8 要求"用 ARKit 训练折在真实数据上训练逆向 JEPA"。在 182 个 ARKit 扫描上训练（3 个种子，V1 配方），在 160 个扫描（ARKIT_VAL80 + ARKIT_CONFIRM80）上评估。
- **结果：**
  - R1 H3 − JA（FAR）：+0.210，90/14/55，PARTIAL。R2 J − JA：+0.231，104/13/42，PARTIAL。FAR AbsRel 从 0.793 降到 0.583，JA 覆盖了 65% 的 FAR。
  - 修订 1：约 47% 的 ARKit 查询与上下文完全没有重叠，调和插值在这些查询上整张图都是 Hypersim 的补洞常数 2.37 m。换成训练折拟合的 1.05 m，FAR 就降到 0.418。
    - 在这个校准基线之上，JA 有害：−0.029，57/14/88。
    - 在有重叠的查询上，JA 与调和插值打平：+0.027，48/0/50。
  - 探索性：归一化 G1 集成在有重叠的查询上 +0.051，67/0/31。
- **意义：**
  - 真实数据训练让模型学会了"室内场景大概多深"，但没有学会基于上下文的补全。
  - 之前 ARKit 上的补全评估有一半是无重叠查询，需要按重叠重新解读。TTT 的 7 帧评估不受影响。
- **进行中：**
  - [EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-BASE-TTT-ARKIT-V1/PROTOCOL.md)（10/9 01:19 锁定）：在 ARKIT_VAL 上比较以 JA1 为起点的 TTT 与以发布权重为起点的 TTT。GPU 运行中。
  - [EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/PROTOCOL.md)（10/9 01:19 锁定，早于 JA 结果）：只在有重叠的目标上训练，在有重叠的新查询帧（第 6、7 帧）上评估。CPU 封存运行中，GPU 排在上一个实验之后。

2026-10-08 13:50（hit CPU + ARKit 下载）：[归一化 G1 补全集成在全新真实扫描上的确认](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-ARKIT-CONFIRM-V1/README.md)（预注册，13:46 锁定，sha256 `b5b24308…`）。**NORMALIZED_ENSEMBLE_CONSISTENT。**
- **来源：** ARKIT_VAL 上发现 G1 的未归一化读出在真实数据上塌缩，"补全集成"的表面通过来自偏差抵消；改用会弃权的归一化读出。
- **数据：** 80 个从没用过的 ARKit 训练折扫描（`outputs/EXP-3D-RGBD-ARKIT-CONFIRM-DATA-V1`，种子顺序第 360–519 位场馆）。
- **结果（FAR）：**
  - (调和 + 归一化 G1) / 2 相对调和：+0.064 [+0.036, +0.095]，48/19/12，SUPPORTED；ALL 上也 SUPPORTED；
  - G1 弃权 49%。
- **意义：** 学到的补全在真实数据上有一个小而一致的增益。真正的大误差来自调和插值在 ARKit 上偏远：减半就能大幅降低均值，但不稳定。

2026-10-08 13:31（hit：数据下载与准备 + GPU 4 共用 + CPU）：[两个集成在真实传感器数据（ARKitScenes）上的检验](docs/experiments/EXP-3D-RGBD-ENSEMBLES-ARKIT-V1/README.md)（预注册，12:08 锁定，sha256 `6947f14f…`）。**TTT 集成复现，补全集成没有可信地复现。**
- **数据：** ARKitScenes 3dod Validation 折，80 个合格的真实 iPad RGB-D 扫描（`outputs/EXP-3D-RGBD-ARKIT-VAL-DATA-V1`）。
  - 修复了 10/5 的 unzip 问题：`-d` 不会创建缺失的父目录。
  - 按种子顺序扩展到第 136 个场馆。1% 的源几何阈值对真实数据偏严，但没有放宽。
- **TTT 集成：**
  - √(REPROJ7 × TTT) 相对 REPROJ7：+0.0049，71/9，SUPPORTED；TTT 必要：60/20，SUPPORTED；
  - TTT 单独也稳定更好（62/18）；
  - 绝对收益小（ALL 0.121 → 0.116）。
- **补全集成：**
  - 主检验 (调和 + J + G1) / 3：+0.422，56/24，PARTIAL；(调和 + G1) / 2：60/20，恰好 SUPPORTED。
  - 事后检查：G1 在 ARKit 上迁移失败，53% 的 FAR 像素塌缩到接近 0（未归一化的深度读出）。调和插值偏远，平均的收益主要是偏差抵消。
  - 在 Hypersim 上同样的检查确认，那里的收益来自 G1 的信息。
  - 逆向 JEPA 在 ARKit 上不比调和插值好，也不为集成增加价值。

2026-10-08 09:44（hit GPU 4 共用 + CPU）：[TTT 集成](docs/experiments/EXP-3D-RGBD-TTT-ENSEMBLE-V1/README.md)（预注册，06:31 锁定，sha256 `5154f900…`）。**T1 = TTT_ENSEMBLE_CONSISTENT，T2 = TTT_NEEDED。**
- **问题：** 新帧到达后，TTT 后的 DepthSplat 状态与 7 帧无模型重投影取几何平均，能否稳定胜过无模型重投影？TTT 在其中是否必要？
- **检验：** EVAL34，83 个场景，流帧同 Core B V8。
- **结果（ALL）：**
  - T1：+0.029 [+0.015, +0.047]，71/12，SUPPORTED；
  - T2（只缓存的状态替换 TTT 状态）：+0.019，80/3，SUPPORTED；
  - TTT 单独相对无模型：+0.040，59/24，PARTIAL；TTT 相对只缓存：+0.032，79/4，SUPPORTED。
- **意义：** Core B 第一次在样本外队列上显示出稳定的实用价值。用法是把梯度写入后的状态与测量驱动的重投影结合；收益在已测区域。

2026-10-08 07:59（hit GPU 4 共用 + CPU）：[TTT 消融：无网络高斯 + 同样的 TTT](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1/README.md)（预注册，06:25 锁定，sha256 `cebe6e23…`）。**LEARNED_BASE_MATTERS。**
- **问题：** V1 中 TTT 的收益是来自学到的 DepthSplat 基座，还是只来自"高斯表示 + 测试时优化"？
- **无网络高斯：** 每个测得像素一个点状高斯，尺度、步长都在 DEV 上选。
- **结果（FRESH78，ALL）：**
  - 同样的 TTT 下，学到的状态更好：+0.024 [+0.008, +0.044]，61/17，SUPPORTED；
  - 不做 TTT 时没有区别：+0.006，47/31；
  - TTT 对无网络高斯也有用（+0.012，69/9），但只有学到状态收益（+0.030）的四成。
- **意义：** 学到的基座的价值在于给测试时优化提供更好的起点和参数化，而不在前馈预测本身；收益仍只在已测区域。

2026-10-08 06:27（hit CPU + GPU 4 推理）：[补全集成](docs/experiments/EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1/README.md)（预注册，05:41 锁定，sha256 `d6248ece…`）。**E1 = ENSEMBLE_CONSISTENT，E2 = JEPA_NO_ADD。**
- **来源：** V1 之后在 FRESH78 上的探索性分析。把学到的补全与调和插值逐像素取平均，场景一致性大幅提高。
- **检验：** EVAL34（83 个场景），从未评估过 J、JG、HG 或 HJG。
- **结果（FAR）：**
  - (调和 + 逆向 JEPA + G1) / 3：+0.131 [+0.079, +0.193]，60/20，SUPPORTED；
  - (调和 + G1) / 2：+0.133，65/15，SUPPORTED，在 ALL 上也 SUPPORTED；
  - G1 单独：+0.148，52/28，PARTIAL；
  - 逆向 JEPA 单独：+0.033，41/39，不成立；加入集成也没有贡献（−0.002）。
- **意义：** 项目里第一个在 FAR 上通过 V2 gate 的补全结果。学到的补全要与测量驱动的插值取平均才稳定；有贡献的是旧载体 G1，不是新基座上的逆向 JEPA。

2026-10-08 04:44（hit GPU 4 与另一用户的常驻进程共用 + CPU）：[以 DepthSplat 为基座检验逆向 JEPA 与 TTT](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/README.md)（预注册，01:16 锁定，sha256 `fa87f117…`）。**Core A = A_PARTIAL，Core B = B_MECHANISM_ONLY。**
- **问题：** 用户 10/7 要求"在新的基座上，验证一下逆向jepa和ttt"。新基座即 DepthSplat，评估队列为 FRESH78 = EVAL-V5 + TRAIN-EXT（78 个场景）。
- **逆向 JEPA：**
  - 从发布权重出发，用"渲染到其他帧的深度与 RGB"训练整个编码器，3 个种子，各 4000 步；
  - FAR 覆盖率从 2% 升到 68%，FAR AbsRel 从 0.425 降到 0.360；
  - 相对调和补全（0.429）+0.069 [+0.014, +0.138]，但只有 45/77 个场景变好，判定 PARTIAL；
  - 在 G1 之上的增量 +0.015，不显著：补全水平与旧项目的 G1 相当。
- **TTT：**
  - 7 帧 RGB-D 状态上做 100 步测试时优化，ALL 从 0.148（只缓存）降到 0.118，73/78 个场景变好，SUPPORTED；
  - 相对 7 帧无模型重投影（0.175）+0.057，但只有 54/78 个场景变好，判定 PARTIAL，中位数只好 0.005；
  - 收益全部来自已测区域，对 7 帧都没看到的区域无作用。
- **事故与修订：**
  - 第一次正式 GPU 作业因显存不足，三个训练在第一步退出，按同一命令重跑（修订 1）；
  - 应用户要求先读 TTT 的真值（修订 2）；
  - 两者都没有改动设置。
- **后续：** 无网络高斯加同样的 TTT 的消融（EXP-3D-RGBD-DEPTHSPLAT-TTT-ABLATION-V1）已在 DEV 上开始选参。

2026-10-06 23:25（hit，只用 CPU，不训练）：新研究线 BUDGETED-MEASUREMENT-PLANNING 的第一阶段 [oracle 余量审计](outputs/EXP-MEASUREMENT-PLANNING-ORACLE-V1/README.md)：**HEADROOM_STATUS = NOT_ESTABLISHED，POLICY_TRAINING_ALLOWED = false**，方向停止。
- **问题：** RGB 与相机免费、深度按预算 K 请求时，"测哪几帧"是否有简单规则吃不掉的决策余量。
- **设计：**
  - 主队列 EVAL-V3 ∪ V4（83 个场景）；
  - 每条记录 9 个候选帧，穷举全部子集；
  - 后端为冻结的无模型 REPROJ_HARMONIC；
  - 10 个可部署规则；
  - 判定 = 交叉查询 oracle 通过 V2 gate，且余量比例 ≥ 0.20；
  - 23:08 锁定，sha256 `140364ae…`。
- **结果：**
  - 只测 1 帧深度，ALL AbsRel 就从 0.289 降到 0.197（LATEST）或 0.172（oracle）；
  - 最强的规则在 K = 1、2、4 时都是"测最新的帧"（LATEST）；
  - 样本内 oracle 比它好 0.025、0.021、0.013，但交叉查询只剩 +0.005 [−0.004, +0.016]、+0.004、−0.001，都不显著；
  - 排除相邻帧（GAP3）、远处查询（EVAL-FAR）都没有可推广的余量；远处查询上覆盖类贪心最强，交叉查询 oracle 反而更差。
- **意义：** 测量价值很大，但在这批数据上"测哪一帧"几乎就是"测离查询最近、也就是最新的帧"。不训练效用预测器，理由见 `negative_result.md`。
- **约束：** 保留集未打开，未下载新数据，未推送；代码在本地分支 `research/measurement-planning`。

2026-10-06 01:10（hit GPU 4 与另一用户的常驻进程共用 + CPU）：[DepthSplat 探针](docs/experiments/EXP-3D-RGBD-DEPTHSPLAT-PROBE-V1/README.md)（探索性）。
- **背景：** 用户 10/5 指示以 DepthSplat 为基座进行尝试。未预注册；没有动用 EVAL-V5、TRAIN-EXT、Replica 和保留集。
- **原任务（3 个 RGB-D 上下文）：**
  - 零样本只用 RGB 时，远差于调和插值（DEV FAR 0.877 对 0.328）；
  - 注入实测深度后与调和插值相同，因为像素对齐的高斯碰不到 FAR。
- **额外的只有 RGB 的帧（每条记录 9 个未使用的帧），在 EVAL-V3/V4（83 个场景）上：**
  - FAR 从 0.487 降到 0.352，与 G1 相当；
  - 与 G1 组合后为 0.314，相对 FAR_G1 为 +0.041 [−0.025, +0.096]（49/29）；
  - 但收益主要来自紧邻查询的帧：只保留帧序距查询 ≥ 3 的帧后，相对 FAR_G1 为 −0.005；
  - 同样这些帧带实测深度时，仍有 +0.078 [+0.009, +0.131]（62/10）。
- **微调（TRAIN72 + SEG-V1）：** 单视角深度 AbsRel 从 0.142 降到 0.109，FAR 的结论不变。
- **意义：** DepthSplat 的作用相当于估计相邻 RGB 图的深度，不能补全未观测区域；测量的覆盖仍是 FAR 的主导因素。DEV（8 个场景）与 EVAL-V3/V4 的方向不同，DEV 不足以评价这类方法。

2026-10-05（hit）：ARKitScenes 小规模试点（72 对 288 次扫描）已暂停。
- 已完成：选择阶段（`outputs/EXP-3D-RGBD-ARKIT-DATA-V1/selection.json`）。
- 未解决：下载后解压第一个压缩包时出错（unzip 退出码 2）。
- 暂停原因：用户改为指示以 DepthSplat 为基座进行尝试。

2026-10-04 06:36（hit）：[V33](outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33/PLAN_BEFORE_V33.md) 按预先写定的条件未运行：V32 的 P1 均值为 −0.0020，不大于 0（`run_condition.json`）。

2026-10-04 06:34（hit GPU 训练 + CPU 评价）：[V32](outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/README.md)（覆盖整个体积的三维上下文分支，分支 C；两组的预注册锁在冻结后未变）。
- **主判定（EVAL-V3 ∪ EVAL-V4，80 个场景，第 24000 步）：**
  - P1 FAR(G1) − FAR(U) = −0.0020 [−0.0200, +0.0206]，36/0/44，中位数 −0.0028，NOT_ESTABLISHED；
  - P2 调和插值 − AHFILL8(U) = +0.0329 [+0.0127, +0.0564]，45/3/35，NOT_DISTINGUISHABLE。
- **描述性：**
  - 学得更快：第 6000 步 U 好 +0.0178 [+0.0070, +0.0288]（57/23），第 3000–20000 步的 FAR 误差都低于 G1；到第 24000 步 G1 追上（0.353 对 0.355）；
  - 第 24000 步的其他队列上 U 更差：EVAL-FAR −0.0137（29/49），FRESH-V2 −0.0376（5/9）；
  - NEAR 与全部像素几乎相同（+0.0018）；U 共 426,094 个参数，每个 seed 训练约 1.85 小时。
- **意义：** 覆盖整个体积的三维上下文只让学习更快，没有提高最终的补全水平，在其他队列上反而略差。结合 V31，图像特征与三维感受野都不是限制最终补全的主要瓶颈；容量更大时泛化略差，更像是 72 个训练场景的数据限制或任务本身的歧义。

2026-10-04 05:30（hit）：[V33](outputs/EXP-3D-RGBD-ARMS-CONFIRM-V33/PLAN_BEFORE_V33.md) 已冻结（只用 CPU，不训练），在 V32 出结果之前。
- **问题：** 在三个新队列上确认三维上下文分支 U（V32）相对 G1 的效果：EVAL-V5（23）、TRAIN-EXT（55）、Replica（8），都与各组的 TRAIN72 不重叠。P1 = FAR(G1) − FAR(U)（EVAL-V5 ∪ TRAIN-EXT，V2 gate），P2 = 调和插值 − AHFILL8(U) 的标签；P（V31）与 Replica 只作描述。
- **预先写定的运行条件：** 只在 V32 的 P1 均值大于 0 时运行；运行脚本等待 V32 的评价结束后自动检查。
- **准备：** 通用多组评价脚本 `scripts/evaluate_rgbd_carrier_arms.py` 已在 TRAIN 替代队列上预演通过；测试 3 个通过；完整测试 1159 个通过、3 个跳过；预注册 sha256 `9ebd3e79…`。

2026-10-04 04:35（hit GPU 训练 + CPU 评价）：[V31](outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/README.md)（冻结的 DINOv2 特征，分支 C；两组的预注册锁在冻结后未变）。
- **主判定（EVAL-V3 ∪ EVAL-V4，80 个场景，第 24000 步）：**
  - P1 FAR(G1) − FAR(P) = +0.0041 [−0.0139, +0.0235]，36/0/44，中位数 −0.0045，NOT_ESTABLISHED；
  - P2 调和插值 − AHFILL8(P) = +0.0347 [+0.0159, +0.0568]，47/3/33，NOT_DISTINGUISHABLE（G1 为 +0.0338）。
- **描述性：**
  - FAR AbsRel：G1 0.353，P 0.349；第 6000 步时 P 更差（−0.024）；
  - 其他队列：EVAL-FAR −0.015、FRESH-V2 −0.022，P 略差；
  - NEAR +0.014、全部像素 +0.012，P 略好，但 CI 都跨 0；
  - 三个 seed 都完成，每个约 1.55 小时。
- **意义：** 预训练图像特征没有让测量覆盖不到的像素上的补全变好。逐视角的特征只沿光线提升到体素上，到不了没有观测支持的区域，这与 V32 要检验的三维感受野瓶颈一致。

2026-10-04 02:12（hit）：[V32](outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/PLAN_BEFORE_V32.md) 已冻结（[修订 1](outputs/EXP-3D-RGBD-UNET-CONTEXT-V32/PLAN_AMENDMENT_1.md)），GPU 训练排在 V31 的训练之后。
- **问题：** V17 L carrier 在三维上的补全只靠两层 3×3×3 卷积（感受野 5³ 个体素），离有观测支持的体素超过 2 个体素的位置得不到场景上下文。加上覆盖整个体积的三维 U-Net 残差分支（输出层初值为零）之后，FAR 上的补全能否稳定变好（P1 = FAR(G1) − FAR(U)，第 24000 步，V2 gate）？混合方法能否稳定胜过调和插值（P2 标签）？对照是 V21 的 G1。
- **准备：** 测试 6 个通过（输出层为零时状态与 V17 L 逐位相同、每个输出体素都能看到整个体积等）；CPU 冒烟测试第 1 步的损失与 V17 L 原始 CPU 训练逐位相同；GPU 冒烟测试按修订 1 在 GPU 任务的第一步进行；完整测试 1156 个通过、3 个跳过；freeze.json sha256 `8df99840…`。

2026-10-04 01:45（hit）：[V31](outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/PLAN_BEFORE_V31.md) 已冻结，GPU 训练在 4 号卡上排队（空闲即开始）。
- **方向：** 用户 10-04 选择接入预训练视觉特征，并同意下载权重（DINOv2 ViT-S/14 reg4，Apache 2.0，88.3 MB，放在仓库外的 `_models/dinov2/`，代码与权重都按哈希锁定）。
- **问题：** carrier 每个视角只有 8 个从零学出的图像通道。接入冻结的 DINOv2 patch 特征（投影到 24 个通道，经初值为零的旁路进入融合）之后，FAR 上的补全能否稳定变好（P1 = FAR(G1) − FAR(P)，第 24000 步，V2 gate）？混合方法能否稳定胜过调和插值（P2 标签）？对照是 V21 的 G1。
- **准备：** 测试 6 个通过（旁路为零时状态与 V17 L 逐位相同、骨干冻结且不进入 checkpoint 等）；[冒烟测试](outputs/EXP-3D-RGBD-PRETRAINED-FEATURES-V31/audit/smoke_v31.md)中 3 个 seed 第 1 步的损失与 G1 逐位相同，每步约 0.23 秒，CPU 与 GPU 的 DEV 结果最大差 3.6e-8；完整测试 1150 个通过、3 个跳过；freeze.json sha256 `a49ca180…`。

2026-10-03 22:03（hit GPU）：[Core B V11](outputs/EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11/README.md)（逐帧在线写入，分支 B，完整性 PASS）。
- **全部像素的 AbsRel（EVAL-V3 ∪ EVAL-V4）：** NO_STREAM 0.269；OFF 0.207；TTT_100 0.186；ONLINE_100 0.193；ONLINE_NEW_100 0.190；ALL_TTT_100 0.184。
- **主判定：**
  - P1 OFF − ONLINE_100 = +0.0145 [+0.0076, +0.0225]，68/0/15，SUPPORTED；
  - P2 ½(OFF + TTT_100) − ONLINE_100 = +0.0038 [−0.0002, +0.0081]，52/0/31，NOT_ESTABLISHED（在线保住一次性写入增益的约 2/3）；
  - P3 ONLINE_NEW_100 − ONLINE_100 = −0.0027 [−0.0082, +0.0017]，41/0/42，NOT_ESTABLISHED（回放已到达的旧帧没有帮助）。
- **描述性：** 在线比一次性差 0.0068（70/13）；一次性写入再加上下文监督只多 +0.0020（CI 跨 0）；特异性 +0.0191；NO_STREAM 与 OFF 的 996 行与 Core B V8 逐一相同。

2026-10-03 20:01（hit GPU + CPU）：[V30](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30/README.md)（只有 RGB 的新帧只作写入的监督，分支 D，完整性 PASS）。
- **全部像素的 AbsRel：** NO_STREAM 0.269；CTX_TTT_100 0.251；RGB_SUP_100 0.349；CTX_RGB_SUP_100 0.250；REPROJ3_HARMONIC 0.289。
- **主判定：**
  - P1 NO_STREAM − RGB_SUP_100 = −0.0796 [−0.1180, −0.0464]，20/0/63，HARMFUL；
  - P2 CTX_TTT_100 − CTX_RGB_SUP_100 = +0.0013 [−0.0052, +0.0106]，35/0/48，NOT_ESTABLISHED。
- **描述性：** 只用新帧 RGB 监督时 FAR 与 NEAR 都变差（−0.066 / −0.080）；混合方法相对 REPROJ3_HARMONIC +0.0354（51/3/29），与不加新帧的混合（+0.0339）、只用上下文精修的混合（+0.0358）相同；封存前流帧只以 RGB 读取 1992 次，没有读取深度。
- **意义：** 只有光度监督时，写入为了拟合新帧的颜色而破坏几何；有上下文深度约束时，新帧的 RGB 也不增加信息。结合 V29：现有 carrier 用不上没有深度的新帧。

2026-10-03 18:44（hit GPU + CPU）：[V28](outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/README.md)（Replica 8 个场景上的流帧写入，分支 B，完整性 PASS）。
- **全部像素的 AbsRel：** NO_STREAM 0.475；OFF 0.470；TTT_100 0.445；TTT_1000 0.425；常数预测 0.411；REPROJ3_HARMONIC 0.331；REPROJ7_HARMONIC 0.062。
- **主判定：**
  - P1 OFF − TTT_100 = +0.0250 [+0.0074, +0.0454]，7/0/1，SUPPORTED（换了数据集写入仍然有效；1000 步再多 +0.020）；
  - P2 REPROJ7_HARMONIC − HFILL8_7(TTT_100) = −0.1468 [−0.2637, −0.0531]，1/0/7，HARMFUL。
- **描述性：** 7 帧 NEAR 只占 70.0%（Hypersim 上约 94%），carrier 在其余像素上远不如调和插值（FAR7 −0.437）；carrier 与常数预测相当（−0.034，3/5）；归一化读出下写入增益只剩 +0.0046（CI 跨 0）。
- **意义：** 只在 Hypersim 上训练的 carrier 迁移到 Replica 基本失败；写入能在这个很差的起点上稳定改进，但离无模型基线很远。

2026-10-03 18:23（hit）：[V30](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30/PLAN_BEFORE_V30.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-SUPERVISION-V30/PROTOCOL.md)），排在 4 号卡的队列中（V28 之后、Core B V11 之前）。
- **问题：** V29 中缓存只有 RGB 的新帧有害。本轮所有策略都不缓存这些帧，状态只由带深度的 3 个上下文帧构造，新帧的 RGB 只作测试时写入的监督。P1：只用新帧 RGB 的写入能否稳定改进状态（NO_STREAM − RGB_SUP_100）？P2：在上下文 RGB-D 监督之外再加新帧的 RGB 监督，能否稳定胜过只用上下文监督的写入（CTX_TTT_100 − CTX_RGB_SUP_100）？
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景，与 V29 相同），V17 L；NO_STREAM、CTX_TTT_100、RGB_SUP_100、CTX_RGB_SUP_100 与 CTX_RGB_SUP_100_WRONG_SCENE；流帧读取时不打开深度文件；基线与 3 帧混合与 V29 相同。
- **准备：** 测试 4 个通过（没有策略缓存流帧、评价后 Core B V11 的函数恢复原样）；CPU 预演通过；完整测试 1144 个通过、3 个跳过；预注册 sha256 `5b51493e…`。

2026-10-03 18:01（hit GPU 训练 + CPU 评价）：[V26](outputs/EXP-3D-RGBD-FAR-FOCUS-V26/README.md)（训练光线集中在测量覆盖不到的像素上，分支 C；两组的预注册锁在冻结后未变）。
- **主判定（主队列 EVAL-V3 ∪ EVAL-V4，80 个场景，第 24000 步）：**
  - P1 FAR(G1) − FAR(GF) = +0.0198 [−0.0045, +0.0511]，42/0/38，中位数 +0.0054，NOT_ESTABLISHED；
  - P2 调和插值 − AHFILL8(GF) = +0.0348 [+0.0128, +0.0605]，49/3/31，NOT_DISTINGUISHABLE（G1 为 +0.0338，几乎相同）。
- **描述性：**
  - GF 的 FAR AbsRel 在 12 个检查点上的平均值都低于 G1（第 24000 步 0.333 对 0.353），但改善集中在少数场景；
  - 其他队列上不成立：第 24000 步 EVAL_FAR −0.0124（32/46），FRESH_V2 −0.0194（8/6），GF 反而更差；
  - NEAR 上没有明显代价（−0.0068，CI 跨 0）。
- **意义：** 只改变训练光线的分布，不能让 carrier 在测量覆盖不到的像素上稳定变好；混合方法相对调和插值的优势，仍然只在约 60% 的场景成立。

2026-10-03 17:52（hit GPU + CPU）：[V29](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29/README.md)（只有 RGB 的新帧，分支 D，完整性 PASS）。
- **全部像素的 AbsRel（EVAL-V3 ∪ EVAL-V4）：** NO_STREAM 0.269；CTX_TTT_100 0.251；OFF_RGB 0.308；TTT_RGB_100 0.295；3 帧无模型基线 REPROJ3_HARMONIC 0.289。
- **主判定：**
  - P1 NO_STREAM − TTT_RGB_100 = −0.0261 [−0.0448, −0.0087]，27/0/56，HARMFUL；
  - P2 REPROJ3_HARMONIC − HFILL8_3(TTT_RGB_100) = +0.0205 [+0.0008, +0.0434]，43/3/37，PARTIAL。
- **描述性：**
  - 只缓存没有深度的新帧就让状态变差 −0.0386（32/51）；写入挽回 +0.0125（38/45，CI 跨 0）；
  - 不加新帧的混合（+0.0339，52/3/28）与只用上下文精修的混合（+0.0358）都比加了 RGB 新帧的混合好；
  - 若流帧有深度，7 帧无模型基线还要好 0.094（73/10）；
  - 封存前流帧共以 RGB 读取 1992 次，没有一次读取深度；4,980 个指标重新计算一致。
- **意义：** 现有的 carrier 只在带深度的视图上训练过，缓存没有深度的帧对它是分布外的输入，反而有害；只用 RGB 的写入不足以抵消。要利用只有 RGB 的新帧，需要不缓存它们、只用于监督，或者让 carrier 在训练中见过没有深度的视图。

2026-10-03 15:03（hit GPU）：[Core B V10](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10/README.md)（四个 carrier 上的写入增益，分支 A，完整性 PASS）。
- **OFF − TTT_100（EVAL-V3 ∪ EVAL-V4，83 个场景）：**
  - V18 L（宽度 32，TRAIN127）+0.0244 [+0.0147, +0.0366]，77/0/6，SUPPORTED；
  - V19 L（宽度 64，TRAIN72）+0.0286 [+0.0184, +0.0406]，70/0/13，SUPPORTED；
  - V20 L（宽度 64，TRAIN127）+0.0327 [+0.0211, +0.0465]，75/0/8，SUPPORTED；
  - 参照 V17 L +0.0213（73/0/10），与 Core B V8 相同。
- **描述性：**
  - 宽度对增益的作用 +0.0078 [+0.0026, +0.0134]（60/23），更宽的 carrier 从写入中得到的更多；训练数据 +0.0036、交互 +0.0010，CI 都跨 0；
  - 四个 carrier 的 FAR 区域都改善 +0.017 到 +0.026，特异性 +0.026 到 +0.034；
  - V17 L 的 NO_STREAM、OFF 共 996 行与 Core B V8 逐位相同，TTT_100 只差 GPU 非确定性带来的 ≤3.4×10⁻⁴；15,936 个指标重新计算一致，NEAR 掩码 332/332 与 V17 相同。
- **意义：** 写入胜过只缓存，在宽度与训练数据不同的四个 carrier 上都成立，宽度越大增益越大。这仍是 carrier 内部的结论；与只用同样测量的无模型基线相比的实际价值见 V24、V25、V27。

2026-10-03 14:28（hit）：[V29](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29/PLAN_BEFORE_V29.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-RGB-ONLY-V29/PROTOCOL.md)），排在 4 号卡的队列中。
- **问题：** 新帧只有 RGB 与相机、没有深度时（很多流式场景没有深度传感器），无模型基线只能用 3 个上下文帧的测得深度，而 carrier 仍可以缓存这些 RGB 帧并用它们的 RGB 写入。P1：只有 RGB 的新帧加写入是否稳定改进状态（NO_STREAM − TTT_RGB_100）？P2：带写入的 3 帧混合是否稳定胜过只能用 3 帧的无模型基线（REPROJ3_HARMONIC − HFILL8_3(TTT_RGB_100)）？
- **设计：** EVAL-V3 ∪ EVAL-V4，V17 L；NO_STREAM、CTX_TTT_100、OFF_RGB、TTT_RGB_100 与 TTT_RGB_100_WRONG_SCENE。流帧读取时不打开深度文件，深度记为无效，carrier 的深度统计与损失都只用它们的 RGB；封存之前的读取记录逐一核对。7 帧（含流帧深度）的无模型基线在封存之后计算，只作参照。
- **准备：** 测试 4 个通过（读取器从不打开深度、评价后读取器恢复原样）；CPU 预演通过；完整测试 1140 个通过、3 个跳过；预注册 sha256 `f3a1c9bb…`。

2026-10-03 14:06（hit）：[V28](outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/PLAN_BEFORE_V28.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-REPLICA-V28/PROTOCOL.md)），排在 4 号卡的队列中（Core B V10 之后，与 V11、V26 依次运行）。
- **问题：** 跨数据集（Replica，8 个场景）：冻结的 carrier 在那里比常数还差。100 步流帧写入能否稳定胜过只缓存（P1）？带写入的混合方法能否稳定胜过只用同样 7 帧的无模型基线（P2）？
- **设计：** NO_STREAM、OFF、OFF_CTX_TTT_100、TTT_100、TTT_1000（描述性剂量）与 TTT_100_WRONG_SCENE；评价用 Core B V8 的封存评价；基线与混合直接调用 V27 冻结的代码；另加常数预测，并与 REPLICA-TRANSFER-V1 的结果行核对。Replica 只在本机使用。
- **准备：** CPU 预演通过；完整测试 1136 个通过、3 个跳过；预注册 sha256 `a1019f23…`。

2026-10-03 13:43（hit CPU）：[V27](outputs/EXP-3D-RGBD-HYBRID-CONFIRM-V27/README.md)（两个新队列上的混合方法与"取更远深度"规则，分支 D，完整性 PASS）。
- **做法：** 在 TRAIN-EXT（55 个场景，Core B V9 封存的预测）与 EVAL-V5（23 个场景，Core B V7 封存的预测）上，重做 V24 的比较，并第一次预注册检验探索性分析得到的 MAXFILL 规则（NEAR7 用调和插值，其余取 carrier 与调和插值中更远的深度）。只在 CPU 上重新计分。
- **覆盖：** 7 帧 NEAR 占有效像素 95.8%（TRAIN-EXT）与 93.2%（EVAL-V5）。
- **全部像素的 AbsRel：** REPROJ7_HARMONIC 0.1747；carrier TTT_100 0.1713（只在 19/78 个场景更好）；HFILL8_7(TTT_100) 0.1734；MAXFILL8_7(TTT_100) 0.1744。
- **主判定：**
  - P1 REPROJ7_HARMONIC − MAXFILL8_7(TTT_100) = +0.0003 [−0.0015, +0.0022]，29/31/18，NOT_ESTABLISHED；
  - P2 REPROJ7_HARMONIC − HFILL8_7(TTT_100) = +0.0013 [−0.0096, +0.0144]，36/19/23，NOT_ESTABLISHED。
- **描述性：** FAR7 像素上混合平均 +0.060（37/22，CI 跨 0），MAXFILL 比原混合差 −0.049；写入在 FAR7 上 +0.028（35/1/23，CI 不含 0）；AHFILL8 在 FAR7 上 +0.097（CI 不含 0），全部像素只有 +0.006；220 个 3 帧掩码与 GPU 试点相同，carrier 的 AbsRel 与两轮的结果行逐一相同。
- **意义：**
  - 探索中发现的"取更远深度"规则没有在新队列上重现；V24 中混合方法的 PARTIAL 优势在新队列上也没有重现；
  - 新帧带着测得深度时，carrier（含写入）相对只用同样测量的无模型基线，没有稳定的实际优势。

2026-10-03 13:23（hit）：[V26](outputs/EXP-3D-RGBD-FAR-FOCUS-V26/PLAN_BEFORE_V26.md) 已冻结（[修订 1](outputs/EXP-3D-RGBD-FAR-FOCUS-V26/PLAN_AMENDMENT_1.md)），GPU 训练排在 Core B V10、V11 之后。
- **问题：** 混合方法里 carrier 只负责测量覆盖不到的像素，但它一直按全部像素训练。把训练光线集中到这些像素上，carrier 在那里的补全能否稳定变好，混合方法能否稳定胜过调和插值？
- **设计：** GF 组 = V17 L 配方（GPU，24000 步，3 个 seed），每个角色每步 1024 条光线中至多 768 条取自 FAR 像素（距构造状态所用的全部视图的重投影命中 8 像素以外的有效像素），其余均匀；对照为 V21 已在 GPU 上训练好的 G1（均匀采样）。评价用冻结的 V17 EVAL 代码。
- **主判定：** P1 = FAR(G1) − FAR(GF)（第 24000 步，EVAL-V3/V4，V2 gate）；P2 = 调和插值 − AHFILL8(GF) 的 INPAINT 标签；分支 A/B/C 与 V21 相同。
- **修订 1（冻结前）：** CPU 冒烟测试只看训练日志的 FAR 像素数：加入 0–4 个额外视图后中位数只有 245，25% 的角色-步为 0。原规则（可放回抽 768 条）会把光线反复打在几个像素上，所以改为不放回抽取 min(768, FAR 像素数) 条。
- **准备：** 训练脚本由 V17 的派生，只改采样；两次冒烟（CPU）都正常，第一次到第 100 步的 checkpoint 与 DEV 评价；完整测试 1131 个通过、3 个跳过。

2026-10-03 12:55（hit CPU）：[V25](outputs/EXP-3D-RGBD-STREAM-FAR-V25/README.md)（远处 query 上的 Core B 状态与使用同样帧的无模型基线，分支 D，完整性 PASS）。
- **做法：** 把 Core B V8 封存的状态在 EVAL-FAR 的远处 query 上渲染（79 个场景，每个场景 2 个远处帧；V8 构造状态时从未接触这些帧），在 CPU 上计分，不构造新的状态。EVAL-FAR 原记录会让流帧规则选中远处帧（158 个 episode 全部如此），所以读取时把远处帧放进 query 角色，流帧与 V8 逐一核对相同。
- **覆盖：** 7 帧 NEAR 占远处 query 有效像素的 87.1%（3 帧为 77.3%）。
- **全部像素的 AbsRel：** REPROJ7_HARMONIC 0.1939；carrier OFF 0.2242、TTT_100 0.2139；HFILL8_7(OFF) 0.1798、HFILL8_7(TTT_100) 0.1794；归一化读出的混合 HFILL8_7N(TTT_100) 0.1738。
- **主判定：**
  - P1 REPROJ7_HARMONIC − HFILL8_7(TTT_100) = +0.0145 [+0.0020, +0.0301]，35/14/30，PARTIAL（62% 的场景不变差）；
  - P2 HFILL8_7(OFF) − HFILL8_7(TTT_100) = +0.0004 [−0.0007, +0.0016]，NOT_ESTABLISHED。
- **描述性：**
  - carrier 内部，写入在远处 query 上仍然稳定：OFF − TTT_100 +0.0102（58/21），特异性 +0.0189（66/13）；但只用 carrier 时比无模型基线差（−0.0200，25/54）；
  - 混合的变体相对无模型基线：AHFILL8_7 +0.0180（37/14/28），归一化读出 +0.0202 [+0.0077, +0.0355]（40/14/25）；
  - FAR7 像素上 carrier 平均好于调和插值（+0.066，36/29，CI 跨 0）；
  - 核对：316 个 3 帧掩码与调和插值与 V17 相同；NO_STREAM 的渲染与 V17 第 24000 步的预测逐图最大差的中位数 1.2e-5；CPU 重渲染与 V8 的 GPU 预测一致（中位数 2.9e-5）。
- **意义：**
  - 在测量覆盖更少的远处 query 上，carrier 对无模型基线的补充平均为正，但与 Core A 一样只在约 60% 的场景成立；
  - 写入改善了 carrier 本身，但在实际可用的混合方法里几乎不起作用，因为它改变的主要是测量已经覆盖的像素；
  - Core A 与 Core B 的结论汇合到同一个问题：测量覆盖不到的区域上，carrier 的补全在场景之间不一致。

2026-10-03 12:30（hit GPU）：[Core B V9](outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9/README.md)（300 步流帧写入与监督来源的重现，分支 D，完整性 PASS）。
- **query AbsRel（TRAIN-EXT，55 个场景）：** NO_STREAM 0.230；OFF 0.191；OFF_CTX_TTT_100 0.172；TTT_100 0.156；TTT_300 0.151。
- **主判定：**
  - P1 TTT_100 − TTT_300 = +0.0041 [−0.0033, +0.0096]，44/0/11 个场景，NOT_ESTABLISHED（第三个 seed 为负）；
  - P2 OFF_CTX_TTT_100 − TTT_100 = +0.0162 [+0.0038, +0.0296]，37/0/18，PARTIAL（67% 的场景不变差，与 Core B V8 的 69% 相近）。
- **描述性：**
  - 超出只缓存的增益：100 步 +0.0355（51/4），300 步 +0.0396（48/7）；上下文监督本身 +0.0193（45/10）；
  - 特异性 TTT_300 +0.0651（48/7）；FAR 区域上剂量 +0.0028、监督来源 +0.0171，CI 都跨 0；
  - NO_STREAM 与 OFF 的 1320 行 AbsRel、660 个状态哈希都与 Core B V6 逐位相同；NEAR 掩码 220/220 与 GPU 试点相同。
- **意义：** 流帧写入的剂量在 100 步附近基本饱和；"新帧监督胜过上下文监督"在第二个队列上平均仍然成立，但仍只有约 2/3 的场景一致。

2026-10-03 11:49（hit CPU）：[V24](outputs/EXP-3D-RGBD-STREAM-BASELINES-V24/README.md)（Core B 与使用同样帧的无模型基线比较，分支 D，完整性 PASS）。
- **做法：** 只在 CPU 上重新计分 Core B V8 封存的 EVAL-V3/V4 预测（83 个场景），不运行模型。无模型基线把 3 个上下文帧与 4 个流帧（共 7 帧）的测得深度重投影到 query，再用调和插值补洞（REPROJ7_HARMONIC）；混合方法 HFILL8_7(p) 在 7 帧 NEAR 像素（距重投影命中 8 像素以内）用调和插值，其余像素用策略 p 的 carrier 深度。
- **覆盖：** 7 帧 NEAR 占 query 有效像素的 93.8%（3 帧为 79.3%），留给 carrier 的 FAR7 像素约 6%；332 个 query 中 200 个有 FAR7 像素。
- **全部像素的 AbsRel：** REPROJ7_HARMONIC 0.1746；carrier OFF 0.2071、TTT_100 0.1858；HFILL8_7(OFF) 0.1667、HFILL8_7(TTT_100) 0.1666。
- **主判定：**
  - P1 HFILL8_7(OFF) − HFILL8_7(TTT_100) = +0.0001 [−0.0004, +0.0007]，35/20/28，NOT_ESTABLISHED；
  - P2 REPROJ7_HARMONIC − HFILL8_7(TTT_100) = +0.0081 [+0.0000, +0.0189]，31/19/33，PARTIAL。
- **描述性：**
  - 只用 carrier 时，TTT_100 比 7 帧无模型基线差 −0.0112（只有 19/83 个场景更好）；
  - FAR7 像素上 carrier 平均更好：TTT_100 比调和插值好 +0.0601 [−0.0034, +0.1342]（34/30），TTT_100 比 OFF 好 +0.0139（37/27），都不一致；
  - 流帧让无模型基线从 0.289 降到 0.175（+0.114，78/5）；3 帧混合相对调和插值 +0.034，与 V17 的 P2 一致；
  - 332 个 3 帧 NEAR 掩码与调和插值都与 V17 封存的数组完全相同。
- **意义：**
  - 新帧带着测得深度到达时，直接重投影已覆盖约 94% 的 query 像素，单靠它就胜过 carrier 加 100 步写入；
  - "写入胜过只缓存"在 carrier 内部成立（Core B V5–V8），但在实际可用的混合方法里，写入的作用只剩远处约 6% 的像素，整体增益约为 0；
  - carrier 真正可能有价值的地方，是测量覆盖不到的远处区域。今后的 Core B 对比应把同样帧的无模型基线作为必备的参照。

2026-10-03 11:32（hit GPU）：[Core B V11](outputs/EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11/PLAN_BEFORE_COREB_V11.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-ONLINE-V11/PROTOCOL.md)），排在 Core B V9、V10 之后。
- **问题：** 帧一个一个到达时逐帧写入（每到一帧缓存并写 25 步，共 100 步，快速权重与 Adam 动量接着用），能否保住一次性写入（4 帧都到达后写 100 步）的增益？
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景），V17 L；NO_STREAM、OFF、TTT_100（与 Core B V7/V8 相同）、ONLINE_100（回放已到达的帧）、ONLINE_NEW_100（只监督最新一帧）、ALL_TTT_100（上下文与流帧一起监督）与 ONLINE_100_WRONG_SCENE。
- **主判定：** P1 OFF − ONLINE_100；P2 ½(OFF + TTT_100) − ONLINE_100（保住一半以上的增益）；P3 ONLINE_NEW_100 − ONLINE_100（回放的作用，单独报告）；分支 A（P1、P2 都 SUPPORTED）/B/C。
- **准备：** 只有 1 个流帧时在线写入与一次性写入逐位相同（测试）；CPU 预演通过；报告里补了一句说明（在线写入首末两步监督的帧不同，损失不能直接比较）之后，完整测试重跑通过（1122 个通过、3 个跳过）；预注册 sha256 `b23f1cb9…`。

2026-10-03 11:08（hit GPU）：[Core B V10](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10/PLAN_BEFORE_COREB_V10.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CARRIER-V10/PROTOCOL.md)），排在 Core B V9 之后，4 号卡空闲时开始。
- **问题：** 100 步流帧写入超出只缓存的增益，是否在另外三个 carrier 上也成立？四个 carrier 都训练到第 24000 步，构成宽度（32/64）× 训练数据（TRAIN72/TRAIN127）的 2×2：V17 L（Core B V4–V9 所用）、V18 L、V19 L、V20 L。
- **设计：** EVAL-V3 ∪ EVAL-V4（83 个场景）；每个 carrier 做 NO_STREAM、OFF、TTT_100 与 TTT_100_WRONG_SCENE；P1–P3 = V18 L、V19 L、V20 L 上的 OFF − TTT_100，三者都 SUPPORTED 为分支 A；V17 L 作本轮内的参照，并与 Core B V8 的行逐行比较（描述性）。
- **准备：**
  - 新的载入函数用各自实验的 `load_checkpoint` 读出 checkpoint，再放进 Core B 的 carrier 类；对 V17 L 与 Core B V3 的载入逐位相同（测试）；12 个 checkpoint 的 sha256 都与各自 `dev_curve.json` 的记录相同；
  - CPU 预演通过；完整测试 1116 个通过、3 个跳过；预注册 sha256 `aba2a5f1…`；
  - 冻结时 Core B V9 正在运行，它的目录不在本轮对旧产物的封存范围内。

2026-10-03 10:45（hit GPU）：[Core B V9](outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9/PLAN_BEFORE_COREB_V9.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-DOSE-V9/PROTOCOL.md)），正在 4 号卡上运行。
- **问题：** 在 TRAIN-EXT（55 个场景，V17 L 没有在上面训练过；Core B V6 只在上面算过 10 步的策略）上：
  - P1：流帧写入从 100 步加到 300 步是否稳定更好（TTT_100 − TTT_300）？
  - P2：100 步时新帧监督是否稳定胜过上下文监督（OFF_CTX_TTT_100 − TTT_100，重现 Core B V8 的 P2）？
- **设计：** NO_STREAM、OFF、OFF_CTX_TTT_100、TTT_100、TTT_300 与 TTT_300_WRONG_SCENE；除 TTT_300 外的策略都与 Core B V7 相同；NO_STREAM 与 OFF 和 Core B V6 逐行比较（描述性）。
- **准备：**
  - 第一次 CPU 预演被运行器拒绝：预演脚本漏设了 `NVIDIA_TF32_OVERRIDE=0`。这是预演环境的问题，记录保留在 `~/cv_work/coreb9_rehearsal_cpu_attempt1_tf32_env`；补上后预演通过；
  - 完整测试 1108 个通过、3 个跳过；预注册 sha256 `25644e43…`。

2026-10-03 10:14（hit GPU）：[Core B V8](outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/README.md)（新帧监督与上下文监督，分支 B，完整性 PASS）。
- **query AbsRel（EVAL-V3 ∪ EVAL-V4，83 个场景）：** NO_STREAM 0.269；OFF 0.207；OFF_CTX_TTT_100 0.197；TTT（10 步）0.199；TTT_100 0.186。
- **主判定：**
  - P1 OFF − TTT_100 = +0.0213 [+0.0141, +0.0297]，73/0/10 个场景，SUPPORTED；
  - P2 OFF_CTX_TTT_100 − TTT_100 = +0.0114 [+0.0053, +0.0180]，57/0/26，PARTIAL（CI 不含 0，但只有 69% 的场景不变差）。
- **描述性：**
  - 10 步 OFF − TTT = +0.0079，与 Core B V5 一致；剂量 10 → 100 步 +0.0134（70/13）；
  - 上下文监督本身 OFF − OFF_CTX_TTT_100 = +0.0099（58/25），但在 FAR 区域为 0（36/44）；
  - FAR 区域：TTT_100 相对只缓存 +0.0190，相对上下文监督 +0.0190，都不含 0；
  - 特异性 +0.0265（73/10）；EVAL-V3、EVAL-V4 分别为 +0.0181、+0.0283；NEAR 掩码 332/332 与 V17 相同。
- **意义：**
  - 100 步的流帧写入在第二个大队列上稳定胜过只缓存，是目前最大、最一致的 Core B 增益；
  - 这部分增益约一半是对状态的一般性精修，另一半来自新帧的信息；在上下文没看到的区域，改善全部来自新帧监督；
  - "新帧监督胜过上下文监督"平均上成立，但还没达到场景一致性的门槛。

2026-10-03 09:03（hit GPU）：[Core B V8](outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/PLAN_BEFORE_COREB_V8.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/PROTOCOL.md)），正在 4 号卡上运行。
- **问题：** 在 EVAL-V3 ∪ EVAL-V4（83 个场景，从未计算过上下文监督的写入与 100 步的流帧写入）上：
  - P1：100 步流帧 TTT 是否稳定胜过只缓存（OFF − TTT_100）？
  - P2：新帧都已缓存时，用新帧监督是否稳定胜过用上下文监督（OFF_CTX_TTT_100 − TTT_100）？
- **设计：** NO_STREAM、OFF、OFF_CTX_TTT_100、TTT（10 步）、TTT_100 与 TTT_100_WRONG_SCENE；策略表与 Core B V7 相同。
- **准备：** CPU 预演通过；完整测试 1104 个通过、3 个跳过；预注册 sha256 `e4848ca4…`。

2026-10-03 08:46（hit GPU）：[Core B V7](outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/README.md)（全新队列 EVAL-V5 上的 100 步测试时写入，分支 B，完整性 PASS）。
- **query AbsRel（23 个场景）：** NO_STREAM 0.286；CTX_TTT_100 0.280；OFF 0.224；OFF_CTX_TTT_100 0.220；TTT 0.217；TTT_100 0.209。
- **主判定：**
  - P1 NO_STREAM − CTX_TTT_100 = +0.0059 [−0.0003, +0.0132]，17/0/6 个场景，NOT_ESTABLISHED；
  - P2 OFF − TTT_100 = +0.0150 [+0.0069, +0.0243]，19/0/4，SUPPORTED。
- **描述性：**
  - 10 步的重现：OFF − TTT = +0.0068（17/6，PARTIAL）；剂量 10 → 100 步，流帧 +0.0081（19/4），上下文 +0.0037；
  - 特异性：TTT_100 +0.0228（22/1），CTX_TTT_100 +0.0090（21/2）；把别的场景的 100 步增量写进来，比只缓存还差；
  - 监督来源 OFF_CTX_TTT_100 − TTT_100：整体 +0.0107（CI 跨 0），FAR 区域 +0.0239 [+0.0093, +0.0417]；
  - 分区：TTT_100 的增益在 NEAR +0.0208，FAR +0.0171 [+0.0035, +0.0301]；CTX_TTT_100 在 FAR 上没有增益（−0.0033）。
- **意义：**
  - 用新帧监督的梯度写入在第四个独立队列上胜过只缓存，100 步的增益约为 10 步的两倍，并扩展到上下文没看到的区域；
  - 只用上下文的精修时有时无（TRAIN-EXT 支持，EVAL-V3/V4 与 EVAL-V5 未确立），对补全没有帮助；
  - 新帧监督相对上下文监督的额外作用主要体现在 FAR 区域，整体上还未确立。

2026-10-03 08:11（hit GPU）：[Core B V7](outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/PLAN_BEFORE_COREB_V7.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-FRESH-V7/PROTOCOL.md)），正在 4 号卡上运行。这是 EVAL-V5 的首次使用。
- **问题：** 依据 V23 的剂量结果选定 100 步之后，在全新的 EVAL-V5（23 个场景）上：
  - 100 步只用上下文的 TTT 能否稳定改进静态状态（P1 = NO_STREAM − CTX_TTT_100）？
  - 100 步用流帧监督的 TTT 能否稳定胜过只缓存新帧（P2 = OFF − TTT_100）？
- **设计：** 8 种策略（0/10/100 步 × 是否缓存流帧 × 监督上下文或流帧）与 4 个"写入别的场景"的对照；10 步的各项对比作为重现，剂量、监督来源与特异性都是描述性的。
- **准备：** 5 个基本策略与 Core B V6 逐位相同（测试）；CPU 预演通过；完整测试 1101 个通过、3 个跳过；预注册 sha256 `e7ede177…`。

2026-10-03 07:54（hit GPU）：[V23](outputs/EXP-3D-RGBD-STATIC-TTT-EVAL-V23/README.md)（只用上下文的测试时精修，Core A 主队列，分支 C，完整性 PASS）。
- **队列与 carrier：** EVAL-V3 ∪ EVAL-V4（83 个场景）；V17 L 第 24000 步。
- **全部像素的 AbsRel：** NO_STREAM 0.269；CTX_TTT（10 步）0.264；30 步 0.258；100 步 0.251。
- **主判定：**
  - P1 NO_STREAM − CTX_TTT（10 步）= +0.0054 [+0.0019, +0.0101]，59/0/24 个场景，PARTIAL（只有 71% 的场景不变差）；
  - P2 特异性 +0.0062 [+0.0023, +0.0109]，61/0/22，PARTIAL；
  - 按预注册，Core B V6 中 10 步上下文 TTT 的结论在这里没有得到严格确认。
- **描述性的剂量臂：**
  - 30 步 +0.0112，100 步 +0.0176 [+0.0079, +0.0321]，都是 66/0/17 个场景；100 步比 10 步多 +0.0121（64/19）；
  - 100 步时 FAR 也改善：+0.0124 [+0.0024, +0.0279]（NEAR +0.0195）；
  - 调和插值 − carrier（FAR）从 +0.144 增加到 +0.157；AHFILL8 从 +0.035 到 +0.037。
- **与模型无关的基线：** 332 个 NEAR 掩码与调和插值都与 V17 封存的数组完全相同。
- **意义：**
  - Core B V4 在 TRAIN 上选出的 10 步，对没见过的场景太少：carrier 正是在 TRAIN 场景上训练的，在那里更多的步数有害；在没见过的场景上，步数越多越好；
  - 下一步（Core B V7）在全新的 EVAL-V5 上，把 100 步作为主判定的设置来检验，这一选择只依据 V23 在 EVAL-V3/V4 上的描述性结果。

2026-10-03 07:26（hit）：[EVAL-V5 数据](outputs/EXP-3D-RGBD-EVAL-V5-DATA/README.md)准备完成（PASS）。
- **规则：** 按用户的决定放宽：允许 val 划分，允许与 DEV/FRESH 共享 volume；其余与 TRAIN-EXT/EVAL-V4 相同，并排除所有曾被观察、使用、读取或判为无效的场景。
- **结果：**
  - 锁定 34 个候选，23 个有效（全部来自官方 train 划分），分布在 7 个 volume；下载 332 MB；
  - 7 个 volume 都只与 DEV 共享，与 TRAIN72、FRESH、EVAL-V3/V4、TRAIN-EXT 都不共享，因此与 V17 L 的训练集按 volume 不相交；
  - 23 个场景都能取到 Core B 的 4 个流帧。
- **用途：** 机制确认队列。首个使用者在其计划中写明；在此之前不在它上面运行任何模型。

2026-10-03 07:15（hit GPU）：[V23](outputs/EXP-3D-RGBD-STATIC-TTT-EVAL-V23/PLAN_BEFORE_V23.md) 已冻结并在 4 号卡上运行：在 Core A 的主队列（EVAL-V3 ∪ EVAL-V4，83 个场景）上确认只用上下文的 TTT，并用 FAR、调和插值与 AHFILL8 评估它对补全的作用；30/100 步的剂量臂只作描述。预注册 sha256 `e629b9b9…`。

2026-10-03 06:59（hit GPU）：[Core B V6](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/README.md)（测试时写入需要新帧吗，分支 A，完整性 PASS）。
- **队列：** TRAIN-EXT，55 个场景；V17 L 从未训练过，Core B 从未使用过。设置沿用 Core B V4（10 步，学习率 1e-3）。
- **query AbsRel：** NO_STREAM 0.230；CTX_TTT（只用 3 个上下文帧做 TTT）0.222；OFF 0.191；OFF_CTX_TTT（缓存新帧，但用上下文监督）0.184；TTT（用新帧监督）0.176。
- **主判定：**
  - P1 STREAM_TTT = OFF − TTT = +0.0148 [+0.0054, +0.0245]，49/0/6 个场景，SUPPORTED：第三个独立队列上重现；
  - P2 CONTEXT_TTT = NO_STREAM − CTX_TTT = +0.0078 [+0.0026, +0.0133]，45/0/10，SUPPORTED：不加新帧、只用上下文视图做测试时优化，也能稳定改进静态状态。
- **描述性：**
  - 监督来源 OFF_CTX_TTT − TTT = +0.0073 [−0.0027, +0.0167]，CI 跨 0：缓存新帧之后，用新帧监督平均更好，但未确立；
  - 缓存新帧之后再做上下文 TTT 也有用：OFF − OFF_CTX_TTT = +0.0075 [+0.0024, +0.0133]；
  - 特异性：TTT +0.0144（46/9），CTX_TTT +0.0075（43/12）；
  - 分区：增益以 NEAR 为主（STREAM_TTT +0.016，CONTEXT_TTT +0.009）；FAR 上 STREAM_TTT +0.012 [+0.0004, +0.0242]，CONTEXT_TTT +0.003（CI 跨 0）；
  - NEAR 掩码 220/220 与 GPU 试点封存的掩码完全相同。
- **意义：**
  - 梯度式测试时写入首先是一种一般性的测试时精修，对静态 carrier 同样有效，这也是 Core A 的一个一致的改进（82% 的场景不变差）；
  - 新帧监督带来的额外增益还需要更多证据；
  - 增益主要让已观测的表面更准，对未观测区域的补全帮助有限。

2026-10-03 06:42（hit GPU）：[Core B V6](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/PLAN_BEFORE_COREB_V6.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-CONTEXT-V6/PROTOCOL.md)），正在 4 号卡上运行。
- **问题：**
  - 流帧 TTT 能否在第三个独立队列上重现？
  - TTT 的增益需要新帧吗？只用 3 个上下文帧自己的 RGB-D 做 TTT，能否改进不加流帧的静态状态？若能，就是对静态 carrier 的一般性测试时精修，也属于 Core A。
- **设计：**
  - 队列 TRAIN-EXT（55 个场景）：V17 L 从未在上面训练过，Core B 从未使用过；
  - 策略 NO_STREAM、CTX_TTT（新）、OFF、OFF_CTX_TTT（新）、TTT；对照 TTT_WRONG_SCENE、CTX_TTT_WRONG_SCENE；
  - 设置沿用 Core B V4（10 步，学习率 1e-3），不重新选择；
  - P1 = OFF − TTT，P2 = NO_STREAM − CTX_TTT，冻结的 V2 gate，分支 A/B/C/D；
  - 描述性：监督来源（OFF_CTX_TTT − TTT）、特异性、归一化读出、NEAR/FAR 分区（与模型无关的掩码，并与 GPU 试点封存的掩码逐一核对）。
- **准备：** 4 个测试（NO_STREAM/OFF/TTT 与 Core B V4 逐位相同）；CPU 预演通过；完整测试 1089 个通过、3 个跳过；06:42:46 冻结（预注册 sha256 `136d777a…`）。

2026-10-03 06:18（hit GPU）：[Core B V5](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/README.md)（在 83 个独立场景上确认基于梯度的测试时写入，分支 A，完整性 PASS）。
- **设计：** EVAL-V3 ∪ EVAL-V4 的 83 个场景，从未用于 Core B，也从未用于 carrier 的训练或选择；沿用 Core B V4 在 TRAIN 上选出的设置（10 步，学习率 1e-3），不重新选择。
- **query AbsRel：** 不加流帧 0.269；只缓存（OFF）0.207；TTT 0.199；写入别的场景的 TTT 增量 0.206。
- **主判定：**
  - TTT_BEYOND_BASELINE = OFF − TTT = +0.0079 [+0.0030, +0.0135]，65/0/18 个场景，SUPPORTED；
  - TTT_SPECIFICITY = TTT_WRONG_SCENE − TTT = +0.0068 [+0.0015, +0.0122]，67/0/16，SUPPORTED；
  - 三个 seed 都为正（+0.010、+0.004、+0.009）；归一化读出下结果相同。
- **描述性：**
  - EVAL-V3 +0.0053（44/13），EVAL-V4 +0.0135（21/5）；EVAL-V3 上特异性的 CI 跨 0；
  - 缓存新帧本身 +0.062（68/15）；92% 的 episode 中，流帧自身的损失下降。
- **事后（探索性）：** 借用 V17 的 NEAR 掩码分区：
  - TTT 超出缓存的增益集中在上下文已经看到的 NEAR 区域（+0.0082，67/16；特异性 +0.0075）；
  - 在没看到的 FAR 区域只有 +0.0047（CI 跨 0），特异性约为 0；
  - FAR 区域的改善几乎都来自新帧本身的观测（+0.089）。
- **意义：**
  - Core B 第一个在独立数据上确认的正结果：基于梯度的测试时写入胜过只缓存新帧，并且携带本场景的信息；
  - 效果小（约为 OFF 误差的 4%），主要让已观测的表面更准，而不是补全未观测的区域；
  - 第一次正式运行因封存屏障的平方级校验过慢而停止，修正后重新冻结运行（见事故记录）。

2026-10-03 05:55（hit）：[V21](outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/README.md)（同一批训练场景里更多样的片段，分支 C）。
- **设计：** G1 是 TRAIN72 的原片段（V17 L 在 GPU 上的重跑）；G5 是原片段加 240 个 SEG-V1 新片段。各 3 个 seed，24000 步，都在 GPU 上。
- **P1：** DIVERSITY_GAIN = FAR(G1) − FAR(G5)，第 24000 步：−0.023 [−0.063, +0.014]，32/0/48 个场景，NOT_ESTABLISHED（G5 略差）。
- **P2：** 调和插值 − AHFILL8(G5) = +0.031 [+0.013, +0.054]，47/3/33，NOT_DISTINGUISHABLE。
- **描述性：**
  - 每个检查点上 G5 的 FAR 都比 G1 差（第 24000 步 0.377 对 0.353）；
  - EVAL-FAR 上 G5 明显更差：−0.034 [−0.060, −0.006]；FRESH-V2 上 +0.024，CI 跨 0；
  - 设备检查：V17 L（CPU）与 G1（GPU）在第 24000 步的 FAR 相差 −0.005 [−0.016, +0.005]，GPU 重跑与 CPU 一致。
- **意义：**
  - 同一批场景里的内容多样性不能解决补全的不一致，反而略有损害，与 V14/V18 中"更多场景在 6000 步时有害"一致；
  - 加宽、加场景、加片段多样性都不是答案；
  - 结合 V22 与事后诊断：长训练的 FAR 增益大部分是读出偏差，而补全的不一致主要来自一部分稳定变差的难场景。

2026-10-03 05:25（hit CPU）：[V22](outputs/EXP-3D-RGBD-DEPTH-READOUT-V22/README.md)（长训练的补全增益是否来自未归一化的深度读出，分支 C，完整性 PASS）。
- **背景（事后、探索性，[诊断记录](outputs/EXP-3D-RGBD-POSTHOC-DIAG-V1/README.md)）：**
  - 渲染器的深度是 Σ wᵢtᵢ，没有除以不透明度，所以不透明度低于 1 时深度偏近；
  - 在 EVAL-V3/V4 上把 V17–V20 已封存的预测改用归一化读出（深度除以不透明度）重新计分，长训练的 FAR 增益从 +0.04–0.06 降到 +0.003–0.011。
- **预注册的确认（EVAL-FAR，79 个场景，远处 query，只重新计分，不运行模型）：**
  - V17 长训练增益：RAW +0.058 [+0.036, +0.079]；归一化后 +0.021 [+0.003, +0.039]，CI 仍不含 0；
  - 主判定一 NOT_EXPLAINED：残差的 CI 上界 0.039 超过 RAW 增益的一半 0.029；
  - 主判定二：第 6000 步的归一化增益 +0.050 [+0.033, +0.068]，55/0/23 个场景，PARTIAL；
  - FAR 平均不透明度：第 6000 步 0.888，第 24000 步 0.955；
  - RAW 读出与 V17–V20 的记录逐位一致。
- **描述性：**
  - V18–V20 的残差为 +0.009 到 +0.026；V18 为分支 A，V19、V20 为 C；
  - 第 24000 步的归一化增益为 +0.013 到 +0.029；NEAR 区域几乎不受影响。
- **意义：**
  - 长训练的 FAR 增益大部分（约 2/3）来自深度读出的偏差，但在远处 query 上还剩约 1/3 是真实的补全改进；
  - 探索性分析中"几乎全部来自读出"的说法没有得到确认；
  - 归一化读出几乎没有代价，平均改进明显，但还不是一致的改进（约 70% 的场景）。此后的实验应当同时报告两种读出。

2026-10-03 05:15（hit GPU）：[Core B V5](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/PLAN_BEFORE_COREB_V5.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/PROTOCOL.md)），正在 4 号卡上运行。
- **问题：** Core B V4 的 TTT（10 步，学习率 1e-3，在 TRAIN 上选出，不重新选择）在 Core B 从未使用过的 EVAL-V3 ∪ EVAL-V4（83 个场景）上，是否仍然胜过只缓存新帧，并且携带场景特异的信息？
- **设计：** carrier、episode 规则、策略、对照与判定都与 Core B V4 相同；EVAL 记录的上下文由同一个只读上下文的加载器读取，范围规则与 DEV 相同。
- **准备：**
  - 测试证明：同一批合成场景上，本轮的队列路径与 Core B V4 的 DEV 路径得到逐位相同的缓存状态；
  - CPU 预演用 4 个 TRAIN 场景充当两个 EVAL 队列，冻结与运行全流程通过；
  - 完整测试 1081 个通过、3 个跳过（隐藏 CUDA）；
  - 05:15:01 第一次冻结（预注册 sha256 `a0183207…`），05:17 开始正式运行；
  - **事故：** 冻结的封存屏障每次读取都重新计算全部 664 个状态（约 3 GB）的哈希，开销是平方级的，三个 seed 要到 12:00 以后才能完成。05:57 在写出任何指标之前停止，第一次尝试的产物原样保留（[事故记录](outputs/EXP-3D-RGBD-DYNAMIC-TTT-EVAL-V5/audit/incident_attempt1_slow_seal_barrier.md)）；
  - **修正：** 本轮新模块中加入 `CachedSealBarrier`：封存关闭时与最后一个 query 之后各完整校验一次，其余语义不变。测试证明它与每次都完整校验的屏障得到逐项相同的输出；
  - 完整测试 1085 个通过、3 个跳过，CPU 预演通过；06:06:00 重新冻结（预注册 sha256 `f2a6f0e9…`），06:06 开始 GPU 预演，通过后自动开始正式运行。

2026-10-03 04:56（hit GPU）：[Core B V4](outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4/README.md)（基于梯度的测试时写入，分支 A，完整性 PASS）。
- **TRAIN 选择：** 4 组设置在 TRAIN 上的平均增益都不为正（10 步/1e-3：−0.0001；10/1e-2：−0.011；50/1e-3：−0.004；50/1e-2：−0.010），按规则选 10 步、学习率 1e-3。
- **DEV AbsRel：** 不加流帧 0.330；只缓存（OFF）0.236；TTT 0.224；写入别的场景的 TTT 增量 0.239。
- **主判定：**
  - TTT_BEYOND_BASELINE = OFF − TTT = +0.0115 [+0.0050, +0.0187]，8/0/0 个场景，SUPPORTED；
  - TTT_SPECIFICITY = TTT_WRONG_SCENE − TTT = +0.0146 [+0.0072, +0.0224]，8/0/0，SUPPORTED；
  - 三个 seed 都为正。
- **意义：**
  - Core B 四轮中第一次：写入胜过只缓存新帧，并且携带本场景的信息。
  - 但增益不大（约为 OFF 误差的 5%），而且只在 8 个反复使用的 DEV 场景上。
  - TRAIN 选择集上没有增益，可能因为 carrier 正是在这些场景上训练的；DEV 场景是 carrier 没见过的。
  - 按计划，下一步在独立数据上确认。
- **事后检查（探索性）：** 用归一化的深度读出（深度除以不透明度）重新计分，TTT 的优势不变（+0.0120，7/0/1；特异性 +0.0152，8/0/0）。OFF 与 TTT 的 query 平均不透明度几乎相同（0.977、0.976），所以增益不是读出偏差。

2026-10-03 02:26（hit）：[Core B V4](outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4/PLAN_BEFORE_COREB_V4.md) 已冻结（[协议](outputs/EXP-3D-RGBD-DYNAMIC-TTT-V4/PROTOCOL.md)），排队等 4 号卡。
- **问题：** 在 V17 第 24000 步的 carrier 上，用新帧自己的 RGB 与测得深度，对快速权重做几步梯度下降（测试时训练，TTT），能否胜过只缓存新帧？
- **设计：**
  - 策略：NO_STREAM、OFF（只缓存）、TTT；对照 TTT_WRONG_SCENE（用别的场景的 TTT 增量）；
  - 超参数（步数 10/50 × 学习率 1e-3/1e-2）只在 32 个 TRAIN episode 上选择，选定之后才开始 DEV 评价；
  - 主判定：TTT_BEYOND_BASELINE = OFF − TTT 与 TTT_SPECIFICITY = TTT_WRONG_SCENE − TTT，用冻结的 V2 gate。
- **准备：**
  - 冻结前检查代码时发现，场景数据类初始化时会读取实验目录里的 `train_depth_prior.json`。冻结流程改为从 V7 复制它，并核对与 Core B V3 的副本相同；
  - CPU 预演用 2 个 TRAIN 场景临时充当 DEV，冻结与运行全流程通过；
  - 完整测试 1077 个通过、3 个跳过（隐藏 CUDA）。
- **执行：** V21 训练结束、释放 4 号卡之后，先在 GPU 上预演一次，通过后自动开始正式运行，预计约 2 小时。
- **V21：**
  - 01:31 冻结；第一次启动因缺 `checkpoints/` 目录在第 0 步失败（[事故记录](outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/audit/incident_launch_missing_checkpoints.md)）；
  - 02:03 以同一冻结命令重启，预计 05:15 前后训练结束，CPU 上的评价随后自动开始。

2026-10-03 01:59（hit GPU）：[Core B V3](outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V3/README.md)（在 V17 第 24000 步的 carrier 上重新检验学到的写入，分支 C，完整性 PASS）。
- **DEV AbsRel：** 不加流帧 0.330；只缓存 4 个新帧（OFF）0.236；学到的写入（ALL）0.237；未训练的写入 0.236；写入别的场景 0.236。
- **主判定：**
  - BEYOND_BASELINE = OFF − ALL = −0.0015 [−0.0079, +0.0031]（4/4 个场景），NOT_ESTABLISHED；
  - SPECIFICITY = ALL_WRONG_SCENE − ALL = −0.0011，NOT_ESTABLISHED。
- **描述性：** 缓存新帧带来 +0.094 [+0.021, +0.189]（7/8 个场景），比 Core B V2 的 +0.052 更大；训练写入规则的作用为 −0.0015。
- **意义：** carrier 变强以后，结论与 Core B V2 相同。收益全部来自把新帧纳入多视角融合，学到的快速权重写入没有额外贡献，也不携带场景信息。按现有设计，Core B 的核心主张不成立；要继续，需要重新设计写入（例如基于梯度的测试时训练，或利用新帧测得深度的写入统计）。

2026-10-03 01:4x（hit）：[V19](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-V19/README.md)、[V20](outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20/README.md)（宽度 64 × 长训练，都是分支 C），以及 V17–V20 的宽度 × 数据 × 步数 2×2×2（描述性，`outputs/EXP-3D-RGBD-LONG-TRAIN-WIDTH-DATA-V20/audit/two_by_two_by_two_*.json`）。
- **训练长度（6000 → 24000 步）：**
  - 四个组合的 FAR 改善都在 +0.042 到 +0.062 之间，CI 都不含 0；
  - 但只有约 2/3 到 3/4 的场景变好，gate 都是 PARTIAL。
- **宽度 32 → 64：** 任何组合下都没有作用（−0.002 到 +0.007，CI 都跨 0）；长训练下也一样（V19 P2 +0.004，V20 P2 −0.002）。
- **训练数据 72 → 127 个场景：** 第 6000 步时有害（−0.027、−0.032，CI 不含 0）；第 24000 步时劣势缩小到 −0.009、−0.016（CI 跨 0），没有转为有利。
- **交互项：** 两两交互与三阶交互都跨 0。最好的组合是宽度 64、TRAIN72、第 24000 步（FAR 0.344），与宽度 32 的 0.348 几乎相同。
- **[GPU 试点](outputs/EXP-3D-RGBD-GPU-LENGTH-PILOT-V1/README.md)（探索性，96000 步，TRAIN-EXT 的 55 个场景）：** 24000 步之后不再改善，宽度 64 也没有优势。
- **意义：** 在本项目的规模内，只有训练长度有用，而且约 24000 步就饱和。补全在场景之间的不一致，不能靠加宽、加数据或加长训练来解决。
- **正在进行（GPU，4 号卡）：**
  - [Core B V3](outputs/EXP-3D-RGBD-DYNAMIC-WRITE-V3/PLAN_BEFORE_COREB_V3.md)：在 V17 第 24000 步的 carrier 上重新检验学到的写入；
  - [V21](outputs/EXP-3D-RGBD-SEGMENT-DIVERSITY-V21/PLAN_BEFORE_V21_DATA.md)：用同一批训练场景相机轨迹后段的 240 个新片段，检验内容多样性。

2026-10-02 12:4x（hit）：[V17](outputs/EXP-3D-RGBD-LONG-TRAIN-V17/README.md)、[V18](outputs/EXP-3D-RGBD-LONG-TRAIN-DATA-V18/README.md)（训练到 24000 步，都是分支 C）。
- **内置对照：** 两者前 6000 步分别与 V11 C1、V14 C1 逐位相同。
- **长训练增益（FAR，第 6000 → 24000 步）：**
  - TRAIN72：+0.044 [+0.018, +0.069]，54/26 个场景；
  - 127 个场景：+0.062 [+0.030, +0.095]，55/25 个场景。
  - CI 都不含 0，留一场景都 > 0；但只有约 2/3 的场景变好，未达 75% 的一致性门槛，gate 为 PARTIAL。
- **组合方法（调和插值 − AHFILL8，第 24000 步）：**
  - TRAIN72：+0.035 [+0.015, +0.057]；127 个场景：+0.032 [+0.012, +0.055]。
  - 这是本项目最好的组合结果，但同样因场景一致性不足，标为 NOT_DISTINGUISHABLE。
- **数据 × 长度（描述性 2×2）：** 更多数据的补全劣势从第 6000 步的 −0.027 [−0.049, −0.007]，缩小到第 24000 步的 −0.009 [−0.031, +0.014]，没有反转。
- **DEV 选择：** 冻结的 DEV 规则仍选早期 checkpoint（750–5000 步），选出的模型补全比第 6000 步还差。
- **意义：**
  - 训练长度是真实但不稳定的杠杆：平均改善明确，约三分之一的场景不改善。
  - FAR 在 16000 步之后基本走平（TRAIN72：0.356 → 0.348；127 个场景：0.354 → 0.358）。
  - V19、V20（宽度 64）预计 10/3 凌晨出结果；GPU 试点（96000 步）今晚出结果。

2026-10-02 04:3x（hit）：工作从 zonghan_o 迁移到 hit（amax，8 张 RTX A6000，128 个 CPU 核）。迁移说明见镜像根目录下的 `_archives_cv/MIGRATION_cv.md`。
- **复现：** 在 hit 的 CPU 上限制 AVX2 指令集后，单线程训练与 zonghan_o 逐位一致：V15 C0 的 6000 步训练记录、第 5000/6000 步参数、12 个 DEV 检查点全部相同（[HIT-REPRO-V1](outputs/EXP-3D-HIT-REPRO-V1/README.md)）。不限制时第 1 步就不同，到第 6000 步参数明显不同，DEV 最多差 0.047，选出的 checkpoint 也不同。GPU（4 号卡）上训练每步快 7–20 倍，但无法逐位复现：强制确定性时 3D `grid_sample` 的反向没有确定性实现；两次 GPU 运行从第 2 步起就不同，选出的 checkpoint 也不同。所以需要逐位对照的正式实验都用 CPU。
- **V17–V20（宽度 × 数据 × 训练长度）：** 都已冻结，在 CPU 上并行训练到 24000 步。V17：宽度 32、TRAIN72；V18：宽度 32、127 个场景；V19：宽度 64、TRAIN72；V20：宽度 64、127 个场景。每个的前 6000 步分别必须与 V11 C1、V14 C1、V12 C1、V15 C1 逐位相同，开头的步数都已确认相同。V17、V18 预计 2026-10-02 中午出结果，V19、V20 预计当晚。
- **GPU：** cv 只用 4 号卡，并且只在它完全空闲时使用；cv 任务优先于本账号的 TTT 任务。

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

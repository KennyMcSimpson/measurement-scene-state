# EXP-3D-RGBD-EVIDENCE-CARRIER-V5 协议

起草：Claude（接手 cv 对话后，经用户授权持续推进），2026-09-29。本协议在任何 V5 训练之前冻结。

## 0. 背景

- V2：16³ learned carrier 上表面监督收益未建立；状态近似场景无关先验。
- V3：证据候选 128→4096，DEV 收益未建立（分支 C）；训练集深度 AbsRel 0.490→0.339 而 DEV 不变。
- V4：给 carrier 加零初始化的光度 plane-sweep 旁路。
  - 主对比：SWEEP_GAIN +0.0072，95% CI [−0.0041, +0.0168]，`NOT_ESTABLISHED`；三个 seed 方向一致（+0.0096/+0.0070/+0.0051）；预注册分支 C。
  - 区域诊断（seal 后，描述性）：OBS1 +0.0169（CI [+0.0005, +0.0344]），OBS2PLUS +0.0403（CI [+0.0038, +0.0871]），OBS0 −0.0007。几何证据正好在多视角可观测区域起作用，全图收益被大量不可观测区域稀释。
- 预注册事后二次分析 EXP-3D-READOUT-REFERENCE-REANALYSIS-V1（不训练，只读 V2–V4 已封存的 DEV 预测）：
  - 总体：9 个 RGB-only carrier 无一优于只由 TRAIN 拟合的无几何常数，`NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE`。
  - RAW：AbsRel 最优常数 2.37 m 的 DEV query AbsRel≈0.507，carrier 为 0.61–0.65，全部显著更差（BELOW）。
  - MEDIAN_SCALED：给定 GT 尺度之后，carrier 的深度形状仍不如平面常数。
  - 排除 NO_HIT 场景 ai_009_001 之后，仍没有 carrier 优于常数。
- V4 之后的 TRAIN-only 探索诊断（不属于任何冻结协议，未打开 DEV）：
  - 32×40 光度 plane-sweep 在真实 3 视角上下文上几乎不含深度信息：argmax 深度 AbsRel 0.518，而 TRAIN 中位数常数为 0.557；δ<1.25 为 0.29，常数为 0.43。
  - 视差不是原因：中位基线 1.26 m、旋转 18°；深度变化 25% 时，最佳源视角的中位像素位移为 2.8 px。
  - 更强的经典匹配也只有有限改善：128×160、64 平面、11×11 窗口 ZNCC 的 AbsRel 为 0.49，只在最自信的 20% 像素上优于常数。
- 已冻结的 EXP-3D-DIRECT-STATE-CAPACITY-V1 给出了关键上限：
  - 只用上下文 RGB 逐场景直接优化 16³ 状态，query AbsRel 为 0.617，与 learned carrier 相同；
  - 用 RGB+D 直接优化则为 0.403；
  - 以 query 监督的 oracle 约 0.15–0.25。
- 结论：在这一表示与 3 个 RGB 视角下，光度拟合本身恢复不出几何，RGB-only carrier 已经到达 RGB-only 直接优化的上限。问题出在输入信息，而不只是 learner。因此本轮进入 **RGB-D 轨道**：上下文视角附带测得的深度（像深度传感器那样），检验 learned shared state 能否把测得的几何融合成可泛化的场景状态。RGB-only 轨道的结论（V2–V4）不因此改变。

## 1. 研究问题

- **Q1**：给 carrier 加入测得的上下文深度线索后，DEV query 几何是否稳定改善？
- **Q2**：状态是否变得场景专属（wrong-scene 与 spatial shuffle 损伤的 CI 下界 > 0）？3 个 RGB-D 视角是否稳定优于 1 个 RGB-D 视角（anchor）？
- **Q3**：改善是否出现在可观测区域（OBS1、OBS2PLUS）？
- **Q4**：与只由 TRAIN 拟合、不含几何信息的常数深度相比，各变体处于什么位置？（描述性，不作 gate）
- **Q5（次要）**：有了测得深度后，表面监督是否还有作用？

## 2. 测得深度线索（固定定义）

对状态的每个上下文视角 r，设测得的射线距离图为 D_r（数据集约定：公制欧氏射线距离），对每个体素中心 x：

1. 投影到视角 r，取最近像素（全分辨率 128×160）的 D_r(p)。只有投影位于相机前方、落在图像内、且 D_r(p) 有限且大于 0 时，该视角对该体素有效。
2. d = ‖x − c_r‖（体素中心到相机中心的距离）；s = exp(log_sigma) × 平均体素边长。
3. 表面似然 a_r = exp(−(d − D)² / 2s²)；自由空间似然 f_r = sigmoid((D − d)/s)，表示体素位于测得表面之前。
4. 体素统计量：对有效视角取平均的 a、f，以及有效视角比例 q，共 3 维；没有有效视角时三者都为 0。log_sigma 是唯一可学习的深度参数，初值 log(0.5)。

深度线索只读上下文视角的测得深度与相机。上下文深度只通过只含上下文帧的 RGBD context-only loader 进入状态构建；query 深度只用于 TRAIN loss，以及全部 DEV 状态 seal 之后的评价。

## 3. 变体

全部沿用 V3 C1 / V4 C0 的配方：16³、4096 个证据候选、64 samples 固定 renderer、冻结 GT-free bounds、同 seed 下共享参数初值完全相同。

| ID | 名称 | 深度旁路 | loss | 角色 |
|---|---|---|---|---|
| C0 | DENSE_SURFACE | 无（只读 RGB+相机） | V2 C1 loss（含 0.1×surface） | 基线，即 V3 C1 / V4 C0 |
| C1 | DEPTH_SURFACE | 有 | 同 C0 | **PRIMARY_METHOD** |
| C2 | DEPTH_NO_SURFACE | 有 | V2 C0 loss（无 surface） | 次要，只回答 Q5 |

旁路：3 维深度统计量经一个**零初始化、无偏置**的线性层（3→8）加到融合隐变量上，位置在 ≥2 视角支持门**之后**，因为单个测得视角已经有信息。另有 log_sigma。旁路共 25 个参数（C1/C2 为 5150，C0 为 5125），其余结构完全相同；零初始化保证训练开始时 C1 与 C0 的状态完全一致。

与 V4 的差别只在证据来源：V4 旁路的统计量来自光度 plane-sweep（位于支持门之前、需要 ≥2 个视角），V5 来自测得深度（位于支持门之后、1 个视角即可）。anchor 状态（只有 anchor 视角）因此也含单视角深度线索，所以 STATIC 检验在本轮是“3 个 RGB-D 视角对 1 个 RGB-D 视角”。

## 4. 数据与训练

与 V4 完全相同：
- 数据：V2 锁定的 24 TRAIN / 8 DEV、帧角色与 GT-free bounds 规则。DEV 已在 V2/V3/V4 中评价，本轮是 DEV 机制诊断，`FRESH_QUALIFICATION_INCLUDED=false`。
- 训练：3 seeds [20260928, 20260929, 20260930]；2000 步；checkpoint [100, 200, 300, 500, 750, 1000, 1500, 2000]；Adam lr 1e-3、常数学习率；每步 1 个 TRAIN 场景、1024 条射线；梯度裁剪 1.0；FP32。
- 执行：seed 顺序执行，同一 seed 的三个变体并行；CPU 绑定 8–23 号核，单线程数值库；每次启动前等待共享 GPU 至少 3072 MiB 空闲；基础设施故障最多重跑 2 次。

## 5. 评价、统计与判定

评价与统计完全沿用 V2–V4：
- 状态构造：A、B、anchor、wrong-scene、spatial shuffle、zero；
- 掩码与区域：共同 opacity mask、静态共 mask、OBS 区域；
- 统计：10,000 次 scene paired bootstrap，seed 20260928。

主对比 `DEPTH_GAIN = AbsRel(C0) − AbsRel(C1)`。

- **DEPTH_STATUS**：对 C0−C1 应用 V2 的主对比 gate，其中包括：
  - C1 wrong-scene 与 shuffle 损伤的 CI 下界 > 0；
  - opacity 检查；
  - 状态完整性。
- **SCENE_SPECIFICITY_STATUS**（C1）、**STATIC_DEV_STATUS**（C1）：同 V4。
- **GEOMETRY_FREE_REFERENCE**（预注册；不是 gate，但进入 §6 的分支判定），回答 Q4：
  - 参考值：冻结前只由 TRAIN primary-query GT 拟合两个常数。AbsRel 最优常数是 1/t 加权中位数；另一个是普通中位数。
  - 比较时机：全部 DEV 状态 seal、预测文件哈希核验之后。
  - 比较方式：各变体在 RAW、OPACITY_NORMALIZED、MEDIAN_SCALED 三种读出下，与常数做 scene-paired 比较。
  - 标签：ABOVE、BELOW 或 NOT_DISTINGUISHABLE。
- 描述性：
  - `SURFACE_WITH_DEPTH_GAIN = AbsRel(C2) − AbsRel(C1)`；
  - 训练末 100 步深度 AbsRel；
  - DEV 曲线最优步；
  - OBS 区域收益；
  - 状态相关性；
  - 学到的 s。

## 6. 预先写定的解读

- **分支 A**：同时满足三项：
  - DEPTH_STATUS=SUPPORTED；
  - SCENE_SPECIFICITY_STATUS=SUPPORTED；
  - C1 的 RAW AbsRel 相对 REF_TRAIN_ABSREL_OPTIMAL 为 ABOVE，即显著优于无几何常数。V2–V4 从未做到这一点，所以“优于 C0”本身不足以说明 carrier 携带几何。

  满足 A 说明测得的几何能让 learned carrier 形成场景专属且优于平凡预测的几何。下一轮在 RGB-D 轨道上建立经过独立性审计的 fresh 队列做资格验证，并准备 Core B 的前置检查。
- **分支 B**：DEPTH_GAIN 的 CI 下界 > 0，或 SCENE_SPECIFICITY_STATUS=SUPPORTED，或 C1 优于常数参考，但未满足 A。深度证据有部分作用；下一轮检验 carrier 容量（grid 分辨率）或训练规模。
- **分支 C**：其余情况。即使输入测得深度，16³ carrier 在 24 个 TRAIN 场景上仍不能泛化，说明瓶颈在 carrier 或训练规模；下一轮另写协议。
- 无论哪个分支：本轮不运行 Dynamic TTT，`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`。

## 7. 禁止项

- **方法**：Dynamic TTT、writer、controller、轨迹搜索；预训练 backbone；学习式匹配或 transformer 融合；32³ carrier；renderer > 64 samples。
- **数据与检索**：bounds 或帧选择搜索；在状态构建中读取 query 深度。
- **调参**：深度线索超参数（采样方式、s 的初值与定义、旁路位置）的任何调参。
- **事后修改**：根据 DEV 结果修改 loss、checkpoint 规则、seed、预算或主方法。

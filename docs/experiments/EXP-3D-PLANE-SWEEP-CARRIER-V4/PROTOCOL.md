# EXP-3D-PLANE-SWEEP-CARRIER-V4 协议

起草：Claude（接手 cv 对话后，经用户授权持续推进），2026-09-29。本协议在任何 V4 训练之前冻结。

## 0. 背景

- V2：16³ learned carrier 上表面监督收益未建立；wrong-scene 与 spatial shuffle 不造成损伤，状态近似场景无关先验。
- V3：证据候选从 128 加密到 4096（约 30 倍的有效证据体素）后，DEV 收益仍未建立（+0.0101，CI [−0.0142, +0.0394]），预注册分支 C。训练集深度 AbsRel 从 0.490 降到 0.339，DEV 几乎不变；三个 seed 的稠密模型都在第一个 checkpoint（500 步）最好，之后 DEV 误差上升；观测最充分的 OBS2PLUS 区域没有改善。
- 结论：证据数量不是主因。现有融合只对 8 维学习特征求逐体素的跨视角均值和方差，没有“同一条视线只有一个表面”的射线竞争，也没有可见性，在 24 个训练场景上学到的是场景记忆。按 V2 协议 §35–36 与 V3 分支 C，本轮引入最小化的几何推理：不需要学习匹配的 plane-sweep 光度代价。

## 1. 研究问题

- **Q1**：给 carrier 加入显式的 plane-sweep 几何线索（射线竞争与可见性）后，DEV query 几何是否稳定改善？
- **Q2**：状态是否变得场景专属（wrong-scene 与 spatial shuffle 损伤的 CI 下界 > 0），full context 是否稳定优于 anchor？
- **Q3**：改善是否出现在多视角可观测的 OBS2PLUS 区域（这是几何推理应当起作用的地方）？
- **Q4（次要）**：有了几何线索后，表面监督是否仍有作用？

## 2. plane-sweep 几何线索（固定定义）

对一个状态的每个上下文视角 r（作为参考视角），其余上下文视角为源视角：

1. 所有上下文图像（原始 RGB，[0,1]）按面积平均下采样到 32×40，内参按像素中心约定相应缩放。在训练集冻结先验给出的 [near, far] 之间取 32 个逆深度等间距的正平行平面（z 深度）。
2. 对每个参考像素 p 与平面深度 d，沿参考射线取 3D 点，投影到各源视角，在同样 32×40 的源图像上双线性采样颜色。只统计投影在图像内且位于相机前方的源视角。
3. 代价 c(p, d)：参考颜色与有效源颜色一起计算的逐通道总体方差，再对 RGB 取平均；有效源视角少于 1 个时该 (p, d) 无效。
4. 深度概率 P(d | p) = softmax_d(−c / τ)，只在有效平面上归一化；整列无效的像素不产生线索。τ = exp(log_tau)，log_tau 是唯一可学习的 plane-sweep 参数，初值 log(0.01)。
5. 对每个体素 v：投影到参考视角，得到下采样网格坐标与相机深度 z。若在图像内且 z ∈ [near, far]，沿逆深度三线性插值得到表面似然 s_r = D·P（均匀分布时为 1），以及自由空间似然 f_r = P(表面深度 > z)，即体素位于该参考视线的表面之前。
6. 体素统计量：对有效参考视角取平均的 s、f，以及有效参考视角比例 q，共 3 维。没有有效参考视角时三者都为 0。

整个过程只使用上下文 RGB、上下文相机与训练集先验，不使用任何深度、标签或 query 信息。只有一个视角的 anchor 状态没有源视角，线索恒为 0，因此 anchor 仍只含先验。

## 3. 变体

全部为 V3 C1 的配方：16³、4096 个证据候选、64 samples 固定 renderer、冻结 GT-free bounds、同 seed 下共享参数的初始值完全相同。

| ID | 名称 | plane-sweep 旁路 | loss | 角色 |
|---|---|---|---|---|
| C0 | DENSE_SURFACE | 无 | V2 C1 loss（含 0.1×surface） | 基线，即 V3 C1 |
| C1 | SWEEP_SURFACE | 有 | 同 C0 | **PRIMARY_METHOD** |
| C2 | SWEEP_NO_SURFACE | 有 | V2 C0 loss（无 surface） | 次要，只回答 Q4 |

旁路：3 维 plane-sweep 统计量经一个**零初始化、无偏置**的线性层（3→16）加到 fuse_up 的输出上（GELU 之前）；另有 log_tau。旁路共 49 个参数（C1/C2 为 5174，C0 为 5125），其余结构完全相同。零初始化保证训练开始时 C1 与 C0 的输出完全一致。

## 4. 数据与训练

- 数据：沿用 V2 锁定的 24 TRAIN / 8 DEV、帧角色与 GT-free bounds 规则；DEV 已在 V2/V3 中评价，本轮是 DEV 机制诊断轮，`FRESH_QUALIFICATION_INCLUDED=false`。
- 训练：3 seeds [20260928, 20260929, 20260930]；**2000 步**；Adam lr 1e-3、常数学习率；每步 1 个 TRAIN 场景、1024 条射线；梯度裁剪 1.0；FP32。
- checkpoint：鉴于 V3 的最优点落在第一个 checkpoint，本轮在前期加密：[100, 200, 300, 500, 750, 1000, 1500, 2000]。各 variant×seed 用 V2 的 DEV 资格规则独立选择。
- 执行：seed 顺序执行，同一 seed 的三个变体并行；CPU 绑定 8–23 号核，单线程数值库；每次启动前等待共享 GPU 至少 3072 MiB 空闲；基础设施故障最多重跑 2 次，其他错误不重试，C0/C1 失败即停止。

## 5. 评价、统计与判定

评价与统计完全沿用 V2/V3（A、B、anchor、wrong-scene、spatial shuffle、zero；共同 opacity mask；静态共 mask；OBS 区域；10,000 次 scene paired bootstrap，seed 20260928）。主对比 `SWEEP_GAIN = AbsRel(C0) − AbsRel(C1)`。

- **SWEEP_STATUS**：对 C0−C1 应用 V2 的主对比 gate（含 C1 wrong-scene 与 shuffle 损伤 CI 下界 > 0、opacity 检查、状态完整性）。
- **SCENE_SPECIFICITY_STATUS**（C1）、**STATIC_DEV_STATUS**（C1）：同 V3。
- 描述性：`SURFACE_WITH_SWEEP_GAIN = AbsRel(C2) − AbsRel(C1)`；训练末 100 步深度 AbsRel；DEV 曲线最优步；OBS 区域收益；状态相关性；学到的 τ；plane-sweep 线索的覆盖率。

## 6. 预先写定的解读

- **分支 A**：SWEEP_STATUS=SUPPORTED 且 SCENE_SPECIFICITY_STATUS=SUPPORTED。显式几何推理能让 learned carrier 形成场景专属几何；下一轮建立经过独立性审计的 fresh 队列做资格验证，并检验训练规模。
- **分支 B**：SWEEP_GAIN 的 CI 下界 > 0，或 SCENE_SPECIFICITY_STATUS=SUPPORTED，但未满足 A。几何线索有部分作用；下一轮检验更多上下文视角或训练规模。
- **分支 C**：其余情况。在 3 个上下文视角、32×40 分辨率下，最小 plane-sweep 不足以让 carrier 泛化；下一轮需要另写协议，考虑预训练几何先验或更多视角。
- 无论哪个分支：本轮不运行 Dynamic TTT，`DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false`，`FINAL_STATIC_STATUS=NOT_ESTABLISHED`。

## 7. 禁止项

Dynamic TTT、writer、controller、轨迹搜索；预训练 backbone（DINOv2、VGGT、DUSt3R 等）；学习式匹配特征或 transformer 融合；32³ carrier；renderer > 64 samples；bounds 或帧选择搜索；plane-sweep 超参数（平面数、分辨率、代价定义、τ 初值）的任何调参；根据 DEV 结果修改 loss、checkpoint 规则、seed、预算或主方法。

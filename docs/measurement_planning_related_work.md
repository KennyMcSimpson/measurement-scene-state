# 沿给定 RGB 轨迹的预算化按需深度采集：相关工作边界综述

- 检索日期：2026-10-06（含 2026 年 arXiv 预印本）。
- 待界定的问题（下文简称"本问题"）：相机轨迹给定、不由方法控制；RGB 帧与位姿免费且持续可得，在决策时可见；从少量 RGB-D 帧起步，对每个到达的 RGB 帧在线决定是否 REQUEST_DEPTH，总预算为 K；深度只有在被请求后才揭示，并融合进 3D 状态/重建；目标是最小化未参与选择的未来/留出查询视角上的深度重建误差（如 AbsRel）。与 NBV 的关键区别在于：动作不是"相机去哪"，而是"在哪些已拍到的 RGB 帧上花深度预算"。
- "已核对"的含义：我打开了论文的 arXiv / CVF / ECVA / 出版社 / 项目页 / 专利页（或在 CVF 页面拒绝抓取时改用 ML Anthology 与 Crossref 元数据），并确认了 venue/年份和关键事实；括号内写明依据。全文未逐字引用摘要，均为转述。

---

## 1. 摘要

在本次检索范围内，没有找到与本问题完全同构的已发表工作，即同时满足"轨迹给定、RGB 与位姿免费且决策时可见、逐帧决定是否请求稠密深度、总预算 K、深度融入 3D 状态、以未参与选择的留出/未来查询视角深度误差为目标"的在线决策问题。但"何时打开深度传感器"本身并不新：Noraky & Sze（IEEE TCSVT 2020）以及同一技术路线的 Analog Devices 专利 US10841491B2（优先权 2016）已经用 RGB 与先前深度判断当前帧深度能否被估计，估计不了才开 ToF；Ericsson 专利 US12423859B2 按单目 SLAM 信息决定何时开启深度传感器；NeuroLiDAR（ICRA 2026）用事件相机触发 LiDAR 关键帧。另一侧，Depth on Demand（ECCV 2024）与 ToF-Splatting（ICCV 2025）给出了"低帧率/时间稀疏深度 + 全帧率 RGB"的强系统，后者还把深度融合进 3DGS 地图，但两者的深度调度都是用户设定的固定比率；FisherRF（ECCV 2024）和 ActiveNeRF（ECCV 2022）在预算下从候选池选视角并在留出视角上评测，但被"购买"的是选中前不可见的 RGB 视图本身，并不存在"RGB 已知、只缺深度"的不对称。Brandão 等（IJRR 2020）在计算预算下以信息最大化调度机器人身上多个深度传感器的采集，是"给定运动下调度深度测量"最接近的先例，但没有 RGB 引导，也不以留出视角深度误差为目标。因此剩余缺口是精确而狭窄的：(1) 在硬预算 K 下跨时间做在线、不可撤销的分配，而不是逐帧阈值触发；(2) 以融合 3D 状态对未来/留出查询的地图级、非短视价值作为准则，而不是"当前帧深度能否被估计"；(3) 在 RGB 已知的条件下估计深度的条件信息价值，并在等预算下稳定胜过均匀、覆盖、可估计性触发与不确定性基线。需要明确：已有证据显示均匀/覆盖式选择往往很强（NeRF Director, CVPR 2024；ISMAR 2024 中 TSDF 融合的均匀降帧几乎无损），所以本方向能否成立取决于实证余量，不能宣称"首次提出按需深度采集"。

---

## 2. 主表（最接近的 32 篇）

行首编号表示类别：A = 按需/触发式深度采集（决定"何时测"）；B = 固定低帧率深度 + RGB（不做调度决策）；C = 预算化视角选择（候选池 + 留出视角评测）；D = 空间自适应深度采样（决定"测哪里"）；E = NBV/主动建图（方法控制相机）；F = 模态选择与预算化获取（抽象上同构）。

列含义补充："camera moves?" 指方法是否控制相机运动；"RGB free?" 指廉价模态是否持续免费且在决策时可见；"depth acquisition optional?" 指深度（或其照明/扫描）是否是由方法决定的可选测量；"sensor modality selected?" 指是否在多种传感器/模态间选择（含单一深度传感器的开/关）。

| paper | venue/year | input | action space | camera moves? | RGB free? | depth acquisition optional? | sensor modality selected? | state representation | objective | online/deployable? | what differs from our formulation | verified? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 [Low Power Depth Estimation of Rigid Objects for Time-of-Flight Imaging](https://arxiv.org/abs/1810.01930) | IEEE TCSVT 2020（arXiv 2018） | 同步采集的 RGB 流 + 可开关 ToF + 上一帧深度 | 每帧二元决策：开/不开 ToF。用块匹配光流和上一帧深度以 RANSAC 估刚体位姿，内点比例不足（实现中约 10%）才开 ToF，否则按位姿重投影上一帧深度 | 否（手持轨迹给定） | 是 | 是 | 仅 ToF 开/关 | 上一帧深度图 + 帧间刚体位姿（逐帧传播，无全局 3D 地图） | 逐帧深度误差（MRE 等）与 ToF 使用率的折中；阈值按数据集调到中位 MRE 约 1%，无硬预算 | 是（嵌入式平台 640×480@30fps；ToF 使用率降约 85%） | "能否由刚体运动外推当前帧"是短视、逐帧准则；没有预算 K 的跨时分配，没有融合 3D 状态，也不在留出/未来视角评测 | 已核对（arXiv 摘要 + ar5iv 正文；期刊信息见 [DSpace](https://dspace.mit.edu/handle/1721.1/130109)） |
| A2 [Low Power Adaptive Time-of-Flight Imaging for Multiple Rigid Objects](https://dspace.mit.edu/entities/publication/31f098ae-f7c3-4011-83f9-ea2f06b6c9f6) | ICIP 2019 | ToF 测量（高/低功率帧） | 自适应改变每帧发射脉冲数：偶尔取高功率深度，用它帮助估计后续低功率深度 | 否 | 摘要未提 RGB（据 DSpace 摘要，运动由 ToF 数据估计；未读全文） | 部分（始终有低功率深度，高功率帧可选） | 否（同一传感器的功率档位） | 最近高功率深度 + 多个独立刚体运动 | 低功率深度误差（最多降 64%）vs 脉冲数（最多降 81%） | 是（嵌入式 30fps） | 选的是功率档而不是"有/无深度"；无 RGB 引导；逐帧评测，无 3D 状态与留出视角 | 已核对（DSpace 摘要与 DOI；触发细则未读全文） |
| A3 [Reducing power consumption for time-of-flight depth imaging（US10841491B2）](https://patents.google.com/patent/US10841491B2/en) | 美国专利，Analog Devices；优先权 2016，授权 2020 | 当前 RGB + 低功率或先前深度 | 每帧：神经网络从 RGB 与先前深度预测高质量深度并给出置信度；置信度高于阈值则不拍新深度，否则按置信度函数设定激光功率 | 否 | 是 | 是 | ToF 照明开/关与功率 | 先前深度 + 当前 RGB（逐帧网络） | 在预测置信度约束下省照明功耗；无预算 K | 专利描述在线系统，无公开评测 | 与 A1 同属"能估计就不测"的逐帧短视触发（发明人含 Noraky）；无 3D 地图与留出视角目标；但它是"RGB 引导深度开关"的早期先例 | 已核对（Google Patents 全文页） |
| A4 [Depth sensor activation for localization based on data from monocular camera（US12423859B2）](https://patents.google.com/patent/US12423859B2/en) | 美国专利，Ericsson；申请 2020，授权 2025 | 常开单目相机 + 可开关深度传感器 | 何时开启深度传感器：由单目数据估计开启的"收益水平"（结构信息密度、可检测 3D 特征数、特定物体、历史地图），满足激活规则即开启 | 否（设备随用户/机器人运动） | 是 | 是 | 是（深度传感器开/关） | SLAM 地图（单目稀疏点云，开启后更稠密） | 定位/建图性能与能耗，并考虑能耗预算 | 专利描述在线系统，无公开评测 | 收益以定位为中心、为手工启发式；没有学到的深度信息价值，也不评测留出视角深度误差；但"单目常开、深度按需"的系统设定与本问题最像 | 已核对（Google Patents 全文页） |
| A5 [NeuroLiDAR: Adaptive Frame Rate Depth Sensing via Neuromorphic Event-LiDAR Fusion](https://arxiv.org/abs/2605.16805) | ICRA 2026（据 arXiv 注释） | 事件相机流 + 低基础帧率 LiDAR | 每个事件窗口由轻量 CNN 判断是否为关键帧并触发 LiDAR；关键帧标签来自车速、近距离物体、新物体进入等场景动态规则 | 否（自车/机器人轨迹给定） | 事件流免费（不是 RGB） | 是（实验中以对固定频率 LiDAR 子采样来模拟触发） | 否 | 最近一帧 LiDAR 深度 + 事件体素（逐帧外推，无地图） | 各时刻深度 RMSE/AbsRel；相对 10Hz LiDAR 的 RMSE 约降 29% | 是（Jetson 实时） | 触发是场景变化驱动，不看下游 3D 状态的价值；无预算 K，无 3D 融合，只评当前流 | 已核对（arXiv 摘要 + HTML 正文） |
| A6 [Thinking Fast, Thinking Slow: Adaptive Multimodal Transformer-based Sensor Fusion for Depth Estimation on Ultra-low-power MCUs](https://arxiv.org/abs/2607.18897) | IEEE Sensors Journal 2026（录用，据 arXiv 注释） | 低分辨率灰度图 + 超声 + 8×8 ToF（在 NYUv2 上仿真） | 每帧逐级决定是否加入更耗能的传感器，置信度门控提前退出 | 否 | 部分（低分辨率图像始终可用） | 是 | 是（逐级加入模态） | 逐帧 Transformer + 跨帧记忆 token | 逐帧深度 δ1 vs 每帧能耗（约省 90%） | 是（GAP9 MCU） | 单帧深度、短视门控；没有 3D 状态和跨时预算分配 | 已核对（arXiv 摘要 + HTML 正文） |
| A7 [Placing and scheduling many depth sensors for wide coverage and efficient mapping in versatile legged robots](https://isr.tecnico.ulisboa.pt/?p=25807) | IJRR 2020 | 腿式机器人上的 20 个深度传感器 | 设计期：传感器布局；运行期：在计算约束下调度每时刻采集/处理哪些传感器（信息最大化，配合采样加速） | 否（运动由步态/任务决定） | 无 RGB | 是（逐传感器） | 是（多个深度传感器之间） | 3D 地图（以信息最大化调度） | 地图覆盖/信息 vs 计算 | 是（真实机器人） | 预算是计算而非采集能耗；在多个深度传感器之间调度，无 RGB 引导；未见留出视角深度误差评测 | 已核对（ISR 页 + Crossref 摘要；调度细节未读全文） |
| B1 [Depth on Demand: Streaming Dense Depth from a Low Frame Rate Active Sensor](https://arxiv.org/abs/2409.08277) | ECCV 2024 | 高帧率 RGB + 低帧率稀疏深度（ScanNet 训练约 500 点）+ 相对位姿 | 无调度决策：深度/RGB 帧率比 τ 由用户固定（实验覆盖 7Scenes 上 0.1–1.0、KITTI 上 1Hz/0.5Hz LiDAR 等） | 否 | 是 | 由固定 τ 决定，不是方法的决策 | 否 | 逐帧网络：当前 RGB + 最近一帧带深度的 RGB 及其稀疏深度 + 相对位姿 | 各帧稠密深度误差；把预测深度做 TSDF 融合后的网格指标 | 是（约 30ms/帧） | 不决定何时测；是"均匀/固定频率"对照下最强的致密化模块；没有评测留出查询视角 | 已核对（arXiv 摘要 + HTML 正文；ECVA 页见检索） |
| B2 [ToF-Splatting: Dense SLAM using Sparse Time-of-Flight Depth and Multi-Frame Integration](https://mlanthology.org/iccv/2025/conti2025iccv-tofsplatting/) | ICCV 2025 | RGB + 极稀疏 ToF；时间稀疏实验中只有部分帧有深度 | 无调度决策：时间稀疏比固定为 2/3/4/6/8，前 50 帧全部给深度 | 否 | 是 | 由固定比率决定 | 否 | 3DGS SLAM 地图 + 多帧融合的稠密深度 | 跟踪与网格精度、完整度、F-score；稀疏比增大时性能平缓下降 | 是（SLAM） | 与本问题最接近的"融合 3D 状态 + 时间稀疏深度"系统，但调度固定；适合作系统底座与均匀基线 | 已核对（ML Anthology + arXiv HTML 正文） |
| C1 [ActiveNeRF: Learning where to See with Uncertainty Estimation](https://arxiv.org/abs/2209.08546) | ECCV 2022 | 少量初始 RGB 视角 | 从候选视角中选得分最高的 k 个加入训练；候选图像在选中前不可见 | 否（从候选位姿中选，不做运动规划） | 否（被采集的就是 RGB） | 否（无深度） | 否 | 带不确定性的 NeRF | 先验到后验的方差下降（信息增益）；留出测试视角 PSNR/SSIM/LPIPS | 否（离线迭代训练） | 购买的是 RGB 视图，候选内容未知；本问题中 RGB 已知、只缺深度；非流式 | 已核对（arXiv + ar5iv 正文） |
| C2 [FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information](https://arxiv.org/abs/2311.17874) | ECCV 2024 | 初始 4 个视角；候选池为数据集训练视角，只用相机参数打分 | 从池中顺序或批量选到 20 个视角；另有 Habitat 中的主动建图（RGB-D） | 视角选择：否；主动建图：是 | 否 | 否（视角选择中无深度；建图中深度随视角而来） | 否 | NeRF / 3DGS | 基于 Fisher 信息的期望信息增益；留出视角 PSNR 等 | 3DGS 下评分约 70fps；视角选择本身为离线池式 | 预算 K + 留出视角目标与本问题相同，但选的是隐藏的 RGB 视图；EIG 只需相机参数，可直接改成"对到达帧的深度算 EIG"的强基线 | 已核对（arXiv HTML 正文） |
| C3 [NeRF Director: Revisiting View Selection in Neural Volume Rendering](https://arxiv.org/abs/2406.08839) | CVPR 2024 | 有位姿候选视角池（合成数据 300 个、真实场景 251–313 个，图像都已拍到） | 选 30–150 个训练视角；比较随机、最远视角采样 FVS、误差引导、不确定性引导 | 否 | 是（候选图像全部可用） | 否 | 否 | NeRF（Instant-NGP 等） | 约 200 个留出测试视角的渲染质量 | 离线 | 免训练的覆盖式 FVS 常胜过误差/不确定性选择；资源是训练视角而不是深度采集；提醒本问题的覆盖基线可能很强 | 已核对（arXiv 摘要 + HTML 正文） |
| D1 [Deep Adaptive LiDAR: End-to-end Optimization of Sampling and Depth Completion at Low Sampling Rates](https://computationalimaging.org/publications/deep-adaptive-lidar/) | ICCP 2020 | 单帧 RGB（+ 单目深度先验） | 单帧内像素级采样模式，与补全网络端到端训练 | 否 | 是 | 是（像素级） | 否 | 单帧深度补全网络 | 低采样率下单帧深度补全误差 | 需可编程 LiDAR | 选的是像素不是帧；单帧补全而非多视角 3D 状态；无时间预算分配 | 已核对（项目页 + Crossref） |
| D2 [Adaptive LiDAR Sampling and Depth Completion using Ensemble Variance](https://arxiv.org/abs/2007.13834) | IEEE TIP 2021 | 单帧，RGB 可选 | 多阶段像素采样，采样概率正比于集成预测方差 | 否 | 是 | 是（像素级） | 否 | 单帧深度补全（模型集成） | 单帧深度误差 vs 测量数（同精度下测量减少 4–10 倍） | 需可编程 LiDAR | 同 D1；集成方差可迁移为帧级"请求深度"打分基线 | 已核对（arXiv + Crossref） |
| D3 [Uncertainty-Aware Interactive LiDAR Sampling for Deep Depth Completion](https://mlanthology.org/wacv/2023/taguchi2023wacv-uncertaintyaware/) | WACV 2023 | 单帧 RGB + 可编程扫描 LiDAR | 交互式逐点采样：用深度不确定性评估候选点，并近似更新不确定性 | 否 | 是 | 是（点级） | 否 | 单帧深度补全 | 单帧补全误差（KITTI，NYUv2 泛化） | 需可编程 LiDAR | 同 D1 | 已核对（ML Anthology；CVF 页面抓取被拒） |
| D4 [Prior based Sampling for Adaptive LiDAR](https://arxiv.org/abs/2304.07099) | arXiv 2023（未见正式发表信息） | 序列：前一时刻深度样本 + 当前帧（补全网络用 RGB） | 每帧预测像素级采样掩码（SampleDepth） | 否 | 是 | 是（像素级） | 否 | 冻结的补全网络 + 前帧先验 | 下游深度补全（KITTI、SHIFT） | 是（逐帧） | 用了时间先验，但每帧都测，只选在哪里测；没有帧级"测/不测"与跨帧预算 | 已核对（arXiv，预印本） |
| D5 [Adaptive LiDAR Scanning: Harnessing Temporal Cues for Efficient 3D Object Detection via Multi-Modal Fusion](https://arxiv.org/abs/2508.01562) | AAAI 2026（据 arXiv 注释） | 历史 LiDAR + 相机 | 为下一帧生成 ROI 二值掩码（ROI 内稠密扫描、其余稀疏） | 否 | 是 | 是（区域级） | 否 | 时空查询嵌入 | 3D 检测精度 + LiDAR 能耗（降 65% 以上） | 是 | 空间密度调制，任务是检测；不决定整帧是否测 | 已核对（arXiv） |
| D6 [Event Guided Depth Sensing](https://arxiv.org/abs/2110.10505) | 3DV 2021 | 事件相机 + 激光点投射器 + 第二个事件传感器 | 按事件活动决定哪些区域密集照明，其余稀疏 | 否 | 事件流免费 | 是（区域级） | 否 | 逐帧结构光深度 | 照明功耗（运动边缘少时可省约 90%）与深度质量 | 原型系统 | 空间选择且由动态驱动；无 3D 融合，无预算 | 已核对（arXiv） |
| D7 [Energy-Efficient Adaptive 3D Sensing](https://mlanthology.org/cvpr/2023/tilmon2023cvpr-energyefficient/) | CVPR 2023 | 被动双目 + 可编程投射（SLM 或 MEMS+DOE） | 只向被动双目失败且应用需要深度的 ROI 投射图案 | 否 | 是（被动图像） | 是（区域级） | 否 | 逐帧深度 | 功耗、量程与人眼安全 | 硬件原型 | 空间选择、单帧；无预算分配与 3D 状态 | 已核对（ML Anthology） |
| D8 [Active Perception using Light Curtains for Autonomous Driving](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4458_ECCV_2020_paper.php) | ECCV 2020 | 可控光幕深度传感器 + 3D 点云检测器 | 每步把光幕放到检测器不确定的区域（约束图上动态规划） | 否 | 不适用 | 是（测哪里） | 否 | 检测器点云 | 3D 检测精度 | 是 | "测哪里"的可控深度传感器，以不确定性引导；目标是检测而非重建，也不选帧 | 已核对（ECVA） |
| E1 [An Information Gain Formulation for Active Volumetric 3D Reconstruction](https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf) | ICRA 2016 | 深度/RGB-D 观测 | 从离散候选视角中选 NBV 并移动 | 是 | 否 | 否（每个视角都测深度） | 否 | 概率体素地图 | 多种体素信息增益（可见性、看到新部分的概率等）→ 重建完整度 | 是 | 控制相机去哪；深度随视角必然获得；候选观测是假想的 | 已核对（venue 经 DOI 10.1109/ICRA.2016.7487527 元数据；要点经 UZH 软件页与检索摘要） |
| E2 [GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction](https://arxiv.org/abs/2402.16174) | CVPR 2024 | RGB-D + IMU | 5D 连续空间视角，RL 策略 | 是 | 否 | 否 | 否 | 概率占据栅格 + 图像语义特征 + 动作历史 | 覆盖率（在未见物体上测试） | 是（仿真） | 学习式 NBV，同 E1 | 已核对（arXiv + HTML 正文） |
| E3 [NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering](https://arxiv.org/abs/2303.01284) | IROS 2023 | RGB | 从候选视角中选渲染不确定性最大者并移动 | 是 | 否（RGB 即测量） | 否 | 否 | 无显式地图（图像式神经渲染） | 测量预算下的渲染质量 | 是 | "测量预算"计的是 RGB 图像数，且控制相机 | 已核对（arXiv） |
| E4 [Active Neural Mapping](https://arxiv.org/abs/2308.16246) | ICCV 2023 | 有位姿深度图（Habitat） | 离散导航动作与探索目标 | 是 | 否 | 否 | 否 | 持续学习的神经场；不确定性取自权重扰动下的神经变异性 | 地图精度与完整度 | 是（在线） | 控制运动；每步都有深度 | 已核对（arXiv + ar5iv 正文） |
| E5 [NARUTO: Neural Active Reconstruction from Uncertain Target Observations](https://arxiv.org/abs/2402.18771) | CVPR 2024 | RGB-D | 6DoF 不确定性目标搜索 + RRT 路径；另提出"主动光线采样"，即从已拍关键帧中选不确定性最高的光线参与优化 | 是 | 否 | 否 | 否 | 哈希网格神经 SDF + 不确定性 | 网格精度、完整度、完整率 | 是 | 主动光线采样属于计算选择而非采集选择 | 已核对（arXiv + HTML 正文） |
| E6 [ActiveGS: Active Scene Reconstruction Using Gaussian Splatting](https://arxiv.org/abs/2412.17769) | IEEE RA-L 2025 | 机载 RGB-D | 视角选择 + 无碰撞路径规划 | 是 | 否 | 否 | 否 | 3DGS + 粗体素地图，GS 置信度建模 | 重建质量与覆盖 | 是（UAV 实测） | 同 E1 | 已核对（arXiv） |
| E7 [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](https://arxiv.org/abs/2501.06897) | CVPR 2025 | RGB-D（Habitat） | NBV，渲染式信息增益（候选视角渲染剪影中缺失像素数）；另有全局-局部关键帧选择（新像素超过 10% 或渲染质量低于阈值才入选） | 是 | 否 | 否 | 否 | 3DGS | 几何精度/完整度 + 新视角轨迹上的 PSNR 等 | 是 | 关键帧规则可直接改作"请求深度"的启发式基线；但本身控制相机且每帧都有深度 | 已核对（arXiv + HTML 正文） |
| E8 [OccamView: Object-Conditioned View Selection for Frame-Budgeted Active 3D Gaussian Reconstruction](https://arxiv.org/abs/2608.16499) | arXiv 2026（预印本） | RGB-D | 用遮挡感知的代理覆盖分数给候选视角打分并移动 | 是 | 否 | 否（预算按 RGB-D 帧数计） | 否 | 3DGS + 在线物体记忆 | 不同帧预算下的完整度 | 是（仿真） | "帧预算"概念最接近，但预算计的是相机去哪拍 RGB-D 帧 | 已核对（arXiv，未经同行评审） |
| F1 [Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting](https://arxiv.org/abs/2410.04680) | ICRA 2025 | RGB-D 视角 + 触觉 | 先选视角，再用 FisherRF 深度不确定性选触摸位姿（两阶段） | 是（机械臂） | 否 | 触觉是额外的可选测量 | 是（视觉/触觉，但分阶段） | 3DGS | 留出视角 PSNR/SSIM/LPIPS 与深度误差 | 是（真实机器人） | 有"第二种昂贵测量"的选择，但不是统一预算下的逐帧取舍，且相机受控 | 已核对（arXiv + HTML 正文） |
| F2 [Multi-Modal Active Perception for Information Gathering in Science Missions](https://arxiv.org/abs/1712.09716) | Autonomous Robots 2019 | 异构科学传感器 | 路径规划 + 传感器调度，以信息增益为目标，MCTS 求解 | 是 | 不适用 | 是 | 是 | 贝叶斯网络 | 关于潜在科学变量的信息增益 | 是（野外数据与硬件实验） | 同时控制路径与传感器；不是视觉 3D 重建 | 已核对（arXiv + Crossref） |
| F3 [Active Feature Acquisition with Generative Surrogate Models](https://proceedings.mlr.press/v139/li21p.html) | ICML 2021 | 部分观测的特征向量 | 推理时逐实例序贯决定获取哪个特征（MDP + 生成式代理模型） | 不适用 | 不适用 | 是（特征级） | 是（抽象意义上） | 生成式代理模型 | 预测性能与获取代价 | 是（推理时） | 抽象同构（"深度 = 昂贵特征"），但非 3D、非沿轨迹流式 | 已核对（PMLR 页） |
| F4 [Attention and Anticipation in Fast Visual-Inertial Navigation](https://arxiv.org/abs/1610.03344) | ICRA 2017；扩展版 IEEE T-RO 2019 | 视觉特征 + IMU | 在资源预算内选特征子集，并前向模拟未来时域评估效用（子模贪心） | 否 | 是 | 否（选测量子集，资源为计算） | 否 | VIN 状态估计 | 未来时域上的估计性能（带近似保证） | 是（实时） | "以未来时域为目标的预算化测量选择"思想相同，但对象是特征与计算，不是深度采集 | 已核对（arXiv + [DSpace](https://dspace.mit.edu/handle/1721.1/126592)；T-RO DOI 10.1109/TRO.2018.2872402） |

统计：32 篇，32 篇已核对（其中 A2、A7、E1 的部分细节仅核对到摘要或元数据层面，已在格内注明；D4、E8 是预印本）。

---

## 3. 分类小结

**(1) Next-best-view（经典与学习式）。** 经典 NBV 在离散候选位姿上计算体素信息增益并移动传感器（[Isler et al., ICRA 2016](https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf)）；学习式方法用 3D CNN 预测视角效用（[Learn-to-Score, ECCV 2018](https://mlanthology.org/eccv/2018/hepp2018eccv-learntoscore/)）或用 RL 在连续空间直接输出视角（[GenNBV, CVPR 2024](https://arxiv.org/abs/2402.16174)）。它们优化的是"传感器下一步去哪"，每个视角的 RGB 与深度一起获得，候选观测在决策时是假想的。本问题恰好相反：位姿由轨迹给定，决策时该帧 RGB 已经可见，只决定是否为它额外购买深度。因此 NBV 的效用函数（未知体素数、可见性）可以借来当打分基线，但"控制相机"这一核心贡献不适用，NBV 文献也不构成本问题的直接先例。

**(2) 主动 3D 重建/主动建图（神经隐式、NeRF）。** [ActiveNeRF](https://arxiv.org/abs/2209.08546)、[NeU-NBV](https://arxiv.org/abs/2303.01284)、[Active Neural Mapping](https://arxiv.org/abs/2308.16246)、[NARUTO](https://arxiv.org/abs/2402.18771)、[Neural Visibility Field (CVPR 2024)](https://arxiv.org/abs/2406.06948) 用神经场的不确定性（方差、权重扰动、可见性）来选下一个视角或探索目标，预算计的是图像数或路径。NARUTO 的"主动光线采样"是在已有关键帧里挑光线来训练，属于计算分配，不是传感采集。与本问题的差别：它们都控制相机，而且深度不被单独定价（RGB-D 一起来，或者只有 RGB）。

**(3) 3DGS 主动建图/视角选择。** [FisherRF](https://arxiv.org/abs/2311.17874) 用 Fisher 信息计算候选视角的期望信息增益，只需要相机参数；[ActiveGS](https://arxiv.org/abs/2412.17769)、[ActiveGAMER](https://arxiv.org/abs/2501.06897)、[GauSS-MI (RSS 2025)](https://roboticsconference.org/2025/program/papers/30/)、[ActiveSplat (RA-L 2025)](https://arxiv.org/abs/2410.21955)、[OccamView](https://arxiv.org/abs/2608.16499) 把置信度、渲染剪影缺失像素、互信息或代理覆盖用于 NBV 与路径规划；[ToF-Splatting](https://mlanthology.org/iccv/2025/conti2025iccv-tofsplatting/) 则是 RGB + 稀疏 ToF 的 3DGS SLAM。这一类给了本问题最合适的 3D 状态（3DGS）和现成的打分器：FisherRF 的 EIG、ActiveGAMER 的"新像素比例/渲染质量"关键帧规则都能改造成"是否为到达帧请求深度"的准则。但它们的预算单位是相机去拍的帧（OccamView 明确按 RGB-D 帧计预算），而不是在给定帧上是否加购深度。

**(4) 可见性/不确定性驱动的视角选择与信息增益。** 这一线量化的是"一次观测（RGB 或 RGB-D）"对模型的信息价值：体素熵与可见性（Isler）、辐射场方差下降（ActiveNeRF）、Fisher EIG（FisherRF）。[NeRF Director (CVPR 2024)](https://arxiv.org/abs/2406.08839) 的系统比较表明，免训练的覆盖式最远视角采样常常胜过误差或不确定性选择，这对本问题是一个重要警示。理论上，以"已选位置与未选位置之间的互信息"为准则的传感器布置（[Krause et al., JMLR 2008](https://www.jmlr.org/papers/v9/krause08a.html)）正好对应"对未参与选择的查询位置负责"的目标。与本问题的差别：本问题需要的是条件价值，即在全部 RGB 与已购深度已知的条件下，再买这一帧深度能为留出查询带来多少信息；现有方法估计的是一个视角整体观测的价值，而且这些视角不是在流上到达的。

**(5) 选择性深度传感、Depth on Demand、自适应深度/LiDAR 采样。** [Depth on Demand (ECCV 2024)](https://arxiv.org/abs/2409.08277) 证明了"全帧率 RGB + 低帧率稀疏深度"可以流式输出稠密深度，并报告了 TSDF 网格指标，但帧率比 τ 是用户固定的，论文也没有讨论自适应触发；ToF-Splatting 在固定的时间稀疏比下做 3DGS SLAM。空间自适应采样一线，包括 [Deep Adaptive LiDAR (ICCP 2020)](https://computationalimaging.org/publications/deep-adaptive-lidar/)、[Super-Pixel Sampler (ICRA 2020)](https://arxiv.org/abs/1908.01379)、[集成方差采样 (TIP 2021)](https://arxiv.org/abs/2007.13834)、[交互式不确定性采样 (WACV 2023)](https://mlanthology.org/wacv/2023/taguchi2023wacv-uncertaintyaware/)、[SampleDepth](https://arxiv.org/abs/2304.07099)、[AAAI 2026 的 ROI 扫描](https://arxiv.org/abs/2508.01562)、[事件引导深度传感 (3DV 2021)](https://arxiv.org/abs/2110.10505)、[CVPR 2023 自适应 3D 传感](https://mlanthology.org/cvpr/2023/tilmon2023cvpr-energyefficient/)、[光幕主动感知 (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4458_ECCV_2020_paper.php)，优化的是单帧内在哪些像素或区域测深度，目标是单帧补全或检测。与本问题的差别：逐像素稀疏采样对逐帧稠密深度；单帧补全对多视角 3D 状态；固定调度或空间选择对时间上的"测/不测"决策。

**(6) 节能深度传感、传感器调度、RGB-D SLAM 关键帧选择。** 直接决定"何时开深度"的工作有：[Noraky & Sze (TCSVT 2020)](https://arxiv.org/abs/1810.01930)，内点不足才开 ToF；[ICIP 2019](https://dspace.mit.edu/entities/publication/31f098ae-f7c3-4011-83f9-ea2f06b6c9f6)，自适应脉冲数；[ADI 专利](https://patents.google.com/patent/US10841491B2/en)，置信度门控；[Ericsson 专利](https://patents.google.com/patent/US12423859B2/en)，按收益开启深度；[NeuroLiDAR](https://arxiv.org/abs/2605.16805)，事件触发；[Thinking Fast, Thinking Slow](https://arxiv.org/abs/2607.18897)，逐级加入 ToF。它们都用阈值换能耗，目标都是当前帧的深度或定位，没有预算 K，也没有用融合地图在留出视角上评测。[Brandão et al. (IJRR 2020)](https://isr.tecnico.ulisboa.pt/?p=25807) 在计算约束下以信息最大化调度多个深度传感器，是"给定运动下调度深度测量"的先例。控制领域的事件触发估计（[Trimpe & D'Andrea, IEEE TAC 2014](https://doi.org/10.1109/tac.2014.2351951)，元数据经 Crossref 核对，"预测方差超过阈值才发送测量"这一规则据检索摘要）提供了"按方差决定何时测"的经典范式。关键帧选择是在全部帧都有深度的前提下分配计算，例如 [信息驱动 RGB-D 里程计 (CVPR 2020)](https://mlanthology.org/cvpr/2020/fontan2020cvpr-informationdriven/)、[Gaussian Splatting SLAM (CVPR 2024)](https://mlanthology.org/cvpr/2024/matsuki2024cvpr-gaussian/) 的共视 IoU/平移阈值、ActiveGAMER 的新像素比例。另外，[ISMAR 2024 的能耗三难研究](https://arxiv.org/abs/2409.04018)发现，ScanNet 上 TSDF 融合均匀降到约 3.75 FPS 时 F-score 几乎不降（只计算力能耗，所有帧本来都有深度）。这说明均匀调度可能留给自适应方法的余量不大。

**(7) 主动 RGB-D 感知与传感器模态选择。** 主动感知的经典定义本就包括选择感知参数，而不仅是位姿（[Bajcsy et al., Autonomous Robots 第 42 卷（2017 在线，2018 卷期）](https://arxiv.org/abs/1603.02729)）。模态选择的代表工作：[Next Best Sense](https://arxiv.org/abs/2410.04680) 先视觉、后触觉，分两个阶段；[Arora et al.](https://arxiv.org/abs/1712.09716) 联合规划路径与调度传感器；[AdaMML (ICCV 2021)](https://mlanthology.org/iccv/2021/panda2021iccv-adamml/) 按视频片段选择处理 RGB、音频或光流以省计算；[EcoFusion (DAC 2022)](https://arxiv.org/abs/2202.11330) 按情境选融合配置以省能耗；Thinking Fast, Thinking Slow 逐级加入耗能传感器。与本问题的差别：多数是在识别任务里省计算，或者同时控制运动；没有工作在固定轨迹上、以统一预算逐帧决定是否购买稠密深度来服务 3D 重建。

**(8) 预算化感知与 3D 预算化主动学习。** 预算化主动学习的信息结构与本问题相似：输入（RGB）免费，"标签"（深度）昂贵，见 [LiDAL (ECCV 2022)](https://arxiv.org/abs/2211.05997) 与面向深度标注的主动学习。流式版本，即实例到达后立即决定是否在预算 K 内查询，在 [Fujii & Kashima (NIPS 2016)](https://proceedings.neurips.cc/paper/2016/hash/07cdfd23373b17c6b337251c22b7ea57-Abstract.html) 中有基于自适应子模性的框架与保证；逐实例特征获取见 [Active Feature Acquisition (ICML 2021)](https://proceedings.mlr.press/v139/li21p.html)。差别在于：这些方法为训练跨场景模型而采标签，本问题是测试时对单个场景的重建，效用在留出查询视角上计量。它们提供的是可以借用的算法形式，如预算节奏控制、阈值、子模贪心，而不是本问题的先例。要把它们的近似保证用到本问题上，需要另外证明重建效用满足（自适应）子模性。

---

## 4. 边界结论

**可以辩护的贡献（在本次检索范围内未见先例）：**
- 问题设定本身：在轨迹给定、RGB 与位姿免费且决策时可见的条件下，对逐帧稠密深度做硬预算 K 的在线、不可撤销分配，并以融合 3D 状态在未参与选择的留出/未来查询视角上的深度误差为目标。已有工作要么阈值触发且只评当前帧（A1–A6），要么调度固定（B1–B2），要么购买的是隐藏的 RGB 视图（C1–C2），要么由方法控制相机（E 类）。
- 地图级、非短视的价值准则：一帧深度的价值取决于它对未来查询的贡献，而不是"当前帧能否被估计"。当前帧可预测但覆盖了后续查询依赖的几何时，深度仍然值钱；当前帧难估计但与查询无关时，深度不值钱。这与 Noraky/ADI/NeuroLiDAR 的准则有本质区别，必须用实验把两者区分开。
- 在 RGB 已知条件下估计深度的条件信息价值（例如结合当前 3D 状态在该帧的渲染不确定性、基于 RGB 的 MVS/单目预测与其分歧），作为可学习或有原则的打分器。
- 跨时间的预算节奏控制（secretary/knapsack 式分配），区别于"阈值换能耗"的做法；如果能对所用效用证明（自适应）子模性，近似保证可以作为理论贡献。
- 评测协议与基准：等预算比较、留出查询不参与选择、严格因果（只用过去信息）、报告预言机余量。即使最后只得到"自适应在哪些轨迹/场景下才优于均匀"的结论，这也是可辩护的实证贡献。

**不可辩护（已被做过）的说法：**
- "首次根据 RGB/廉价模态决定何时开启深度传感器以省电"：已有 Noraky & Sze（arXiv 2018 / TCSVT 2020）、ADI 专利（优先权 2016）、Ericsson 专利（2020 申请）、NeuroLiDAR（事件触发）、Thinking Fast, Thinking Slow（置信度门控加 ToF）。
- "首次用低帧率深度 + 全帧率 RGB 做稠密深度或建图"：Depth on Demand、ToF-Splatting（含 3DGS SLAM 下的时间稀疏实验）、Noraky 系列。
- "首次在预算下用信息增益或 Fisher 信息做视角选择，并在留出视角评测"：ActiveNeRF、FisherRF、NeU-NBV、NeRF Director。
- "首次用不确定性自适应地采深度"：空间版本已有 Gofer（TIP 2021）、Taguchi（WACV 2023）、Bergman（ICCP 2020）、SampleDepth。
- "首次以信息最大化调度深度传感器"：Brandão et al.（IJRR 2020）。
- "场景变化/事件触发深度采集"：NeuroLiDAR、Event Guided Depth Sensing。
- 不宜默认自适应必然优于均匀：NeRF Director 与 ISMAR 2024 的证据都显示覆盖/均匀策略很强，必须用等预算实验证明余量。

**公平评测必须包含的基线（全部在相同 K 下比较；阈值类方法需改为 top-K 或预算节奏版本）：**
- 均匀/固定频率：每 N 帧请求一次，对应 DoD/ToF-Splatting 的协议；同时报告随机选择（多种子）。
- 位姿空间覆盖：在位姿空间做最远视角采样（NeRF Director 的 FVS），以及按平移/旋转距离设置关键帧。
- 共视/新颖性：与已有深度帧的共视 IoU 低于阈值即请求（Gaussian Splatting SLAM 式）；新像素比例超过阈值即请求（ActiveGAMER 式）。
- 当前帧可估计性触发：Noraky 式刚体外推内点比例、ADI 式 RGB→深度预测置信度、NeuroLiDAR 式场景变化触发。
- 不确定性/信息增益：在到达帧上渲染当前 3D 状态的深度，计算集成方差（Gofer 式）；对到达帧相机计算 FisherRF 的 EIG；ActiveNeRF 式方差下降；Isler 式未知体素或可见性增益。
- 学习式按需深度：以 DoD（或 ToF-Splatting 的多帧融合模块）作为致密化器，在均匀调度下的结果；如果提出学习式策略，还要与"同一模型 + 简单阈值"做消融。
- 上下界：全部帧有深度、只有 RGB 不加深度、非因果的事后贪心（直接用 GT 查询误差选帧）、离线互信息贪心（Krause 式），用来量化自适应可能达到的余量。
- 报告方式：留出/未来查询视角上的 AbsRel 与 δ，加网格 F-score；误差-K 曲线；深度帧占比作为能耗代理；决策时延；覆盖多种轨迹类型（平滑扫描、回环、快速转动）。

**检索局限：** 专利检索只做了关键词抽查；商用系统（如手机或头显）中可能存在未公开的深度占空比策略；2026 年的预印本更新很快，正式立项前建议再做一次增量检索。

---

## 5. 检索记录

**检索式（WebSearch，按执行顺序归并；"[ext]"表示使用 extended 模式）：**
1. Depth on Demand: Streaming Dense Depth from a Low Frame Rate Active Sensor ECCV 2024
2. low power time-of-flight depth estimation adaptively turn on ToF sensor RGB previous depth Noraky Sze
3. adaptive LiDAR sampling depth completion ensemble variance
4. Deep Adaptive LiDAR end-to-end optimization of sampling and depth completion at low sampling rates
5. "Uncertainty-Aware Interactive LiDAR Sampling for Deep Depth Completion" arXiv
6. Noraky Sze "Low Power Adaptive Time-of-Flight Imaging for Multiple Rigid Objects" ICIP 2019
7. energy-efficient depth sensing mobile AR when to turn on LiDAR ToF duty cycling monocular depth prediction
8. "Depth Map Estimation of Dynamic Scenes Using Prior Depth Information" Noraky（及其发表信息检索）
9. adaptive depth sensing reinforcement learning when to activate depth sensor RGB-D power saving
10. Event Guided Depth Sensing Muglikar Scaramuzza 3DV 2021
11. Active Perception using Light Curtains for Autonomous Driving ECCV 2020 Ancha
12. Super-Pixel Sampler data-driven approach for depth sampling and reconstruction ICRA 2020
13. ActiveNeRF learning where to see with uncertainty estimation ECCV 2022
14. FisherRF active view selection and uncertainty quantification radiance fields Fisher information ECCV 2024
15. NeU-NBV next best view planning using uncertainty estimation in image-based neural rendering
16. NARUTO neural active reconstruction from uncertain target observations CVPR 2024
17. [ext] learning when to query depth sensor per frame budget RGB video reinforcement learning policy depth acquisition
18. [ext] selective depth acquisition fixed camera trajectory budget which frames to capture depth 3D reconstruction
19. [ext] "on-demand" depth sensing mobile augmented reality LiDAR energy MobiSys OR SenSys OR MobiCom
20. [ext] uncertainty-triggered LiDAR activation monocular depth estimation energy saving robot "turn on" depth sensor only when needed
21. ActiveGS active scene reconstruction using Gaussian splatting
22. Active Neural Mapping ICCV 2023 Yan neural field uncertainty exploration
23. Next Best Sense guiding vision and touch with FisherRF 3D Gaussian splatting
24. An information gain formulation for active volumetric 3D reconstruction Isler ICRA 2016
25. EcoFusion energy-aware adaptive sensor fusion autonomous vehicle perception
26. Selective Sensor Fusion for Neural Visual-Inertial Odometry CVPR 2019
27. Attention and Anticipation in Fast Visual-Inertial Navigation Carlone Karaman feature selection budget
28. AdaMML adaptive multi-modal learning efficient video recognition modality selection per segment
29. NeRF Director revisiting view selection in neural volume rendering CVPR 2024
30. [ext] frame selection from video for Gaussian splatting reconstruction budget informative frames subset selection 2025
31. active view selector cross reference image quality assessment view selection 3D reconstruction
32. Learn-to-Score efficient 3D scene exploration by predicting view utility ECCV 2018 Hepp
33. [ext] adaptive LiDAR sampling video sequence reinforcement learning where to sample next frame depth completion
34. active learning depth estimation which frames to annotate with LiDAR ground truth budget
35. sensor scheduling active sensing robotics information-theoretic when to measure cost-aware perception survey
36. [ext] active depth supervision selection NeRF Gaussian splatting which views to acquire depth uncertainty budget
37. "Multi-Modal Active Perception for Information Gathering in Science Missions" Autonomous Robots（及 DOI 检索）
38. Neural Visibility Field for uncertainty-driven active mapping CVPR 2024
39. information-theoretic keyframe selection RGB-D SLAM entropy measurement selection budget
40. resource-aware SLAM measurement selection submodular budget "anticipation" future views map quality
41. Near-optimal sensor placements in Gaussian processes mutual information Krause Singh Guestrin JMLR
42. [ext] adaptive frame rate LiDAR or ToF depth sensor scheduling learned policy video depth estimation energy
43. [ext] "active depth sensing" budget "when" acquire depth frames RGB video uncertainty arXiv 2025
44. [ext] RGB-D SLAM switch off depth camera save power monocular mode adaptive depth sensor usage energy-efficient SLAM
45. "Adaptive Illumination based Depth Sensing using Deep Superpixel and Soft Sampling Approximation"
46. ActiveGAMER active Gaussian mapping through efficient rendering CVPR 2025
47. GauSS-MI Gaussian splatting Shannon mutual information active 3D reconstruction RSS 2025
48. ActiveSplat high-fidelity scene reconstruction through active Gaussian splatting RA-L
49. Uncertainty guided policy for active robotic 3D reconstruction using neural radiance fields RA-L 2022
50. [ext] path-constrained view selection when to trigger sensor along fixed trajectory 3D reconstruction budget measurements
51. event-triggered state estimation variance-based triggering sensor communication budget Trimpe
52. Active Feature Acquisition with Generative Surrogate Models ICML 2021
53. [ext] cost-aware next best view choose sensing modality RGB or depth per view active reconstruction energy
54. [ext] active sparse depth query monocular depth "where to measure" LiDAR budget video temporal policy 2024 2025 arXiv
55. [ext] intermittent depth frames RGB-D SLAM only some frames have depth mixed monocular RGB-D Gaussian splatting mapping
56. "sensing on demand" OR "sense on demand" depth camera robot mapping learned trigger energy budget
57. [ext] patent selectively activate depth sensor based on image data scene change uncertainty augmented reality power
58. Araújo Gonzalez Morin Ericsson energy-efficient SLAM sensor activation depth sensor monocular camera paper
59. GenNBV generalizable next-best-view policy active 3D reconstruction CVPR 2024
60. LiDAR frame skipping adaptive sensing rate energy 3D perception autonomous driving learned when to scan
61. keyframe selection from casually captured video for NeRF or Gaussian splatting training subset informative frames coverage uncertainty
62. [ext] "depth acquisition" policy budget RGB stream reconstruction "novel view" OR "held-out view" active selection which frames depth sensor 2026
63. [ext] active learning of depth measurements for 3D Gaussian splatting from RGB video LiDAR frames selection uncertainty "budget"
64. [ext] "selective depth sensing" OR "budgeted depth sensing" OR "depth sensing on a budget" OR "adaptive depth sensing" RGB camera
65. [ext] camera-triggered LiDAR acquisition adaptive temporal sampling RGB guided decide whether to fire LiDAR this frame energy depth estimation
66. "temporal sparsification" OR "temporally sparse" depth sensor RGB video densification adaptive schedule which frames depth 2025
67. "ToF-Splatting" Conti Poggi Cambareri Oswald Mattoccia ICCV 2025 openaccess
68. active learning monocular depth estimation selecting images to label with depth ground truth paper
69. test-time adaptation depth estimation sparse depth on selected frames online selective supervision LiDAR budget video
70. smartphone LiDAR energy consumption AR scene reconstruction adaptive LiDAR usage duty cycle research paper mobile 3D scanning
71. energy-aware perception drone mapping power gating LiDAR or depth camera selectively enable sensors trajectory information gain
72. "measurement scheduling" OR "observation scheduling" depth sensor 3D reconstruction mapping robot RGB-D information gain fixed path
73. streaming submodular selection frames online budget secretary problem view selection 3D reconstruction video
74. Budgeted stream-based active learning via adaptive submodular maximization Fujii Kashima NIPS 2016
75. Brandão Figueiredo Bernardino "scheduling" "depth sensors" legged robot information maximization
76. depth supervision on a subset of training views Gaussian splatting NeRF "only a few views" have depth mixed RGB and RGB-D views selection
77. reinforcement learning toggle LiDAR on off energy-efficient navigation mapping robot camera always on sensor activation policy

此外用 Crossref API 与 Semantic Scholar API 核对了若干论文的 venue/年份/DOI（Isler 2016、Gofer 2021、Bergman 2020、Super-Pixel Sampler 2020、Arora 2019、Trimpe 2014、Brandão 2020、Bajcsy 2018、Makarov 2021）。

**检查过但未放入主表的论文（每行一条及原因）：**
- [Depth Map Estimation of Dynamic Scenes Using Prior Depth Information](https://ar5iv.labs.arxiv.org/html/2002.00297)（Noraky & Sze，arXiv 2020）：只给出两次测量之间的估计算法，未规定何时采集深度；主表已收录同组的 A1、A2。
- [Monocular camera activation for localization based on data from depth sensor（US12056893B2）](https://patents.google.com/patent/US12056893B2/en)：方向相反，是按深度数据决定何时开单目相机；与 A4 同族。
- [Learn-to-Score (ECCV 2018)](https://mlanthology.org/eccv/2018/hepp2018eccv-learntoscore/)：学习式 NBV 效用，控制无人机视角；E 类已有 GenNBV 代表。
- [Neural Visibility Field (CVPR 2024)](https://arxiv.org/abs/2406.06948)：基于 NeRF 可见性不确定性的 NBV，控制相机；与 E 类重复。
- [GauSS-MI (RSS 2025)](https://roboticsconference.org/2025/program/papers/30/)：3DGS 互信息 NBV，控制相机；与 E6/E7 重复。
- [ActiveSplat (RA-L 2025)](https://arxiv.org/abs/2410.21955)：3DGS 主动建图（视角选择 + 拓扑路径），控制相机；与 E6 重复。
- [Informative Rays Selection for Few-Shot NeRF / KeyNeRF (VISAPP 2024)](https://arxiv.org/abs/2312.17561)：从固定图像池选视角与像素，资源是训练视角而非深度；与 C3 重复。
- [Super-Pixel Sampler / Image-Guided Depth Sampling and Reconstruction（ICRA 2020，经 Crossref 核对）](https://arxiv.org/abs/1908.01379)：单帧超像素深度采样；D 类已有多个代表，在第 3 节引用。
- [Adaptive Illumination based Depth Sensing using Deep Superpixel and Soft Sampling Approximation](https://arxiv.org/abs/2103.12297)（arXiv 2021，未见正式发表）：单帧自适应照明掩码；与 D1 重复。
- [Towards a MEMS-based Adaptive LIDAR (3DV 2020)](https://arxiv.org/abs/2003.09545)：空间自适应扫描硬件原型；不涉及帧级决策。
- [Virtually increasing the measurement frequency of LIDAR sensor utilizing a single RGB camera](https://arxiv.org/abs/2302.05192)（arXiv 2023）：固定低频 LiDAR 加 RGB 补帧；属 B 类，但比 DoD 弱。
- [Sense Less, Generate More (R-MAE)](https://arxiv.org/abs/2406.07833)（arXiv 2024）：随机角度区域开激光，配合掩码自编码预训练；随机而非决策式。
- [Neural RGB→D Sensing (CVPR 2019)](https://arxiv.org/abs/1901.02571)：从单目视频估计深度及其不确定性；可作为"RGB 引导的不确定性打分器"组件，本身不做采集决策。
- [EcoFusion (DAC 2022)](https://arxiv.org/abs/2202.11330)：按情境选择多传感器融合配置以省能耗，任务是检测；在第 3 节引用。
- [AdaMML (ICCV 2021)](https://mlanthology.org/iccv/2021/panda2021iccv-adamml/)：按视频片段选模态以省计算；在第 3 节作为类比引用。
- [Selective Sensor Fusion for Neural VIO (CVPR 2019)](https://arxiv.org/abs/1903.01534)：特征级软/硬掩码融合，不决定是否采集。
- [Information-Driven Direct RGB-D Odometry (CVPR 2020)](https://mlanthology.org/cvpr/2020/fontan2020cvpr-informationdriven/)：信息论点选择与关键帧，资源是计算；在第 3 节引用。
- [Gaussian Splatting SLAM (CVPR 2024)](https://mlanthology.org/cvpr/2024/matsuki2024cvpr-gaussian/)：共视 IoU/平移阈值关键帧管理；作为覆盖类基线来源引用。
- [EliGSiR: Continual RGB-D Mapping with Gaussian Splatting under Bounded Compute](https://arxiv.org/abs/2609.20348)（arXiv 2026）：已知位姿的 RGB-D 流中按地图状态调度视图，但资源是计算，且每帧都有深度。
- [Towards Energy-Efficiency by Navigating the Trilemma of Energy, Latency, and Accuracy (ISMAR 2024)](https://arxiv.org/abs/2409.04018)：TSDF 融合的均匀降帧等能耗研究，只计计算能耗；作为"均匀基线很强"的证据在第 3、4 节引用。
- [ActMVS: Active Scene Reconstruction with Monocular Multi-View Stereo (ICRA 2026)](https://arxiv.org/abs/2606.01367)：纯 RGB 主动重建，结果接近 RGB-D 方法；控制相机，但说明深度的边际价值需要实测。
- [LiDAL (ECCV 2022)](https://arxiv.org/abs/2211.05997)：LiDAR 语义分割的标注预算主动学习，属跨场景训练而非测试时重建。
- [Learning Loss for Active Learning in Depth Reconstruction Problem (CINTI 2021)](https://doi.org/10.1109/cinti53070.2021.9668436)：为训练深度网络选择待标注样本；只核对了 Crossref 元数据。
- [Budgeted stream-based active learning via adaptive submodular maximization (NIPS 2016)](https://proceedings.neurips.cc/paper/2016/hash/07cdfd23373b17c6b337251c22b7ea57-Abstract.html)：流式预算 K 查询的理论框架，非视觉；在第 3、4 节作为形式化工具引用。
- [Near-Optimal Sensor Placements in Gaussian Processes (JMLR 2008)](https://www.jmlr.org/papers/v9/krause08a.html)：以已选/未选位置互信息为准则的布置，非流式、非视觉；作为离线上界基线引用。
- [Revisiting Active Perception（Autonomous Robots 第 42 卷，2017 在线，经 Crossref 核对）](https://arxiv.org/abs/1603.02729)：主动感知的定义性综述，不是具体方法。

**核对方式说明：** CVF open-access 的 HTML 页面对抓取工具返回 403，因此 CVPR/ICCV/WACV 论文改用 arXiv 摘要页、ML Anthology 页或 Crossref 元数据核对 venue/年份；方法细节尽量读 arXiv HTML 或 ar5iv 正文。几个 PDF（Isler 2016、Brandão 2020、Kerl 2013）无法被抓取工具解析，相应结论只依据摘要或元数据，并在表中注明。DVO-SLAM（Kerl et al., IROS 2013）的关键帧准则因此未能核实，正文未将其作为依据。

**复核：** 2026-10-06 再次打开了 A1（arXiv 1810.01930）与 A5（arXiv 2605.16805）的摘要页，题目、作者、触发方式与上表一致；A5 的注释栏写明 ICRA 2026 录用。

---

## 6. 与 EXP-MEASUREMENT-PLANNING-ORACLE-V1 基线的对应

第一阶段用的是无模型后端：调和插值作用于初始上下文加上已请求帧的 z-buffer。第 4 节要求的基线族在本实验中这样落实（定义见该实验的 `PROTOCOL.md`）：

| 第 4 节的基线族 | 本实验中的规则 | 说明 |
| --- | --- | --- |
| 均匀/固定频率、随机 | UNIFORM、TEMPORAL_EVEN、RANDOM_SEEDED；另报告精确的随机期望 RANDOM_EXPECTED | 固定频率按候选序或帧序等距 |
| 位姿空间覆盖（FVS） | POSE_FARTHEST | 贪心最远：平移 + 1 m/rad × 旋转 |
| 共视/新像素 | MAX_UNCOVERED、GREEDY_COVERAGE | 候选视角中离已测命中点超过 8 像素的比例，即"新像素比例" |
| 当前帧可估计性触发 | GREEDY_PHOTOMETRIC | 已测 RGB-D 重投影到候选视角后与候选 RGB 的残差：解释不了的帧优先 |
| 不确定性/信息增益 | GREEDY_FILL_DISTANCE、MAX_NOVELTY3D | 无模型后端没有概率模型，用"到已测点的距离"和"预测表面离已测点的比例"作代理 |
| 上下界 | K = 0（不加深度）、K = 9（全部候选帧）、ORACLE（穷举子集）、XORACLE（交叉查询）、QUERY_NEAREST（用查询相机，不可部署） | ORACLE 是穷举而不是贪心，比"事后贪心"更强 |
| 在线时间节奏 | LATEST（最新的 k 帧） | 在线逐帧决策留到以后的阶段 |

**本阶段没有覆盖的部分：**

- FisherRF 式的 EIG 与集成方差，需要概率或学习式的 3D 模型；
- 学习式按需深度（DoD 一类致密化器）；
- 在线、不可撤销的预算节奏控制。

如果第一阶段确立了余量，第二阶段需要换用有不确定性的状态表示，再补上这些基线。

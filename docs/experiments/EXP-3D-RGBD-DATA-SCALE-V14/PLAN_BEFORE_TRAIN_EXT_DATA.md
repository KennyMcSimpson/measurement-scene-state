# EXP-3D-RGBD-DATA-SCALE-V14 计划：只增加训练场景

- **写定：** Claude，2026-09-30（远端时间），在 V13 完成之后、读取任何新训练数据之前。
- **资源：** 全程 CPU（8–23 号核，单线程 BLAS），GPU 暂停。
- **修改规则：** 本计划写定后，数据规则、配方、评价与判定都不得根据任何新数据或结果修改。

## 0. 问题

- **已有结果：** 在 TRAIN72 上，加宽（V11 宽度 8→32、V12 32→64）、多尺度 3D 细化、补全加权都没有改善补全。组合方法相对经典补洞在 EVAL-V3、EVAL-V4 与 EVAL-FAR 上都只是数值领先（+0.010 到 +0.016，CI 跨零）。
- **剩下的杠杆：** 训练场景的数量与多样性。V6 把训练场景从 24 增加到 72，DEV 收益 +0.033，CI 跨零。
- **本轮问题：** 同一配方只增加训练场景，补全能否改善？改善后的组合能否胜过经典补洞？

## 1. 新训练数据 EXP-3D-RGBD-TRAIN-EXT-DATA

- **候选：** 只用元数据确定，锁定之后才读取任何媒体。
  - Hypersim 官方 train 或 val 划分（protocol partition 相同），曾被观察过的场景也可以用。
  - 不在受保护或排除的名单中（final holdout 与 diagnostic test 永不读取）。
  - 不属于任何评价队列：DEV、FRESH-V1 与 FRESH-V2 的候选、EVAL-V3、EVAL-V4；也不在 TRAIN72 中。
  - 不是以往数据准备中已判定无效的场景（这些规则是确定性的，重读只会再失败）。
  - 与 DEV 场景不共享 volume，保证 DEV 上的 checkpoint 选择不偏向新模型。FRESH 队列已经用完，不再隔离。
  - 源资产与所有评价场景、受保护场景、TRAIN72 以及彼此之间都不相同。
  - cam_00 至少 16 个公开帧，轨迹不是 BAD。
- **顺序与取舍：**
  - 按 sha256("TRAIN-EXT-20260930:" + scene) 排序，逐个分段下载。
  - 套用与 TRAIN-X、EVAL-V3 相同的冻结规则：校准、源几何、帧质量、只看相机的角色规则。
  - 全部有效场景都保留，每个 volume 不设上限；数据预算 2 GiB。
- **门槛：** 有效场景少于 40 个时不训练，V14 不运行。
- **预计：** 元数据统计有 86 个候选，分布在 39 个 volume；按以往的失败率，约 60–70 个有效，训练集约扩大到 1.9 倍。

## 2. 训练

- **配方：** V11 C1，不做任何改动。
  - 模型：hidden 32、expansion 64，32³ 网格，CONTEXT_DEPTH bounds。
  - 训练：可变 3–7 视角，6000 步，Adam 1e-3，梯度裁剪 1.0。
  - 选择：checkpoint 用冻结的 V2 DEV 规则逐 seed 选择。
- **变体：**
  - C0：TRAIN72。必须与 V11 C1 逐位相同，包括训练记录、DEV 曲线与选中的 checkpoint，否则停止。
  - C1：TRAIN72 + TRAIN-EXT。
- **执行：** 3 个 seed，6 个训练并行，每个占一个核。
- **代码：**
  - 由 V11（含 Amendment 1）的代码逐字派生，只有两处改动：
    - 训练集名单；
    - DEV 曲线评价只记录状态哈希，不再保存状态文件，只有选中的 checkpoint 保存状态（节省约 10 GB 磁盘）。
  - 派生之后先检查：C0 的前 300 步训练损失与 V11 C1 逐位相同。

## 3. 评价

- **主队列：** EVAL-V3 ∪ EVAL-V4 的近处 query，共 83 个场景。
  - 两个队列的场景互不重叠。
  - 它们与 TRAIN72、TRAIN-EXT 按场景和源资产都不相交，但可能共享 volume（TRAIN72 也是如此）。
- **封存：** 与 V13 相同的代码路径。锁定之后先重算 V11 的封存预测，再封存全部预测，最后读取 query 深度。
- **主判定：**
  - **P1：** DATA_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1)，使用冻结的 V2 gate（与 V11 的 COMPLETION_GAIN 同一定义和同一函数）。
  - **P2：** AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 of C1)，使用 INPAINT-V1 的标签规则。
    - 选 HFILL8 是因为它没有可调参数。V13 的弃权阈值是在 EVAL-V3/V4 上选定的，所以 AHFILL8 只作描述性结果。
- **分支：**
  - A：P1 SUPPORTED 且 P2 ABOVE。
  - B：只满足其一。
  - C：两者都不满足。
- **描述性：**
  - ALL/NEAR/FAR 各方法误差；
  - AHFILL8（阈值 0.5）；
  - DEV 结果与训练曲线；
  - EVAL-FAR 远处 query；
  - carrier 与常数、与 REPROJ_NN 的差；
  - C1 − C0 在 ALL 上的差。

## 4. 禁止项

- 不得根据任何新数据或结果修改候选规则、有效性规则、门槛、配方、步数、选择规则、评价队列、判定或分支。
- 不打开受保护的 final holdout。

# EXP-3D-RGBD-WIDTH-DATA-V15 协议：更多数据时宽度是否起作用

- **依据：** [`PLAN_BEFORE_V15_TRAINING.md`](PLAN_BEFORE_V15_TRAINING.md)（sha256 `4c0cc1b7…`），在任何 V15 训练之前写定。本协议只把计划落实到代码，不改变任何规则。
- **冻结：** 训练开始之前冻结（`preregistration.json`）。冻结时哈希协议、计划、全部输入、源码，以及此前所有实验的输出。
- **资源：** 全程 CPU（8–23 号核，单线程 BLAS），GPU 暂停。

## 1. 设计

- **训练集：** 两个变体都用 TRAIN72 + TRAIN-EXT，共 127 个场景，与 V14 C1 相同。
- **C0：** 宽度 32（hidden 32、expansion 64，64,238 个参数），就是 V14 C1 的配方。训练记录、DEV 曲线、选中的 checkpoint 与评价预测都必须与 V14 C1 逐位相同。
- **C1：** 宽度 64（hidden 64、expansion 128，250,542 个参数），每个 seed 的初始参数与 V12 C1 相同。
- **配方：** 与 V11 C1 完全相同（32³ 网格，CONTEXT_DEPTH bounds，冻结的 V2 C1 损失，6000 步，Adam 1e-3，梯度裁剪 1.0，可变 3–7 视角）。两个宽度共享同一数据流：场景顺序、query 槽位与光线索引都相同。
- **选择：** 冻结的 V2 DEV 规则，逐 seed 选择。
- **执行：** seed 为 20260928、20260929、20260930；6 个训练并行，每个占一个核。

## 2. 代码

由 V14 的文件逐字派生（`/tmp/v15_derive.py`，每处替换都检查出现次数），只改变体表、参数量、对照与文字。

- **保留 V14 的两处修正：** 按本变体自己的训练集取场景；DEV 曲线不存状态文件。
- **EVAL 阶段：** `scripts/evaluate_rgbd_wide_scale_eval.py`。
  - 封存之前，C0 的每个预测都必须与 V14 已封存的 C1 预测一致（最大差 ≤ 1e-6），否则停止、不读取任何 query 深度。
  - 在主队列上额外渲染 V12 C1（宽度 64 / 72 个场景）与 C0、C1 第 6000 步的 checkpoint；宽度 32 / 72 个场景的格子直接取 V14 封存的 C0 预测（= V11 C1）。这些都只作描述。
- **命名：** 文件名用 `rgbd_wide_scale` 前缀，避免被 V12 的 `*rgbd_width*` 通配规则匹配。
- **验证：**
  - 7 个单元测试；
  - 两轮合成数据全流程排练（第二轮含 finalize），都 PASS；
  - 排练中确认：两个宽度共享数据流；DEV 曲线没有状态文件；C0 与按 V13 代码路径生成的对照逐位一致；初始参数哈希与 V11 C1、V12 C1 一致。

## 3. 判定（与计划相同）

- **主队列：** EVAL-V3 ∪ EVAL-V4 的近处 query，共 83 个场景。
- **P1：** WIDTH_AT_SCALE_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1)，冻结的 V2 gate。
- **P2：** AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 of C1)，INPAINT-V1 的标签规则。
- **分支：**
  - A：P1 SUPPORTED 且 P2 ABOVE；
  - B：只满足其一；
  - C：两者都不满足。
- **次要：** HFILL8_WIDTH_GAIN（V2 gate）。
- **描述性：**
  - 2×2（宽度 32/64 × 72/127 个训练场景）与交互作用 [FAR(32,72) − FAR(64,72)] − [FAR(32,127) − FAR(64,127)]；
  - 第 6000 步的结果；
  - AHFILL8；
  - 各队列（含 EVAL-FAR）；
  - DEV 与训练曲线。
- **聚合：** 与 V11 相同。每个场景内 seed 等权，再取角色均值；场景 bootstrap 10,000 次，seed 20260928。

## 4. 禁止项

- 不得根据任何结果修改配方、步数、选择规则、评价队列、判定或分支。
- 基础设施故障按冻结策略处理；主训练失败则整个实验停止。
- 不打开 FRESH 与受保护的 final holdout。

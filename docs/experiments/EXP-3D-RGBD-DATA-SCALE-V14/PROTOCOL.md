# EXP-3D-RGBD-DATA-SCALE-V14 协议：只增加训练场景

- **依据：** [`PLAN_BEFORE_TRAIN_EXT_DATA.md`](PLAN_BEFORE_TRAIN_EXT_DATA.md)（sha256 `526c3623…`），在读取任何新训练数据之前写定。本协议只把计划落实到数据与代码，不改变任何规则。
- **冻结：** 训练开始之前冻结（`preregistration.json`）。冻结时哈希协议、计划、全部输入、源码，以及此前所有实验的输出。
- **资源：** 全程 CPU（8–23 号核，单线程 BLAS），GPU 暂停。

## 1. 数据

- **新训练数据 TRAIN-EXT：** 见 [`outputs/EXP-3D-RGBD-TRAIN-EXT-DATA`](../EXP-3D-RGBD-TRAIN-EXT-DATA/README.md)。
  - 按计划的规则锁定了 77 个候选；读取之后，55 个有效，22 个未通过冻结的有效性规则。
  - 55 个场景来自 29 个 volume：31 个来自官方 train，24 个来自官方 val。
  - 与 DEV 不共享 volume；源资产与所有评价场景、受保护场景和 TRAIN72 都不相同。
  - 每个场景都有冻结的角色规则要求的 4 个空闲帧（可变视角训练用）。
- **训练集：**
  - C0：TRAIN72，与 V11 相同。
  - C1：TRAIN72 + TRAIN-EXT，共 127 个场景，约为 1.76 倍。
- **DEV：** 8 个场景，只用于选择 checkpoint 和描述性结果。
- **评价队列：**
  - 主队列：EVAL-V3（57 个场景）与 EVAL-V4（26 个场景）的近处 query 合并，共 83 个场景。两者场景互不重叠；都与两个训练集按场景和源资产不相交，但可能共享 volume。
  - 描述性：EVAL-FAR（79 个场景的远处 query，场景与 EVAL-V3/V4 相同）。
- **不使用：** FRESH-V1/V2；受保护的 final holdout 不打开。

## 2. 训练（配方与 V11 C1 完全相同）

- **模型：** hidden 32、expansion 64，64,238 个参数；32³ 网格，CONTEXT_DEPTH bounds，冻结的 V2 C1 损失。
- **优化：** 6000 步，Adam 1e-3，梯度裁剪 1.0，每步 1 个场景、1024 条光线。
- **视角：** 可变 3–7 视角（V11 的额外视角规则，独立随机数生成器）。
- **初始化：** 两个变体的每个 seed 都与 V11 C1 的初始参数完全相同。
- **选择：** 在 12 个 checkpoint 中，用冻结的 V2 DEV 规则逐 seed 选择。
- **执行：** seed 为 20260928、20260929、20260930；6 个训练并行，每个占一个核。

## 3. 代码

由 V12 的文件逐字派生（`/tmp/claude_v14_derive.py`，每处替换都检查出现次数），只有以下实质改动：

1. **训练集：** C1 的训练集为 TRAIN_EXT。V14 自带 `train_records`，按 V14 的变体表取训练集。
   - 原因：V12 借用的 V8 版本查的是 V8 的变体表（所有变体都是 TRAIN72），照抄会让 C1 悄悄地仍在 TRAIN72 上训练。
   - 测试 `test_train_records_follow_the_v14_table_and_not_the_v8_table` 专门检查这一点。
2. **状态文件：** DEV 曲线评价只记录状态哈希，不保存状态文件；只有选中的 checkpoint 保存状态（节省约 10 GB 磁盘）。所有读取状态文件的代码都只读选中的那一份。
3. **EVAL 阶段：** 新写的 `scripts/evaluate_rgbd_data_scale_eval.py`（seal / score / analyze）。
   - 三个队列一次封存，渲染深度与不透明度。
   - 封存之前，C0 的每个预测都必须与已封存的 V11 C1 预测一致（最大差 ≤ 1e-6），否则停止、不读取任何 query 深度。对照的封存分别来自：EVAL-V3 为 V11，EVAL-V4 为 EVAL-V4 复现实验，EVAL-FAR 为 V13。

- **finalize 的附加检查：** 每个变体的训练记录只能出现本变体训练集中的场景。
- **验证：**
  - 7 个单元测试；
  - 两轮合成数据全流程排练（第二轮含 finalize），都 PASS；
  - 排练中确认：C0 只用了 TRAIN72 的场景，C1 用到了新场景；DEV 曲线没有状态文件；C0 与按 V13 代码路径生成的对照逐位一致。

## 4. 判定（与计划相同）

- **P1：** DATA_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1)，合并的 83 个场景，冻结的 V2 gate（与 V11 COMPLETION_GAIN 同一函数）。
- **P2：** AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 of C1)，同一批场景，INPAINT-V1 的标签规则。
- **分支：**
  - A：P1 SUPPORTED 且 P2 ABOVE；
  - B：只满足其一；
  - C：两者都不满足。
- **次要与描述性：**
  - HFILL8_DATA_GAIN（V2 gate）；
  - AHFILL8（不透明度阈值 0.5），因为阈值是在 EVAL-V3/V4 上选定的，只作描述；
  - 各队列分别的结果（含 EVAL-FAR）；
  - DEV 的 DATA_GAIN_DEV 与训练曲线；
  - carrier 与常数、与 REPROJ_NN 的差。
- **聚合：** 与 V11 相同。每个场景内 seed 等权，再取角色均值；场景 bootstrap 10,000 次，seed 20260928。

## 5. 禁止项

- 不得根据任何结果修改训练集、配方、步数、选择规则、评价队列、判定或分支。
- 基础设施故障（信号终止）按冻结的策略归档后重跑；其他错误不重试，主训练失败则整个实验停止。
- 不打开 FRESH 与受保护的 final holdout。

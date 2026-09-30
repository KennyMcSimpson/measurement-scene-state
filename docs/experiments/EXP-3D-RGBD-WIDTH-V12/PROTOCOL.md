# EXP-3D-RGBD-WIDTH-V12 协议

起草：Claude（经用户授权持续推进），2026-09-29 19:1x（远端时间），写于 V11 出任何评价结果之前。按用户要求，GPU 暂停使用，全程只用 CPU（8–23 号核）。

## 0. 是否运行（由计划规定）

- **依据：** `outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/PLAN_BEFORE_V11_RESULTS.md`（sha256 `b726d336…`）。
- **规则：** 只有当 V11 在 EVAL-V3 上的 COMPLETION_GAIN 点估计 > 0 时，本实验才运行；prepare 会读取 V11 的 `eval_v3_results.json` 检查这一条件，不成立就拒绝冻结。
- **问题：** 在 V11 C1 的配方上把宽度从 32 加到 64，能否继续改善未观测区域的补全？

## 1. 变体

| ID | 名称 | 宽度 | 参数 | 角色 |
|---|---|---|---|---|
| C0 | WIDTH32 | hidden 32，expansion 64 | 64,238 | 机制对照。重新训练 V11 C1 的配方，前 300 步必须与 V11 C1 逐位相同（损失、场景、额外视角） |
| C1 | WIDTH64 | hidden 64，expansion 128 | 250,542 | **PRIMARY_METHOD** |

- **相同的部分：** 其余配方与 V11 C1 完全相同，包括 TRAIN72、32³、CONTEXT_DEPTH bounds、冻结的 V2 C1 loss、6000 步、checkpoint 步、学习率、可变 3–7 视角规则及其独立的随机数生成器（seed + 7919）、seed 20260928–30、场景顺序与射线。
- **执行：** 6 个训练同时运行，各占 1 个 CPU 核。
- **重新训练 C0 的原因：** 这样 DEV 上的选择与诊断流程可以原样沿用 V11 的双变体代码。按计划，EVAL-V4 上主对比的对照仍是 **V11 C1 已选中的 checkpoint**，C0 只用于检查逐位复现，并作为 DEV 描述。

## 2. DEV（checkpoint 选择与次要描述）

- **checkpoint 选择：** 8 个 DEV 场景，标准 3 视角，规则与 V7–V11 相同。
- **描述：** 报告 WIDTH_GAIN_DEV = AbsRel(C0) − AbsRel(C1) 以及 V11 流程中的其余描述。这些结果不进入分支判定。
- **舍入容差（19:5x 补充，仍在 V11 出结果之前）：** DEV 评价从一开始就采用 V11 Amendment 1 的做法：渲染颜色超出 [0, 1] 不超过 1×10⁻⁶ 时截回，超出更多仍然报错。原因是宽 carrier 在 float32 下会把亮面颜色算成 1.0000001；评价模块由 V11 Amendment 1 的评价模块派生，只改了 carrier 的导入。

## 3. EVAL-V4 评价（主）

- **队列：** EVAL-V4，26 个场景（`outputs/EXP-3D-RGBD-EVAL-V4-DATA`，清单 sha256 `15f104af…`）。
- **被比较的对象：** V11 C1 已选中的 checkpoint（对照，按计划不重新训练）与本实验 C1 已选中的 checkpoint（WIDTH64），每个各 3 个 seed。本实验 C0 的 EVAL-V4 预测也会计算，但只用来检查它与 V11 C1 是否一致。
- **流程：**
  - 状态构建、渲染、REPROJ_NN、NEAR/FAR（8 像素）以及 REPROJ_HARMONIC 与组合，都与 V11 和 INPAINT-V1 的冻结函数完全相同；
  - 全部预测在读取 query 深度之前封存。
- **主 gate：** WIDTH64_COMPLETION_GAIN = FAR AbsRel(V11 C1) − FAR AbsRel(V12 C1)，使用冻结的 V2 gate（与 V11 的 COMPLETION_STATUS 同一函数），得到 WIDTH64_COMPLETION_STATUS。
- **次要：**
  - AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8 of V12 C1)，沿用 INPAINT-V1 的标签规则；
  - AbsRel(HFILL8 of V11 C1) − AbsRel(HFILL8 of V12 C1)，沿用冻结的 V2 gate。
- **分支：**
  - A：WIDTH64_COMPLETION_STATUS 为 SUPPORTED，且 HFILL8(V12 C1) 相对调和补洞为 ABOVE；
  - B：只满足其一；
  - C：两者都不满足。

## 4. 与 V10 并行运行

- **安排：** V10 已经在运行时，本实验可以与它并行。
- **封存范围：** 本实验冻结时封存 `outputs/` 与 `docs/experiments/` 下的全部已有文件，但以下两类目录除外，并在冻结文件中逐一列出：
  - 正在运行的 V10（`outputs/EXP-3D-RGB-PRIOR-BOUNDS-V10`、`docs/experiments/EXP-3D-RGB-PRIOR-BOUNDS-V10`）；
  - 计划中会在本实验运行期间执行的 EXP-3D-RGBD-REPLICATION-EVAL-V4。
- **原因：** 这些目录在本实验运行期间会合法地写入新文件或追加日志。
- **收尾检查：** 本实验的收尾只重新核对被封存的文件。

## 5. 禁止项

- 使用 GPU；使用 FRESH-V1/V2 或 EVAL-V3 作评价；打开 final holdout。
- 根据任何结果修改宽度、配方、区域定义、FILL8 阈值、gate、seed 或训练集。

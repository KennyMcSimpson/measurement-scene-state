# EXP-3D-RGBD-COMPLETION-V11：V9 出结果之前写定的计划

写于 2026-09-29 17:04（远端时间），17:07 修订一处（C0 由复用改为重新训练），修订时仍未查看任何 V9 结果。此时 V9 仍在训练最后一个 seed；它在训练中写出的 checkpoint DEV 评价没有被任何人查看过。V11 的 prepare 会核对本文件的 sha256。

## 动机（均为探索性，见各自 README）

- 上限分析（DEV，已封存预测）：被看到的区域用重投影、其余用 carrier，只比重投影好约 0.03；瓶颈在补全质量。
- 容量试验（TRAIN24 训练，48 个 TRAIN-X 场景评价）：hidden 32 相对 hidden 8，在离测量较远的像素上好 0.059 [+0.021, +0.098]；测量保留组合（FILL8）比重投影好 0.028 [−0.003, +0.063]。
- 容量试验使用了 TRAIN-X，所以 V11 不能再用 TRAIN-X 做确认；8 个 DEV 场景的统计功效不足。V11 的主评价放在新的 EVAL-V3 队列上（`scripts/prepare_rgbd_eval_v3_data.py`）。

## 1. 基础配方（在 V9 出结果之前固定）

- 与 Core B V2 相同的规则：V9 VIEWCOUNT_GAIN 的点估计 > 0 时用 V9 C1（VARIABLE3TO7）的配方，否则用 V9 C0（FIXED3）的配方。
- V11 C0 用 V11 的代码重新训练该 V9 变体的配方（宽度 8），必须与该 V9 变体逐位相同（完整性检查）；这样两个变体走同一条代码路径。
- V11 C1 用同一配方训练，只把 hidden_dim 8→32、expansion_dim 16→64；其余（TRAIN72、32³、CONTEXT_DEPTH bounds、loss、6000 步、checkpoint 步与 DEV 选择规则、seed、数据流）不变。
- 6 个训练（2 个变体 × 3 个 seed）同时运行，各占 1 个 CPU 核。

## 2. 评价与主问题

- 主评价队列：EVAL-V3（按场景与源资产与全部已用场景不相交，与 DEV、FRESH 的 volume 不相交；可能与 TRAIN72 共享 volume，只作机制队列）。所有状态在读取 GT 之前封存。
- 区域：FAR 为离上下文深度重投影命中点超过 8 像素的像素，NEAR 为其余；FILL8 为 NEAR 用 REPROJ_NN、FAR 用 carrier（不学习、可部署的组合）。
- 主 gate（冻结的 V2 gate，按 EVAL-V3 场景 bootstrap）：
  - COMPLETION_GAIN = FAR AbsRel(C0) − FAR AbsRel(C1)；
  - HYBRID_GAIN = AbsRel(REPROJ_NN) − AbsRel(FILL8 of C1)。
- 分支：A 两者都 SUPPORTED；B 其中一个 SUPPORTED；C 其余。

## 3. 队列

V11 排在 Core B V2 之后、V10 之前；EVAL-V3 的下载在 Core B V2 运行期间进行。用户另行指定时以用户为准。

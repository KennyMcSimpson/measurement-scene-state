# EXP-3D-RGBD-DYNAMIC-WRITE-V2：V9 出结果之前写定的计划

写于 2026-09-29 15:52（远端时间），15:56 更正一处描述。此时 V9 仍在训练；它的训练脚本在每个 checkpoint 写出 DEV 评价文件，但没有人查看过 V9 的任何评价输出或指标。Core B V2 的 prepare 会核对本文件的 sha256。

## 1. Core B V2 使用哪个 carrier

- 若 V9 的 VIEWCOUNT_GAIN 点估计 > 0：使用 V9 C1（VARIABLE3TO7）的 3 个 seed。
- 否则：使用 V9 C0（FIXED3，与 V8 C1 相同）的 3 个 seed。
- 每个 seed 的 checkpoint 沿用 V9 的 DEV 选择结果，与 Core B V1 使用 V7 C1 的方式相同。

## 2. 非学习基线与主 gate

- 策略集合与 Core B V1 相同：NO_STREAM、OFF、OFF_CLAMP3、ALL、FUSE、COMPLETE、ALL_UNTRAINED、WRONG_SCENE。
- 非学习基线 BASELINE_POLICY 预先固定，不按 DEV 结果挑选：
  - carrier 为 V9 C1 时取 OFF（7 个视角在其训练分布内）；
  - carrier 为 V9 C0 时取 OFF_CLAMP3。
- 主 gate，均使用冻结的 V2 估计器：
  - BEYOND_BASELINE = AbsRel(BASELINE_POLICY) − AbsRel(ALL)；
  - SPECIFICITY = AbsRel(WRONG_SCENE) − AbsRel(ALL)。
- 分支：
  - A：两者都 SUPPORTED。
  - B：BEYOND_BASELINE 的点估计 > 0，但不满足 A。
  - C：其余情况。
- 写入规则的训练（TRAIN72、固定步数、不做 DEV 选择）与 stream 规则（4 个均匀空闲帧）沿用 Core B V1。

## 3. V9 收尾之后的队列

队列顺序不取决于 V9 的结果；用户另行指定方向时以用户为准。

1. 几何基线补充分析 2（描述性，`outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/ADDENDUM2.md`）。
2. EXP-3D-RGB-PRIOR-BOUNDS-V10：RGB-only，bounds 先验单因素（冻结先验对 TRAIN72 q01/q99 先验），32³，固定 3 视角训练。它不依赖 V9 的任何结果。
3. Core B V2（本计划）。
4. Core A 保留测量的新一轮（设计中）。

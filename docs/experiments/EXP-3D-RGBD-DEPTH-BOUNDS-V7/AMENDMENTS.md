# EXP-3D-RGBD-DEPTH-BOUNDS-V7 修订记录

## 修订 1（2026-09-29，在查看任何 V7 DEV 指标之前写定）

### 发生了什么

- 训练（9 个 run）、选中 checkpoint 的 DEV 评价、状态相关诊断和无几何参考都已按冻结流程完成。
- 统计分析这一步在冻结的 V2 估计器（`geometry_carrier_statistics.analyze`）里报错退出，错误为 `Matched variants must share frozen geometry ray-hit fractions`。
- 写本修订时，我只看过这条报错和训练日志，没有打开任何 V7 的 DEV 指标、参考比较或分析结果。

### 原因

- 这条检查要求 C0 与 C1 在每个场景上的几何射线命中比例（query 射线与状态 bounds 相交的比例）完全相同。
- 这是 V2–V6 的前提：那几轮所有变体都用同一个 bounds 规则，命中比例必然相同，所以这条检查用来防止几何不匹配。
- V7 检验的因素恰恰是 bounds 规则，C1 与 C0 的命中比例按设计就不同。这条检查在 V7 中不适用。
- 合成演练没有暴露这个问题，因为在合成数据上两种规则都覆盖了全部射线。这是协议设计中的疏漏。

### 修订内容

- **只删一条断言：** 分析仍调用冻结的 V2 估计器，只删除这一条相等断言，也就是两行代码。
- **其余逐字不变：** 主对比、全部 gate、bootstrap、NO_HIT 口径和预注册分支都保持原样。
- **依赖命中比例的 opacity 检查仍按冻结定义执行，每个变体用自己的命中比例：**
  - 每个场景的覆盖率 ≥ 0.99 × 该变体的命中比例；
  - C1 的覆盖率比 C0 低不超过 0.01；
  - 在 C0∩C1 共同 opacity 掩码上计算的收益。
- **描述性输出：** 另外报告每个变体、每个场景的命中比例（`ray_hit_fractions.json`）。

### 实现

冻结文件一个都不改，finalize 仍逐个核对它们的哈希。新增以下文件：

- `scripts/analyze_rgbd_bounds_carrier_amend1.py`：从冻结估计器的源码中删掉这两行，并断言删掉的恰好是这两行，然后用冻结的分析脚本产出全部结果。
- `scripts/audit_rgbd_bounds_statistics_amend1.py`：在隔离目录中用修订后的分析重新计算，并与保存的结果逐字节比对。
- `scripts/run_rgbd_bounds_experiment_amend1.py`：只补跑分析和审计，然后执行冻结的 finalize。失败的分析日志移到 `audit/failed_attempts/`。
- `tests/test_rgbd_bounds_amend1.py`，验证三点：
  - 命中比例相同时，修订版与冻结版输出逐位相同；
  - 命中比例不同时，冻结版报错，修订版不报错；
  - 两份源码只差这两行。

这次报错不属于基础设施重试，不计入 `INFRASTRUCTURE_RETRIES`。

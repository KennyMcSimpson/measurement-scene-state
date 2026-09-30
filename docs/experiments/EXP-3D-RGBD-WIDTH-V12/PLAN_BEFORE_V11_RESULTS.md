# V11 出结果之前写定的后续计划

写于 2026-09-29 19:00（远端时间）。写作时的状态：

- V11 的 6 个训练仍在进行：C0 约 5200/6000 步，C1 约 3000/6000 步。
- V11 没有任何评价输出，`raw/` 目录不存在。
- 我没有读取任何 V11 的 DEV 曲线或 checkpoint 指标，只看过训练损失和步数。
- INPAINT-V1 已于 18:55:58 锁定（见其 `lock.json` 与 `audit/LOCK_NOTE.md`）。

## 1. 固定顺序（不因任何结果改变）

1. V11 收尾。
2. INPAINT-V1 `--stage run`。
3. V10 冻结并运行。启动命令先创建 `checkpoints/`，以免重复 V11 的启动事故。
4. **EVAL-V4 数据**（`EXP-3D-RGBD-EVAL-V4-DATA`，脚本 `scripts/prepare_rgbd_eval_v4_data.py`）：本计划写定后立即锁定并准备，只下载和验证，不运行模型。
   - 规则：沿用 EVAL-V3 的规则，并把 EVAL-V3 已读取的 78 个候选（57 个有效、21 个无效）加入已用集合，所以只剩从未读取过的场景。
   - 参数：新的排序种子 `EVAL-V4-20260929`；每个 volume 最多 2 个场景；至少 20 个有效场景；数据预算 2 GiB。
   - 定位：与 EVAL-V3 一样是机制队列，按场景和源资产与训练集不相交，可能与 TRAIN72 和 EVAL-V3 共享 volume，不作资格验证。
5. **EXP-3D-RGBD-REPLICATION-EVAL-V4**（不训练）：在 EVAL-V4 上复现 V11 与 INPAINT-V1 的主对比。
   - 冻结不变的部分：所用 checkpoint 为 V11 C0/C1 已选中的 checkpoint；区域定义、封存顺序、补洞方法、统计与标签规则全部沿用。
   - 主对比：HARMONIC_HYBRID_GAIN = AbsRel(REPROJ_HARMONIC) − AbsRel(HFILL8_C1)，沿用 INPAINT-V1 的标签规则。
   - 次要：V11 的 COMPLETION_GAIN 与 HYBRID_GAIN，使用冻结的 V2 gate。
   - 无论 V11 或 INPAINT-V1 的结果如何都运行。

## 2. V12（EXP-3D-RGBD-WIDTH-V12）是否运行

- **判定条件：** V11 在 EVAL-V3 上的 COMPLETION_GAIN 点估计 > 0。
- **条件成立时：**
  - 变体：V12 = hidden 64、expansion 128（约 25 万参数）。
  - 相同的部分：其余配方与 V11 C1 完全相同，包括 TRAIN72、32³、CONTEXT_DEPTH bounds、可变 3–7 视角、6000 步、seed，以及 checkpoint 的选择规则。
  - 对照：V11 C1 已选中的 checkpoint，不重新训练。
  - 主评价在 EVAL-V4 上进行：WIDTH64_COMPLETION_GAIN = FAR AbsRel(V11 C1) − FAR AbsRel(WIDTH64)，使用冻结的 V2 gate；次要为 HFILL8(WIDTH64) 相对 REPROJ_HARMONIC 的差。
  - 成本：在当前负载下测得 1.56 s/步（hidden 32 为 0.61 s/步），3 个 seed 并行约需 4–5 小时，全程 CPU。
- **条件不成立时：** 不运行 V12。宽度在 TRAIN72、32³ 上不是补全的杠杆，停止容量方向。
- **已排除的替代因素：** 只用 TRAIN 的探索性试验表明，多尺度 3D 细化和补全加权的射线采样都没有改善 FAR（见 INPAINT-V1 的 `audit/exploratory_completion_pilot_train_only/`）。所以不把它们列为下一轮因素。

## 3. 关于 final holdout 的建议（由用户决定）

- **如果 INPAINT-V1 为分支 A，且 EVAL-V4 复现的主对比也为 ABOVE：** 建议冻结方法（HFILL8 + V11 C1；若 V12 成立，另行讨论是否改用更宽的版本），起草 final holdout 的预注册分析计划，交用户批准后才打开。
- **否则：** 不建议为这条主张打开 final holdout，并如实报告"学到的补全相对经典几何补洞的优势没有确立"。
- **始终如此：** final holdout 只打开一次，时间由用户决定。

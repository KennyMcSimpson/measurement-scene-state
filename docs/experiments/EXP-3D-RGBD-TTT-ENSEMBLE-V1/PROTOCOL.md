# EXP-3D-RGBD-TTT-ENSEMBLE-V1：TTT 后的状态与无模型重投影取平均，能否稳定胜过无模型重投影？

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 来源与问题

- **V1 的 Core B：** 在 FRESH78 上，TTT 后的 DepthSplat 状态平均好于 7 帧无模型重投影 REPROJ7_HARMONIC（+0.057），但只有 54/78 个场景更好（PARTIAL）。
- **探索性分析：** V1 结束后，在 FRESH78 上做了探索（`coreb_ensemble_explore.py`，样本内，不是检验）。
    - 把 TTT 状态与 REPROJ7 逐像素取几何平均：+0.044，66/78 个场景更好。
    - 算术平均：+0.030，64/14。
    - 不做 TTT 的状态（CACHE）取几何平均后：只有 26/78 更好。
- **要回答的问题：**
    1. 这种"TTT 状态与测量驱动的重投影取平均"能否在另一个队列上稳定胜过 REPROJ7（V2 gate）？
    2. TTT 在其中是不是必要的，即把 TTT 换成只缓存是否明显更差？
- **背景：** 把学到的补全与调和插值取平均，已在 EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1 中通过 V2 gate。

## 2. 队列与输入

- **EVAL34 = EVAL-V3 + EVAL-V4：** 83 个场景、166 个 episode、332 个查询。
    - 流帧 = Core B 帧号规则（4 帧，两角色共享），必须与 Core B V8 封存的流角色逐一相同。
    - 这个队列以前被旧项目的 Core B 实验（旧载体）多次使用，但从没评估过 DepthSplat 的 CACHE、TTT 或它们与 REPROJ7 的平均。所以对这个在 FRESH78 上提出的假设是样本外的。
- **状态构建器：** 发布版 DepthSplat（ZS，尺度上限 0.1），注入测得深度。用 V1 锁定的 `ds_eval.py`：
    - CACHE：3 帧上下文加 4 帧流作为 7 视角输入；
    - TTT：在 CACHE 上做 100 步 Adam，m = 1，与 V1 完全相同。
- **无模型：** REPROJ7_HARMONIC、NEAR7 与 cos，由 V1 锁定的 `geometry_seal.py` 计算，并对 EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1 的 EVAL34 arms 封存复核 REPROJ_HARMONIC 与 NEAR。

## 3. 读出

- **R7：** REPROJ7_HARMONIC。
- **CACHE、TTT：** (depth_z / α) / cos，只在 α ≥ 0.5 处使用，其余用 R7。这就是 V1 的 `_F7` 读出。
- **R7_TTT_GEO：** √(R7 × TTT)。R7_TTT 为 (R7 + TTT) / 2。
- **R7_CACHE_GEO：** √(R7 × CACHE)。R7_CACHE 为 (R7 + CACHE) / 2。

指标为 AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 4. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| T1 | R7 − R7_TTT_GEO | ALL | TTT_ENSEMBLE_CONSISTENT（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| T2 | R7_CACHE_GEO − R7_TTT_GEO | ALL | TTT_NEEDED（SUPPORTED）/ TTT_NEEDED_PARTIAL / TTT_NOT_NEEDED / TTT_HARMFUL |

## 5. 次要分析（只给标签）

- R7 − TTT（V1 的 B2 在 EVAL34 上的复现）；
- CACHE − TTT（V1 的 B1 的复现）；
- R7 − R7_TTT（算术平均）；
- R7 − R7_CACHE_GEO；
- R7 − CACHE；
- T1 在 NEAR7 与 FAR7 上的版本；
- TTT − R7_TTT_GEO（平均相对 TTT 单独）。

## 6. 完整性与局限

- **完整性：**
    - 读真值之前，几何封存（复核通过）与 ZS 的 DepthSplat 封存必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 主检验选几何平均，是因为它在 FRESH78 的探索中更好。算术平均作为次要分析一并报告。
    - 假设是在 FRESH78 上事后形成的；EVAL34 本身以前被多次使用过。

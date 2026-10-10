# EXP-3D-RGBD-COMPLETION-SELECTOR-ARKIT-V1：用上下文深度范围决定何时使用学到的补全

本文件即预注册。分两步锁定：

1. 写定本文件后、下载任何新数据之前，先把哈希记入 `audit/plan_locked_at.txt`。
2. 新队列准备好、评估根目录建好之后，正式锁定（`preregistration.json`），然后才运行任何模型、读取任何真值。

## 1. 问题

此前的发现：

- 在真实数据（ARKit）有重叠的近处视角上，四种学到的补全都把 FAR 误差平均降低约 1/4，但只有约 54% 的场景变好。这四种是 Hypersim 训练的小载体 G1、ARKit 训练的小载体 GA，以及 ARKit 训练的 DepthSplat 逆向 JEPA（JA、JO）。
- EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 的事后探索（样本内，ARKIT160_OVERLAP）发现，有一个不需要真值的量能预测"学到的补全是否有用"：**上下文深度范围**。
    - 定义：S = (p90 − p10) / 中位数，在该 episode 3 个上下文帧的有效测得深度上合并计算。
    - S 与增益的 Spearman 相关，在两个来源队列、三种模型上都是 +0.42 到 +0.56。
    - 规则"S ≥ 1.2 时用学到的补全，否则用调和插值"（1.2 约为样本内的中位数），在三种模型上都让 86–92% 的场景不变差，并几乎保留了全部平均增益。

阈值是在样本内选出来的，所以本实验在全新的扫描上检验这条规则，规则与阈值事先固定。

## 2. 数据

- **新队列 ARKIT_SELECT**（EXP-3D-RGBD-ARKIT-SELECT-DATA-V1，`scripts/prepare_rgbd_arkit_select_cohort.py`）：
    - 取 ARKitScenes 训练折，按试点的种子顺序取第 520–679 位场馆，每个场馆取 video id 最小的扫描；
    - 准备规则与 CONFIRM 队列完全相同，按顺序准备，直到 80 个扫描合格。
- **不相交：**
    - 第 0–359 位留给试点，所有 ARKit 模型都只在其中训练；
    - 第 360–519 位是 CONFIRM 队列；
    - Validation 折是 VAL 队列。
    - 新队列与这三者都不相交，此前从未被任何实验使用。
- **episode：** 记录 × 角色（A、B）；上下文是该角色的 3 帧，无流式帧。
- **查询：** 第 6、7 帧；若记录已把其中一帧用作上下文帧或旧查询帧，就按 9、11、12 的顺序顺延。规则与 OVERLAP-V1 相同。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 模型（全部冻结，本实验不训练任何模型）

- **GA：** EXP-3D-RGBD-CARRIER-ARKIT-OVERLAP-V1 的载体，3 个种子，第 24000 步。
- **G1：** V21 的 G1，3 个种子，第 24000 步。
- 以上两者都用 arms 评估器在 CPU 上评估。
- **JO：** EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 的 JO1–JO3，用 V1 的 `ds_eval.py` 只跑 CTX，尺度上限 3.0，在 GPU 上评估。

## 4. 读出

- **H3：** REPROJ_HARMONIC。
- **GAN、G1N：** 每个种子在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余取 H3，再对种子平均。
- **JO：** 至少 2 个种子覆盖（α ≥ 0.5）处取覆盖种子的平均射线深度，其余取 H3。
- **HGAN：** (H3 + GAN) / 2。
- **选择器 SEL_X：** 若该 episode 的 S ≥ 1.2，取 X；否则取 H3。按 episode 决定，同一 episode 的两个查询相同。
- **分歧选择器 DSEL_GAN：** 若查询的 median |log(GAN / H3)|（NEAR 之外）≥ 0.18，取 GAN；否则取 H3。阈值约为样本内的中位数。
- **区域：** FARH，即至少有一个上下文命中的查询中、NEAR 之外的有效像素。
- **指标：** AbsRel。场景值 = 各角色的查询平均后，再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。平局（选择器取 H3 的场景）按 V2 gate 的规则计为"不变差"。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | H3 − SEL_GAN | FARH | SELECTOR_CONSISTENT（SUPPORTED）/ SELECTOR_PARTIAL / SELECTOR_NOT_ESTABLISHED / SELECTOR_HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 含义 |
| --- | --- | --- |
| S1 | H3 − SEL_G1N | 选择器用于 Hypersim 小载体 |
| S2 | H3 − SEL_JO | 选择器用于 DepthSplat 逆向 JEPA |
| S3 | H3 − GAN | 不选择时的复现 |
| S4 | H3 − G1N | 不选择时的复现 |
| S5 | H3 − JO | 不选择时的复现 |
| S6 | GAN − SEL_GAN | 选择是否损失平均增益 |
| S7 | H3 − HGAN | |
| S8 | H3 − DSEL_GAN | 分歧选择器 |

另外报告：被选中（S ≥ 1.2）的 episode 比例，以及 S 与增益的 Spearman 相关（描述性）。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，arms 封存（G1、GA）和 JO 的 DepthSplat 封存都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只有 80 个扫描。
    - 规则来自样本内探索；本实验是它第一次在全新数据上的检验。
    - 真值是 iPad LiDAR 的测量。

## 修订 1（2026-10-09 11:05:48 +0800 记录；只修复打分代码的一个错误）

- **错误：** 第一次运行 `sel_score.py score` 时，有些查询的 NEAR 覆盖了整张图，NEAR 之外没有像素。这时"分歧"（NEAR 之外 |log(GAN / H3)| 的中位数）是对空集求中位数，得到 NaN，写 `scores/query_rows.json` 时被 JSON 编码器拒绝，运行中止。
- **影响：** 这次运行读取了查询深度，但在写出第一个文件之前就中止了。`scores/` 是空的，`results/` 不存在，没有产生、也没有看到任何结果。
- **修复：** 没有 NEAR 之外像素的查询，分歧记为空值，分歧选择器对它取 H3。这类查询本来也没有 FARH 像素，不进入任何对比，所以所有读出、区域、对比与判定都不变。
- **存档：** 原文件保留为 `audit/sel_score.attempt1.py`，失败日志保留为 `audit/score.attempt1_nan.log`。

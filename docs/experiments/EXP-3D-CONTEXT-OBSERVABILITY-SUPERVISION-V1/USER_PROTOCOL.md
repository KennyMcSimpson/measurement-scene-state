你现在位于：

https://github.com/KennyMcSimpson/measurement-scene-state.git

请基于已经完成的：

EXP-3D-DIRECT-STATE-CAPACITY-V1
EXP-3D-16G-OPTIMIZATION-BOUNDS-V1

执行下一阶段：

EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1

============================================================
0. 本轮唯一研究目标
============================================================

当前最重要的已知事实：

1. 当前主要表示：
   RGB+D / 16^3 / renderer64 / frozen GT-free bounds。

2. context-direct 10k query AbsRel：
   0.361480

3. 同 representation / renderer / bounds 下，
   query-supervised diagnostic：
   0.148045

4. context-direct 与 query oracle gap：
   +0.213435
   95% CI [0.134787, 0.302639]

5. context 优化 1k→10k：
   context error 持续下降，
   query 没有稳定改善。

6. RGB-only 明显弱于 RGB+D。

因此本轮不再研究：

- 更多 optimizer steps
- 更多 renderer samples
- 更高 grid
- 新 bounds
- Dynamic TTT
- carrier training

本轮只回答：

Q1.
sealed query geometry 中，
哪些区域实际上能够由 context observations 几何约束？

Q2.
当前 context-direct 的主要错误来自：

- 已经充分观察但推断错误的区域；
还是
- 未观察 / 遮挡 / 弱约束区域？

Q3.
当前 renderer-level depth loss
是否没有充分约束内部 density geometry？

Q4.
显式 free-space 和 surface supervision
能否在不使用 query GT 的情况下，
缩小 context → query-oracle gap？

Q5.
下一版 learned carrier 最需要增加的是：

- visibility-aware lifting
- explicit surface/free-space supervision
- scene completion
- 还是其他 context inference mechanism？

============================================================
1. 本轮研究纪律
============================================================

禁止：

- new learned carrier training
- Dynamic TTT
- writer
- FC/CF
- controller
- trajectory search
- grid > 16^3
- renderer > 64 samples
- bounds search
- candidate allocation search
- context frame selection search
- final holdout
- 使用 query GT 优化 context-only state
- 根据 query result 调 loss weight
- 根据 query result 调 surface-band width
- 删除低可观测 query pixels
- 只报告 observable region 而隐藏 overall metric

本轮是：

OBSERVABILITY ATTRIBUTION
+
GEOMETRIC SUPERVISION ATTRIBUTION

不是新方法 benchmark。

============================================================
2. 数据角色
============================================================

继续使用上一轮同一批：

17 个 CAPACITY_DEV_EXPOSED scenes

只能作为 attribution/dev。

不得称为：

- independent validation
- final test
- final qualification

上一轮已经锁定的 FINAL_HOLDOUT：

继续完全禁止：

- model forward
- optimization
- observability scoring
- query evaluation

必须输出：

FINAL_HOLDOUT_TOUCHED=false

============================================================
3. 冻结主配置
============================================================

主 representation 完全冻结为：

GRID = 16^3
RENDER_SAMPLES = 64
BOUNDS = 上一轮 frozen GT-free bounds
OPTIMIZATION_BUDGET = 10000 steps

沿用上一轮：

- state semantics
- density logits
- sigmoid-bounded RGB
- normalized ray directions
- metric ray-distance depth
- RGB background
- optimizer
- LR
- clip
- regularization
- context/query frame IDs
- scene IDs

除“新增几何 supervision”外，
不得修改其他主因素。

============================================================
4. Baseline 必须精确复现
============================================================

先复现上一轮：

BASELINE_RGBD_RENDER_LOSS

预期：

Context AbsRel ≈ 0.083607
Query AbsRel ≈ 0.361480

以及：

query-oracle diagnostic ≈ 0.148045

允许正常数值误差，
但必须报告：

BASELINE_REPRO_STATUS=

PASS
FAIL

如果 FAIL：

停止新增 supervision formal run，
先调查复现。

============================================================
5. Phase A：Query Geometry Observability Audit
============================================================

这一阶段不改变任何模型或 state。

只分析：

“query surface 到底能否从 context 被看到？”

允许使用 query GT depth，
但仅在 state 完成 seal 后用于：

DIAGNOSTIC REGION LABELING

不能用于：

- optimization
- bounds
- loss
- checkpoint selection
- scene selection

============================================================
6. 从 query pixel 恢复真实 3D surface point
============================================================

对每个 query GT-valid pixel：

使用：

query camera
query GT ray-distance depth

恢复 world-space surface point：

X_q

保存：

scene
query id
pixel
world coordinate
GT depth

这一步只在 seal 之后。

============================================================
7. 将 query surface 投回每个 context view
============================================================

对每个 X_q：

投影到所有 context cameras。

每个 context view 记录：

- projected pixel
- in_front
- inside_image
- context_depth_valid
- projected ray distance
- context GT ray distance
- depth disagreement

使用与旧 common-visible 诊断一致的
visibility tolerance：

abs_tol = 0.05 m
relative_tol = 1%

一个 context view 只有满足：

inside image
+
in front
+
valid context depth
+
|d_projected - d_context_GT|
<= 0.05m + 0.01*d_context_GT

才标记：

VISIBLE_SUPPORT=true

不得仅因为投影落在图像内
就称作 visible。

============================================================
8. Query pixel observability 分类
============================================================

每个 query surface point 得到：

VISIBLE_VIEW_COUNT = 0,1,2,...

至少报告：

OBS0:
0 visible context views

OBS1:
exactly 1 visible view

OBS2PLUS:
>=2 visible views

不要只使用一个二元可见/不可见分类。

另外保存连续变量：

- visible view count
- max triangulation angle
- median triangulation angle
- camera baseline
- nearest context angle
- distance to context camera
- bounds inside flag

============================================================
9. Triangulation angle 分桶
============================================================

仅在 >=2 visible views 时计算
view-ray triangulation geometry。

提前固定描述性 bins：

0–5°
5–15°
15–30°
>30°

这些 bins 只作诊断。

不得看结果后改变 threshold
再重新定义 primary category。

============================================================
10. Occlusion / conflict 分类
============================================================

如果 X_q：

- 投影进 context image
- context depth valid
- 但 projected distance 明显大于 context depth

说明 query surface 在该 context 中
可能位于已观察 surface 后方。

标记：

OCCLUDED_BY_CONTEXT_SURFACE

如果 projected distance 明显小于
context depth：

标记：

DEPTH_CONFLICT_FRONT

不要把这两类直接混入
VISIBLE_SUPPORT。

============================================================
11. Observability 分区上的误差
============================================================

对以下 state：

A. context-direct RGB+D
B. query-supervised diagnostic oracle
C. frozen learned carrier
D. frozen anchor

分别计算：

overall AbsRel

OBS0 AbsRel
OBS1 AbsRel
OBS2PLUS AbsRel

以及各 triangulation-angle bin。

同时报告每个区域：

pixel fraction
scene coverage
empty-mask scene count

不允许：

只在有像素的 scenes 上重新平均
而静默删除 empty scenes。

============================================================
12. 最关键的 Observability Gap
============================================================

定义：

GAP_obs_k =
ContextDirect_AbsRel(obs_k)
-
QueryOracle_AbsRel(obs_k)

重点回答：

如果 OBS2PLUS 中 gap 仍然很大：

说明即使 geometry 被多视图真实观察，
当前 context inference 仍没有充分恢复。

如果 OBS2PLUS gap 很小，
但 OBS0 / OBS1 gap 很大：

说明主要瓶颈更像：

scene completion /
underconstrained geometry

而不是基本 multi-view inference。

============================================================
13. Observability 状态分类
============================================================

输出：

OBSERVABILITY_STATUS=

OBSERVED_REGION_FAILURE
UNOBSERVED_REGION_DOMINANT
MIXED
INCONCLUSIVE

建议解释：

OBSERVED_REGION_FAILURE：

>=2 visible views 区域
仍有显著 context→oracle gap。

UNOBSERVED_REGION_DOMINANT：

OBS2PLUS 接近 oracle，
主要误差集中在 OBS0/OBS1。

MIXED：

两者都明显。

============================================================
14. Phase B：Explicit Ray Geometry Supervision
============================================================

这一步仍然是：

direct-state optimization

不是 learned carrier。

保持：

- 16^3
- same bounds
- same renderer
- same contexts
- same 10k steps
- same optimizer
- same RGB loss
- same rendered-depth loss

只改变新增 geometry supervision。

============================================================
15. 四个正式 supervision variants
============================================================

必须运行：

S0 — RENDER_ONLY

当前 baseline：

RGB reconstruction
+
rendered depth AbsRel
+
旧 regularization

S1 — FREE_SPACE

S0
+
explicit free-space loss

S2 — SURFACE

S0
+
explicit surface-band mass loss

S3 — FREE_SPACE + SURFACE

S0
+
两个 geometry losses

先不要加 TSDF、normal、
point feature matching。

本轮只分析最基础的：

free space
+
surface location

============================================================
16. Surface band 定义
============================================================

不能手调固定米制 epsilon
追求最好结果。

使用由当前 16^3 voxel geometry
确定的固定规则。

对每个 scene：

计算 voxel physical spacing。

定义：

tau_surface =
0.5 × voxel_diagonal

如现有实现更适合：

1.0 × half voxel diagonal

必须在 formal 前冻结一种，
写入：

supervision_contract.json

不得按 scene/model score 调节。

============================================================
17. Free-space loss
============================================================

对于每条合法 context depth ray：

GT surface distance：

d*

renderer samples：

t_i

对于：

t_i < d* - tau_surface

明确视为：

FREE SPACE

惩罚这些 sample 的：

density / alpha / occupancy

推荐实现为：

L_free =
mean(alpha_i)

或等价稳定形式。

具体数学形式在 formal 前冻结。

不能监督：

t_i > d* + tau_surface

为空。

因为 surface 后方可能是：

unknown / occluded geometry。

============================================================
18. Surface loss
============================================================

对：

|t_i - d*| <= tau_surface

要求 ray 在 surface band 内
具有足够的 rendering mass。

推荐：

surface_mass =
sum_{surface band} w_i

L_surface =
-log(surface_mass + eps)

或数值等价的稳定形式。

不要直接强迫：

某个唯一 voxel density = 1

除非 state semantics 明确需要。

目的是：

让 density mass
集中在真实 surface 附近。

============================================================
19. 不要破坏 rendered-depth baseline
============================================================

S1/S2/S3 都继续保留：

原 RGB loss
原 rendered-depth loss

这是新增结构监督，
不是替代原任务。

这样才能判断：

显式内部 geometry constraint
是否提供额外价值。

============================================================
20. Loss scale 不能使用 query 调
============================================================

主 formal 使用预先冻结权重。

推荐先通过：

- loss normalization
- per-ray mean normalization

使：

L_free
L_surface

数量级合理。

允许极小 smoke
检查：

- 梯度非零
- 没有 NaN
- state 不立即 collapse

但不能：

看 query score
选择 λ。

如果必须选择 λ：

只能根据：

context-side loss scale
和数值稳定性

不能使用 sealed-query metric。

============================================================
21. Matched initialization
============================================================

对每个 scene/context：

S0/S1/S2/S3

使用：

- 相同 initial state
- 相同 random seed
- 相同 ray minibatch sequence
- 相同 optimizer
- 相同 steps

尽量做 matched-pair comparison。

============================================================
22. 主结果
============================================================

对每个 variant 报告：

Context AbsRel
Query AbsRel
Query RGB MSE
RMSE
δ1
Opacity
Coverage

并按 observability region 报告：

OBS0
OBS1
OBS2PLUS

特别关注：

S0 vs S1
S0 vs S2
S0 vs S3

的 query AbsRel paired gain。

============================================================
23. Geometry supervision gap closure
============================================================

沿用同一 query-oracle diagnostic reference。

定义：

BASE_GAP =
S0 query AbsRel
-
QueryOracle AbsRel

NEW_GAP =
Sx query AbsRel
-
QueryOracle AbsRel

GAP_CLOSED =
BASE_GAP - NEW_GAP

同时报告：

GAP_CLOSED_FRACTION =
GAP_CLOSED / BASE_GAP

这个比例只作描述性诊断。

Query oracle 不是严格 mathematical upper bound，
不能称为“恢复了 x% 的理论最优”。

============================================================
24. 关键判断：改善发生在哪些区域？
============================================================

如果 S3 主要改善：

OBS2PLUS

说明：

明确几何监督帮助
已观测区域的 state recovery。

如果主要改善：

OBS0

需要非常谨慎。

因为 context depth 没有直接观察这些表面，
可能是 regularization / prior / indirect consistency。

不能声称“从 observation 恢复了未见真值”
而没有机制证据。

============================================================
25. Free-space / surface attribution
============================================================

输出：

GEOMETRY_SUPERVISION_STATUS=

FREE_SPACE_DOMINANT
SURFACE_DOMINANT
COMPLEMENTARY
NO_CLEAR_GAIN
HARMFUL
INCONCLUSIVE

只有：

S1/S2/S3 的 paired comparison
和 scene distribution

支持时才能分类。

============================================================
26. Scene-specific information
============================================================

对 S0 和最佳预注册 supervision variant
都做：

wrong-scene replacement

保持 recipient query camera。

报告：

WRONG_SCENE_DAMAGE

如果 geometry supervision
降低 query error，
但 wrong-scene damage 接近0：

需要警惕结果是否主要来自
更强 global prior。

============================================================
27. Spatial structure diagnostic
============================================================

对最佳 supervision variant
继续做：

spatial shuffle
zero state

只作 diagnostic。

如果 shuffle 几乎不影响：

不能宣称学到了有意义
spatial scene state。

============================================================
28. Context generalization
============================================================

继续同时报告：

context error
query error

如果 S3：

context 更差一点
但 query 明显更好，

这可能是非常有价值的结果。

不要因为 context fit 下降
就自动判定方法更差。

真正目标是：

shared state 对 sealed query 的泛化。

============================================================
29. 不做 learned carrier training
============================================================

无论 S3 多好：

本轮都不训练新 carrier。

只输出：

GEOMETRY_AWARE_CARRIER_RECOMMENDED=
true / false

真正的 carrier 设计
下一轮单独进行。

============================================================
30. 是否需要 visibility-aware carrier
============================================================

根据 Phase A 给出：

VISIBILITY_AWARE_FUSION_RECOMMENDED=

true
false
inconclusive

建议：

如果 OBS2PLUS 明显优于 OBS0/1，
并且多视图 support 与 error 强相关，

未来 carrier 应考虑：

visibility-aware fusion /
view selection。

如果连 OBS2PLUS 都差：

仅增加 visibility gate
可能不够，
应优先提升 geometry inference。

============================================================
31. 下一版 carrier 决策矩阵
============================================================

CASE A

OBS2PLUS 已接近 oracle，
OBS0/1 很差。

→ completion 是核心问题。
→ 下一 carrier 强化
   geometry completion / learned prior。

CASE B

OBS2PLUS 也远离 oracle，
S3 显著改善。

→ explicit geometry supervision
   是主要下一步。
→ 训练 16^3 geometry-aware carrier。

CASE C

OBS2PLUS 差，
S3 也无明显改善。

→ 当前 density-state/context inference
   设计不足。
→ 考虑 MVS/cost-volume/
   visibility-aware lifting。

CASE D

只有 free-space 有效。

→ 主要问题是 density 漂到
   surface 前方。

CASE E

只有 surface 有效。

→ depth rendering loss
   没有把 mass 集中在真实表面。

CASE F

free + surface 联合明显最好。

→ 下一 learned carrier
   应显式加入 ray-geometry supervision。

============================================================
32. 不碰 Final Holdout
============================================================

本轮：

FINAL_HOLDOUT_TOUCHED=false

这些仍是 exposed attribution scenes。

未来真正 learned carrier
冻结后才允许打开 fresh holdout。

============================================================
33. 统计
============================================================

正式统计单位：

scene

10,000 paired scene bootstrap
seed = 20260927
95% percentile CI

所有正式 comparison 报告：

mean
median
CI
improved/tied/worse
LOSO
top1/top3 contribution

Observability pixel masks
不能被当成独立统计样本。

============================================================
34. Regression tests
============================================================

至少新增：

1. observability labels computed only after state seal.
2. query GT never enters context optimizer.
3. query GT never enters supervision loss.
4. visibility diagnostic cannot alter model/state.
5. free-space loss only supervises before surface band.
6. post-surface samples receive no free-space label.
7. surface loss only uses context GT depth.
8. surface-band rule deterministic from voxel spacing.
9. supervision weights frozen before formal query evaluation.
10. S0/S1/S2/S3 use matched initialization.
11. same ray minibatch stream across variants.
12. query metric cannot select lambda.
13. wrong-scene preserves recipient query camera.
14. final holdout loader disabled.
15. scene bootstrap resamples scenes.
16. raw→statistics deterministic.

运行完整 suite。

============================================================
35. 输出目录
============================================================

outputs/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1/

docs/experiments/EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1/

至少：

README.md
STATUS.md
preregistration.json
scene_manifest.json
observability_contract.json
supervision_contract.json
observability_analysis.json
supervision_analysis.json
gap_closure_analysis.json
wrong_scene_analysis.json
bootstrap_results.json
cost_analysis.json
integrity.json
tests.json
commands.sh

raw/
figures/
audit/

============================================================
36. 必须生成的图
============================================================

1.
query pixel fraction by observability class

2.
context-direct vs query-oracle AbsRel
by OBS0 / OBS1 / OBS2PLUS

3.
query AbsRel:
S0 / S1 / S2 / S3

4.
geometry supervision gain
by observability class

5.
per-scene S3−S0 query gain

6.
context error vs query error
for all supervision variants

7.
query-oracle gap before/after supervision

============================================================
37. README 必须回答
============================================================

1. query geometry 中多少是 OBS0 / OBS1 / OBS2PLUS？

2. 当前误差主要集中在哪种可观测性区域？

3. OBS2PLUS 上 context-direct
   与 query-oracle 还差多少？

4. triangulation angle 与 query error
   有什么描述性关系？

5. occluded regions 的 error 是否更高？

6. Free-space supervision 是否改善 query？

7. Surface supervision 是否改善 query？

8. 两者联合是否互补？

9. 最好的 geometry supervision
   缩小多少 context→oracle gap？

10. 改善主要来自 observed
    还是 unobserved regions？

11. explicit supervision 是否牺牲
    context fit 但改善 query？

12. wrong-scene damage 是否仍然成立？

13. 当前主要问题更像：
    - observability
    - geometry supervision
    - scene completion
    - lifting/fusion
    - representation
    - mixed

14. 下一版 carrier
    应该具体增加什么？

============================================================
38. 最终终端摘要
============================================================

打印：

============================================================
CONTEXT GEOMETRY OBSERVABILITY + SUPERVISION FINAL
============================================================

N_SCENES=

BASELINE_QUERY_ABSREL=
QUERY_ORACLE_ABSREL=
BASE_CONTEXT_ORACLE_GAP=

OBS0_FRACTION=
OBS1_FRACTION=
OBS2PLUS_FRACTION=

OBS0_BASE_ABSREL=
OBS1_BASE_ABSREL=
OBS2PLUS_BASE_ABSREL=

OBS0_ORACLE_ABSREL=
OBS1_ORACLE_ABSREL=
OBS2PLUS_ORACLE_ABSREL=

OBSERVABILITY_STATUS=

S0_RENDER_ONLY_ABSREL=
S1_FREE_SPACE_ABSREL=
S2_SURFACE_ABSREL=
S3_FREE_SURFACE_ABSREL=

BEST_SUPERVISION=
BEST_SUPERVISION_GAIN=
BEST_SUPERVISION_CI=

GAP_CLOSED=
GAP_CLOSED_FRACTION=

WRONG_SCENE_DAMAGE_BEST=

GEOMETRY_SUPERVISION_STATUS=
VISIBILITY_AWARE_FUSION_RECOMMENDED=
GEOMETRY_AWARE_CARRIER_RECOMMENDED=

NEXT_CARRIER_DIRECTION=

FINAL_HOLDOUT_TOUCHED=false
NEW_CARRIER_TRAINED=false
DYNAMIC_TTT_RUN=false

TESTS=
FINAL_INTEGRITY=

REPORT_DIR=

============================================================

============================================================
39. 最重要的科学纪律
============================================================

不要问：

“加哪个 loss 可以把分数调最好？”

而是问：

“context 为什么不能确定一个
对 unseen views 也正确的 shared state？”

必须区分：

A.
query 区域本来没被 context 看到

vs

B.
query 区域明明多视图可见，
但 geometry inference 仍失败

vs

C.
renderer depth loss 太弱，
允许错误 density field
同样拟合 context

vs

D.
显式 free-space/surface
可以解决大部分问题

vs

E.
即使显式 geometry supervision
也无法缩小 gap

如果是 E，
下一步不要继续堆 loss。

应转向：

visibility-aware multi-view lifting /
cost volume /
MVS-style geometry reasoning /
更强 carrier architecture。

不要为了得到 geometry-aware carrier
的正面结论修改 protocol。
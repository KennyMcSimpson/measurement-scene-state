你现在位于：

https://github.com/KennyMcSimpson/measurement-scene-state.git

请基于已完成实验：

EXP-3D-DIRECT-STATE-CAPACITY-V1

执行下一阶段：

EXP-3D-16G-OPTIMIZATION-BOUNDS-V1

============================================================
0. 本轮唯一目的
============================================================

上一轮已经得到：

1. RGB+D direct-state：
   8^3 query AbsRel = 0.460804
   16^3 = 0.402736
   32^3 = 0.389382

2. 8^3 → 16^3：
   gain = +0.058068
   95% CI [0.028670, 0.091763]
   13 improve / 1 tie / 3 worse

3. 16^3 → 32^3：
   gain = +0.013353
   CI [-0.004395, 0.032370]
   尚未建立稳定收益。

4. Renderer 64 → 128 samples
   几乎没有收益。

5. 425 / 425 direct states
   都选择了最大预算 step=1000，
   且最后三次 objective 仍下降。

6. 8^3 bounds sensitivity：
   CURRENT RGB+D = 0.460804
   GT-free alternative bounds = 0.398010

7. Query-supervised oracle 也对 bounds 敏感。

因此本轮只回答两个问题：

Q1.
16^3 direct-state 在优化更充分以后，
真正可达的 context→query 几何性能是多少？

Q2.
使用一个冻结、部署可合法构造、
不读取当前 scene GT/query GT 的 bounds 规则，
是否能稳定优于 CURRENT bounds？

本轮结束后才决定：

是否值得训练新的 16^3 learned carrier。

============================================================
1. 本轮禁止事项
============================================================

严禁：

- 新 carrier training
- Dynamic TTT
- writer
- FC/CF
- controller
- history
- trajectory search
- 32^3 carrier
- 更高 renderer samples sweep
- support redesign
- candidate allocation redesign
- context selection redesign
- final holdout
- 根据 query 分数选择 optimizer step
- 根据 query 结果调整 bounds
- 使用 current-scene GT depth 构造 deployable bounds
- 使用 query camera 构造 pre-seal bounds
- 看结果后增加一个新的优化 budget

本轮不是做新方法。

本轮是：

OPTIMIZATION SUFFICIENCY
+
GT-FREE BOUNDS ATTRIBUTION

============================================================
2. 数据角色
============================================================

沿用上一轮 17 个：

CAPACITY_DEV_EXPOSED

场景。

这些场景已经曝光，
只能用于 attribution。

不得称为：

- validation
- independent test
- final qualification

上一轮锁定的 final holdout：

继续完全禁止模型 forward / direct optimization / query score。

输出：

FINAL_HOLDOUT_TOUCHED=false

============================================================
3. 创建独立实验目录
============================================================

创建：

outputs/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/

docs/experiments/EXP-3D-16G-OPTIMIZATION-BOUNDS-V1/

至少包含：

README.md
STATUS.md
preregistration.json
config.json
scene_manifest.json
optimization_budget_lock.json
bounds_contract.json
bounds_lock.json
optimization_results.json
optimization_sufficiency.json
bounds_analysis.json
query_oracle_diagnostic.json
bootstrap_results.json
cost_analysis.json
integrity.json
tests.json
commands.sh
git_commit.txt
dirty.patch

raw/
figures/
audit/

============================================================
4. 冻结主 representation
============================================================

本轮主 representation：

GRID = 16^3

Renderer：

64 samples

保持与上一轮完全一致：

- density logits
- sigmoid-bounded RGB
- fixed renderer
- normalized ray direction
- metric ray-distance depth
- unnormalized sum(w*t)
- RGB background=0
- state semantics
- context/query IDs
- scene set

不要重新定义 state。

8^3 只保留上一轮结果作 reference，
本轮不重新大规模 sweep。

32^3 不运行。

============================================================
5. 主轨只做 RGB+D
============================================================

本轮主要科学轨：

RGB+D context-supervised direct state

允许：

- context RGB
- context depth
- context camera

禁止：

- query RGB
- query depth
- query camera before seal
- future/query-derived geometry

RGB-only 不再做完整 budget sweep。

如果需要，
只在最后选定的 16^3 配置上
做一个 secondary sanity，
但不能影响主决策。

============================================================
6. 优化充分性：预注册固定 budget sweep
============================================================

上一轮 1000 steps 未收敛。

本轮预先锁定：

STEPS =
1000
3000
10000
30000

如果预计 30000 成本过高，
可以在正式运行前，
基于上一轮单 state wall-clock
预先改成：

1000
3000
10000

但一旦 formal 开始，
不得看 query 结果后增加 budget。

对所有 scenes / A/B contexts
使用完全相同的 budget list。

============================================================
7. optimizer 必须冻结
============================================================

沿用上一轮：

Adam
LR = 0.03
clip = 1

以及相同：

- RGB loss
- valid-depth AbsRel
- density regularization
- physical-spacing TV

不要同时调：

LR
loss weights
regularization
optimizer

本轮只改变：

optimization budget

和：

bounds rule

否则无法归因。

============================================================
8. checkpoint selection 规则
============================================================

不能使用 sealed-query performance early stop。

对于每个 state：

保存固定 steps：

0
100
300
1000
3000
10000
30000

按实际锁定 budget 调整。

主分析同时报告两种规则：

A. FIXED-BUDGET

直接使用每个预注册 budget 的最终 state。

B. CONTEXT-OBJECTIVE SELECTED

在不超过该 budget 的 checkpoint 中，
只根据 context full objective
选择最佳 checkpoint。

绝不能使用 query score。

主 qualification 优先使用：

FIXED-BUDGET

context-selected 作为辅助诊断。

============================================================
9. 必须检查真正的优化 plateau
============================================================

不能只看：

最后三次 objective 是否下降。

对每个 state / budget 记录：

- total context objective
- RGB component
- depth component
- regularization
- gradient norm
- density norm
- color norm
- opacity
- context AbsRel
- context RGB MSE

定义预注册 plateau diagnostic：

例如最后 10% checkpoints 中：

relative objective improvement < threshold

并且：

gradient norm 不再系统下降

具体 threshold 必须在 formal 开始前写入：

optimization_budget_lock.json

它只用于判断优化充分性，
不得自动提前读取 query。

============================================================
10. 训练预算比较必须看 query
============================================================

所有 states 在该 budget 完成并 seal 后，
才允许 evaluator 读取 query。

对每个 budget 报告：

Context AbsRel

Query AbsRel

Generalization gap

Wrong-scene damage

Context RGB MSE

Query RGB MSE

Opacity

Coverage

主要曲线：

optimization steps
→
context AbsRel

optimization steps
→
query AbsRel

optimization steps
→
generalization gap

============================================================
11. 检查“继续优化 context 是否反而伤害 query”
============================================================

这是本轮很关键的分析。

对每个 scene/state：

计算：

ΔContext(step_i → step_j)

ΔQuery(step_i → step_j)

如果：

context objective 持续变好

但：

query 先改善后恶化

则说明：

更多优化可能增加 context overfitting，
不能简单认为“1000 steps没收敛，所以继续跑一定更好”。

统计：

- context improves & query improves
- context improves & query worsens
- both flat

报告 scene count。

============================================================
12. GT-free bounds：只允许两个主配置
============================================================

比较：

B0 = CURRENT_BOUNDS

B1 = FROZEN_GT_FREE_BOUNDS

不要增加第三、第四种 bounds。

============================================================
13. CURRENT_BOUNDS
============================================================

保持上一轮：

[-6,-4,-6]
to
[6,4,6]

或源码对应的完全相同定义。

============================================================
14. FROZEN_GT_FREE_BOUNDS
============================================================

沿用上一轮已经预先定义过的 GT-free 思路：

只使用：

- context camera rays
- training-set global depth prior

global depth prior：

只能来自原 TRAIN scenes。

例如已经存在：

near = 0.214252 m
far = 5.672045 m

或当前冻结 artifact 中对应的精确值。

使用训练集：

q01 / q99

深度 prior。

当前 scene 测试时：

只根据：

context camera rays
+
冻结 near/far prior

构造 AABB。

额外规则：

- 两侧扩 5%
- 最小边长 1m

必须与上一轮 alternative bounds
实现保持一致，
除非发现代码 bug。

不能读取：

- current scene depth
- query camera
- query GT
- model prediction
- model score

============================================================
15. bounds contract 必须程序化审计
============================================================

生成：

bounds_contract.json

包含：

inputs_allowed:
  context_intrinsics
  context_extrinsics
  frozen_training_depth_prior

inputs_forbidden:
  context_depth_current_scene
  query_camera
  query_depth
  query_rgb
  carrier_prediction
  query_score

为每个 scene 保存：

bounds_hash
camera_hash
prior_hash

============================================================
16. 2 × K 主实验矩阵
============================================================

正式矩阵：

CURRENT_BOUNDS × budgets

GT_FREE_BOUNDS × budgets

即例如：

CURRENT × 1k
CURRENT × 3k
CURRENT × 10k
CURRENT × 30k

GT_FREE × 1k
GT_FREE × 3k
GT_FREE × 10k
GT_FREE × 30k

所有设置：

- 16^3
- same renderer
- same state parameterization
- same context
- same query
- same optimizer
- same loss

============================================================
17. Query-supervised oracle 只做少量诊断
============================================================

为了区分：

context inference
vs
optimization sufficiency

允许运行：

QUERY_ORACLE_16

但仅在：

CURRENT_BOUNDS

GT_FREE_BOUNDS

两个 bounds 下。

同样使用预注册 budget sweep。

oracle：

- 可以使用两个 query GT
- 两 query 必须共享同一 state
- 明确标记 DIAGNOSTIC ONLY

不能与 context-direct
放在 deployable 方法排名里。

============================================================
18. Oracle 的作用
============================================================

主要回答：

如果 query-supervised oracle
随着 1k→10k/30k
明显改善：

说明上一轮 oracle 也受到有限优化影响。

如果 oracle plateau 很早，
但 context-direct gap 仍大：

context inverse problem /
visibility /
evidence sufficiency
嫌疑更大。

如果 context-direct 与 oracle
随着 budget 增大一起接近：

优化不足是主要贡献因素之一。

============================================================
19. 主要统计
============================================================

正式统计单位：

scene

使用：

10,000 paired scene bootstrap
seed 20260927
95% percentile CI

对于：

step_i → step_j

报告：

mean query gain
CI
median
improved/tied/worse
LOSO
top1/top3 contribution

不能把 A/B 当独立 scene。

============================================================
20. 优化充分性判定
============================================================

输出：

OPTIMIZATION_STATUS=

UNDEROPTIMIZED
PARTIALLY_SATURATED
SATURATED
QUERY_OVERFIT
INCONCLUSIVE

建议：

UNDEROPTIMIZED：

更大 budget
持续稳定降低 query error，
且最大 budget 仍无 plateau。

PARTIALLY_SATURATED：

前期有稳定 query gain，
后期 gain 已很小或CI跨0。

SATURATED：

更大 budget 对 context/query
均无实际改善，
objective也 plateau。

QUERY_OVERFIT：

context 持续改善，
但 query 随后系统变差。

============================================================
21. bounds 判定
============================================================

输出：

BOUNDS_STATUS=

GT_FREE_BETTER
CURRENT_BETTER
NO_CLEAR_DIFFERENCE
INTERACTION_WITH_OPTIMIZATION

主判断不能只看 1k。

需要检查：

bounds × budget interaction。

例如：

GT-free 1k 很好，
但 10k 后 current 追上，

就不能简单说 bounds 是主要瓶颈。

============================================================
22. 预注册 practical threshold
============================================================

延续上一轮：

stable practical gain：

mean gain >= 0.02

paired CI lower bound > 0

>=75% scenes 不变差

engineering-good：

mean AbsRel <= 0.25

并且：

>=75% scenes AbsRel <= 0.35

这些只是工程门槛，
不是官方 benchmark。

本轮不能事后修改。

============================================================
23. 重新评估 16^3 是否值得训练 carrier
============================================================

定义：

CARRIER_TRAINING_READINESS=

READY
PROMISING_BUT_NOT_READY
NOT_READY

READY 至少需要：

1. 16^3 context-direct 在最大充分 budget
   的 query 表现稳定；

2. optimization 已达到
   SATURATED 或 PARTIALLY_SATURATED；

3. GT-free bounds 规则已经冻结；

4. full context query 几何
   明显优于 frozen carrier；

5. wrong-scene damage > 0
   且 CI lower > 0；

6. 结果不是单scene贡献；

7. context→query gap
   不再大到无法解释。

不要求 engineering-good
必须完全通过，
但如果没通过：

只能 READY_WITH_LIMITATION，
README 必须明确。

============================================================
24. 如果 GT-free 16^3 最终仍很差
============================================================

不要训练 carrier。

进一步判断：

如果 query oracle 很好，
context-direct 差：

CONTEXT_INFERENCE_BOTTLENECK=true

下一轮研究：

lifting / geometry inference /
visibility / context supervision

如果 query oracle 也差：

REPRESENTATION_OR_BOUNDS_BOTTLENECK=true

下一轮才考虑：

state schema / resolution / bounds / renderer redesign

============================================================
25. 如果 16^3 context-direct 已经很好
============================================================

也不要在本轮训练 carrier。

只输出：

MATCHED_CARRIER_TRAINING_RECOMMENDED=true

下一轮单独建立：

EXP-3D-16G-MATCHED-CARRIER-TRAINING-V1

这样 attribution 和 training
不会污染在同一轮里。

============================================================
26. RGB-only secondary check
============================================================

只有主 RGB+D 分析完成后，
允许在最终选定的：

BOUNDS
+
BUDGET

上重新跑一次：

RGB_ONLY 16^3

只作为 secondary diagnostic。

禁止为 RGB-only 单独调：

LR
budget
bounds
loss

报告：

RGBD vs RGB-only

差异。

如果 RGB-only 仍然失败：

不能用 RGB+D 成功
宣称 RGB-only geometry adaptation 已解决。

============================================================
27. 计算成本
============================================================

对每个 budget 报告：

seconds/state
total optimization seconds
peak allocated VRAM
peak reserved VRAM
render latency

估算未来 carrier training
不属于本轮实测，
必须明确标记 estimate。

============================================================
28. Final holdout 继续禁止
============================================================

无论本轮结果多好：

FINAL_HOLDOUT_TOUCHED=false

因为本轮仍然是：

exposed attribution.

真正 holdout 只能给未来
冻结的 learned carrier 使用。

============================================================
29. Regression tests
============================================================

至少新增：

1. query GT cannot affect context-only optimization.
2. query camera unavailable before state seal.
3. current-scene depth cannot enter GT-free bounds.
4. query camera cannot enter GT-free bounds.
5. bounds deterministic from same context cameras/prior.
6. budget list frozen before formal run.
7. query metric cannot select optimization checkpoint.
8. optimizer/loss identical across bounds.
9. state/query IDs identical across budgets.
10. query-oracle marked privileged diagnostic.
11. final holdout loader disabled.
12. scene bootstrap resamples scenes only.
13. raw→statistics deterministic.
14. 16^3 state semantics identical to previous experiment.
15. renderer remains 64 samples.

运行完整 suite。

============================================================
30. 图表
============================================================

至少生成：

1.
steps vs context AbsRel

2.
steps vs query AbsRel

3.
steps vs generalization gap

4.
CURRENT vs GT-free bounds
across budgets

5.
context-direct vs query-oracle
across budgets

6.
per-scene 1k→max-budget query gain

7.
compute vs query quality

============================================================
31. README 必须回答
============================================================

1. 1000 steps 到底是不是明显 underoptimized？

2. 3k / 10k / 30k
   分别带来多少 query gain？

3. context objective 继续下降时，
   query 是否也继续改善？

4. 是否出现 context overfit？

5. 16^3 最大充分 budget
   最终 query AbsRel 是多少？

6. GT-free bounds 是否稳定优于 current？

7. bounds 收益是否依赖 optimization budget？

8. Query oracle 在更多优化后能到多少？

9. Context-direct 与 query-oracle gap
   是否缩小？

10. 主要剩余瓶颈是：
    optimization
    bounds
    context inference
    representation
    还是混合？

11. 现在是否真的值得训练新的16^3 carrier？

12. 下一轮应该训练 carrier，
    还是先修 context inference？

============================================================
32. 最终终端摘要
============================================================

打印：

============================================================
16G OPTIMIZATION SUFFICIENCY + GT-FREE BOUNDS FINAL
============================================================

N_SCENES=

GRID=16
RENDER_SAMPLES=64

CURRENT_1K_ABSREL=
CURRENT_3K_ABSREL=
CURRENT_10K_ABSREL=
CURRENT_30K_ABSREL=

GTFREE_1K_ABSREL=
GTFREE_3K_ABSREL=
GTFREE_10K_ABSREL=
GTFREE_30K_ABSREL=

BEST_CONTEXT_ONLY_CONFIG=
BEST_CONTEXT_ONLY_ABSREL=

QUERY_GAIN_1K_TO_MAX=
QUERY_GAIN_CI=

CONTEXT_GAIN_1K_TO_MAX=
GENERALIZATION_GAP_MAX=

QUERY_ORACLE_CURRENT_MAX=
QUERY_ORACLE_GTFREE_MAX=
CONTEXT_TO_ORACLE_GAP=

OPTIMIZATION_STATUS=
BOUNDS_STATUS=

ENGINEERING_GOOD=
CARRIER_TRAINING_READINESS=
MATCHED_CARRIER_TRAINING_RECOMMENDED=

RGB_ONLY_SECONDARY_ABSREL=

FINAL_HOLDOUT_TOUCHED=false
NEW_CARRIER_TRAINED=false
DYNAMIC_TTT_RUN=false

TESTS=
FINAL_INTEGRITY=

REPORT_DIR=

============================================================

============================================================
33. 最重要的科学纪律
============================================================

本轮真正要区分：

CASE A

更多优化显著继续改善query，
最大budget仍未饱和。

→ optimization insufficiency
  是重要因素。

CASE B

context继续改善，
query开始恶化。

→ direct-state正在context overfit；
  单纯增加训练步数不是答案。

CASE C

GT-free bounds在所有充分budget上
稳定更好。

→ bounds design 是真实可部署因素。

CASE D

query oracle很好，
context-direct仍明显差。

→ state表达能力存在，
context inference / visibility / supervision
是主要问题。

CASE E

query oracle和context-direct
都在充分优化后仍然很差。

→ representation/bounds本身需要redesign。

CASE F

16^3 + frozen GT-free bounds +
充分优化后表现稳定，
并且scene-specific state证据成立。

→ 才建议下一轮做
matched learned-carrier training。

不要为了进入 CASE F
改变 optimizer、loss、bounds 或工程门槛。
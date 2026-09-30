你现在位于：

https://github.com/KennyMcSimpson/measurement-scene-state.git

请基于已经完成的：

EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1
EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1

执行下一阶段：

EXP-3D-DIRECT-STATE-CAPACITY-V1

============================================================
0. 本轮唯一研究问题
============================================================

当前 frozen B-final carrier 在 unseen/dev 场景上的 shared-state
静态资格没有建立。

上一轮已经证明：

1. volume 会显著改变结果；
2. oracle volume 的正 relative gain 大部分来自 anchor 恶化；
3. oracle candidate support 单独增加并没有改善任务；
4. oracle volume + support 也没有通过预注册稳定性门槛；
5. support / spatial discretization / renderer prior / training distribution
   仍然耦合；
6. 当前还不能把失败归因于 support，
   也不能直接归因于 representation/training。

因此本轮不再改 support 方法。

本轮只回答：

Q1.
当前 typed scene state + fixed renderer 本身有没有足够的表示能力？

Q2.
如果完全绕过 encoder / lifting / fusion / completion，
直接优化 scene state，
能否从 context observations 得到一个在 sealed query 上表现良好的状态？

Q3.
当前 8^3 resolution 是否构成主要容量瓶颈？

Q4.
如果 direct-state 很好但 carrier 很差，
问题是否主要来自 state construction / training？

Q5.
如果 direct-state 自己也很差，
是否应该停止重训当前 carrier，
转而 redesign scene representation / renderer？

============================================================
1. 研究纪律
============================================================

本轮禁止：

- Dynamic TTT
- FC/CF
- controller
- writer
- fast-weight updates
- action policy
- trajectory search
- support redesign
- candidate allocation redesign
- context selector redesign
- new carrier training
- matched retraining
- train-scale-up
- final protected holdout
- 根据 query score 改 scene/state resolution
- 使用 query GT 优化 scene state
- 用 test query 选择 optimizer step

本轮不是为了得到正结果。

本轮是 representation-capacity attribution。

============================================================
2. 数据角色
============================================================

旧 7 个 exposed scenes 和上一轮 10 个 new DEV scenes
都已经被观察过，因此本轮统一作为：

CAPACITY_DEV_EXPOSED

只用于 attribution。

不得把它们称为独立验证。

上一轮锁定的 FINAL_HOLDOUT：

继续完全封存。

不得：

- forward
- render
- query score
- direct-state optimization
- hyperparameter selection

本轮不打开 final holdout。

============================================================
3. 创建独立实验目录
============================================================

创建：

outputs/EXP-3D-DIRECT-STATE-CAPACITY-V1/

docs/experiments/EXP-3D-DIRECT-STATE-CAPACITY-V1/

至少包含：

README.md
STATUS.md
preregistration.json
config.json
scene_manifest.json
capacity_contract.json
optimization_contract.json
renderer_contract.json
resolution_contract.json
direct_state_results.json
capacity_analysis.json
resolution_analysis.json
optimization_curves.json
bootstrap_results.json
integrity.json
tests.json
commands.sh
git_commit.txt
dirty.patch

raw/
figures/
audit/
checkpoints/

不得覆盖旧实验。

============================================================
4. 冻结旧 carrier baseline
============================================================

首先复现当前 frozen carrier：

1000A+600B B-final

checkpoint SHA256：

be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42

对 CAPACITY_DEV_EXPOSED 场景重新生成：

- A
- B
- anchor
- prior

旧统计必须精确或在数值容差内复现。

如果不能：

停止正式 capacity 实验，
先调查 reproducibility。

============================================================
5. 定义 Direct State Optimization
============================================================

核心要求：

完全绕过：

- image encoder learning
- lifting network
- fuse network
- completion network
- fast-weight writer

直接把 shared scene state 本身作为可优化变量。

保持 fixed renderer 不变。

对于每个 scene / context：

优化：

density logits z(x)
RGB color c(x)

以及只有在当前 typed state 中已经正式存在、
且 fixed renderer 必需的最小辅助字段。

不要偷偷加入新的 query-conditioned network。

Direct state 必须仍然满足：

同一个 optimized state
→ fixed renderer
→ 多个不同 sealed query

不能为每个 query 单独优化 state。

============================================================
6. Query leakage 禁止
============================================================

Direct-state optimization 只能读取：

- context RGB
- context camera
- 如果本轨允许：context depth

不能读取：

- sealed query RGB
- sealed query depth
- sealed query score
- sealed query camera，除非 renderer evaluation 阶段
- future/query visibility mask
- query GT geometry
- query-derived bounds

query camera 和 GT 必须在 state optimization 完成并 seal 后
才能进入 evaluator。

增加强制 loader 隔离。

============================================================
7. 两条 capacity 轨道
============================================================

必须分开运行：

----------------------------------------
TRACK A — RGB+D DIRECT STATE
----------------------------------------

允许 context：

- RGB
- depth
- camera

直接优化 state。

这是较强的 representation-capacity diagnostic。

它回答：

“如果 context geometry 已知，
当前 state + renderer 能否表示一个能泛化到 query 的 scene？”

不能包装成 RGB-only 方法。

----------------------------------------
TRACK B — RGB-ONLY DIRECT STATE
----------------------------------------

只允许：

- context RGB
- camera

通过 differentiable fixed renderer
做 inverse rendering。

不使用 context depth。

这条轨用于检查：

仅 RGB supervision 下，
representation + renderer 是否仍有可达能力。

TRACK A / B 必须分别报告。

============================================================
8. 优化目标
============================================================

TRACK A 推荐：

L =
λ_rgb L_rgb(context render, context RGB)
+
λ_depth L_depth(context render, context depth)
+
预先固定的最小正则项

TRACK B：

L =
L_rgb(context render, context RGB)
+
预先固定的最小正则项

正则只能用于：

- density stability
- bounded color
- mild spatial smoothness

不能加入 query-derived regularizer。

所有 λ 必须：

- 在正式 query evaluation 前冻结
- 只通过 training-side / exposed capacity-dev
  的 context reconstruction 行为选择
- 不能使用 sealed query score 调

============================================================
9. 先做 Optimization Sanity
============================================================

在跑 resolution sweep 前，
先确认 direct-state optimizer 真正在工作。

至少检查：

- context RGB loss 是否下降
- TRACK A context depth loss 是否下降
- state parameters 是否有梯度
- renderer 输出是否变化
- 没有 NaN/inf
- density 没有立即 collapse
- opacity 没有全部变成0或1

保存：

optimization_trace.csv

包含：

step
loss_rgb
loss_depth
regularization
opacity
grad_norm
state_norm
context metric

但不能在 query 上 early-stop。

============================================================
10. 优化停止规则
============================================================

必须预先冻结：

- optimizer
- LR
- max steps
- checkpoint intervals
- selection rule

推荐：

Adam

先用一个小 pilot 确定稳定 LR，
然后整个正式 capacity study 使用同一规则。

禁止：

为不同 scene 单独调 LR。

正式 checkpoint selection
只能根据 context-side objective。

例如：

best context objective checkpoint

或者：

fixed final step

二选一，预先冻结。

不能根据 sealed query 表现选择。

============================================================
11. 第一核心实验：8^3 Direct State
============================================================

首先完全保持当前 state resolution：

8^3

以及：

- current fixed renderer
- current ray semantics
- current query protocol
- current scene bounds protocol

但要注意：

上一轮 oracle volume 表明 bounds 会影响 renderer/prior。

因此本轮必须至少分开：

A. CURRENT_BOUNDS
B. PREDECLARED_GEOMETRY_BOUNDS

其中 PREDECLARED_GEOMETRY_BOUNDS
必须是 GT-free 或 training-side frozen 规则。

不能使用 sealed query GT bbox。

如果无法定义公平 GT-free alternative：

主 capacity result 使用 CURRENT_BOUNDS，
bounds sensitivity 单独做 diagnostic。

============================================================
12. Direct-state vs carrier
============================================================

对每个 scene/context：

比较：

CARRIER_FIXED

vs

DIRECT_STATE_RGBD

vs

DIRECT_STATE_RGB_ONLY

所有方法使用：

- 相同 context
- 相同 query
- 相同 fixed renderer
- 相同 state tensor semantic
- 相同 resolution

报告：

Depth AbsRel
RMSE
δ1
RGB MSE
SSIM
Opacity
Coverage

主比较：

DIRECT_STATE_RGBD − CARRIER

DIRECT_STATE_RGB_ONLY − CARRIER

误差指标中负值 = direct-state 更好。

============================================================
13. 最重要的 capacity gap
============================================================

定义：

CAPACITY_GAP =
Carrier query AbsRel
-
DirectState query AbsRel

正值越大：

说明表示本身能做得比 carrier 好，
瓶颈更可能在 state construction / training。

同时定义：

ANCHOR_GAP =
Anchor query AbsRel
-
DirectState full-context query AbsRel

如果 direct-state full context
仍然不能稳定优于 anchor：

representation/renderer capacity 的嫌疑增大。

============================================================
14. Resolution Sweep
============================================================

只有 8^3 direct-state pipeline 通过 sanity 后，
运行：

8^3
16^3
32^3

如果 32^3 资源过高：

允许：

8^3
12^3
16^3

但必须在开始正式 sweep 前锁定。

不要看结果后增加一个刚好最优的分辨率。

对每个 resolution：

- direct state independent
- same contexts
- same queries
- same loss
- same optimizer family
- same evaluation
- same scene set

记录：

state parameter count
VRAM
optimization time
renderer time

============================================================
15. Resolution 不能偷偷增加其他能力
============================================================

改变 resolution 时不能同时：

- 改 renderer semantics
- 改 loss
- 改 bounds rule
- 改 context selection
- 改 optimization objective
- 改 scene list

如果更高 resolution 需要更多 renderer samples：

必须单独记录。

最好先固定 renderer sample count，
另做 renderer-resolution diagnostic。

============================================================
16. Renderer Capacity Control
============================================================

为了区分：

state resolution
vs
ray integration resolution

增加一个次要二维矩阵：

STATE GRID:
8^3 / 16^3

RENDER SAMPLES:
64 / 128

如果资源允许。

但优先级：

state resolution > renderer samples

输出：

renderer_capacity_analysis.json

如果增加 renderer samples
几乎不改善 query：

说明 ray discretization 不是主要限制。

============================================================
17. Context Memorization Control
============================================================

Direct-state optimization 很容易记住 context。

因此必须同时报告：

CONTEXT SCORE

和

SEALED QUERY SCORE

定义：

GENERALIZATION_GAP =
query error - context error

如果：

context 很好
query 很差

说明 representation 只是记住 context，
不代表形成可复用 shared state。

============================================================
18. Wrong-scene direct-state control
============================================================

对 optimized direct state 继续做：

wrong-scene replacement

保持 query camera 不变。

如果一个 scene 的 optimized state
换成另一个 scene 后 query performance
几乎不变：

说明 direct-state 也可能主要输出 prior。

报告：

WRONG_SCENE_DAMAGE_DIRECT

============================================================
19. State interpolation / shuffle diagnostic
============================================================

只作为 diagnostic：

A. spatial shuffle
B. density-only
C. color-only
D. zero-density / zero-color
E. wrong-scene state

检查 fixed renderer output
是否依赖 scene-specific state structure。

不能把 latent difference 本身当结果。

必须评价 query prediction。

============================================================
20. Direct-state oracle variant
============================================================

允许一个明确标记的：

QUERY-SUPERVISED ORACLE STATE

但它只能作为 renderer/state expressivity
的极限诊断。

它允许直接优化 query GT。

目的：

如果连 query-supervised direct state
都做不好：

说明 state + renderer expressivity 本身非常有限。

如果 query-supervised 很好，
而 context-trained direct-state 很差：

说明主要问题更可能是 inverse problem /
context information / training objective。

必须放在：

DIAGNOSTIC ORACLE

区块。

绝不能与 deployable或context-only结果混排。

============================================================
21. 三层 capacity decomposition
============================================================

最终必须把结果拆成：

LEVEL 1:
QUERY-SUPERVISED DIRECT STATE

回答：

state + renderer 最大表达能力怎样？

LEVEL 2:
CONTEXT-SUPERVISED DIRECT STATE

回答：

给真实 context evidence，
直接优化 state 能泛化到 query 多好？

LEVEL 3:
LEARNED CARRIER

回答：

网络能否从 context 一次/有限步骤生成好 state？

于是：

L1 → L2 gap

主要反映：

context inverse problem /
visibility /
optimization

L2 → L3 gap

主要反映：

learned state construction /
training

不要声称这是严格因果分解，
但作为 attribution framework 使用。

============================================================
22. Resolution capacity 判定
============================================================

输出：

STATE_CAPACITY_STATUS=

SUFFICIENT
RESOLUTION_LIMITED
RENDERER_LIMITED
CONTEXT_INFERENCE_LIMITED
LEARNER_LIMITED
MIXED
INCONCLUSIVE

建议解释：

SUFFICIENT:
8^3 context-direct-state 已稳定优于 carrier/anchor，
query表现合理。

RESOLUTION_LIMITED:
8^3 direct-state差，
更高grid显著稳定改善。

RENDERER_LIMITED:
state grid增加作用小，
renderer samples增加明显改善。

CONTEXT_INFERENCE_LIMITED:
query-supervised state很好，
context-direct-state明显差。

LEARNER_LIMITED:
context-direct-state很好，
carrier明显差。

MIXED:
多个因素均有明显gap。

============================================================
23. 统计
============================================================

正式统计单位继续：

scene

使用：

10,000 paired scene bootstrap
seed 20260927
95% percentile CI

报告：

mean
median
LOSO
improved/tied/worse
top1 contribution
top3 contribution

不要把每个 query 当独立样本。

============================================================
24. 不要碰 final holdout
============================================================

即使某个 direct-state resolution
结果很好：

也不要打开上一轮 final holdout。

因为本轮只是 attribution，
不是方法 qualification。

final holdout 保留给未来：

真正冻结的 redesigned carrier。

============================================================
25. 下一阶段 gate
============================================================

根据结果自动选择下一步。

CASE A
Query-supervised direct-state 也很差。

→ STOP current representation.
→ 下一轮 redesign state/renderer。

CASE B
Query-supervised 好，
context-direct-state 差。

→ representation能表达，
但从context恢复state困难。
→ 下一轮研究 context lifting / geometry inference。

CASE C
Context-direct-state 好，
carrier 差。

→ representation和context information都够，
主要是 learner/training 问题。
→ 下一轮做 matched carrier retraining。

CASE D
8^3差，16^3/32^3明显好。

→ resolution bottleneck。
→ 下一轮重新设计高分辨率 carrier。

CASE E
8^3 direct-state 已经很好。

→ 不要增加 resolution。
→ 直接修 learner/training。

CASE F
RGB+D好，RGB-only差。

→ depth supervision / geometric ambiguity
   是重要因素。
→ 方法主线需明确 RGB+D 与 RGB-only 边界。

============================================================
26. 本轮禁止训练新 carrier
============================================================

即使发现：

LEARNER_LIMITED

本轮也只报告。

不要马上开始 retraining。

因为我们需要先冻结 attribution，
再根据结果单独设计下一训练实验。

============================================================
27. Regression tests
============================================================

至少新增：

1. query GT cannot enter context-only optimizer.
2. query camera unavailable before state seal.
3. RGB-only track cannot load context depth.
4. RGB+D track cannot read query depth.
5. query-supervised oracle explicitly marked diagnostic.
6. state optimization does not mutate carrier checkpoint.
7. same state serves all sealed queries.
8. resolution sweep uses same scene/query IDs.
9. resolution sweep changes only allowed fields.
10. optimizer checkpoint selection uses context objective only.
11. wrong-scene control keeps query camera unchanged.
12. final holdout loader stays disabled.
13. scene bootstrap resamples scenes.
14. raw results regenerate statistics exactly.

运行完整 suite。

============================================================
28. 输出图表
============================================================

至少生成：

1. carrier vs direct-state query AbsRel
2. context vs query error
3. resolution vs query AbsRel
4. per-scene capacity gap
5. query-supervised vs context-supervised vs carrier
6. state resolution vs compute/memory

图中不能只展示有利 scenes。

============================================================
29. README 必须回答
============================================================

1. 8^3 state + fixed renderer 本身能不能表示正确 geometry？
2. Query-supervised direct state 能做到什么水平？
3. Context RGB+D direct state 能做到什么水平？
4. RGB-only direct state 能做到什么水平？
5. Carrier 与 direct-state 的 gap 多大？
6. 这个 gap 是否跨多个 scenes？
7. Resolution 提高是否稳定改善 query？
8. Renderer samples 是否重要？
9. Direct-state 是否只是记住 context？
10. Wrong-scene replacement 是否损害 direct-state？
11. 主要瓶颈是 representation、renderer、
    context inference 还是 learner？
12. 下一轮到底应该：
    - redesign representation
    - redesign lifting
    - increase resolution
    - retrain carrier
    - 还是停止当前路线？

============================================================
30. 最终摘要
============================================================

打印：

============================================================
3D DIRECT-STATE CAPACITY ATTRIBUTION FINAL
============================================================

N_CAPACITY_DEV_SCENES=

CARRIER_ABSREL=

DIRECT_RGBD_8_ABSREL=
DIRECT_RGBONLY_8_ABSREL=
QUERY_ORACLE_8_ABSREL=

CAPACITY_GAP_RGBD=
CAPACITY_GAP_RGBONLY=

DIRECT_RGBD_16_ABSREL=
DIRECT_RGBD_32_ABSREL=

RESOLUTION_8_TO_16_GAIN=
RESOLUTION_16_TO_32_GAIN=

RENDER64_ABSREL=
RENDER128_ABSREL=

CONTEXT_RGBD_ERROR=
QUERY_RGBD_ERROR=
GENERALIZATION_GAP_RGBD=

WRONG_SCENE_DAMAGE_DIRECT=

STATE_CAPACITY_STATUS=

REPRESENTATION_BOTTLENECK=
RESOLUTION_BOTTLENECK=
RENDERER_BOTTLENECK=
CONTEXT_INFERENCE_BOTTLENECK=
LEARNER_TRAINING_BOTTLENECK=

FINAL_HOLDOUT_TOUCHED=false
DYNAMIC_TTT_RUN=false
NEW_CARRIER_TRAINED=false

TESTS=
FINAL_INTEGRITY=

REPORT_DIR=

============================================================

============================================================
31. 最重要的科学纪律
============================================================

不要问：

“怎样让 direct-state 结果尽量好？”

而是问：

“当前系统失败到底发生在哪一层？”

必须真正区分：

scene representation本身表达不了

vs

representation能表达，
但context无法恢复

vs

context能够恢复，
但learner没学会

vs

renderer离散化限制

vs

resolution限制。

只有把这层 attribution 做清楚，
下一轮才值得重新训练 carrier。

不要为了得到某个想要的归因而修改 protocol。
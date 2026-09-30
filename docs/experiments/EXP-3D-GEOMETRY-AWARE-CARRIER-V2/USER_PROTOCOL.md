你现在位于：

https://github.com/KennyMcSimpson/measurement-scene-state.git

请基于已完成实验：

EXP-3D-DIRECT-STATE-CAPACITY-V1
EXP-3D-16G-OPTIMIZATION-BOUNDS-V1
EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1

执行下一阶段：

EXP-3D-GEOMETRY-AWARE-CARRIER-V2

============================================================
0. 本轮研究目标
============================================================

前序归因已经得到：

1. 8^3 → 16^3 的 RGB+D direct-state 几何改善稳定；
2. 16^3 → 32^3 没有建立稳定额外收益；
3. renderer 64 → 128 samples 无实际收益；
4. 更多 context optimization 不会稳定改善 sealed-query；
5. 16^3 query-supervised state 可达到明显更好的几何水平，
   因此 representation 具有较强表达能力；
6. context-direct 与 query oracle 之间仍存在显著 gap；
7. 显式 surface supervision 稳定改善 sealed-query：

   S0 query AbsRel ≈ 0.361556
   S2 ≈ 0.313772
   S3 ≈ 0.312183

   S0−S2 ≈ +0.047784
   S0−S3 ≈ +0.049372

8. free-space 单独收益很小，且 S2→S3 的额外收益不稳定；
9. observability 问题为 MIXED；
10. surface supervision 在 OBS2PLUS 等已观测区域也有明显收益；
11. 旧 learned carrier 的核心问题仍是：
    context observations → good shared scene state。

因此本轮第一次训练新的 learned carrier。

核心问题：

Q1.
将 16^3 state resolution 与显式 surface-aware ray supervision
加入 learned carrier 后，
是否能在未参与训练的新 scene 上稳定改善 shared-state geometry？

Q2.
这种改善是否来自 surface supervision，
而不是单纯 16^3、更多训练或不同 checkpoint selection？

Q3.
新 carrier 是否真正依赖 scene context，
能够由同一个 sealed shared state 服务多个 query？

Q4.
如果新 carrier 静态资格通过，
是否已经具备下一轮重新开启 Dynamic TTT 的条件？

============================================================
1. 最重要的科学原则
============================================================

本轮禁止：

- Dynamic TTT
- writer updates
- FC/CF
- action policy
- controller
- trajectory search
- visibility-aware fusion redesign
- cost volume
- transformer fusion
- VGGT / DUSt3R backbone
- 32^3 carrier
- renderer >64 samples
- bounds search
- context frame selection search
- 根据最终 holdout 分数改 loss
- 根据 holdout 换 checkpoint
- 只训练 geometry-aware 不训练 matched baseline
- 用旧 8^3 B-final 作为唯一 baseline 后宣称 surface supervision 有效

本轮的中心要求是：

**严格 matched carrier training。**

============================================================
2. 创建独立实验目录
============================================================

创建：

outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/

docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/

至少包含：

README.md
STATUS.md
preregistration.json
training_contract.json
architecture_contract.json
loss_contract.json
scene_split.json
frame_role_lock.json
checkpoint_selection_lock.json
data_provenance.json
baseline_training.json
surface_training.json
training_curves.json
static_results.json
wrong_scene_results.json
state_use_results.json
bootstrap_results.json
qualification_results.json
cost_analysis.json
integrity.json
tests.json
commands.sh
git_commit.txt
dirty.patch

raw/
figures/
audit/
checkpoints/

不能覆盖旧实验。

============================================================
3. 新方法只改两个主要因素
============================================================

与旧 8^3 carrier 相比，
新训练 family 使用：

GRID = 16^3

和冻结的：

GT_FREE_BOUNDS

renderer：

64 samples

但本轮的 matched scientific comparison
不是旧8^3 vs 新16^3。

正式比较必须是：

C0 — BASELINE16

16^3
+
原始 carrier training loss
+
无 explicit surface loss

C1 — SURFACE16

16^3
+
与 C0 完全相同
+
explicit surface supervision

这是本轮 primary comparison。

============================================================
4. Free-space 的角色
============================================================

前序归因显示：

surface supervision 是主要有效成分，
free-space 单独没有建立稳定 practical gain。

因此本轮主方法：

C1 = SURFACE16

不把 free-space 加入主模型。

允许训练一个次要 ablation：

C2 — FREE_SURFACE16

16^3
+
surface
+
free-space

但只能作为 secondary ablation。

不能因为 C2 最终比 C1 低一点，
事后把 C2 改成主方法。

PRIMARY_METHOD = SURFACE16

必须在 formal training 前锁定。

============================================================
5. Architecture 必须尽量 matched
============================================================

C0/C1/C2：

必须使用完全相同的：

- image encoder
- lifting
- fusion
- complete branch
- channel widths
- MLP structure
- fixed renderer
- parameter count
- candidate semantics
- context inputs
- output typed state

唯一允许不同：

surface/free-space loss 是否启用。

如果 8^3 → 16^3
导致某些 tensor dimensionality
自然改变：

记录在 architecture_contract.json。

不得给 C1 额外 hidden channels。

============================================================
6. State 语义
============================================================

继续保持：

query-independent typed scene state。

同一个 sealed state：

必须回答多个 query。

query camera：

只能在 state seal 后进入 fixed renderer。

禁止：

query-conditioned carrier state。

禁止：

每个 query 单独 materialize
不同 scene state。

============================================================
7. GT-free bounds 冻结
============================================================

沿用上一轮已经使用的：

FROZEN_GT_FREE_BOUNDS

只能使用：

- context intrinsics
- context extrinsics
- training-set frozen global near/far prior

不能使用：

- current-scene GT depth
- query camera
- query depth
- model prediction
- query score

正式训练和 evaluation
都使用同一 bounds rule。

============================================================
8. 数据 split 必须重新整理
============================================================

当前已有大量 exposed attribution scenes。

它们可以用于：

- training
- development
- analysis

但不能再称为 fresh test。

正式建立：

TRAIN
DEV
FRESH_QUALIFICATION

按 physical scene 划分。

要求：

TRAIN：
推荐至少 24 scenes

如果资源允许：
32–64 scenes

DEV：
至少 8 scenes

FRESH_QUALIFICATION：
至少 8 scenes

FRESH_QUALIFICATION：

必须在整个训练与选择期间完全封存。

============================================================
9. 旧曝光场景怎样使用
============================================================

已曝光的：

旧3 train
旧7 unseen
17 attribution scenes

不能被重新包装成 fresh qualification。

允许分配进入：

TRAIN
或
DEV

但必须记录：

historically_exposed=true

最终 fresh qualification
必须是新的锁定 scene cohort。

============================================================
10. Fresh qualification 独立性
============================================================

此前某些 holdout 存在：

historical project exposure
无法完全证明的问题。

所以本轮在正式 training 前
必须先解决 fresh qualification provenance。

每个 fresh scene 至少确认：

- 不在 B-final training manifest
- 不在旧 100-scene pool exposed list
- 不在旧 365-scene execution logs，如果可恢复
- 不在旧实验 result manifests
- 不在旧 context/query manifests
- 本轮此前没有模型 forward

如果无法建立足够可信的独立性：

不要声称 independent test。

状态：

FRESH_QUALIFICATION_STATUS=
BLOCKED_INDEPENDENCE_UNRESOLVED

但允许完成 train/dev。

============================================================
11. Frame role 必须冻结
============================================================

每个 scene 在模型训练/评价前固定：

context A
context B
anchor
sealed query

frame selection 只能依据：

- source validity
- camera validity
-预先规定的几何 feasibility

不能依据：

- carrier prediction
- loss
- query score

A/B：

尽量相同 frame 数
并满足基本 multi-view feasibility。

============================================================
12. Training supervision
============================================================

C0 baseline loss：

L_C0 =
L_RGB
+
L_render_depth
+
旧 regularization

C1：

L_C1 =
L_C0
+
lambda_surface * L_surface

C2 secondary：

L_C2 =
L_C1
+
lambda_free * L_free

============================================================
13. Surface loss 完全沿用已验证定义
============================================================

不要重新设计 surface loss。

沿用上一轮：

tau_surface =
0.5 × voxel diagonal

surface band：

|t_i - d*| <= tau_surface

surface mass：

sum of rendering weights in band

L_surface =
-log(surface_mass + eps)

lambda_surface = 0.1

除非上一轮 artifact 中精确定义不同，
以冻结 artifact 为 authority。

不得根据 DEV query
重新调 lambda。

============================================================
14. Free-space secondary loss
============================================================

只用于 C2：

t_i < d* - tau_surface

施加 free-space penalty。

surface 后方：

不标记为空空间。

lambda_free = 0.1

沿用上一轮定义。

============================================================
15. Test-time 输入边界
============================================================

训练时：

可以使用 TRAIN scene RGB + depth。

测试/qualification 时：

carrier 输入只能使用：

- context RGB
- context camera

不得向 carrier 输入：

- context depth
- query depth
- query RGB target
- future information

这是非常重要的。

Geometry supervision 是：

OFFLINE TRAINING SUPERVISION

不是 test-time sensor input。

README 必须明确写：

Test-time input = RGB + calibrated cameras.

============================================================
16. 为什么训练时允许 depth
============================================================

本项目当前主线为：

RGB+D training utility track

即：

训练阶段可使用 depth
塑造 shared-state geometry。

部署时：

只给 RGB + camera。

不能将其描述成：

RGB-only training。

如果后续需要严格 RGB-only，
另开实验，不在本轮混入。

============================================================
17. Training 配置必须 matched
============================================================

C0/C1/C2 使用：

相同：

- training scenes
- frame tuples
- optimizer
- LR schedule
- batch size
- total steps
- random seeds
- initialization scheme
- data augmentation
- context/query sampling
- checkpoint frequency
- gradient clipping
- loss normalization
- GPU precision

只有 loss term 区别。

============================================================
18. 至少使用多个 seeds
============================================================

推荐：

3 seeds

最低：

2 seeds

如果计算资源限制严重：

可以主训练 1 seed，
但必须明确：

SEED_ROBUSTNESS=NOT_ESTABLISHED

不能把单 seed 差异包装成稳定训练提升。

最好：

SEEDS = [20260928, 20260929, 20260930]

或预先固定等价列表。

============================================================
19. Checkpoint selection
============================================================

checkpoint 只能用：

TRAIN-side objective
和 DEV scene predefined metric

选择。

不能访问 FRESH_QUALIFICATION。

主 checkpoint selection：

DEV scene-macro depth AbsRel

但需要同时检查：

RGB
opacity
coverage
context/query gap

不能因为一个 checkpoint
query AbsRel 低但 geometry collapse
就选择。

selection rule
必须训练前写入：

checkpoint_selection_lock.json

============================================================
20. C0/C1 必须独立选择 checkpoint
============================================================

不能强迫二者使用同一步，
除非预注册明确规定 fixed training step。

如果允许 early selection：

两者必须使用同一选择规则。

报告：

selected step
training curve
DEV curve

不能根据 final qualification
重新选择。

============================================================
21. Primary DEV 评价
============================================================

对 C0 / C1 / C2：

评价：

A
B
anchor
wrong_scene

所有 scene 使用：

同一 frame roles。

主指标：

scene-macro query depth AbsRel

同时报告：

RGB MSE
SSIM
RMSE
δ1
opacity
coverage

============================================================
22. Primary carrier comparison
============================================================

正式 primary：

C0 BASELINE16
vs
C1 SURFACE16

定义：

SURFACE_GAIN =
AbsRel(C0)
-
AbsRel(C1)

正值代表 surface supervision 更好。

统计单位：

scene

使用：

10,000 paired scene bootstrap
seed fixed
95% percentile CI

============================================================
23. Practical success gate
============================================================

SURFACE_TRAINING_STATUS=

SUPPORTED
PARTIAL
NOT_ESTABLISHED
HARMFUL

SUPPORTED 至少需要：

1.
SURFACE_GAIN > 0

2.
paired CI lower > 0

3.
至少 75% DEV scenes
C1 不比 C0 差

4.
不是单 scene 主导

5.
wrong-scene damage 保持正

6.
state spatial shuffle 明显恶化

7.
没有通过 opacity/coverage artifact
解释全部 gain

============================================================
24. Static carrier qualification gate
============================================================

除了 C1 > C0，
还需要判断 C1 本身够不够格。

STATIC_DEV_STATUS=

SUPPORTED
PARTIAL
NOT_ESTABLISHED

SUPPORTED 至少要求：

1.
Full context > anchor

定义：

anchor AbsRel
-
mean(A,B) AbsRel

正值。

2.
scene-bootstrap lower > 0

3.
至少 75% DEV scenes
full context >= anchor

4.
wrong_scene damage >0
且 CI lower >0

5.
同一个 sealed state
服务多个 query

6.
geometry 不是 opacity artifact

============================================================
25. Observability diagnostics 继续保留
============================================================

复用上一轮：

OBS0
OBS1
OBS2PLUS

但本轮它只是：

diagnostic。

比较：

C0
vs
C1

在各区域的 query AbsRel。

目标：

检查 learned carrier
是否真正复制了 direct-state
surface supervision 在 OBS2PLUS 上的收益 signature。

如果 C1 只改善 OBS0
而 OBS2PLUS 无改善：

说明 learned carrier 行为
与 direct-state 机制结果不一致，
需要谨慎解释。

============================================================
26. Context-query gap
============================================================

对 C0/C1 都报告：

context AbsRel
query AbsRel

GENERALIZATION_GAP =
query - context

Surface supervision 的理想结果：

不一定要求 context error 更低。

核心是：

query error 更低，
generalization gap 缩小。

============================================================
27. Wrong-scene 和 spatial shuffle
============================================================

对 C0/C1：

都做：

wrong-scene replacement

和：

spatial shuffle

recipient query camera 保持不变。

如果 C1 query error 更好
但 wrong-scene/shuffle damage 消失：

要警惕：

模型是否退化成 global prior。

============================================================
28. Old B-final 只作为 reference
============================================================

允许报告：

OLD_B_FINAL_8

但它不是 surface supervision
的主要 attribution baseline。

正式核心比较必须：

C0 16³
vs
C1 16³

因为这样：

grid
architecture
training scenes
budget

全部 matched。

============================================================
29. C2 free+surface 的用途
============================================================

C2 只回答：

free-space 在 learned carrier
中是否提供额外价值。

比较：

C1 vs C2

但不影响 PRIMARY_METHOD=C1。

如果 C2 明显更好：

记录为后续信息。

不能事后改 primary。

============================================================
30. Fresh qualification 打开条件
============================================================

只有：

SURFACE_TRAINING_STATUS=SUPPORTED

并且：

STATIC_DEV_STATUS=SUPPORTED
或至少非常明确的 PARTIAL with frozen rationale

才允许打开：

FRESH_QUALIFICATION。

否则：

FRESH_QUALIFICATION_OPENED=false

不看结果。

============================================================
31. Fresh qualification 只能运行一次
============================================================

打开前冻结：

- C0 checkpoint
- C1 checkpoint
- architecture
- loss
- bounds
- frame roles
- metrics
- scene list
- seed handling
- query definitions

然后：

一次运行 C0/C1。

不能：

- 调 lambda
- 换 checkpoint
- 改 frame
- 改 bounds
- 加 train scenes
- 调 selection rule

============================================================
32. Fresh qualification 主要问题
============================================================

在真正 fresh scenes 上回答：

Q1.
C1 是否稳定优于 C0？

Q2.
C1 full context 是否稳定优于 anchor？

Q3.
wrong-scene 是否明显更差？

Q4.
surface supervision signature
是否在多个新 scenes 重现？

============================================================
33. Final status
============================================================

如果 fresh qualification 可运行：

FINAL_STATIC_STATUS=

SUPPORTED
PARTIAL
NOT_ESTABLISHED

SUPPORTED 至少要求：

Surface16 > Baseline16
并且

FullContext > Anchor
并且

WrongScene > Correct
并且

多 scene 一致。

============================================================
34. 什么时候重新打开 Dynamic TTT？
============================================================

只有：

FINAL_STATIC_STATUS=SUPPORTED

才输出：

DYNAMIC_TTT_NEXT_STAGE_ALLOWED=true

否则：

false

不要因为 DEV 看起来不错
就提前开 controller。

============================================================
35. 如果 C1 失败怎么办？
============================================================

CASE A

Direct-state surface supervision 有效，
learned carrier C1 无收益。

→ 说明 carrier 没学会利用 surface supervision。
→ 下一步研究：
   lifting / fusion / geometry reasoning。

CASE B

C1 > C0，
但 full context 仍不如 anchor。

→ surface helps，
  但 shared-state carrier 仍不够格。
→ 不开 Dynamic TTT。

CASE C

C1 > C0，
full context > anchor，
但 wrong-scene 不显著。

→ 可能 reliance on global prior。
→ 加强 scene-specific state construction。

CASE D

C1 全面成立。

→ 进入 Dynamic TTT V2。

============================================================
36. 不要这轮引入现代大模块
============================================================

本轮明确禁止：

VoRTX transformer
MVSNeRF cost volume
GeoNeRF
MVSplat
VGGT
DUSt3R

原因：

我们已经有一个干净的
surface-supervision positive signal。

先验证 learned carrier
能否复制它。

只有 C1 训练失败，
下一轮才有理由引入
visibility-aware / cost-volume geometry reasoning。

============================================================
37. 训练成本
============================================================

记录：

per-step time
epoch/step walltime
GPU memory
checkpoint size
inference latency
renderer latency

C0/C1/C2：

成本必须分别记录。

不能声称：

surface supervision 推理更贵，

如果该 loss 只在训练阶段存在。

测试时如果 architecture 完全相同：

明确：

inference architecture identical.

============================================================
38. Regression tests
============================================================

至少新增：

1. C0/C1 architecture identical.
2. C1 differs only by allowed loss term.
3. test-time carrier input contains no depth.
4. surface loss only active during training.
5. query GT never enters carrier input.
6. fresh qualification scenes never enter training/dev.
7. checkpoint selection cannot access fresh qualification.
8. frame roles deterministic.
9. GT-free bounds cannot read current scene depth.
10. same state serves all queries.
11. wrong-scene preserves recipient camera.
12. spatial shuffle modifies state but not camera.
13. scene bootstrap resamples scenes.
14. old exposed scenes cannot be marked fresh.
15. fresh qualification runner requires frozen lock.
16. raw→statistics deterministic.

运行完整 suite。

============================================================
39. 图表
============================================================

至少生成：

1.
C0 vs C1 query AbsRel

2.
per-scene C1 gain

3.
Full context vs anchor
for C0/C1

4.
OBS0 / OBS1 / OBS2PLUS
C0 vs C1

5.
training curves
C0/C1

6.
context vs query gap
C0/C1

7.
wrong-scene / shuffle damage

8.
quality vs compute

============================================================
40. README 必须回答
============================================================

1.
Surface supervision 在 learned carrier
中是否仍然有效？

2.
它是否跨多个 scenes？

3.
是否复现 direct-state 中
OBS2PLUS 改善 signature？

4.
16³ baseline retraining
本身提升多少？

5.
surface supervision 相对 matched baseline
额外提升多少？

6.
full context 是否终于稳定优于 anchor？

7.
wrong-scene 是否稳定更差？

8.
state 是否真正 scene-specific？

9.
context-query gap 是否缩小？

10.
free-space 在 learned carrier 中
有没有额外价值？

11.
DEV 是否达到 static qualification？

12.
fresh qualification 是否被允许打开？

13.
fresh scenes 是否复现主要 signature？

14.
现在是否可以重新打开 Dynamic TTT？

============================================================
41. 最终终端摘要
============================================================

打印：

============================================================
16G GEOMETRY-AWARE CARRIER V2 FINAL
============================================================

N_TRAIN_SCENES=
N_DEV_SCENES=
N_FRESH_QUALIFICATION_SCENES=

BASELINE16_CHECKPOINT=
SURFACE16_CHECKPOINT=
FREE_SURFACE16_CHECKPOINT=

BASELINE16_QUERY_ABSREL=
SURFACE16_QUERY_ABSREL=
SURFACE_GAIN=
SURFACE_GAIN_CI=

FREE_SURFACE_QUERY_ABSREL=
FREE_SPACE_EXTRA_GAIN=

BASELINE16_FULL_CONTEXT_GAIN=
SURFACE16_FULL_CONTEXT_GAIN=

BASELINE16_WRONG_SCENE_DAMAGE=
SURFACE16_WRONG_SCENE_DAMAGE=

BASELINE16_SHUFFLE_DAMAGE=
SURFACE16_SHUFFLE_DAMAGE=

OBS2PLUS_BASELINE16=
OBS2PLUS_SURFACE16=
OBS2PLUS_SURFACE_GAIN=

SURFACE_TRAINING_STATUS=
STATIC_DEV_STATUS=

FRESH_QUALIFICATION_INDEPENDENCE=
FRESH_QUALIFICATION_OPENED=

FRESH_BASELINE16_ABSREL=
FRESH_SURFACE16_ABSREL=
FRESH_SURFACE_GAIN=
FRESH_SURFACE_GAIN_CI=

FINAL_STATIC_STATUS=

DYNAMIC_TTT_NEXT_STAGE_ALLOWED=

TEST_TIME_INPUT=RGB+CAMERA
TEST_TIME_DEPTH_USED=false

FINAL_HOLDOUT_TOUCHED=
DYNAMIC_TTT_RUN=false

TESTS=
FINAL_INTEGRITY=

REPORT_DIR=

============================================================

============================================================
42. 最重要的研究纪律
============================================================

不要问：

“怎么把新 carrier 调得最好？”

而要回答：

“之前在 direct-state 中发现的 surface supervision，
能不能被 learned carrier真正学到，
并在 fresh scenes 上转化为 shared-state geometry？”

必须区分：

16³ resolution 的收益

vs

surface supervision 的收益

vs

更多训练数据的收益

vs

checkpoint selection 的收益。

因此 primary comparison 必须始终是：

matched 16³ baseline

vs

matched 16³ surface-aware carrier。

只有这个对照成立，
才能说 geometry-aware training
是下一版 carrier 的真正方法增量。

不要为了重新打开 Dynamic TTT
修改 qualification gate。
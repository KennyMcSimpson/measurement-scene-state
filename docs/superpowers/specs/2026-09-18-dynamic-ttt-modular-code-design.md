# 逆向 JEPA + 动态 TTT：模块化代码设计

日期：2026-09-18；复核更新：2026-09-19。状态：第一阶段载体、直接写入、流式运行、独立评估和短训练调试已实现；学习策略、离线动作教师及完整 benchmark 尚未实现。实际交付与验证见 `../../dynamic_ttt_implementation_20260919.md`。

对应研究方案：`D:/My Documents/Kenny/My Documents/cv/CV_逆向JEPA与动态TTT_合作者完整讨论方案_2026-09-17.md`。

## 1. 实现决策

采用**保留静态 MCSS、新增独立动态包**的方式。现有 `engine.py` 继续负责静态训练，现有 V5 的模型、checkpoint 和固定 renderer 保留。动态包提供新的场景载体、每场景快速权重和逐帧运行器；离线教师和最终查询评估放在运行器之外。

这既能复用相机、几何和测量代码，又能分别检验两个研究核心：动态写入可直接关闭；逆向 JEPA 需要另行训练匹配的表示/读出对照，不能用一个布尔开关假装去掉整个研究思想。

- 逆向 JEPA 对应“上下文图像 → 一个查询无关的 SceneState → 固定测量 → 真实目标监督”。它分布在载体、静态训练目标和独立查询评估中，不是一层名为 JEPA 的网络。
- 动态 TTT 对应“旧状态的预测误差 → 动作选择 → 指定位置的快速权重写入 → 重新构建状态”。它由反馈、策略、写入和流式运行器共同实现。

比较过三个工程方向：

| 方向 | 好处 | 成本/问题 | 决定 |
| --- | --- | --- | --- |
| 在 `engine.py` 和旧 `forward` 中加入大量动态分支 | 初期改动少 | 静态训练、部署、教师和查询标签很容易混用 | 不采用 |
| 完全另起工程 | 物理上独立 | 重复相机、renderer、metrics，容易与 V5 定义漂移 | 不采用 |
| 同工程新增动态包，复用稳定底层接口 | 模块可单独测试，旧基线可恢复 | 需要明确数据类型和新 checkpoint 格式 | **采用** |

不需要先重构整个旧工程。新增文件按实现阶段逐步建立，避免一次创建大量空壳。

## 2. 建议目录及责任

以下是完整设计的目标结构；具体已实现范围以本次交付说明为准，不能把该目录清单视为全量实现记录。

```text
src/mcss/
  data/
    episodes.py                 # 将开发清单解析为观察流和独立查询描述
  dynamic/
    types.py                    # 动作、已到达观测、快速权重、反馈与状态类型
    config.py                   # 动态实验配置与组合校验
    cache.py                    # 只缓存已经到达的图像/特征/相机
    lifting.py                  # 图像特征投影、几何有效性、均值/方差/支持数
    carrier.py                  # 共享三维状态、两个明确的可写位置
    write_rule.py               # 冻结投影、通道gate、外积、单步裁剪、同步提交
    feedback.py                 # 旧状态预测与当前观测的更新前误差
    policy.py                   # 固定动作及只看前缀的学习策略
    budget.py                   # 动作可行性、算子调用账单和成本
    runner.py                   # 在线状态机，按规定顺序调用其他模块
    checkpoint.py               # 新格式、初始化映射、慢参数冻结、快状态重置
  training/
    types.py                    # 仅离线可见的监督、教师轨迹和版本绑定类型
    grounded.py                 # 两组上下文、共享query的静态状态训练
    write_unroll.py             # 短轨迹展开，训练载体和写入规则
    action_teacher.py           # 离线分支执行、未来收益标签
    policy_fit.py               # 前缀策略训练与训练场景再采样
    readout_controls.py         # 核心A的匹配读出对照；不进入主方法运行器
  evaluation/
    sealed_queries.py           # 状态封存后加载query并评分
    streaming_report.py         # scene级指标、预算/耗时和轨迹报告

scripts/
  train_grounded_state.py
  train_write_rule.py
  collect_action_targets.py
  train_action_policy.py
  evaluate_streaming.py

tests/
  dynamic/                      # 状态机、更新隔离、动作、预算、checkpoint
  test_episode_access.py        # 未来/查询/深度访问边界
  test_sealed_queries.py        # 封存前后状态完整性
```

脚本只负责解析参数、组装组件和调用入口。网络结构放在 `carrier.py`，写入数学放在 `write_rule.py`；二者都不读取磁盘数据，也不保存结果。`runner.py` 只编排步骤，不计算训练损失或处理下载。

`types.py` 只依赖 PyTorch、标准库和现有 `Cameras/SceneState`。运行时包不得导入 `training.action_teacher`、最终评估标签读取器或旧 `SceneExample`。训练模块可以调用同一个运行时计算内核，依赖方向保持单向。

## 3. 复用边界

直接复用现有 `mcss.geometry`、`Cameras`、`SceneState`、固定测量 renderer、离线损失和指标定义。固定 renderer 继续保持零可训练参数。相机使用 OpenCV c2w，体素顺序 `[D,H,W]`，深度保持米制射线距离。

旧 `ManifestSceneDataset.__getitem__` 会同时读取 context 和 target 标签，因此只继续用于旧静态训练及对照。新在线运行器不能接收它返回的完整 `SceneExample`。

现有 `EvidenceResidualStateEncoder.forward` 同时负责编码、投影、融合、空间处理和状态头。新载体先用公开几何函数实现所需投影接口，不复制整个旧类再加开关，也不把它的私有 helper 当成长期公共 API。若后续提取共享纯函数，单独提交并验证旧输出与 checkpoint 不变。

当前小型 CNN 可作为第一版编码器。预训练图像骨干通过明确的编码器适配接口在后续接入，不作为第一轮数据/状态机测试的依赖。

## 4. 三类状态明确分开

| 对象 | 内容 | 生命周期 |
| --- | --- | --- |
| 慢参数 | 编码器、载体、两个可写矩阵的初始值、写入投影/gate、策略 | 离线训练；部署冻结 |
| 观察缓存 `B_t` | 已到达 RGB、相机、冻结特征、原始帧 ID | 单场景，按到达顺序增长 |
| 快速权重 `DeltaW_t` | fuse 和 complete 的场景私有偏移 | 单场景，换场景归零 |
| 场景状态 `S_t` | 当前缓存与当前权重共同生成的密度、颜色等 | 每次观察到达/写入后重新生成 |

快速权重是显式 Tensor 状态，不注册为全局 optimizer 的 `nn.Parameter`；批量实现时也不能让不同 episode 共用可变 Tensor。部署时不开 backward；短轨迹离线展开需要保留计算图，因此不要在公共 `carrier` / `write_rule` 函数上写死 `@torch.no_grad()`，由部署入口控制推理上下文。

第一版仅要求单 episode 推理正确。批量并行和增量缓存属于后续优化；正确性参考实现每次从已经到达的完整前缀重建，明确记录可能为 `O(T^2)` 的总重读成本。

## 5. 最小公共接口

下面是接口约定，不是已经完成的代码。

```python
# dynamic/types.py
Action = OFF | FUSE | COMPLETE | ALL
OnlineEpisodeSpec = {scene_id, warmup_rgb_camera_refs, stream_rgb_camera_refs,
                     declared_length, coordinate_protocol}
QueryVaultSpec = {scene_id, query_camera_refs, query_label_refs}
OnlineObservation = {scene_id, frame_id, rgb, camera}
FastWeights = {delta_fuse, delta_complete, step}
EpisodeState = {cache, fast_weights, scene_state, history, budget}
ControlInput = {preupdate_feedback, current_image_summary, scene_stats, fast_stats,
                previous_action, observed_camera_stats, remaining_budget}
WriteTrace = {fuse_activation, complete_activation, observed_feature_statistics,
              candidate_ids, support_weights, source_episode_id,
              source_cache_revision, source_fast_step}
SealedScene = {episode_id, scene_id, split_id, query_vault_id,
               scene_state, state_hash, observed_ids,
               checkpoint_hash, config_hash, fast_state_hash}
```

`OnlineObservation` 没有 depth、normal、未来帧、query 路径或完整 dataset 引用。`ControlInput` 中只保存前缀可得值，不持有 evaluator/teacher 对象。数据准备清单是离线索引，不能把整份 JSON 作为在线 batch。

当前 pilot 清单同时列有 stream 和 query 路径，只是准备阶段的索引。第 1 步应将其编译成独立文件：`OnlineEpisodeSpec` 只列 warmup/stream 的 RGB 与相机引用，拒绝 depth、normal、query 和 target 字段；`QueryVaultSpec` 由独立 evaluator 持有。完整未来 stream 引用仅留在数据驱动器中，逐帧交给运行器，策略只接收当前及历史信息。流接口只提供顺序 cursor，不提供未来随机索引或提前读取。两份清单在离线准备时检查 scene/split 归属、frame ID 无交叉；query vault 在收到封存凭据且核对 episode、scene、split 和 vault 身份后才打开相机和标签。公开 vault ID 只是协议标识，不附带查询路径或内容。

| 模块 | 主要接口 | 不能做什么 |
| --- | --- | --- |
| `episodes` | `open_online_episode(OnlineEpisodeSpec) -> OnlineEpisodeSource` | 不接受混合query索引；不向运行器附带未来完整样本 |
| `cache` | `append(observation, features) -> ObservationCache` | 不改历史帧 ID、不自动载入后续帧 |
| `carrier` | `encode(observation)`；`trace(cache, fast)`；`materialize(cache, fast) -> SceneState` | 不知道 query 相机/标签，不选择动作 |
| `feedback` | `compute(previous_state, arrived_observation) -> Feedback` | 不先吸收当前图像，不用更新后误差选已执行动作 |
| `policy` | `choose(control_input, feasible_actions) -> Action` | 不运行离线搜索，不查询真实未来收益 |
| `write_rule` | `propose(trace) -> WriteProposal`；`commit(old_fast, proposal, action) -> FastWeights` | 不原位改慢参数，不读取误差图作为首版写入目标 |
| `budget` | `feasible_actions(ledger, declared_remaining_steps)`；`record(actual_calls)` | 不根据未来图像难度预留成本，不跳过困难帧 |
| `runner` | `reset(warmup)`；`step(arrived)`；`seal()` | 不加载训练/测试query标签 |
| `sealed_queries` | `evaluate(sealed_scene, query_source) -> SceneMetrics` | 不修改缓存、快权重、策略或场景状态 |

工程上的类型边界本身不是安全沙箱；必须配合独立读取路径和访问追踪测试。离线 teacher 因任务需要看到训练场景的未来目标，必须使用另一种 `TrainingSupervision` 类型。

`training/types.py` 中定义 `TrainingSupervision={schema_version, future_observations, query_cameras, query_labels, valid_masks, utility_config}`；它只能由训练入口打开。`TeacherTrace` 保存 schema_version、前缀 ID/摘要、分支身份与随机种子、候选动作、续接策略、收益及真实成本，并绑定载体、写入规则、特征提取、效用、预算和训练清单的哈希。策略拟合只读其中的前缀特征与监督值，不能把未来或query内容拼入特征。运行时包不导入这些类型。

## 6. 载体与两个可写位置

`carrier.py` 实现“观测编码 → 几何投影 → 多视图统计 → fuse 通道 MLP → 三维 refinement → complete 通道 MLP → 状态 heads”。

两个 MLP 都显式提供可写的下投影矩阵，使用 `linear(u, W0 + DeltaW, bias)`，保持 `W[out_dim, in_dim]` 的 PyTorch 约定。第一版令宽度由配置确定，不把旧的 3×3×3 卷积核当作现成可写矩阵。

位置的几何解释由消融验证；`fuse` 名字只表示它位于多视图融合路径，不能提前声称只负责纠错。

新载体允许 density logits 充分正负变化，不能继承旧 V5 在高外观一致性位置的受限修正范围。该改变必须同时用于新载体的 OFF、固定写入和动态策略组；旧 V5 是历史比较点，不能把载体修正收益记作动态控制收益。

首轮机制模型可使用单分辨率颜色和密度场；若启用高分辨率 appearance，所有对照统一启用并计入成本。动态状态先不附带旧 V5 的 `StateEvidence`：旧 payload 的 residual/gate 恒等式不一定适用于新载体。观察支持统计作为独立 trace 记录，防止旧 evidence 被误当作更新后几何置信度。

## 7. 写入数学与提交语义

矩阵约定固定为 `W[d,m]`、`u[N,m]`、`v[N,d]`，因此 `einsum('ni,nj->ij', q[:,None] * v, u)` 产生 `[d,m]` 增量。`v = P.T @ (g * h)` 的向量维度为 `d`。原方案的公式按这一约定实现即可。

写入目标由已经到达的图像特征、支持数、均值/方差和已知坐标组成。至少两个有效视图才支持写入；几何有效性只来自有限值、相机前方和图像边界。首版残差进入动作策略，不直接变成几何修正目标。

`ALL` 的两个 proposal 必须来自相同的旧快权重版本；`commit` 校验 episode、缓存版本和 `source_fast_step` 后同时返回两个新偏移。只校验步数不够，因为不同场景或 teacher 分支可能恰好处于同一步。每个分支须持有独立身份，防止跨分支提交。部署每次 proposal 都计算单步 Frobenius 裁剪，另报累计快权重范数；不能把单步裁剪解释为长期稳定保证。

`OFF` 的含义是本步不新增快权重写入。**新图像仍进入缓存，场景仍必须重新生成**；OFF 不等于状态和输出完全不变。

## 8. 在线步骤的唯一顺序

```text
warmup -> 缓存已见输入 -> DeltaW=0 -> materialize 初始状态

每个新观察到达时：
  1. 用旧 SceneState 在当前相机下 render；只依赖旧状态和当前相机
  2. 编码当前 RGB，按反馈配置编码渲染图，构造更新前反馈及当前图像摘要
  3. 只用已见信息和预算选择动作；此时当前图像尚未进入场景缓存
  4. 将当前 RGB 与第 2 步的特征追加到缓存
  5. 非 OFF：用旧快权重和新缓存产生两个位置的 trace/proposal
  6. 从同一个旧版本同步提交被选择的快权重增量
  7. 用新缓存和提交后的权重重新 materialize 场景
  8. 记录真实调用、延迟、显存、动作与状态版本

最后 seal -> 独立 evaluator 才打开 query -> 只读评分
```

步骤 4 复用步骤 2 的当前图像特征；当前图像只编码一次，渲染图的额外特征编码另计。RGB 像素残差不需要渲染图特征编码，特征残差需要；该选择写入配置，不能隐藏成本。候选写入需要旧权重的额外 forward，同样计入账单。最后一步 commit 后必须有一次 materialize。

`seal()` 保存与活动运行器不共享可变存储的状态快照。独立 evaluator 只接受该快照和自己的查询来源，并在评分前后核对状态哈希；Python 类型标注或 frozen dataclass 本身不能阻止 Tensor 原位修改。

相机参考系由第一张 warmup 的相机确定，空间范围使用预先固定的 `local_bounds_m`。不能由全场景或目标深度生成在线范围。相同已见前缀搭配不同未来/查询时，动作和封存前状态必须相同。

未来观测和最终查询的预算预留只根据协议事先声明的数量、输出分辨率和固定算子成本计算；不打开未来文件或查询元数据。若最小必需成本已经超预算，启动时拒绝该配置，而不是运行中静默跳帧。

## 9. 离线训练分四个入口

| 入口 | 训练什么 | 可以看到的监督 | 输出 |
| --- | --- | --- | --- |
| `train_grounded_state.py` | 编码器/共享场景载体，写入关闭 | 训练scene的真实query RGB/depth | 新载体 OFF checkpoint |
| `train_write_rule.py` | 载体、两个写入位置、Q/P/g | 短轨迹训练目标 | 冻结载体及写入规则 |
| `collect_action_targets.py` | 不进行部署策略决策；离线分支打分 | 训练scene未来RGB/depth | 前缀特征、各动作收益及成本 |
| `train_action_policy.py` | 轻量动作评分网络 | 前缀输入及离线收益标签 | 冻结policy checkpoint |

公共计算内核允许离线训练保留计算图，部署入口在冻结模式调用同一套计算，避免训练和测试写入规则两份实现。teacher 分支要独立复制快状态、缓存索引、随机数和预算，不能让某个分支写入影响其他动作。

静态 A 使用 `StaticContextPairManifest`：两组不重叠上下文分别调用同一个载体，各自构建状态并预测同一组训练query，绝不把两组输入合并成一个状态。动态 B 使用 `StreamEpisodeManifest`，接续 A 的权重，用单条历史流做短序列展开。两者是独立采样协议，载体始终只接收一个已见缓存；两组上下文不是部署时必须同时提供的两条流。4+8+4 pilot 用于动态接口验收，不能代替 A 的双上下文诊断。A/B 的上下文长度与覆盖范围仍需在开发集检查，避免把采样变化当作方法收益。

阶段 A 的 OFF checkpoint 用于状态学习与容量诊断。阶段 B 如果继续训练载体，正式动态比较中的 OFF、固定动作和动态策略必须全部使用**同一个阶段 B checkpoint**；否则“OFF 对比动态”会混入不同训练历程的收益。阶段 A checkpoint 作为独立训练基线另报。

阶段 B 的最终载体及写入规则冻结后，阶段 C 才生成动作收益标签；标签绑定该 B checkpoint，不能直接沿用阶段 A 的教师收益。若之后改动载体或写入规则，原标签失效，需要重新生成。

`readout_controls.py` 只实现研究方案规定的对照组件及训练组合。R-Fixed 主线始终用固定测量，R-Residual 的查询条件残差头是独立对照，不能偷偷进入主方法的场景构建或反馈。两种读出各自训练；每种读出内部的 OFF/固定/动态三组共用该行的载体 checkpoint。所有额外读出参数和计算另计。

按 scene 划分 train/dev/test。当前 6 场景 pilot 仅用于工程/机制开发，不是完整论文证据。365 个训练场景用于后续基础训练，46 个验证场景用于开发；20 个历史诊断场景不是未见测试。26 个新下载场景只用于最终冻结后的测试。

## 10. Checkpoint 与复现

新 checkpoint 使用独立 schema，例如 `mcss.dynamic.v1`，包含载体与写入参数、策略参数、配置、数据划分/预处理标识和训练阶段。部署 episode 快状态默认不放进通用 checkpoint；调试快照另存并明确 scene/step。

复用 teacher 标签时，逐项核对生成它们的载体/写入规则、前缀特征、utility、预算和训练清单版本。policy checkpoint 记录该来源链；部署必须匹配载体、写入规则、特征schema和预算协议。部署测试清单当然不同于训练清单，应核对预先冻结的划分和不相交性，不能错误要求两者哈希相同。

V5 的 `training.resume` 只继续恢复同构旧训练。若热启动新载体，使用单独初始化映射，列明允许复制的 key/shape、实际复制比例及新增参数；对遗漏的必需 key 报错，不能简单 `strict=False` 静默通过。不得带入旧 optimizer、训练步数或场景私有快状态。

预训练 backbone 的来源、版本、输入归一化和冻结策略写进 checkpoint。所有内部对照使用相同骨干和初始化。RGB-only 轨与 RGB+D 轨使用独立配置和 checkpoint；不能复用深度训练 V5 或 depth teacher 冒充严格 RGB-only。

现有工程不是 Git 仓库，本设计没有虚构 commit。实现前应以明确的版本管理/源文件哈希保存改动，冻结 V5 的 `MANIFEST.sha256` 保持 7/7 匹配。

配置也按责任拆分：数据划分/episode 协议、载体、写入规则、策略、预算、训练阶段、评估分别使用自己的配置类型。入口组装它们并保存完整的最终配置与哈希；模型内部不接收含文件路径、教师标签和评估配置的巨型字典。不用一个 `is_train` 开关同时改变标签读取、写入公式和动作选择。

每次运行保存 `resolved_config`、源文件哈希、checkpoint 哈希、split/episode 哈希、允许使用的标签与预训练来源、随机种子和阶段名称；逐步轨迹与逐场景指标分文件保存。聚合报告只读逐场景结果，不重新挑选成功帧。模型 checkpoint、动作教师样本和测试指标各用独立目录，防止下一阶段把错误产物当输入。

## 11. 实施顺序与每步验收

每一步都要产生可运行的小交付，再增加下一层；第一版不同时实现所有外部模型或大型训练框架。

| 顺序 | 新增/涉及文件 | 可独立验收的交付 | 必测失败情形 |
| --- | --- | --- | --- |
| 1 | `data/episodes.py`、`dynamic/types.py`、`cache.py`、`evaluation/sealed_queries.py` | 小型真实episode分为在线流与独立query；无模型也可测试访问顺序 | query/未来/深度读取注入立即失败；跨scene复用被拒绝 |
| 2 | `dynamic/lifting.py`、`carrier.py`、`config.py` | 新载体 OFF：从已见图像产生一个可反复读出的 SceneState | query置换不改状态；高置信错误位置仍有负density修正能力 |
| 3 | `dynamic/write_rule.py`、`checkpoint.py` | 两个写入位置可分别更新，ALL同步提交 | 修改非选中位置、慢参数被写、跨episode alias、旧版本proposal重放 |
| 4 | `dynamic/feedback.py`、`budget.py`、`runner.py`、固定策略 | 4+8+4 真实episode完整跑通，seal后独立评分 | 先吸收当前图再算pre误差、OFF不追加观测、最后一步未重读 |
| 5 | `training/grounded.py`、`write_unroll.py` | 小规模共享状态训练和固定写入训练 | 短轨迹梯度断开、训练与部署公式不同、对照预算不匹配 |
| 6 | `training/action_teacher.py`、`policy_fit.py`、学习策略 | 训练集离线收益标签与前缀policy | 未来信息进入policy特征、分支相互污染、oracle冒充部署动作 |
| 7 | `evaluation/streaming_report.py`、各脚本/配置 | 同一协议下OFF/固定/动态与逐scene指标 | 逐帧伪增样本、隐藏materialize成本、使用诊断集作最终测试 |

首轮测试命令采用项目 `.venv/Scripts/python.exe -m pytest`，临时路径限制在 `outputs`。新增测试按模块运行，旧 V5 兼容性测试及 7 项文件哈希检查作为回归门。

模块拆分以责任为单位，不要求每个函数单独一个文件：只有在线调度留在 `runner.py`，训练循环不放进模型 `forward`，文件读取不放进策略，下载器不导入模型包。第一批只实现第 1 步的数据边界，再接第 2 步载体，避免一开始就维护全部训练脚本。

## 12. 最小防错矩阵

- **数据**：打乱 query 标签或替换未来图像，已见前缀的动作/快状态/场景状态保持不变；修改已见上下文，应能改变状态。
- **状态**：不同 episode 的快权重不共享存储；reset 回到同一个慢 checkpoint 和零快权重。
- **动作**：OFF 不写快权重；FUSE/COMPLETE只写对应位置；ALL候选内部计算顺序不影响同步结果。
- **公式**：外积shape、支持数门槛、固定采样包、单步裁剪、零支持无增量都被断言。
- **读取**：query读出前后封存哈希一致；新增观察后必有materialize；最后一次写入真实影响最后状态。
- **预算**：所有方法记录编码、反馈、策略、候选forward、外积和materialize的实际次数；预算不足只收缩动作集合。
- **训练**：有/无展开梯度的路径明确；部署没有慢参数梯度和optimizer step。
- **比较**：载体修正和两个可写MLP同时存在于新OFF/固定/动态组，固定策略至少包含FUSE、COMPLETE、ALL；V5单独作为历史参考。

这些测试验证软件与信息边界。方法是否提高几何精度、是否存在有效轨迹和是否达到投稿要求，仍需实际实验。

## 13. 当前下载与本设计的关系

用户已授权下载26个完整最终测试ZIP，总计102,405,935,184字节。下载目标为 `data/hypersim_final_holdout_archives`，与训练prepared目录分开；本轮不解压、不评分、不将它们加入训练采样器。

`scripts/download_hypersim_holdout_archives.py` 是独立的数据运维工具；采用断点续传、远端大小及ETag/Last-Modified核对、逐包ZIP目录检查和本地SHA256记录。进度文件位于 `outputs/hypersim_holdout_full_download_20260918/status.json`。本地SHA256用于后续文件一致性，不冒充官方校验和。

模型架构代码没有在本轮按这份设计改写；本轮交付是实际下载任务与上述可审查的模块设计。

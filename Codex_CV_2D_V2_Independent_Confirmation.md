# Codex 提示词：二维动态写入 V2——样本依赖性、任务相关信号与独立确认

你正在处理 measurement-scene-state 仓库。请在当前代码基础上执行一个新的、独立存档的 V2 实验，不重写架构，不覆盖任何 V1 产物。

本轮目的不是追求正结果，而是区分三个问题：

H1a：当前写入家族相对 OFF 的任务收益机会，能否在此前未观察的新序列上复现？
H1b：逐样本最优轨迹是否超过全部样本共用的事后最优全局轨迹，而不仅仅超过四条恒定轨迹？
H2：保持写入规则不变，增加一个预先限定的部署可见任务信号及保守拒写机制，能否获得独立测试净收益？

本轮只提出一个新信号：source-mask round-trip consistency。它是假设，不是已经验证有效的指标。

重要：不能先查看新测试集的 H1 结果，再调整方法并在同一测试集验证 H2。所有方法、超参数、阈值和主要比较必须在首次读取新测试 target GT 前锁定。

## 0. 先审计现状，再写代码

阅读并核对实际存在的文件：
- README.md、ROADMAP.md、EXPERIMENTS.md；
- docs/experiments/EXP-2D-20260921-corrected/；
- outputs/EXP-2D-20260926-opportunity-selector-v1/，以及对应的 docs 报告目录；
- V1 的 README、STATUS、config、split_manifest、preregistration、integrity、raw、selector artifacts、signal_analysis、oracle_analysis、rank_analysis；
- src/mcss/vision_probe/adaptation.py、experiment.py，以及现有相关测试。

路径不存在时按文件名和实验 ID 查找，不得伪造文件或根据 README 想象实现。

记录 git commit、dirty diff、环境、模型/特征缓存/PCA/decoder/数据划分/selector artifact 的哈希。未提交改动也必须存档。未经要求不得 push、删除文件或覆盖旧结果。

V1 报告的参考值：448 pixel-space J，16 sequences、32 pairs、每 pair 16 条轨迹；OFF≈0.552677，constant oracle≈0.554378，dynamic oracle≈0.555129；dynamic−constant≈0.000750；visible selector−OFF≈−0.000470。原始精度以 raw 为准，不要用四舍五入后的差异误报错误。

V1 数字仅用于核验，不是要求 V2 必须复现的性能目标。旧 validation 已被观察，不能当成新 holdout。README 声称 PASS 不能替代独立 integrity 和 artifacts 检查。

创建唯一新目录：
outputs/EXP-2D-opportunity-selector-v2-<实际时间戳>/
docs/experiments/EXP-2D-opportunity-selector-v2-<实际时间戳>/

## 1. 冻结本轮不变项

保持以下内容与 V1 一致：
- DINOv2 backbone、448 输入及真实 32×32 feature grid；
- 64-d PCA/whitening、冻结 RGB decoder，优先直接复用并验证 V1 fit artifacts；
- A/B 快速状态、OFF/A/B/ALL 动作、两步共 16 条轨迹；
- 写入公式、步长、clip、初始化、状态更新语义、candidate isolation；
- 每个 target 独立重置，不宣称跨帧持续学习；
- pixel-space J 为 primary，原有近似 F/JF 为 secondary；
- object → target slots → sequence 的分层等权聚合；
- source-mask 使用约定、mask readout、缺失/空目标的既有处理规则。

不得增加 backbone、SAM/flow 模型、长轨迹、复杂神经控制器、rank 扩展或三维实验。本轮不重复大规模 rank 审计，也不声称 rank≤3 已被证明是性能瓶颈。

## 2. 先用封存 raw 补齐“真正样本依赖性”诊断

这一步只后处理 V1 已保存结果，不重新执行旧 validation，不改变其指标或阈值，不用旧 validation 重新训练 selector。

记 J_i(tau) 为 pair i 的对象平均 J，所有总体均值都使用原有 sequence 等权协议。

计算并严格区分：
1. OFF/OFF。
2. Discovery-fixed16：只在 discovery 上从 16 条轨迹中选一条，供所有测试样本使用。
3. Constant oracle：每 pair 从 OFF/OFF、A/A、B/B、ALL/ALL 中选最优。
4. Dynamic oracle：每 pair 从 16 条轨迹中选最优。
5. Hindsight-global16：在被分析样本上，事后寻找总体均值最高的同一条轨迹。

Hindsight-global16 使用测试答案，只是离线诊断，不是合法部署基线，不能反向用于 selector 训练或阈值选择。

至少报告三个不同差值：
- write_opportunity = dynamic_oracle − OFF；
- nonconstant_extra = dynamic_oracle − constant_oracle；
- sample_dependence_gap = dynamic_oracle − hindsight_global16。

第三项才直接刻画相对所有样本共用同一条轨迹，逐样本选择增加了多少空间。

再报告：
- 最优全局轨迹及全部并列最优者；
- 每条全局轨迹到 pair oracle 的 regret；
- 严格超过全部 constant 的 pair/sequence 数；
- 完整最优轨迹集合，不能只看任意 tie-break 产生的动作熵；
- 是否有一条轨迹属于所有 pair 的最优集合；
- 上述三个 gap 的 sequence contributions、top-1/top-3 share 和 leave-one-sequence-out。

对 sample_dependence_gap 做 bootstrap 和 leave-one-sequence-out 时，必须在每个重采样/删一子集中重新计算 hindsight-global16。固定完整样本上的 winner 会改变统计量，不得混淆。

轨迹非恒定不等于顺序因果效应：OFF→ALL 与 ALL→ALL 的更新次数不同。除非有动作多重集合和更新次数匹配的对照，否则只称 nonconstant-trajectory opportunity，不能称已证明 order mechanism。

若 raw 不含必要的 candidate 行，明确 RAW_INCOMPLETE，并列出缺项；不得用聚合 README 伪造逐 pair 数据。

## 3. 新独立数据：先证明独立，绝不自动拆封旧保留集

先检查当前已获授权可用的数据，生成 discovery_data_audit.json 和 independent_data_audit.json。

合格的新确认序列必须：
- 未用于 V1/V2 fit、discovery、旧 validation、表示拟合、selector 拟合、阈值选择或人工查看任务结果；
- 不能通过同一视频换几个 frame 伪装成独立新序列；
- 与历史集合进行 sequence/video ID、来源、图像精确哈希检查，并尽可能补近重复检查；不确定项标记 UNVERIFIED；
- 来自明确可用、许可与访问权限允许的数据源；不得假造下载地址或访问受限数据；
- 与既有 source-annotated mask-transfer 任务兼容。

本提示词不授权打开现有 reserve 或 DAVIS 官方 validation，也不授权付费、绕过许可或大规模不受控下载。

若历史划分已覆盖所有可用 DAVIS train 序列，就如实记录，不能重新命名旧序列当新数据。

若发现可用的未曝光外部数据，记录其来源和域差异：外部数据上的实验属于跨数据集确认，不写成原分布内重复验证。

固定采样方案：目标 32 个新序列、每序列 2 个 target pair；这是资源规划，不是统计充分性的保证。根据实际可用数据，在打开 target GT 内容前锁定 N、序列列表、source/target frame 和种子。不足目标数时如实记录，不得看显著性再决定补样。

选帧只能根据帧索引及标注可用性元数据，不能根据 target mask 大小、模型分数、motion 难度或预期收益筛选。缺失数据的处理规则也须提前锁定。

新测试 target GT 在所有 selector 预测、动作决策和 predictions manifest 写入并校验哈希之后，才允许进入单独的 evaluator。

若没有真正独立且获授权的新序列：
- 独立确认状态必须为 BLOCKED_NO_INDEPENDENT_DATA；
- 继续完成旧 raw 诊断、discovery 实验、测试、配置和可复现运行命令；
- 不生成伪造的确认表，不把旧 validation 重评包装成泛化，不自动开启 reserve；
- 最终明确缺少的是哪类数据，不能宣称全部科学目标完成。

## 4. 只新增一个部署可见信号：source-mask round-trip consistency

先确认既有任务在部署时是否允许 source image 的标注 mask。只有该约定真实成立，才实现下述信号。

这将是 source-annotated setting，不得宣传为无标注或纯无监督选择器。target GT 仍然完全禁止进入选择器。

对每条候选轨迹 tau：
- F_s：固定 source 特征；
- F_t^tau：候选更新后的 target 特征；
- Y_s：部署时允许使用的 source 对象 mask/概率；
- P_(t←s)：使用既有匹配规则得到的 source 到 target 对应；
- P_(s←t)：独立按同一冻结规则计算的反向对应。

构造：
Yhat_t = P_(t←s) Y_s
Yhat_s = P_(s←t) Yhat_t

定义 source 上的前景对象平均 soft-IoU cycle loss：
L_cycle(tau) = 1 − mean_object softIoU(Yhat_s, Y_s)
f_cycle(tau) = L_cycle(OFF/OFF) − L_cycle(tau)

只新增 f_cycle 这一个预先限定的特征，避免扩展出大量事后候选信号。

使用现有匹配规则和已冻结的 temperature/normalization；如现有实现是 hard matching，保留其对应语义，并明确 cycle 的实现。不得利用新测试表现选择 soft/hard matching、温度、mask 阈值或对象权重。

要求：
- 两个方向都由特征对应得到，不能直接构造单位映射或通过 GT 强行闭环；
- 不能读取 target mask、target J/F、oracle index 或文件名中的标签信息；
- source 与 target 必须是不同帧；
- 背景不能淹没前景对象平均；
- 缺失 source 标注时明确 UNSUPPORTED，不得用 target mask 替代；
- cycle score 高不等于 correspondence 正确，只作为待验证预测信号。

保存各候选的 L_cycle、f_cycle、现有 visible features，以及独立评价侧的 reward label。feature 构建 API 与 reward 构建 API 必须分离。

## 5. 两个新 selector，区分“拒写”与“新信号”

保留且不修改 V1 RGB selector 和 V1 visible selector，作为历史冻结比较方法；artifact 缺失就标记，不得在新测试上重建参数。

实现：

V2-GateOnly：
- 使用 V1 实际存在且部署合法的可见特征；
- 简单 ridge 预测每个 candidate 相对 OFF 的任务收益；
- 加明确的拒写门控。

V2-CycleGate：
- 相同特征，加唯一新增的 f_cycle；
- 相同模型族、训练协议、超参数网格和门控方式；
- 它是预先指定的主要 H2 方法，不能看测试后改选 GateOnly 为“主方法”。

预测目标：
r_i(tau) = J_i(tau) − J_i(OFF/OFF)

OFF 作为固定零收益的回退，不参与非 OFF 候选的最大值预测竞争：
1. 在其余 15 条轨迹中选择预测收益最大的 tau*；
2. predicted_reward(tau*) > threshold 时选择 tau*；
3. 否则选择 OFF/OFF。

阈值的“保守”只表示预设拒写策略，不是已获得统计安全保证。

预先固定最终回退规则：对每个 V2 方法，若其 discovery 外层 OOF 的 sequence 等权净 ΔJ≤0，则最终部署策略冻结为 ALWAYS_OFF；若>0，才按第 6 节拟合完整 discovery 模型。分别报告回退前的外层 OOF 结果和最终策略，不能把使用外层结果做出的回退选择再包装成独立性能估计。不要为产生正例强迫写入，也不要把 ALWAYS_OFF 与 OFF 持平称为 H2 成功。

## 6. 训练和超参数：只使用 discovery，严格按 sequence 分组

监督标签只能来自 discovery 的 target GT。这是 discovery-supervised selector，而不是推理时使用测试答案。

候选网格预先限定：
- ridge alpha = [0.1, 1, 10, 100]；
- abstention threshold = [0, 0.0005, 0.001, 0.002, 0.005]；
- 另有 ALWAYS_OFF 候选。

阈值用 J 的 0..1 单位，不是百分比。例如 0.001 对应 0.1 个百分点。

在 discovery 内做 nested leave-one-sequence-out：
- 外层留出一个完整 sequence，估计选择流程的泛化行为；
- 内层只在外层训练序列中选择 alpha 和 threshold；
- 同一 sequence 的全部 pair 和 16 个 candidates 必须在同一折；
- 标准化仅拟合训练折；
- 每个 sequence 总权重相等，避免 candidate 行多的序列主导；
- 内层目标为 sequence 等权的实际策略净 ΔJ，而不只是 candidate reward 的拟合误差；
- 相同目标值时，按预先固定的规则选择更低写入率、更高 threshold、更强正则。

完成流程评估后，使用全部 discovery 按同一分组调参规则选择最终配置并拟合。外层表现和最终 tuning score 分开报告，不能把调参最优分数当独立验证结果。

所有 artifact 在新测试 target GT 读取前冻结。旧 validation 的回顾性特征排序不得用于重新选择特征、符号、alpha 或阈值。

## 7. 正式确认：只评估一次，不根据结果改方法

在新独立数据上统一评估：
- OFF/OFF；
- Discovery-fixed16（从 discovery 全部 16 条轨迹选定）；
- V1 RGB selector（可恢复时）；
- V1 visible selector（可恢复时）；
- V2-GateOnly；
- V2-CycleGate（主要 H2 方法）；
- per-pair constant oracle；
- per-pair dynamic oracle；
- hindsight-global16（仅诊断）。

先跑不读 target GT 的全部预测，保存并锁定；再由独立 evaluator 读取 GT、评价全部候选并计算 oracle。

完成后不调整特征、标签阈值、pair 选择或方法定义。真实 bug 需记录影响范围、失效结果和修复；已看过标签的数据不能因此重新称为未见 holdout。

## 8. 统计和决策指标

主要统计单位是 sequence。所有方法比较使用相同样本的 paired sequence bootstrap，10,000 次，固定 seed，保留序列内全部 pair。pair bootstrap 只能补充。

主要 H2 比较提前指定：
1. V2-CycleGate − OFF；
2. V2-CycleGate − Discovery-fixed16。

同时提供 95% 描述性 CI。若对“同时胜过两个基线”作确认性结论，为两个比较额外提供预先指定的 Bonferroni 保守区间（各 97.5% 双侧区间），不靠测试后挑一个有利比较。

GateOnly/V1/F/JF/分桶/相关性是次要或探索性分析，明确多重比较限制。

Oracle 方面保存第 2 节全部指标。oracle 包含 OFF、dynamic 包含 constant，故非负有构造性；oracle CI 用于描述机会分布，不解释为部署有效或因果机制的证明。

每个 selector 至少报告：
- mean J、ΔJ vs OFF/fixed、CI；
- write/OFF rate；
- improved/tied/worse pair 和 sequence 数；
- harmful-write rate，明确分母是写入 pair；
- beneficial precision/recall，明确标签和零分母规则；
- positive capture、harmful loss、net capture；
- regret to dynamic oracle；
- per-sequence contributions 和 leave-one-sequence-out。

capture 统一采用与主指标一致的分层权重：
positive_gain = weighted_mean max(selector_J − OFF_J, 0)
harmful_loss = weighted_mean max(OFF_J − selector_J, 0)
net_gain = positive_gain − harmful_loss
oracle_gain = weighted_mean(dynamic_oracle_J − OFF_J)
positive_capture = positive_gain / oracle_gain
net_capture = net_gain / oracle_gain

oracle_gain=0 时 ratio 记 NA，不填 0 或 1。positive capture 不能掩盖负净收益。

冻结数值并列容差，并额外报告 ≥0.5、≥1、≥2 个百分点的收益计数。tiny-object/序列贡献只能用于敏感性分析，不能事后删除样本改善主结果。

## 9. 成本必须如实记录

统计：backbone extraction、候选 proposal、A/B updates、双向 correspondence、cycle signal、selector inference 的时间、调用次数和峰值显存；说明共享缓存与重复计算。

候选试算后选择 OFF，也已经付出试算成本。不得把低写入率宣传为低推理成本。没有 matched-budget baseline 时，不宣称效率优势。

## 10. 测试和审计

新增测试至少覆盖：
- OFF 状态不变、candidate rollback/isolation、同 seed 可复现；
- 所有 target slots 和对象正确进入分层聚合；
- 手工 toy data 验证 constant oracle、dynamic oracle、hindsight-global16 不混淆；
- bootstrap/LOSO 内重新求全局最优轨迹；
- nested CV 的 sequence 隔离、scaler 仅训练折拟合；
- selector 和 feature API 不接受 target GT/reward/oracle；
- 新测试预测 manifest 锁定前，target mask 访问次数必须为 0；
- source-mask cycle 正确处理方向、前景平均、source≠target 和退化输入；
- ALWAYS_OFF 正确返回基线，所有 ratio 零分母明确；
- reserve/官方 validation 访问被拒绝；
- 新旧序列重叠或独立性不明时不得进入确认性结果；
- 仅从封存 raw 重生成分析不会重新读数据、训练或改选择决策。

运行完整 test suite。保留失败与重试日志，不掩盖原生崩溃等问题，不通过关闭审计来得到 PASS。

## 11. 交付物

保存：
README.md、STATUS.md、config.json、preregistration.json、environment.json、git_commit.txt、dirty.patch、split_manifest.json、independent_data_audit.json、feature_contract.json、selector_lock.json、predictions_manifest.json、integrity.json、tests.json、raw_results、per_pair_results.csv、per_sequence_results.csv、global_trajectory_diagnostic.json、nested_cv_results.json、selector_analysis.json、bootstrap_results.json、cost_analysis.json、commands.sh、regenerate.sh。

来源 mask、target GT 的读取日志分开保存。README 中所有数字可从 raw 重生成。

新独立数据不可用时，仍交付已完成的 raw 诊断、discovery 结果、实现、测试和待运行命令，并明确其非确认性边界；不用 NA 替换成虚构数字。

## 12. 最终结论不得超出证据

分别回答：
1. V1 的逐样本 oracle 相比 hindsight-global16，还剩多少真正样本依赖空间？
2. 新独立序列上，写入机会和非恒定机会是否保持，量级多大？
3. 收益是否仍跨多个序列，而非某个 tiny target 或单个视频？
4. Cycle 信号相对原有信号，是否在独立数据上增加净收益？
5. GateOnly 是否主要通过少写降低损害？
6. CycleGate 是否胜过 OFF，并进一步胜过 discovery-fixed16？
7. 若失败，证据只支持当前 selector 未成功，还是连候选家族的实用 headroom 也很小？
8. 哪些结论仍受测试域、样本量、全候选试算、source 标注假设或历史数据曝光限制？

不能从 V1/V2 selector 失败推出所有部署可见信号都不可能有效；也不能从微小 oracle gap 推出已经得到实用方法。

最后打印：

============================================================
EXP-2D V2: INDEPENDENT OPPORTUNITY + TASK-AWARE SELECTION
============================================================
PRIMARY_METRIC=PIXEL_J
PRIMARY_RESOLUTION=448
DATA_INDEPENDENCE_STATUS=
EVALUATION_DOMAIN=
N_NEW_SEQUENCES=
N_NEW_PAIRS=
OLD_RAW_GLOBAL16_DIAGNOSTIC=
OFF_J=
DISCOVERY_FIXED16_J=
HINDSIGHT_GLOBAL16_J=
CONSTANT_ORACLE_J=
DYNAMIC_ORACLE_J=
ORACLE_MINUS_OFF=
DYNAMIC_MINUS_CONSTANT=
ORACLE_MINUS_HINDSIGHT_GLOBAL16=
GATE_ONLY_J=
CYCLE_GATE_J=
CYCLE_GATE_MINUS_OFF=
CYCLE_GATE_MINUS_FIXED=
CYCLE_GATE_MINUS_GATE_ONLY=
CYCLE_GATE_WRITE_RATE=
CYCLE_GATE_HARMFUL_WRITE_RATE=
CYCLE_GATE_POSITIVE_CAPTURE=
CYCLE_GATE_NET_CAPTURE=
PRIMARY_POLICY_ALWAYS_OFF=
TARGET_GT_READ_BEFORE_PREDICTION_LOCK=
RESERVE_TOUCHED=
DAVIS_OFFICIAL_VAL_TOUCHED=
CONFIRMATION_STATUS=
TESTS=
FINAL_INTEGRITY=
REPORT_DIR=
============================================================

现在先审计仓库和数据可用性，然后按上述顺序开发并运行可执行部分。遇到真正的数据或权限阻塞时保留已完成工作、明确标记阻塞；不得静默改变实验问题、打开保护数据集、制造结果或只写一份计划便宣称完成。

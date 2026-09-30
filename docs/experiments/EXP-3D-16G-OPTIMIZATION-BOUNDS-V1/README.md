# EXP-3D-16G-OPTIMIZATION-BOUNDS-V1

主配置为提前指定的 RGBD / 16³ / renderer64 / GT-free bounds / 10000 steps / FIXED_BUDGET。query AbsRel=0.361480，95% CI [0.254429, 0.485632]。OPTIMIZATION_STATUS=INCONCLUSIVE；BOUNDS_STATUS=NO_CLEAR_DIFFERENCE；CARRIER_TRAINING_READINESS=PROMISING_BUT_NOT_READY。

这是 17 个已暴露场景上的归因实验，不是新独立确认。30k 在正式运行前因成本排除，NOT_RUN；不根据 query 增加预算。BEST 字段只是 PREDECLARED_PRIMARY，不是事后最低 query。

| Bounds | Steps | Context AbsRel | Query AbsRel (95% CI) | Query−context gap |
|---|---:|---:|---|---:|
| CURRENT_BOUNDS | 1000 | 0.142544 | 0.402719，95% CI [0.290392, 0.529675] | 0.260175 |
| CURRENT_BOUNDS | 3000 | 0.130557 | 0.403912，95% CI [0.291145, 0.530170] | 0.273355 |
| CURRENT_BOUNDS | 10000 | 0.120814 | 0.406972，95% CI [0.298099, 0.529087] | 0.286158 |
| FROZEN_GT_FREE_BOUNDS | 1000 | 0.101663 | 0.363735，95% CI [0.258207, 0.485022] | 0.262072 |
| FROZEN_GT_FREE_BOUNDS | 3000 | 0.090477 | 0.357089，95% CI [0.251632, 0.481126] | 0.266612 |
| FROZEN_GT_FREE_BOUNDS | 10000 | 0.083607 | 0.361480，95% CI [0.254429, 0.485632] | 0.277873 |

**1. 1000 steps 是否明显 underoptimized？**

不能仅凭 objective 下降判断。GT-free 1k→10k query gain=0.002255，95% CI [-0.014427, 0.020248]；按完整预注册规则得到 INCONCLUSIVE。

**2. 3k / 10k / 30k 带来多少 query gain？**

1000→3000=0.006646，95% CI [-0.004043, 0.017444]；3000→10000=-0.004391，95% CI [-0.013359, 0.004686]。30k=NOT_RUN。两 bounds、两 selection 的全部相邻和首末差值见 bootstrap_results.json。

**3. context objective 下降时 query 是否继续改善？**

各相邻预算的 scene 方向计数：[{"from": 1000, "to": 3000, "both_flat": 0, "context_improves_query_improves": 9, "context_improves_query_worsens": 7}, {"from": 3000, "to": 10000, "both_flat": 0, "context_improves_query_improves": 8, "context_improves_query_worsens": 8}]。这些三类不是全部可能方向；完整每 scene 差值保存在统计 JSON。

**4. 是否出现 context overfit？**

预注册状态为 INCONCLUSIVE。只有 QUERY_OVERFIT 分类建立该项系统性证据；其他状态不排除个别场景 query 变差。

**5. 16³ 最大充分 budget 的 query AbsRel 是多少？**

0.361480，95% CI [0.254429, 0.485632]；最大已运行预算为10k，plateau scene fraction=0.000000。不能把最大已运行预算自动称为充分优化或全局最优。

**6. GT-free 是否稳定优于 current？**

10k CURRENT−GT-free=0.045493，95% CI [-0.030585, 0.141762]；正式分类 NO_CLEAR_DIFFERENCE，稳定收益同时要求均值≥.02、CI下界>0、≥75%场景不变差。

**7. bounds 收益是否依赖 optimization budget？**

收益差的差 (10k bounds gain − 1k bounds gain)=0.006508，95% CI [-0.016917, 0.028806]。判断基于 RGBD 同 role 同 seed 的比较。

**8. Query oracle 更多优化后能到多少？**

10k CURRENT=0.225905，95% CI [0.121294, 0.357272]；GT-free=0.148045，95% CI [0.064474, 0.274001]。两 query 共享 state；监督使用 query GT，只有诊断意义。

**9. Context-direct 与 oracle gap 是否缩小？**

GT-free 10k context−oracle=0.213435，95% CI [0.134787, 0.302639]；1k gap−10k gap=-0.018880，95% CI [-0.035303, -0.002529]。正数表示 gap 缩小，不单独证明因果机制。

**10. 主要剩余瓶颈是什么？**

证据标签：optimization=INCONCLUSIVE，bounds=NO_CLEAR_DIFFERENCE。CONTEXT_INFERENCE_BOTTLENECK=true；REPRESENTATION_OR_BOUNDS_BOTTLENECK=false。Oracle 工程门槛：均值=0.148045，15/17 个场景 AbsRel≤0.35。主 context-direct 尚未达到工程好结果门槛，而同一预定 GT-free/10k 的 query-supervised oracle 达到门槛，支持 context inference / visibility / context evidence 是剩余问题之一。应优先研究如何从合法 context 推断几何。这不是排他的因果定位，也未证明 representation/bounds 没有任何限制。 这两个 flag 仅按用户第24节和既有工程门槛派生，不改变冻结的三项主分类。

**11. 是否值得训练新的16³ carrier？**

PROMISING_BUT_NOT_READY；engineering-good=False。逐项资格检查：{"bounded_generalization_gap": false, "no_stable_late_query_harm": true, "not_single_scene": true, "optimization_saturated_or_partially": false, "scene_specific_state": true, "stable_carrier_gain": true}。READY_WITH_LIMITATION 明确表示尚未通过工程好结果门槛。

**12. 下一轮训练 carrier 还是先修 context inference？**

本轮不建议立即训练新 carrier。先按优化状态及 context−oracle gap 制定独立协议，检查优化充分性、context inference、visibility 与监督；不能用有限预算 oracle 的失败证明表示不可能。

辅助检查：frozen carrier−主配置=0.318346，95% CI [0.207337, 0.427231]；wrong-scene−正确 scene=0.206832，95% CI [0.155688, 0.251813]；query−context=0.277873，95% CI [0.190763, 0.380503]。

RGB-only secondary query AbsRel=0.615155，95% CI [0.529734, 0.697939]；RGB_ONLY−RGBD=0.253675，95% CI [0.189971, 0.320391]。这是主分析之后的固定配置诊断，不改变主决策。

统计单位是 scene：先平均 query，再平均 A/B，再场景等权；10,000 paired scene bootstrap，seed=20260927，95% percentile CI。per_scene、median、improved/tied/worse、LOSO、top1/top3 贡献保存在 bootstrap_results.json。两条 selection 均完整报告，CONTEXT_SELECTED 只使用预算内 full context objective，不能替换主配置。

Oracle CURRENT 的 joint_query seed 与 GT-free A/B seed 不配对；同 bounds 的 RGBD 与 QUERY_ORACLE 也因 track 不同而 seed 不同。Oracle 跨 bounds 差异不是纯 bounds 因果效果，也不参与 BOUNDS_STATUS 主判定。

固定 kernel 与 CPU regression 可以精确检查；CUDA 浮点复算不承诺 bitwise 相等，需报告实际容差与误差。状态 hash 完整性和数值复算容差是不同检查，不能混称。

成本按真实共享轨迹计一次，不能把不同 budget/selection 的累计时间再相加。四子进程共享 GPU，累计 worker seconds 不等于 phase wall time，也不可直接当独占 GPU 延迟。未来 carrier training 成本未实测。

主阶段（context + query oracle，119 条轨迹）累计 worker optimization 10939.038 秒；主阶段 phase wall-time 独立记录为 {"context_evaluation_seconds": 115.67976627498865, "context_seconds": 1631.5848608016968, "device": "cuda", "oracle_evaluation_seconds": 39.160654762992635, "oracle_seconds": 1189.9490562670399, "shared_gpu": true, "timing_basis": {"context": "earliest trajectory lock mtime to context completion marker mtime; excludes initial launcher setup", "oracle": "external perf_counter around optimization CLI; includes process setup"}, "worker_count": 4}。Secondary（34 条轨迹）累计 worker optimization 2063.288 秒；全阶段（153 条轨迹）累计 worker optimization 13002.326 秒；secondary phase wall-time 独立记录为 {"device": "cuda", "secondary_evaluation_seconds": 11.858088374021463, "secondary_seconds": 556.0444912919775, "shared_gpu": true, "timing_basis": "external perf_counter around CLI; includes process setup", "worker_count": 4}。每配置 seconds/state、total seconds、allocated/reserved VRAM 和 render latency 见 cost_analysis.json；所有七组曲线见 figures/。

FINAL_HOLDOUT_TOUCHED=false；NEW_CARRIER_TRAINED=false；DYNAMIC_TTT_RUN=false。没有新增模型训练、没有开启 holdout；direct-state 优化属于已授权归因实验。

复现命令见 commands.sh；raw、配置、锁定、访问日志、测试和完整性证据均在本实验目录，checkpoint 文件以本地 outputs 与发布索引为准。报告生成不加载任何媒体、模型或 checkpoint。

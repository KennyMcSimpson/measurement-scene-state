# EXP-3D-CONTEXT-OBSERVABILITY-SUPERVISION-V1

预先指定的主比较 S0−S3 query AbsRel gain=0.049372，95% CI [0.023642, 0.076404]。OBSERVABILITY_STATUS=MIXED；GEOMETRY_SUPERVISION_STATUS=SURFACE_DOMINANT。

本轮是17个 CAPACITY_DEV_EXPOSED 场景上的可观测性与几何监督归因，不是独立 validation、final test 或资格确认。BEST_SUPERVISION 固定指 S3（PREDECLARED_PRIMARY），没有根据 query 分数更换主方法。

保持16³、renderer64、旧 frozen GT-free bounds、10000 steps、Adam/LR/clip/RGB loss/rendered-depth loss/regularization、context/query IDs 不变。lambda_free=lambda_surface=0.1，tau_surface=0.5×||bounds extent/16||，epsilon=1e-8，均预先冻结，未按 query 调参。

| Variant | Context AbsRel | Overall query AbsRel (95% CI) | RGB MSE | RMSE | δ1 | Opacity | Coverage |
|---|---:|---|---:|---:|---:|---:|---:|
| S0 | 0.083606 | 0.361556，95% CI [0.254601, 0.485549] | 0.068237 | 2.966425 | 0.507299 | 0.776131 | 0.941176 |
| S1 | 0.083505 | 0.357400，95% CI [0.248408, 0.483846] | 0.067283 | 2.956593 | 0.520396 | 0.770818 | 0.941176 |
| S2 | 0.086492 | 0.313772，95% CI [0.203687, 0.441248] | 0.062269 | 2.850940 | 0.567740 | 0.817903 | 0.941176 |
| S3 | 0.086515 | 0.312183，95% CI [0.201723, 0.439961] | 0.062192 | 2.846055 | 0.569792 | 0.817131 | 0.941176 |

Coverage 是预测 opacity>1e-6 的比例，不等于真实表面可见性。下面区域标签是 query GT surface 与 context GT nearest-depth 一致性诊断，只有全部136 states 完成并 seal 后才计算。

| Region | Pixel fraction (scene equal) | S0 conditional AbsRel | Oracle conditional AbsRel | S0−oracle gap | Covered scenes / all | Empty scenes |
|---|---:|---|---|---|---:|---:|
| OBS0 | 0.297232 | 0.508010，95% CI [0.407460, 0.614531] | 0.174750，95% CI [0.084287, 0.299821] | 0.333260，95% CI [0.239141, 0.431207] | 17/17 | 0 |
| OBS1 | 0.275110 | 0.294431，95% CI [0.203123, 0.408603] | 0.141607，95% CI [0.055360, 0.270282] | 0.152824，95% CI [0.098434, 0.217489] | 17/17 | 0 |
| OBS2PLUS | 0.427658 | 0.288511，95% CI [0.167183, 0.442423] | 0.129473，95% CI [0.044983, 0.258881] | 0.159038，95% CI [0.075116, 0.285273] | 17/17 | 0 |

**1. query geometry 中多少属于 OBS0 / OBS1 / OBS2PLUS？**

见上表场景等权 pixel fractions；三类保留全部 GT-valid pixels，未因低可观测性删掉 query。OBS0 不等同于物理上绝对不可见，只表示该容差/采样近似下没有 visible support。

**2. 当前误差主要集中在哪种可观测性区域？**

总体 AbsRel 的可加区域贡献分别为 OBS0=0.185229，95% CI [0.104966, 0.276862]；OBS1=0.078905，95% CI [0.051572, 0.113134]；OBS2PLUS=0.097422，95% CI [0.066381, 0.132997]。贡献相加重建 overall，不能只凭某个小区域 conditional error 很高就说它主导总体误差。正式状态=MIXED。

**3. OBS2PLUS 上 context-direct 与 query oracle 还差多少？**

0.159038，95% CI [0.075116, 0.285273]。这是覆盖场景上的条件差值，空区域/低覆盖会限制分类；不把 A/B 或像素当独立 scene。

**4. triangulation angle 与 error 有什么关系？**

ANGLE_0_5: 0.256862，95% CI [0.152409, 0.387221]，scene coverage=0.941176；ANGLE_5_15: 0.213175，95% CI [0.128833, 0.331197]，scene coverage=1.000000；ANGLE_15_30: 0.175180，95% CI [0.122129, 0.228565]，scene coverage=0.823529；ANGLE_GT30: 0.282474，95% CI [0.119855, 0.537446]，scene coverage=0.647059。只对>=2 visible views计算；主 bins基于最大夹角[0,5],(5,15],(15,30],>30，median angle及bins另存。这里只描述关联，不推断视差因果。

**5. occluded regions 的 error 是否更高？**

OCCLUDED_ANY S0 conditional=0.325539，95% CI [0.243118, 0.432045]；CONFLICT_ANY=0.365629，95% CI [0.253262, 0.498244]；overall=0.361556，95% CI [0.254601, 0.485549]。这些是重叠区域的描述性比较，没有额外预注册的遮挡因果检验；不能把条件均值差自动解释为显著因果伤害。

**6. Free-space supervision 是否改善 query？**

S0−S1=0.004156，95% CI [-0.000492, 0.011804]；正值代表净总体收益，稳定 practical gain 还需均值≥.02、CI下界>0、≥75%场景不变差。

**7. Surface supervision 是否改善 query？**

S0−S2=0.047784，95% CI [0.020090, 0.075833]；使用同一冻结 practical gate，不能仅凭方向为正宣布有效。

**8. 两者联合是否互补？**

S0−S3=0.049372，95% CI [0.023642, 0.076404]；S1−S3=0.045216，95% CI [0.021312, 0.070889]；S2−S3=0.001588，95% CI [-0.000325, 0.005092]。预注册分类=SURFACE_DOMINANT；不把联合均值最低自动称为互补。

**9. 主 geometry supervision 缩小多少 context→oracle gap？**

S3 GAP_CLOSED=0.049372，95% CI [0.023642, 0.076404]；GAP_CLOSED_FRACTION=0.231239，比例CI=[0.11446780547280545, 0.3861702847307981]。比例是描述性 gap closure，不是恢复理论最优的百分比；bootstrap 分母非正的未定义抽样数=0。

**10. 改善来自 observed 还是 unobserved regions？**

OBS0: conditional gain=0.043509，95% CI [0.018290, 0.071561]，overall additive gain contribution=0.009485，95% CI [0.001928, 0.018163]；OBS1: conditional gain=0.056935，95% CI [0.030946, 0.084911]，overall additive gain contribution=0.017785，95% CI [0.008820, 0.027954]；OBS2PLUS: conditional gain=0.081946，95% CI [0.021340, 0.175153]，overall additive gain contribution=0.022103，95% CI [0.006743, 0.039781]。用有符号可加贡献判断总体来源；OBS0改善可能由regularization/prior/间接一致性引起，没有证据时不能称为从观察恢复了未见真值。

**11. 是否牺牲 context fit 但改善 query？**

S0 context=0.083606, query=0.361556; S1 context=0.083505, query=0.357400; S2 context=0.086492, query=0.313772; S3 context=0.086515, query=0.312183。同时检查上方paired query gain，不因context误差单独升高就拒绝方法，也不因context更好就接受。

**12. wrong-scene damage 是否仍然成立？**

S0=0.206738，95% CI [0.155870, 0.251627]；S3=0.274114，95% CI [0.215241, 0.331762]；S3 spatial shuffle damage=0.385853，95% CI [0.290513, 0.480739]；zero damage=0.687817，95% CI [0.560039, 0.798277]。所有差值为control−direct，recipient camera保持不变。Wrong-scene沿旧定义连同donor原bounds一起替换，因此敏感性不能完全归为density内容的独立贡献；S3 spatial shuffle保持recipient同bounds，补充空间结构证据。不能把CI跨0当scene-specific或空间结构成立。

**13. 当前问题更像 observability、supervision、completion、lifting 或混合？**

冻结证据分类：MIXED / SURFACE_DOMINANT。下一轮单独研究16³ geometry-aware carrier，并保留显式 ray free-space/surface 监督与独立验证；本轮不训练。 这些是本次归因条件下的研究方向，不是单一原因的排他证明。

**14. 下一版 carrier 应具体增加什么？**

下一轮单独研究16³ geometry-aware carrier，并保留显式 ray free-space/surface 监督与独立验证；本轮不训练。 GEOMETRY_AWARE_CARRIER_RECOMMENDED=True；VISIBILITY_AWARE_FUSION_RECOMMENDED=inconclusive。没有在本轮启动carrier训练。

区域总体贡献与条件均值是不同估计量：每scene先平均query和A/B的区域fraction、error contribution，再以贡献/fraction得到该scene条件均值。空scene的条件值为null，完整scene名单、coverage和empty counts保留；bootstrap每次重采样所有scene并重算有效覆盖分母。未对空区域填零误差，亦未静默删除场景。

low-support positive gap share={'definition': 'positive scene-region additive contributions; not signed net-gap share', 'positive_contribution_by_region': {'OBS0': 2.086658796573732, 'OBS1': 0.7265123110065906, 'OBS2PLUS': 0.8165165471452888}, 'share': 0.7750449557051449}。该量截取正gap后计算比例，不等于有符号净gap贡献。对应有符号区域gap见observability_analysis.json/signed_additive_gap_by_region。

visible support是投影在前/图内、有效context depth且ray-distance差≤0.05m+1%context depth的nearest-depth近似，不是表面可见性证明。occlusion与front-conflict分别保存，可在不同context views同时发生，不是可加分区。

free loss只作用于t<d−tau的sample，surface后方未知不标为空。surface loss为−log(band内rendering mass+eps)，有效hit但空band/miss仅从这个新增term排除并计数；旧任务loss和总体评价仍保留所有合法ray/pixel。surface目标band局部，但梯度通过transmittance还会影响band之前density。

S0–S3使用相同初始化、seed、ray minibatch stream和固定10k终态。full context objective仅用于诊断，query GT只在seal后标注区域；未用于optimizer、lambda、tau、bounds、checkpoint或scene选择。旧oracle是使用query GT的特权诊断，不是可部署方法，也不是数学能力上界。

BASELINE_REPRO_STATUS=PASS；新S0先完成再通过复现门控，S1–S3才允许启动。正常CUDA浮点复算不承诺bitwise；正式预锁AbsRel容差为scene mean .001、每row .01，实际差值见baseline_reproduction.json。CPU精确回归与CUDA容差检查、state文件/tensor SHA完整性各有不同含义，不能混称。

统计单位=scene, never pixels；draws=10000、seed=20260927、95% percentile CI。mean/median/CI/improved-tied-worse/LOSO/top1/top3贡献见bootstrap_results.json。像素不是独立统计样本。

| Variant | Worker optimization seconds | Seconds/state | Peak allocated bytes | Peak reserved bytes | Direct render seconds |
|---|---:|---:|---:|---:|---:|
| S0 | 5090.043 | 149.707 | 25203712 | 37748736 | 2.370 |
| S1 | 8167.066 | 240.208 | 26070016 | 39845888 | 2.333 |
| S2 | 28584.438 | 840.719 | 25279488 | 39845888 | 2.665 |
| S3 | 26424.146 | 777.181 | 25204224 | 37748736 | 2.462 |

136个state的worker optimization累计=68265.692秒；这是累计worker时间，不是墙钟总耗时。phase wall-time单独记录：{"parallel_workers": 4, "phases": {"baseline_evaluation_seconds": 4.733191018982325, "baseline_optimization_seconds": 1337.688599333982, "formal_evaluation_seconds": 132.37200595095055, "formal_optimization_seconds": 16201.069044506003, "observability_diagnostic_seconds": 51.95835704001365}, "shared_gpu": true, "timing_basis": "external perf_counter around CLI including process setup"}。四进程共享GPU，不能当作独占GPU benchmark；旧参考预测复用，未来carrier成本未实测。

七类曲线见figures/。完整raw、mask NPZ、配置/锁定、访问日志、测试、state/prediction hashes和复现命令保存在本实验目录；发布归档checkpoint索引指向本地outputs。

FINAL_HOLDOUT_TOUCHED=false；NEW_CARRIER_TRAINED=false；DYNAMIC_TTT_RUN=false。未打开保护集，未新训练learned carrier。

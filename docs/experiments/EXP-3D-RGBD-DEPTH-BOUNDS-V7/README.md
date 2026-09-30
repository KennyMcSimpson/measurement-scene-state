# EXP-3D-RGBD-DEPTH-BOUNDS-V7

单因素机制诊断：bounds 规则。三个变体都是 V6 的 C1 配方（冻结的 V5 C1 模型与 loss、72 个训练场景、6000 步、同 seed 初始化），训练和评价都用各自的 bounds 规则：C0 为冻结的相机射线先验规则，C1 为测得上下文深度与相机的外包盒，C2 为去掉 1% 分位尾部的外包盒（次要、不进入 gate）。全程 CPU。BOUNDS_GAIN=AbsRel(C0)−AbsRel(C1)=0.179014，95% CI [0.083609, 0.279266]；BOUNDS_STATUS=SUPPORTED；解读分支=A。

DEV 的 8 个场景已在 V2–V6 中被评价，本轮是 DEV 机制诊断；FRESH-V1 未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，另报不做选择的最后一步 DEV AbsRel。

| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |
|---|---:|---|---:|---:|
| C0 FROZEN_BOUNDS | 72 | 0.459393，95% CI [0.287300, 0.651353] | 0.461957 | 0.301420 |
| C1 DEPTH_BOUNDS | 72 | 0.280380，95% CI [0.173388, 0.396277] | 0.330440 | 0.205038 |
| C2 DEPTH_BOUNDS_TRIM1 | 72 | 0.264993，95% CI [0.145711, 0.392545] | 0.285342 | 0.196315 |

**1. 由测得深度确定 bounds，DEV 几何是否改善？**

BOUNDS_GAIN=0.179014，95% CI [0.083609, 0.279266]；BOUNDS_STATUS=SUPPORTED（V2 冻结主对比 gate）。

**2. 是否跨场景与 seed？**

improved/tied/worse=7/0/1；LOSO 范围=[0.14481456328734899, 0.2052410473149438]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。

**3. 不做 checkpoint 选择时差多少？**

最后一步（6000 步）DEV AbsRel={"C0": "0.461957", "C1": "0.330440", "C2": "0.285342"}；选中步={"C0": {"20260928": 6000, "20260929": 6000, "20260930": 5000}, "C1": {"20260928": 2000, "20260929": 2000, "20260930": 2000}, "C2": {"20260928": 3000, "20260929": 6000, "20260930": 4000}}。

**4. 状态是否场景专属、多视角是否优于单视角？**

C1 wrong-scene=0.363561，95% CI [0.180098, 0.591230]；shuffle=0.467476，95% CI [0.334173, 0.588275]；SCENE_SPECIFICITY_STATUS=SUPPORTED；3 视角相对 anchor=0.171290，95% CI [0.093336, 0.254793]；STATIC_DEV_STATUS=SUPPORTED。

**5. 覆盖率与体素大小怎么取舍（C2 对 C1）？**

TRIM_GAIN=AbsRel(C2 裁 1%)−AbsRel(C1 不裁剪)=-0.015386，95% CI [-0.039325, 0.008834]；正值表示覆盖率比体素大小更重要。仅描述。

**6. 与无几何常数相比如何？**

全部场景标签={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；排除 NO_HIT 场景后={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；C1_ABOVE_REFERENCE_ALL_SCENES=True；C1_ABOVE_REFERENCE_HIT_SCENES=True。

**7. 训练集拟合如何？**

最后100步训练深度AbsRel={"C0": "0.301420", "C1": "0.205038", "C2": "0.196315"}；学到的 σ={"C0": [0.853753, 0.799972, 0.809007], "C1": [0.977929, 0.960373, 0.92726], "C2": [0.912317, 0.933248, 0.896155]}。

**8. 区域诊断？**

C0−C1 条件 AbsRel：OBS0=0.151884，95% CI [0.038633, 0.274406]；OBS1=0.229585，95% CI [0.119363, 0.340301]；OBS2PLUS=0.184795，95% CI [0.072464, 0.298645]。

**9. 结论属于哪个预注册分支？**

分支A：由测得深度确定 bounds 让 RGB-D carrier 在全部 DEV 场景上稳定超过平凡基线。下一轮需要新的独立队列做资格验证（FRESH-V1 已用过一次）；仍不开 Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

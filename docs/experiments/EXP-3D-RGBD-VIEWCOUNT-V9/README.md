# EXP-3D-RGBD-VIEWCOUNT-V9

单因素机制诊断：训练时的视角数。三个变体都是 Core A 配方（冻结的 V5 C1 模型与 loss、32³ 网格、测得深度 bounds、72 个训练场景、6000 步、同 seed 初始化、相同的场景顺序与射线），只有训练上下文的视角数不同：C0 固定 3 个，C1 为 3 个上下文帧加 0–4 个均匀空闲帧（每步随机），C2 固定 7 个（次要、不进入 gate）。DEV 评价与 checkpoint 选择都用标准 3 视角上下文。全程 CPU。VIEWCOUNT_GAIN=AbsRel(C0)−AbsRel(C1)=0.001947，95% CI [-0.007638, 0.012622]；VIEWCOUNT_STATUS=NOT_ESTABLISHED；解读分支=B。

DEV 的 8 个场景已在 V2–V8 与 Core B V1 中被评价，本轮是 DEV 机制诊断；FRESH-V1、FRESH-V2 都未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，另报不做选择的最后一步 DEV AbsRel。

| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |
|---|---:|---|---:|---:|
| C0 FIXED3 | 72 | 0.259437，95% CI [0.140459, 0.403133] | 0.269161 | 0.204620 |
| C1 VARIABLE3TO7 | 72 | 0.257490，95% CI [0.139941, 0.401362] | 0.277366 | 0.191025 |
| C2 FIXED7 | 72 | 0.275281，95% CI [0.151133, 0.432509] | 0.287902 | 0.155803 |

**1. 训练时使用可变视角数，标准 3 视角 DEV 几何是否改善？**

VIEWCOUNT_GAIN=0.001947，95% CI [-0.007638, 0.012622]；VIEWCOUNT_STATUS=NOT_ESTABLISHED（V2 冻结主对比 gate）。

**2. 是否跨场景与 seed？**

improved/tied/worse=4/0/4；LOSO 范围=[-0.002206806743811945, 0.00461720677748708]；SEED_ROBUSTNESS=NOT_ESTABLISHED。

**3. 不做 checkpoint 选择时差多少？**

最后一步（6000 步）DEV AbsRel={"C0": "0.269161", "C1": "0.277366", "C2": "0.287902"}；选中步={"C0": {"20260928": 5000, "20260929": 3000, "20260930": 4000}, "C1": {"20260928": 4000, "20260929": 5000, "20260930": 4000}, "C2": {"20260928": 6000, "20260929": 3000, "20260930": 4000}}。

**4. 状态是否场景专属、多视角是否优于单视角？**

C1 wrong-scene=0.365573，95% CI [0.163060, 0.562776]；shuffle=0.532099，95% CI [0.389468, 0.656437]；SCENE_SPECIFICITY_STATUS=SUPPORTED；3 视角相对 anchor=0.108829，95% CI [0.014469, 0.229068]；STATIC_DEV_STATUS=SUPPORTED。

**5. 总用 7 视角训练（C2）与可变视角（C1）相比如何？**

FIXED7_GAP=AbsRel(C2 固定 7 视角)−AbsRel(C1 可变视角)=0.017791，95% CI [0.001149, 0.035466]；正值表示可变视角更好。仅描述。

**6. 与无几何常数相比如何？**

全部场景标签={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；排除 NO_HIT 场景后={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；C1_ABOVE_REFERENCE_ALL_SCENES=True；C1_ABOVE_REFERENCE_HIT_SCENES=True。

**7. 训练集拟合如何？**

最后100步训练深度AbsRel={"C0": "0.204620", "C1": "0.191025", "C2": "0.155803"}；学到的 σ={"C0": [1.00221, 0.957296, 0.937189], "C1": [1.02714, 0.970737, 0.948644], "C2": [0.99349, 1.028012, 0.984742]}。

**8. 区域诊断？**

C0−C1 条件 AbsRel：OBS0=0.006798，95% CI [-0.009832, 0.026815]；OBS1=-0.001569，95% CI [-0.009729, 0.004983]；OBS2PLUS=0.007085，95% CI [-0.011155, 0.030848]。

**9. 结论属于哪个预注册分支？**

分支B：可变视角训练有部分作用。可在 Core B 第二轮中检验它对流式观测的作用。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

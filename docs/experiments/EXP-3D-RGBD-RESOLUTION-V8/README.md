# EXP-3D-RGBD-RESOLUTION-V8

单因素机制诊断：体素网格分辨率。三个变体都是 V7 的配方（冻结的 V5 C1 模型与 loss、72 个训练场景、6000 步、同 seed 初始化、同一条 bounds 规则），只有网格不同：C0 为 16³，C1 为 32³，C2 为 24³（次要、不进入 gate）。每个体素都是稠密证据候选，共享参数与初始化完全相同。全程 CPU。RESOLUTION_GAIN=AbsRel(C0)−AbsRel(C1)=0.020942，95% CI [-0.017116, 0.051871]；RESOLUTION_STATUS=NOT_ESTABLISHED；解读分支=B。

DEV 的 8 个场景已在 V2–V7 中被评价，本轮是 DEV 机制诊断；FRESH-V1、FRESH-V2 都未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，另报不做选择的最后一步 DEV AbsRel。

| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |
|---|---:|---|---:|---:|
| C0 GRID16 | 72 | 0.280380，95% CI [0.173388, 0.396277] | 0.330440 | 0.205038 |
| C1 GRID32 | 72 | 0.259437，95% CI [0.140459, 0.403133] | 0.269161 | 0.204620 |
| C2 GRID24 | 72 | 0.244301，95% CI [0.148689, 0.343585] | 0.278813 | 0.199770 |

**1. 把体素网格从 16³ 提高到 32³，DEV 几何是否改善？**

RESOLUTION_GAIN=0.020942，95% CI [-0.017116, 0.051871]；RESOLUTION_STATUS=NOT_ESTABLISHED（V2 冻结主对比 gate）。

**2. 是否跨场景与 seed？**

improved/tied/worse=6/0/2；LOSO 范围=[0.012854508817871254, 0.03742389720745951]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。

**3. 不做 checkpoint 选择时差多少？**

最后一步（6000 步）DEV AbsRel={"C0": "0.330440", "C1": "0.269161", "C2": "0.278813"}；选中步={"C0": {"20260928": 2000, "20260929": 2000, "20260930": 2000}, "C1": {"20260928": 5000, "20260929": 3000, "20260930": 4000}, "C2": {"20260928": 3000, "20260929": 3000, "20260930": 4000}}。

**4. 状态是否场景专属、多视角是否优于单视角？**

C1 wrong-scene=0.352994，95% CI [0.150170, 0.554668]；shuffle=0.529256，95% CI [0.385016, 0.659598]；SCENE_SPECIFICITY_STATUS=SUPPORTED；3 视角相对 anchor=0.183122，95% CI [0.101177, 0.272109]；STATIC_DEV_STATUS=SUPPORTED。

**5. 分辨率的剂量-反应如何（C2 24³ 对 C1 32³）？**

GRID24_GAP=AbsRel(C2 24³)−AbsRel(C1 32³)=-0.015136，95% CI [-0.072024, 0.021662]；正值表示 32³ 优于 24³。仅描述。

**6. 与无几何常数相比如何？**

全部场景标签={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；排除 NO_HIT 场景后={"RAW:depth_absrel": "ABOVE", "RAW:depth_delta1": "ABOVE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "ABOVE", "MEDIAN_SCALED:depth_absrel": "ABOVE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；C1_ABOVE_REFERENCE_ALL_SCENES=True；C1_ABOVE_REFERENCE_HIT_SCENES=True。

**7. 训练集拟合如何？**

最后100步训练深度AbsRel={"C0": "0.205038", "C1": "0.204620", "C2": "0.199770"}；学到的 σ={"C0": [0.977929, 0.960373, 0.92726], "C1": [1.00221, 0.957296, 0.937189], "C2": [1.010078, 0.944417, 0.929783]}。

**8. 区域诊断？**

C0−C1 条件 AbsRel：OBS0=0.016994，95% CI [-0.010301, 0.043085]；OBS1=0.027634，95% CI [-0.025380, 0.079632]；OBS2PLUS=0.011741，95% CI [-0.126455, 0.108693]。

**9. 结论属于哪个预注册分支？**

分支B：分辨率有部分作用。下一轮检验更多训练数据或深度读出；仍不开 Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

# EXP-3D-RGB-PRIOR-BOUNDS-V10

单因素机制诊断：RGB-only 轨道上 GT-free bounds 的深度先验。三个变体都是 V5 C0 的 RGB-only 配方（不含深度旁路的稠密 carrier、冻结的 V2 表面 loss、72 个训练场景、6000 步、与 V5–V9 相同的逐 seed 初始化），测试时只输入 RGB 与相机。bounds 都按冻结的 GT-free 规则由上下文相机射线投到先验的近、远距离确定，只有先验不同：C0 用冻结的 V2 先验（3 个训练场景 12 帧拟合），C1 用 TRAIN72 全部帧拟合的先验，二者都是 32³；C2 为 C1 的 16³ 版本（次要、不进入 gate）。全程 CPU。PRIOR_BOUNDS_GAIN=AbsRel(C0)−AbsRel(C1)=0.142302，95% CI [0.022405, 0.261499]；PRIOR_BOUNDS_STATUS=PARTIAL；解读分支=B。

DEV 的 8 个场景已在 V2–V9 中被评价，本轮是 DEV 机制诊断；FRESH-V1、FRESH-V2 都未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，另报不做选择的最后一步 DEV AbsRel。

| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |
|---|---:|---|---:|---:|
| C0 FROZEN_PRIOR_GRID32 | 72 | 0.602888，95% CI [0.449726, 0.755017] | 0.619468 | 0.478809 |
| C1 TRAIN72_PRIOR_GRID32 | 72 | 0.460586，95% CI [0.369435, 0.530782] | 0.542961 | 0.424105 |
| C2 TRAIN72_PRIOR_GRID16 | 72 | 0.466877，95% CI [0.362679, 0.546065] | 0.531067 | 0.378759 |

**1. 改用 TRAIN72 拟合的先验确定 bounds，RGB-only 的 DEV 几何是否改善？**

PRIOR_BOUNDS_GAIN=0.142302，95% CI [0.022405, 0.261499]；PRIOR_BOUNDS_STATUS=PARTIAL（V2 冻结主对比 gate）。

**2. 是否跨场景与 seed？**

improved/tied/worse=7/0/1；LOSO 范围=[0.09697280356095735, 0.1910075876174988]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。

**3. 不做 checkpoint 选择时差多少？**

最后一步（6000 步）DEV AbsRel={"C0": "0.619468", "C1": "0.542961", "C2": "0.531067"}；选中步={"C0": {"20260928": 3000, "20260929": 1000, "20260930": 1500}, "C1": {"20260928": 500, "20260929": 1000, "20260930": 500}, "C2": {"20260928": 500, "20260929": 1000, "20260930": 200}}。

**4. 状态是否场景专属、多视角是否优于单视角？**

C1 wrong-scene=0.012004，95% CI [-0.021395, 0.041488]；shuffle=0.048244，95% CI [0.000831, 0.098012]；SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED；3 视角相对 anchor=0.035904，95% CI [-0.012838, 0.089453]；STATIC_DEV_STATUS=NOT_ESTABLISHED。

**5. TRAIN72 先验下 16³ 与 32³ 差多少（C2 对 C1）？**

GRID16_GAP=AbsRel(C2 16³)−AbsRel(C1 32³)=0.006291，95% CI [-0.016436, 0.030911]；正值表示 32³ 优于 16³。仅描述。

**6. 与无几何常数相比如何？**

全部场景标签={"RAW:depth_absrel": "NOT_DISTINGUISHABLE", "RAW:depth_delta1": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_absrel": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_delta1": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_absrel": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；排除 NO_HIT 场景后={"RAW:depth_absrel": "NOT_DISTINGUISHABLE", "RAW:depth_delta1": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_absrel": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_delta1": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_absrel": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；C1_ABOVE_REFERENCE_ALL_SCENES=False；C1_ABOVE_REFERENCE_HIT_SCENES=False。

**7. 训练集拟合如何？**

最后100步训练深度AbsRel={"C0": "0.478809", "C1": "0.424105", "C2": "0.378759"}；射线命中比例（场景平均）={"C0": "0.875000", "C1": "1.000000", "C2": "1.000000"}。

**8. 区域诊断？**

C0−C1 条件 AbsRel：OBS0=0.188923，95% CI [0.092125, 0.284930]；OBS1=0.101594，95% CI [-0.071189, 0.277424]；OBS2PLUS=0.124855，95% CI [-0.022579, 0.285552]。

**9. 结论属于哪个预注册分支？**

分支B：bounds 先验有部分作用，但 RGB-only 仍未稳定超过平凡基线。下一轮检验光度证据（分辨率或多视角匹配）；仍不开 Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

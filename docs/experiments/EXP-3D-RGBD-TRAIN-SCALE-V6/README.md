# EXP-3D-RGBD-TRAIN-SCALE-V6

单因素机制诊断：训练规模。C0 与 C1 都是冻结的 V5 C1 配方（稠密 16³ carrier + 零初始化测得深度旁路 + 表面监督），同 seed 初始化、同 6000 步预算；唯一差别是训练场景 24 个对 72 个（新增 48 个 TRAIN-X，与 DEV、FRESH-V1 volume 不重叠）。C2 为只用 RGB 的 V5 C0 配方在 72 个场景上训练，次要、不进入 gate。全程 CPU。SCALE_GAIN=AbsRel(C0)−AbsRel(C1)=0.032788，95% CI [-0.007950, 0.083785]；SCALE_STATUS=NOT_ESTABLISHED；解读分支=C。

DEV 的 8 个场景已在 V2–V5 中被评价，本轮是 DEV 机制诊断；FRESH-V1 未使用。checkpoint 仍按 V2 冻结的 DEV 规则选择，因此另报不做选择的最后一步 DEV AbsRel。

| Variant | 训练场景 | Query AbsRel (95% CI) | 最后一步 DEV AbsRel | 训练末100步深度AbsRel |
|---|---:|---|---:|---:|
| C0 DEPTH_TRAIN24 | 24 | 0.492181，95% CI [0.331836, 0.674023] | 0.494604 | 0.194699 |
| C1 DEPTH_TRAIN72 | 72 | 0.459393，95% CI [0.287300, 0.651353] | 0.461957 | 0.301420 |
| C2 RGB_TRAIN72 | 72 | 0.602048，95% CI [0.461856, 0.753296] | 0.611081 | 0.443003 |

**1. 训练场景从 24 增加到 72，DEV 几何是否改善？**

SCALE_GAIN=0.032788，95% CI [-0.007950, 0.083785]；SCALE_STATUS=NOT_ESTABLISHED（V2 冻结主对比 gate）。

**2. 是否跨场景与 seed？**

improved/tied/worse=5/1/2；LOSO 范围=[0.011493342300750806, 0.044543123751166334]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。

**3. 不做 checkpoint 选择时差多少？**

最后一步（6000 步）DEV AbsRel={"C0": "0.494604", "C1": "0.461957", "C2": "0.611081"}；选中步={"C0": {"20260928": 4000, "20260929": 1500, "20260930": 6000}, "C1": {"20260928": 6000, "20260929": 6000, "20260930": 5000}, "C2": {"20260928": 3000, "20260929": 5000, "20260930": 4000}}。

**4. 状态是否场景专属、多视角是否优于单视角？**

C1 wrong-scene=0.197604，95% CI [0.073566, 0.335651]；shuffle=0.295046，95% CI [0.156806, 0.445246]；SCENE_SPECIFICITY_STATUS=SUPPORTED；3 视角相对 anchor=0.084340，95% CI [0.009839, 0.172849]；STATIC_DEV_STATUS=SUPPORTED。

**5. 规模扩大后，测得深度还值多少？**

DEPTH_AT_SCALE_GAIN=AbsRel(C2 RGB)−AbsRel(C1)=0.142655，95% CI [0.068479, 0.210217]。仅描述。

**6. 与无几何常数相比如何？**

全部场景标签={"RAW:depth_absrel": "NOT_DISTINGUISHABLE", "RAW:depth_delta1": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_absrel": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_delta1": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_absrel": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；排除 NO_HIT 场景后={"RAW:depth_absrel": "NOT_DISTINGUISHABLE", "RAW:depth_delta1": "NOT_DISTINGUISHABLE", "OPACITY_NORMALIZED:depth_absrel": "ABOVE", "OPACITY_NORMALIZED:depth_delta1": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_absrel": "NOT_DISTINGUISHABLE", "MEDIAN_SCALED:depth_delta1": "NOT_DISTINGUISHABLE"}；C1_ABOVE_REFERENCE_HIT_SCENES=False。

**7. 训练集拟合如何？**

最后100步训练深度AbsRel={"C0": "0.194699", "C1": "0.301420", "C2": "0.443003"}；学到的 σ={"C0": [0.811635, 0.823107, 0.766124], "C1": [0.853753, 0.799972, 0.809007]}。

**8. 区域诊断？**

C0−C1 条件 AbsRel：OBS0=0.037301，95% CI [-0.026587, 0.108195]；OBS1=0.030892，95% CI [-0.013038, 0.078739]；OBS2PLUS=0.065186，95% CI [0.017672, 0.124657]。

**9. 结论属于哪个预注册分支？**

分支C：在同样的 6000 步下，训练场景扩大 3 倍没有稳定改善 DEV：泛化瓶颈不只是数据量。下一轮另写协议（先验、容量或证据形式）；仍不开 Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

# EXP-3D-RGBD-EVIDENCE-CARRIER-V5

单因素机制诊断（RGB-D轨道）：V3/V4的稠密16³ learned carrier，C1只多一条零初始化的测得深度旁路（每个上下文视角的测距给出体素的表面与自由空间似然）。DEPTH_GAIN=AbsRel(C0 DENSE_SURFACE)−AbsRel(C1 DEPTH_SURFACE)=0.114371，95% CI [0.064940, 0.166409]；DEPTH_STATUS=SUPPORTED；SCENE_SPECIFICITY_STATUS=SUPPORTED；STATIC_DEV_STATUS=SUPPORTED；CARRIER_REFERENCE_STATUS=NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE；解读分支=B。

Test-time input = RGB + 测得的上下文深度 + calibrated cameras（C0只读RGB与相机）。上下文深度只经由只含上下文帧的RGBD loader进入状态构建；query深度只用于TRAIN loss和全部状态seal之后的DEV评价。所有场景都是历史曝光场景，8个DEV场景已在V2、V3、V4中被评价：本轮是DEV机制诊断，不是资格轮，没有fresh队列。

共享参数（5125个）、同seed初始值、4096个稠密证据候选、lifting、融合、refinement、heads、renderer、bounds、数据、预算与checkpoint规则全部相同；深度旁路共25个参数（3→8投影与log_sigma），加在≥2视角支持门之后，零初始化保证训练开始时C1与C0完全相同。C2只去掉表面监督，是次要变体，不进入任何gate。统计复用V2冻结的估计器，JSON中的SURFACE_*字段在本轮表示主对比C0−C1（测得深度）。

| Variant | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | 训练末100步深度AbsRel | Opacity |
|---|---|---:|---:|---:|---:|
| C0 DENSE_SURFACE | 0.617415，95% CI [0.472868, 0.763046] | 0.546693 | 0.070722 | 0.369587 | 0.696121 |
| C1 DEPTH_SURFACE | 0.503044，95% CI [0.352312, 0.677064] | 0.307848 | 0.195196 | 0.267846 | 0.810935 |
| C2 DEPTH_NO_SURFACE | 0.525849，95% CI [0.389746, 0.690650] | 0.363230 | 0.162619 | 0.286681 | 0.800863 |

**1. 测得的上下文深度是否改善DEV query几何？**

DEPTH_GAIN=0.114371，95% CI [0.064940, 0.166409]；DEPTH_STATUS=SUPPORTED。gate沿用V2主对比：均值、CI、75%场景不差、所有LOSO>0、正向top1≤0.5、C1 wrong-scene与shuffle损伤CI>0、opacity与状态审计。

**2. 是否跨多个场景和seed？**

improved/tied/worse=7/1/0；LOSO范围=[0.09622956230182281, 0.1307095132194476]；positive top1/top3=0.26379067650368/0.6172854997934216；seeds=[20260928, 20260929, 20260930]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。每个seed的结果见static_results.json。

**3. 状态是否变得场景专属？**

C1 wrong-scene损伤=0.168744，95% CI [0.046729, 0.301522]；C1 spatial shuffle损伤=0.215871，95% CI [0.106555, 0.339485]；SCENE_SPECIFICITY_STATUS=SUPPORTED。对照C0：wrong-scene=0.018243，95% CI [-0.067355, 0.115700]，shuffle=0.010454，95% CI [-0.083641, 0.104540]。wrong-scene同时替换donor bounds，shuffle保留recipient bounds与相机。

**4. full context是否稳定优于anchor？**

C0=0.028551，95% CI [-0.057627, 0.118350]；C1=0.152435，95% CI [0.074077, 0.237755]。anchor只有一个视角，达不到≥2视角lifting支持；C1/C2的anchor状态仍含单视角深度旁路，所以这是3个RGB-D视角对1个RGB-D视角的融合检验。

**5. 训练集拟合与泛化差距如何？**

最后100步训练深度AbsRel（A/B平均、seed平均）={"C0": "0.369587", "C1": "0.267846", "C2": "0.286681"}；query−context gap：C0=0.070722，95% CI [-0.161171, 0.238946]；C1=0.195196，95% CI [0.046515, 0.350788]；C2=0.162619，95% CI [0.022525, 0.320505]。只是描述，不是gate。

**6. 模型学到了怎样的深度似然宽度？**

exp(log_sigma)（按seed，单位为平均体素边长）={"C1": [0.82161, 0.80141, 0.755275], "C2": [0.849272, 0.813735, 0.787153]}；初值0.5。越小表示越依赖精确的表面位置。

**7. 有了测得深度后，表面监督是否还有作用？**

SURFACE_WITH_DEPTH_GAIN=AbsRel(C2)−AbsRel(C1)=0.022805，95% CI [-0.002059, 0.049856]（正值表示表面监督有益；improved/tied/worse=5/1/2）。仅描述，不参与gate，不改变PRIMARY_METHOD=C1。

**8. 改善是否出现在多视角可观测区域？**

C0−C1条件AbsRel：OBS0=0.103198，95% CI [0.041730, 0.168647]；OBS1=0.103762，95% CI [0.048312, 0.159169]；OBS2PLUS=0.148189，95% CI [0.048108, 0.278535]。测得深度应当主要改善OBS1/OBS2PLUS；这是seal后的区域诊断，不替代全图主指标。

**9. 状态相关性诊断说明了什么？**

同场景A/B与跨场景A状态的密度相关（seed平均）={"C0": {"within": "0.742994", "across": "0.573694", "direct_vs_anchor": "-0.104622"}, "C1": {"within": "0.708843", "across": "0.455922", "direct_vs_anchor": "0.456571"}, "C2": {"within": "0.708468", "across": "0.497952", "direct_vs_anchor": "0.429773"}}。仅描述，不参与gate。

**10. DEV是否达到static qualification？**

SUPPORTED；检查={"full_context_mean_ci_scene_consistency": true, "shared_sealed_state_integrity": true, "static_opacity_artifact_gate": true, "wrong_scene_ci_positive": true}；主对比检查={"every_leave_one_scene_out_gain_positive": true, "mean_ci_and_scene_consistency": true, "opacity_artifact_gate": true, "positive_top1_share_at_most_0_5": true, "spatial_shuffle_ci_positive": true, "state_integrity": true, "wrong_scene_ci_positive": true}。即使通过，本轮DEV已曝光，也不构成最终资格。

**11. 与TRAIN拟合的无几何常数相比如何？**

TRAIN拟合的无几何常数：AbsRel最优常数=2.3736 m，中位数常数=4.1992 m。C0: RAW AbsRel相对参考=-0.110439，95% CI [-0.215715, -0.014460]（BELOW），RAW delta1相对参考=-0.132609，95% CI [-0.351774, 0.112837]（NOT_DISTINGUISHABLE）；C1: RAW AbsRel相对参考=0.003932，95% CI [-0.130434, 0.107879]（NOT_DISTINGUISHABLE），RAW delta1相对参考=-0.008520，95% CI [-0.245152, 0.275363]（NOT_DISTINGUISHABLE）；C2: RAW AbsRel相对参考=-0.018873，95% CI [-0.148655, 0.089800]（NOT_DISTINGUISHABLE），RAW delta1相对参考=-0.047951，95% CI [-0.269211, 0.199321]（NOT_DISTINGUISHABLE）。CARRIER_REFERENCE_STATUS=NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE。正数表示carrier更好；只作描述，不参与gate。

**12. 结论属于哪个预注册解读分支，下一步是什么？现在能否重开Dynamic TTT？**

分支B：深度证据有部分作用。下一轮检验carrier容量（grid分辨率）或训练规模；仍不开Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

**opacity与状态审计**

C0/C1 opacity checks={"all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "all_scenes_coverage_drop_at_most_0_01": true, "common_mask_gain_ci_positive_with_75pct_scene_coverage": true, "opacity_normalized_gain_ci_positive_with_75pct_scene_coverage": true}；C1 static opacity checks={"C1_all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "C1_anchor_direct_common_mask_gain_ci_positive_with_75pct_scene_coverage": true, "C1_anchor_direct_normalized_gain_ci_positive_with_75pct_scene_coverage": true}；NO_HIT scenes=['ai_009_001']；shared-state/seal审计={"fresh_qualification_opened": false, "matched_initialization_and_streams": true, "query_after_state_seal": true, "recipient_camera_preserved": true, "shared_state_multiple_queries": true, "status": "PASS", "test_time_depth_scope": "context frames only; query depth only after seal", "test_time_depth_used": true}。

**选择、统计与成本**

每个variant×seed用同一DEV规则独立选择checkpoint（checkpoint在前期加密）。selected={"C0": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C0_20260928/step_000750.pt", "sha256": "61aa8e9ff75006797923ef85518299f16fb2baa20b59a6f0010315146d63e2f6", "step": 750}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C0_20260929/step_000500.pt", "sha256": "4048848725f679e6404c3c594fb362d0e3e70cfee1657e921830dd517aaa6a05", "step": 500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C0_20260930/step_000500.pt", "sha256": "663fb8884ecf42e0dfbe6ddbcc4c8a2a5e9cdec43ec77f745d0a68681c82f24d", "step": 500}}, "C1": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C1_20260928/step_001500.pt", "sha256": "9206d4326fec3d12f350cba15da4c377fe182261728a8b0d4960207bd4d6bf81", "step": 1500}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C1_20260929/step_001500.pt", "sha256": "1b31f47fb67ec484146602ecfcc3a835ac1798528eeb80fde6bf8abd17fe691d", "step": 1500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C1_20260930/step_002000.pt", "sha256": "a895ac5607b940984dbfb5380bf91b80678efebb7aa28ecc9aa51ecebba2ccd4", "step": 2000}}, "C2": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C2_20260928/step_001500.pt", "sha256": "5d5fe30cc5f5bfec5e3d44c472c4f407278560469f63df4f1cbdabb42bf14768", "step": 1500}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C2_20260929/step_001500.pt", "sha256": "b616a788b14a678d53a474cc1379c00fcfcb3b20ef74bc9600d76ddd1c16c909", "step": 1500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/checkpoints/C2_20260930/step_001500.pt", "sha256": "b5ebefefedb2bcd82fc7ebcb7fa55e47bdf6d64a1fd7c0e1456bcb5194242792", "step": 1500}}}。先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000次paired scene bootstrap，seed 20260928，95% percentile CI。成本见cost_analysis.json；深度旁路只增加构建状态时的计算，renderer与推理参数（除旁路25个）相同。

基础设施故障重跑次数=0（失败尝试归档在audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS=EVALUATED；CPU亲和性固定为8–23号核。协议见PROTOCOL.md。

TEST_TIME_INPUT=RGB+DEPTH+CAMERA；TEST_TIME_DEPTH_USED=true（仅上下文深度）；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。

# EXP-3D-DENSE-EVIDENCE-CARRIER-V3

单因素机制诊断：同一个16³ learned carrier，唯一差别是多视角证据候选点从128个（V2配方，只覆盖3.1%体素）加密到全部4096个体素。DENSITY_GAIN=AbsRel(C0 SPARSE128_SURFACE)−AbsRel(C1 DENSE4096_SURFACE)=0.010057，95% CI [-0.014221, 0.039420]；DENSITY_STATUS=NOT_ESTABLISHED；SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED；STATIC_DEV_STATUS=NOT_ESTABLISHED；解读分支=C。

Test-time input = RGB + calibrated cameras。训练是RGB+D utility track；depth只用于TRAIN loss和seal后的DEV评价。所有场景都是历史曝光场景，8个DEV场景已在V2中被评价：本轮是DEV上的机制诊断，不是资格轮，没有fresh队列。

参数量（5125）、同seed初始参数、lifting规则（≥2视角支持）、融合、refinement、heads、renderer、bounds、数据、预算与checkpoint选择规则全部相同。C2只去掉表面监督，是次要变体，不进入任何gate。统计复用V2冻结的估计器，JSON中的SURFACE_*字段在本轮表示主对比C0−C1（密度）。

| Variant | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | 训练末100步深度AbsRel | Opacity |
|---|---|---:|---:|---:|---:|
| C0 SPARSE128_SURFACE | 0.630097，95% CI [0.491865, 0.767503] | 0.601034 | 0.029064 | 0.489525 | 0.714076 |
| C1 DENSE4096_SURFACE | 0.620040，95% CI [0.475082, 0.765772] | 0.546371 | 0.073669 | 0.339227 | 0.672621 |
| C2 DENSE4096_NO_SURFACE | 0.634788，95% CI [0.502900, 0.772551] | 0.554610 | 0.080177 | 0.345210 | 0.680304 |

**1. 稠密证据是否改善DEV query几何？**

DENSITY_GAIN=0.010057，95% CI [-0.014221, 0.039420]；DENSITY_STATUS=NOT_ESTABLISHED。gate沿用V2主对比：均值、CI、75%场景不差、所有LOSO>0、正向top1≤0.5、C1 wrong-scene与shuffle损伤CI>0、opacity与状态审计。

**2. 是否跨多个场景和seed？**

improved/tied/worse=4/1/3；LOSO范围=[-0.002240751301118516, 0.015662684137865267]；positive top1/top3=0.5983173327986545/0.9119957047014826；seeds=[20260928, 20260929, 20260930]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。每个seed的结果见static_results.json。

**3. 稠密carrier的状态是否变得场景专属？**

C1 wrong-scene损伤=0.021432，95% CI [-0.060513, 0.118758]；C1 spatial shuffle损伤=0.010268，95% CI [-0.080416, 0.101004]；SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED。对照C0：wrong-scene=-0.009360，95% CI [-0.049961, 0.035066]，shuffle=0.001559，95% CI [-0.064121, 0.065254]。wrong-scene同时替换donor bounds，shuffle保留recipient bounds与相机。

**4. full context是否稳定优于anchor？**

C0=0.014251，95% CI [-0.048574, 0.075941]；C1=0.028639，95% CI [-0.058802, 0.116849]。anchor只有一个视角，达不到≥2视角支持，因此anchor状态只含学到的先验。

**5. 训练集拟合是否改善？**

最后100步训练深度AbsRel（A/B平均、seed平均）={"C0": "0.489525", "C1": "0.339227", "C2": "0.345210"}。只是描述，不是gate。

**6. context-query gap是否缩小？**

C0=0.029064，95% CI [-0.281379, 0.241698]；C1=0.073669，95% CI [-0.153774, 0.239296]；C2=0.080177，95% CI [-0.156468, 0.259592]。未预注册gap差值的显著性，不从CI重叠推断。

**7. 稠密证据下表面监督是否有收益？**

SURFACE_AT_DENSE_GAIN=AbsRel(C2)−AbsRel(C1)=0.014747，95% CI [-0.003022, 0.035766]（正值表示表面监督有益；improved/tied/worse=4/1/3）。仅描述，不参与gate，不改变PRIMARY_METHOD=C1。

**8. 改善发生在哪些可观测区域？**

C0−C1条件AbsRel：OBS0=0.015689，95% CI [0.002453, 0.028519]；OBS1=-0.016249，95% CI [-0.060355, 0.020395]；OBS2PLUS=-0.018624，95% CI [-0.102040, 0.044040]。这是seal后的区域诊断，不替代全图主指标。

**9. 状态相关性诊断说明了什么？**

同场景A/B与跨场景A状态的密度相关（seed平均）={"C0": {"within": "0.702161", "across": "0.556373", "direct_vs_anchor": "0.075508"}, "C1": {"within": "0.740314", "across": "0.561396", "direct_vs_anchor": "-0.100623"}, "C2": {"within": "0.758514", "across": "0.565459", "direct_vs_anchor": "-0.170418"}}。仅描述，不参与gate。

**10. DEV是否达到static qualification？**

NOT_ESTABLISHED；检查={"full_context_mean_ci_scene_consistency": false, "shared_sealed_state_integrity": true, "static_opacity_artifact_gate": false, "wrong_scene_ci_positive": false}；主对比检查={"every_leave_one_scene_out_gain_positive": false, "mean_ci_and_scene_consistency": false, "opacity_artifact_gate": false, "positive_top1_share_at_most_0_5": false, "spatial_shuffle_ci_positive": false, "state_integrity": true, "wrong_scene_ci_positive": false}。即使通过，本轮DEV已曝光，也不构成最终资格。

**11. 结论属于哪个预注册解读分支，下一步是什么？现在能否重开Dynamic TTT？**

分支C：排除“候选稀疏”作为主因。下一轮按V2 CASE A引入可见性感知融合或cost-volume几何推理，另写协议；仍不开Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

**opacity与状态审计**

C0/C1 opacity checks={"all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "all_scenes_coverage_drop_at_most_0_01": true, "common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "opacity_normalized_gain_ci_positive_with_75pct_scene_coverage": false}；C1 static opacity checks={"C1_all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "C1_anchor_direct_common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "C1_anchor_direct_normalized_gain_ci_positive_with_75pct_scene_coverage": false}；NO_HIT scenes=['ai_009_001']；shared-state/seal审计={"fresh_qualification_opened": false, "matched_initialization_and_streams": true, "query_after_state_seal": true, "recipient_camera_preserved": true, "shared_state_multiple_queries": true, "status": "PASS", "test_time_depth_used": false}。

**选择、统计与成本**

每个variant×seed用同一DEV规则独立选择checkpoint。selected={"C0": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C0_20260928/step_003000.pt", "sha256": "58647eb33cfc77ad00f59e74a8b65c6006ee69e93799e8d864623c7335009234", "step": 3000}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C0_20260929/step_001000.pt", "sha256": "693bcd55c8b76493a61aee9db5d1eb1dabf7ecbb866938ca7fc234b150021288", "step": 1000}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C0_20260930/step_001000.pt", "sha256": "f6aeff6fe55208bb66cefd6cb92144c09df5c56ec96a763c5a4fe9876b188b87", "step": 1000}}, "C1": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C1_20260928/step_000500.pt", "sha256": "8d0d7968a201e5b5781fa7040e2636e18f513910f2b996ac71665814ef46a9b8", "step": 500}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C1_20260929/step_000500.pt", "sha256": "85ce4cddb4b4717fbf62c7fb8bdc2bc943ad78fb0033002de2856702a7791043", "step": 500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C1_20260930/step_000500.pt", "sha256": "b3aed1deb31c23dfa637b950d02d92095d93cfbcebe9646fab56442ddd275e2a", "step": 500}}, "C2": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C2_20260928/step_001000.pt", "sha256": "e16700441f1f5e4fc25bc987f1eb68a63df86529b138912934848353507f8639", "step": 1000}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C2_20260929/step_001000.pt", "sha256": "8665c9dc65163788e14719a765055561411d5fbd4ef46f17237c6aec74d099ee", "step": 1000}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/checkpoints/C2_20260930/step_000500.pt", "sha256": "ec768ed1c583be1a1c01a8006bab6540048a48a279eba0545406ccc570d99336", "step": 500}}}。先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000次paired scene bootstrap，seed 20260928，95% percentile CI。成本见cost_analysis.json；稠密与稀疏的推理参数相同，只是候选点更多。

基础设施故障重跑次数=0（失败尝试归档在audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS=EVALUATED；CPU亲和性固定为8–23号核。协议见PROTOCOL.md。

TEST_TIME_INPUT=RGB+CAMERA；TEST_TIME_DEPTH_USED=false；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。

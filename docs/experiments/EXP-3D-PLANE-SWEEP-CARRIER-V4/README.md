# EXP-3D-PLANE-SWEEP-CARRIER-V4

单因素机制诊断：V3的稠密16³ learned carrier，C1只多一条零初始化的plane-sweep旁路（不需要学习匹配的光度代价，沿参考射线做深度softmax，得到每个体素的表面与自由空间似然）。SWEEP_GAIN=AbsRel(C0 DENSE_SURFACE)−AbsRel(C1 SWEEP_SURFACE)=0.007209，95% CI [-0.004064, 0.016814]；SWEEP_STATUS=NOT_ESTABLISHED；SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED；STATIC_DEV_STATUS=NOT_ESTABLISHED；解读分支=C。

Test-time input = RGB + calibrated cameras。训练是RGB+D utility track；depth只用于TRAIN loss和seal后的DEV评价。plane-sweep只读上下文RGB、上下文相机与训练集先验。所有场景都是历史曝光场景，8个DEV场景已在V2、V3中被评价：本轮是DEV机制诊断，不是资格轮，没有fresh队列。

共享参数（5125个）、同seed初始值、4096个稠密证据候选、lifting、融合、refinement、heads、renderer、bounds、数据、预算与checkpoint规则全部相同；旁路共49个参数，零初始化保证训练开始时C1与C0完全相同。C2只去掉表面监督，是次要变体，不进入任何gate。统计复用V2冻结的估计器，JSON中的SURFACE_*字段在本轮表示主对比C0−C1（plane-sweep）。

| Variant | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | 训练末100步深度AbsRel | Opacity |
|---|---|---:|---:|---:|---:|
| C0 DENSE_SURFACE | 0.617386，95% CI [0.472811, 0.763034] | 0.546651 | 0.070735 | 0.370086 | 0.696208 |
| C1 SWEEP_SURFACE | 0.610177，95% CI [0.471354, 0.755915] | 0.565687 | 0.044491 | 0.359277 | 0.724697 |
| C2 SWEEP_NO_SURFACE | 0.618356，95% CI [0.467372, 0.765675] | 0.575799 | 0.042557 | 0.362646 | 0.627220 |

**1. plane-sweep几何线索是否改善DEV query几何？**

SWEEP_GAIN=0.007209，95% CI [-0.004064, 0.016814]；SWEEP_STATUS=NOT_ESTABLISHED。gate沿用V2主对比：均值、CI、75%场景不差、所有LOSO>0、正向top1≤0.5、C1 wrong-scene与shuffle损伤CI>0、opacity与状态审计。

**2. 是否跨多个场景和seed？**

improved/tied/worse=5/1/2；LOSO范围=[0.004558701677701457, 0.011724709083353324]；positive top1/top3=0.2945053356110441/0.728379987970284；seeds=[20260928, 20260929, 20260930]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。每个seed的结果见static_results.json。

**3. 状态是否变得场景专属？**

C1 wrong-scene损伤=0.029380，95% CI [-0.038306, 0.104475]；C1 spatial shuffle损伤=0.027144，95% CI [-0.063876, 0.121167]；SCENE_SPECIFICITY_STATUS=NOT_ESTABLISHED。对照C0：wrong-scene=0.018255，95% CI [-0.067358, 0.115724]，shuffle=0.010479，95% CI [-0.083623, 0.104569]。wrong-scene同时替换donor bounds，shuffle保留recipient bounds与相机。

**4. full context是否稳定优于anchor？**

C0=0.028574，95% CI [-0.057598, 0.118392]；C1=0.058761，95% CI [-0.035155, 0.152749]。anchor只有一个视角，既达不到≥2视角支持，也没有plane-sweep源视角，因此anchor状态只含学到的先验。

**5. 训练集拟合与泛化差距如何？**

最后100步训练深度AbsRel（A/B平均、seed平均）={"C0": "0.370086", "C1": "0.359277", "C2": "0.362646"}；query−context gap：C0=0.070735，95% CI [-0.161112, 0.238866]；C1=0.044491，95% CI [-0.255496, 0.247913]；C2=0.042557，95% CI [-0.264384, 0.254803]。只是描述，不是gate。

**6. 模型学到了怎样的plane-sweep温度？**

exp(log_tau)（按seed）={"C1": [0.007951, 0.006526, 0.007589], "C2": [0.010164, 0.007404, 0.009767]}；初值0.01。温度越小表示越依赖尖锐的深度分布。

**7. 有了几何线索后，表面监督是否还有作用？**

SURFACE_WITH_SWEEP_GAIN=AbsRel(C2)−AbsRel(C1)=0.008179，95% CI [-0.022672, 0.039145]（正值表示表面监督有益；improved/tied/worse=4/1/3）。仅描述，不参与gate，不改变PRIMARY_METHOD=C1。

**8. 改善是否出现在多视角可观测区域？**

C0−C1条件AbsRel：OBS0=-0.000682，95% CI [-0.023768, 0.020218]；OBS1=0.016912，95% CI [0.000517, 0.034370]；OBS2PLUS=0.040308，95% CI [0.003822, 0.087142]。几何推理应当主要改善OBS1/OBS2PLUS；这是seal后的区域诊断，不替代全图主指标。

**9. 状态相关性诊断说明了什么？**

同场景A/B与跨场景A状态的密度相关（seed平均）={"C0": {"within": "0.742919", "across": "0.573564", "direct_vs_anchor": "-0.104413"}, "C1": {"within": "0.752164", "across": "0.637163", "direct_vs_anchor": "-0.100920"}, "C2": {"within": "0.764071", "across": "0.607602", "direct_vs_anchor": "-0.072218"}}。仅描述，不参与gate。

**10. DEV是否达到static qualification？**

NOT_ESTABLISHED；检查={"full_context_mean_ci_scene_consistency": false, "shared_sealed_state_integrity": true, "static_opacity_artifact_gate": false, "wrong_scene_ci_positive": false}；主对比检查={"every_leave_one_scene_out_gain_positive": true, "mean_ci_and_scene_consistency": false, "opacity_artifact_gate": false, "positive_top1_share_at_most_0_5": true, "spatial_shuffle_ci_positive": false, "state_integrity": true, "wrong_scene_ci_positive": false}。即使通过，本轮DEV已曝光，也不构成最终资格。

**11. 结论属于哪个预注册解读分支，下一步是什么？现在能否重开Dynamic TTT？**

分支C：在3个上下文视角、32×40分辨率下，最小plane-sweep不足以让carrier泛化。下一轮另写协议，考虑预训练几何先验或更多视角；仍不开Dynamic TTT。 DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false；FINAL_STATIC_STATUS=NOT_ESTABLISHED。

**opacity与状态审计**

C0/C1 opacity checks={"all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "all_scenes_coverage_drop_at_most_0_01": true, "common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "opacity_normalized_gain_ci_positive_with_75pct_scene_coverage": false}；C1 static opacity checks={"C1_all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "C1_anchor_direct_common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "C1_anchor_direct_normalized_gain_ci_positive_with_75pct_scene_coverage": false}；NO_HIT scenes=['ai_009_001']；shared-state/seal审计={"fresh_qualification_opened": false, "matched_initialization_and_streams": true, "query_after_state_seal": true, "recipient_camera_preserved": true, "shared_state_multiple_queries": true, "status": "PASS", "test_time_depth_used": false}。

**选择、统计与成本**

每个variant×seed用同一DEV规则独立选择checkpoint（checkpoint在前期加密）。selected={"C0": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C0_20260928/step_000750.pt", "sha256": "66dbcf9b695a7406c5d35bc5021f9eeed987037903d7275a220be47c30eba7b2", "step": 750}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C0_20260929/step_000500.pt", "sha256": "6ae824597fabc9d7d122f9cca93ef21d01c0961bddbc36c81eb6a1b6d431b988", "step": 500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C0_20260930/step_000500.pt", "sha256": "5a76dc27c3b7dc3cdd13b21f03f83b049a9e210cd18baab01b27bc52cabd3171", "step": 500}}, "C1": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C1_20260928/step_000750.pt", "sha256": "b27a073577bd64e6104f916f9e6cd78190da560aca1090df38470fc145c56bbd", "step": 750}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C1_20260929/step_001500.pt", "sha256": "9f64d8123947dd7aed952764a4cdcd9def39cd6e54ff6bf5748f8fccdf0d4f79", "step": 1500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C1_20260930/step_000200.pt", "sha256": "963c99a16b087a64ff3cdb30c5ecb2176a5ba5136f45354afca19e49af1a68a7", "step": 200}}, "C2": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C2_20260928/step_000500.pt", "sha256": "8e1ad64bbc77547552e2f96bf05c0514a38db1397e37c7c5e2ff6e78aa97aa9d", "step": 500}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C2_20260929/step_000200.pt", "sha256": "b26e7c6fafc3cddddf628b80c03d98b4a86e73274e7c8c46517f3259855a43ba", "step": 200}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-PLANE-SWEEP-CARRIER-V4/checkpoints/C2_20260930/step_000200.pt", "sha256": "6a5232fc267ed2f1148d222f48deaca22d9189ee5d154b306587a9dd46573863", "step": 200}}}。先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000次paired scene bootstrap，seed 20260928，95% percentile CI。成本见cost_analysis.json；plane-sweep只增加构建状态时的计算，renderer与推理参数（除旁路49个）相同。

基础设施故障重跑次数=0（失败尝试归档在audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS=EVALUATED；CPU亲和性固定为8–23号核。协议见PROTOCOL.md。

TEST_TIME_INPUT=RGB+CAMERA；TEST_TIME_DEPTH_USED=false；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。

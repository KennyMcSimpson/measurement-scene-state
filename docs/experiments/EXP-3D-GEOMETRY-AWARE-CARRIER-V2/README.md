# EXP-3D-GEOMETRY-AWARE-CARRIER-V2

主比较是matched16³ BASELINE16(C0) vs SURFACE16(C1)，唯一方法差别是TRAIN-side显式surface loss。SURFACE_GAIN=0.019491，95% CI [-0.025280, 0.076809]；SURFACE_TRAINING_STATUS=NOT_ESTABLISHED；STATIC_DEV_STATUS=NOT_ESTABLISHED。

Test-time input = RGB + calibrated cameras. 训练是RGB+D utility track，不是RGB-only training；depth仅供TRAIN loss及seal后的DEV评价/诊断，未作为test-time carrier输入。

C1固定为PRIMARY_METHOD=SURFACE16；C2=FREE_SURFACE16仅为训练前预注册的secondary ablation（C1＋0.1×free-space），只报告C1−C2，不进入C0/C1共mask、OBS区域或任何gate，不能事后改为主方法。未新算旧8³ B-final，只保留历史上下文；不能把C0改进单独归因为resolution、重训或训练数据增多。各variant推理architecture完全一致，surface/free-space loss只存在训练阶段，不能宣称它们使推理架构更贵。

| Method | Query AbsRel (95% CI) | Context AbsRel | Query−context gap | Opacity | Coverage |
|---|---|---:|---:|---:|---:|
| C0 | 0.649526，95% CI [0.546397, 0.772108] | 0.600661 | 0.048865 | 0.660147 | 0.875000 |
| C1 | 0.630035，95% CI [0.491692, 0.767498] | 0.601192 | 0.028843 | 0.714125 | 0.875000 |
| C2 | 0.630218，95% CI [0.492609, 0.767425] | 0.600730 | 0.029489 | 0.709103 | 0.875000 |

**1. Surface supervision在learned carrier中是否有效？**

NOT_ESTABLISHED；C0−C1=0.019491，95% CI [-0.025280, 0.076809]。本轮主gate是>0而非沿用旧.02门槛，同时检查CI、75%场景不差、所有LOSO>0、positive top1 share≤.5、controls与opacity。

**2. 是否跨多个scenes和seeds？**

improved/tied/worse=3/1/4；LOSO范围=[-0.0038797353582333666, 0.03207572737260692]；positive top1/top3=0.6802632083736532/1.0。seed列表=[20260928, 20260929, 20260930]；SEED_ROBUSTNESS=DIRECTION_REPLICATED。所有seed保留，不选best seed；每seed结果见static_results.json。top1 share≤.5参与gate，top3仅报告不作为gate。

**3. 是否复现OBS2PLUS改善signature？**

OBS2PLUS C0−C1=-0.015232，95% CI [-0.071991, 0.030507]；scene coverage=1.0。这是post-seal条件区域诊断；空scene保留null，不把conditional收益替代全图净收益。

**4. 16³ baseline retraining本身提升多少？**

本轮未新增旧8³模型matched评价，因此不能独立分离grid、更多数据、训练预算和重训收益。正式归因只比较相同architecture/grid/data/seed/budget/selection规则的C0和C1。

**5. surface相对matched baseline额外提升多少？**

0.019491，95% CI [-0.025280, 0.076809]。这才是主方法增量，方向为C0 AbsRel−C1 AbsRel，正值更好。

**6. full context是否稳定优于anchor？**

C0=0.007968，95% CI [-0.036974, 0.054203]；C1=0.014287，95% CI [-0.048558, 0.076079]；STATIC_DEV_STATUS=NOT_ESTABLISHED。anchor_A只编码共同anchor RGB，沿用A的合法context-camera bounds；anchor_B同理，不是只允许单相机几何信息的baseline。

**7. wrong-scene是否稳定更差？**

C0=-0.010667，95% CI [-0.046695, 0.023605]；C1=-0.009385，95% CI [-0.050073, 0.035244]。recipient query camera不变；wrong-scene同时替换donor bounds，不能将全部damage独归density内容。

**8. state是否真正scene-specific并有空间结构？**

C0 shuffle=-0.007271，95% CI [-0.049991, 0.036289]；C1 shuffle=0.001621，95% CI [-0.064117, 0.065403]。shuffle保留recipient bounds。shared-state/seal审计={"fresh_qualification_opened": false, "matched_initialization_and_streams": true, "query_after_state_seal": true, "recipient_camera_preserved": true, "shared_state_multiple_queries": true, "status": "PASS", "test_time_depth_used": false}。多个query必须使用同一sealed state，controls CI不支持时不能声称已证明。

**9. context-query gap是否缩小？**

C0=0.048865，95% CI [-0.186420, 0.226785]；C1=0.028843，95% CI [-0.282134, 0.241743]。这里分别报告两者，未预注册额外gap-change显著性则不从两个CI重叠与否推出差值显著性；query净收益是主目标。

**10. free-space有没有额外价值？**

C1−C2=-0.000183，95% CI [-0.001437, 0.000970]（正值表示加入free-space后AbsRel更低；improved/tied/worse=3/1/4）。C2只是secondary ablation，PRIMARY_METHOD仍为C1；即使C2更低也不改主方法，该对比不参与任何gate。

**11. DEV是否达到static qualification？**

NOT_ESTABLISHED；检查={"full_context_mean_ci_scene_consistency": false, "shared_sealed_state_integrity": true, "static_opacity_artifact_gate": false, "wrong_scene_ci_positive": false}。surface检查={"every_leave_one_scene_out_gain_positive": false, "mean_ci_and_scene_consistency": false, "opacity_artifact_gate": false, "positive_top1_share_at_most_0_5": false, "spatial_shuffle_ci_positive": false, "state_integrity": true, "wrong_scene_ci_positive": false}。DEV资格不是fresh最终资格。

**12. fresh qualification是否允许打开？**

FRESH_QUALIFICATION_OPENED=false；独立性=BLOCKED_INDEPENDENCE_UNRESOLVED。即使DEV良好也不替换fresh cohort或将已曝光scene重新包装。

**13. fresh scenes是否复现主要signature？**

NOT_RUN；没有新fresh结果，FINAL_STATIC_STATUS=NOT_ESTABLISHED。

**14. 现在是否可以重新打开Dynamic TTT？**

DYNAMIC_TTT_NEXT_STAGE_ALLOWED=false；DYNAMIC_TTT_RUN=false。未满足进入下一阶段的全部条件。保留matched对照：surface无收益时研究carrier lifting/fusion/geometry reasoning；full context不胜anchor时改善shared-state构造；wrong-scene不显著时检查global-prior依赖。本轮不引入这些新架构，也不开Dynamic TTT。

**预锁opacity与固定bounds限制**

在V2正式训练前，主agent查阅已曝光历史ai_009_001发现固定bounds query rays全miss，因此未改scene/roles/bounds，改用每scene coverage≥.99×ray_hitfraction、C1−C0coverage≥−.01。NO_HIT不是新model collapse。所有8个DEV场景及全部GT-valid pixel仍计入主指标，固定bounds盲区保留。

共mask要求C0和C1 opacity>1e-6；空scene显式null，共mask统计覆盖至少75%scene。还要求rendered_depth/opacity.clamp_min(1e-6)这一纯诊断上的C0−C1收益CI下界>0。主rendered depth语义和主AbsRel不改变。诊断降低透明度缩放解释的风险，不构成排除所有opacity相关机制的证明。

opacity checks={"all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "all_scenes_coverage_drop_at_most_0_01": true, "common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "opacity_normalized_gain_ci_positive_with_75pct_scene_coverage": false}；NO_HIT scenes=['ai_009_001']。

STATIC_DEV_STATUS使用C1自身的static opacity gate（训练前预注册）：每个scene C1 coverage≥.99×ray_hitfraction；在anchor与direct共同opacity>1e-6的GT-valid mask上，anchor−direct AbsRel及其depth/opacity归一化版本的CI下界均>0，且至少75% scene有非空mask。C1 common-mask anchor−direct=0.016328，95% CI [-0.057630, 0.086972]；归一化=-0.034389，95% CI [-0.103470, 0.032719]；static opacity checks={"C1_all_scenes_coverage_at_least_0_99_ray_hitfraction": true, "C1_anchor_direct_common_mask_gain_ci_positive_with_75pct_scene_coverage": false, "C1_anchor_direct_normalized_gain_ci_positive_with_75pct_scene_coverage": false}。

**选择、统计与成本**

每variant×seed用同一预锁DEV选择规则独立选择checkpoint；不是强行同步步数。selected step/path={"C0": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C0_20260928/step_001000.pt", "sha256": "1abc9c7f79c00a48b62cae4c69fdee76b764126dba4957b88a49d0dfda94d2a8", "step": 1000}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C0_20260929/step_000500.pt", "sha256": "2046ea165338fde67b089ab4f88a6ab3b647b80672c139a38278b707ea27a2b3", "step": 500}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C0_20260930/step_000500.pt", "sha256": "a8c4edb4417f3ff1c46bf22ecc9585733abb4da69011ac50f267593d289a77ec", "step": 500}}, "C1": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C1_20260928/step_003000.pt", "sha256": "473d2daee0668db916daaf4395b9cb72d4ab9115ba3f9fb711564974f69f8f7a", "step": 3000}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C1_20260929/step_001000.pt", "sha256": "d73905660eca840395afae9d097a55199abbc763dec60caa730badc2cb04ac79", "step": 1000}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C1_20260930/step_001000.pt", "sha256": "5e06ebc142c378e3f9104351aefa56ffa51439e3d5f45f701d6240a7b0dbc36a", "step": 1000}}, "C2": {"20260928": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C2_20260928/step_003000.pt", "sha256": "aaddea72bebd6a5dc7133df01cc1d1f32a23b5ce57bcd663888c923131ac105a", "step": 3000}, "20260929": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C2_20260929/step_001000.pt", "sha256": "7c02b9308f5fc09eb521093b3a821da30dd2a34caa9e2535c8b4ee52c18c47da", "step": 1000}, "20260930": {"path": "/home/zonghan/measurement-scene-state/outputs/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/checkpoints/C2_20260930/step_001000.pt", "sha256": "0c9da9e3c0b3d3736fc714620e6051b858d5acbec42536c97a5e186d654c91fd", "step": 1000}}}。lambda_surface=.1与tau=.5×physical voxel diagonal预先冻结，不能由DEV/fresh重新调节。surface-band mass会通过transmittance向前方density传播梯度。

先对每个query/role等权平均所有seed，再平均query/role得到scene值；10,000 paired scene bootstrap，seed20260928，95% percentile CI。seeds、A/B和pixels不能作为独立scene扩充样本量。区域conditional均值、pixel fraction、可加误差贡献与空区域scene coverage见observability_analysis.json；完整mean/median/LOSO/top1/top3在bootstrap_results.json。

training curves和8类图见training_curves.json及figures/。各variant成本原始记录={"cost_interpretation": "training wall includes DEV selection and checkpoint IO; step seconds separate; surface/free-space losses are training-only", "gpu_shared": true, "inference_architecture_identical": true, "variants": {"C0": {"checkpoint_bytes": 99711, "inference_seconds_per_state": 0.01092590832922724, "peak_cuda_allocated_bytes": 101342720, "renderer_seconds_per_query": 0.006186974229422049, "seconds_per_training_step": 0.06878108127576221, "training_seconds": 684.1185701389913}, "C1": {"checkpoint_bytes": 99711, "inference_seconds_per_state": 0.0056023964801473385, "peak_cuda_allocated_bytes": 104656384, "renderer_seconds_per_query": 0.00619391018760022, "seconds_per_training_step": 0.06962988698549775, "training_seconds": 687.7262420679326}, "C2": {"checkpoint_bytes": 99711, "inference_seconds_per_state": 0.007006994580782096, "peak_cuda_allocated_bytes": 104656384, "renderer_seconds_per_query": 0.006233067030431509, "seconds_per_training_step": 0.06964831776545098, "training_seconds": 687.746018462989}}}。training_seconds是worker累计时间，与phase_wall_times分开；inference与renderer latency分开，checkpoint bytes/VRAM亦分别记录，缺失值不伪造。

本报告仅读取保存的JSON，不进行模型forward或新评价。复现见commands.sh；raw、配置、checkpoint选择锁、数据provenance、state/GT访问审计与完整性清单保存在本实验目录。

原生崩溃重跑次数=0（崩溃尝试均归档到audit/failed_attempts且从不使用，见audit/incidents.json）；SECONDARY_C2_STATUS=EVALUATED；CPU亲和性固定为8–23号核。封存前的接手修订见TAKEOVER_AMENDMENTS.md。

TEST_TIME_INPUT=RGB+CAMERA；TEST_TIME_DEPTH_USED=false；FINAL_HOLDOUT_TOUCHED=false；DYNAMIC_TTT_RUN=false。

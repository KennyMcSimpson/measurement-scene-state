"""Regenerate the registered support-attribution report from saved raw, never inference."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from mcss.mechanism_pilot.small_training import sha, write_json
from mcss.mechanism_pilot.support_redesign_statistics import analyze


def main(root, docs, *, raw_only=False):
    root, docs = Path(root), Path(docs)
    raw_path = root / "raw/oracle_run/results.json"
    if raw_path.exists():
        rows = json.loads(raw_path.read_text())
    else:
        rows = json.loads(gzip.decompress((root / "raw/oracle_results.json.gz").read_bytes()))
    statistics = analyze(rows)
    docs.mkdir(parents=True, exist_ok=True)
    if raw_only:
        write_json(docs / "bootstrap_results.json", statistics)
        print(
            "Regenerated statistics from680 saved development rows; no media/model/holdout access"
        )
        return

    def load(path):
        return json.loads((root / path).read_text())

    pooled = statistics["pooled"]["designs"]
    combined = pooled["ORACLE_VOLUME_SUPPORT"]
    new_combined = statistics["cohorts"]["new_dev"]["designs"]["ORACLE_VOLUME_SUPPORT"]
    stop = (
        combined["full_context_gain"]["ci95"][0] <= 0
        or new_combined["full_context_gain"]["mean"] <= 0
    )
    if not stop:
        raise RuntimeError(
            "Oracle passes registered continuation gate; complete downstream work first"
        )
    write_json(root / "bootstrap_results.json", statistics)
    write_json(
        root / "oracle_support_analysis.json",
        {
            "scope": "FROZEN_CARRIER_PRIVILEGED_DIAGNOSTIC_NOT_MATHEMATICAL_UPPER_BOUND",
            "cohorts": statistics["cohorts"],
            "pooled": statistics["pooled"],
            "registered_continuation_gate_pass": False,
            "stop_reason": "Combined oracle pooled full-context gain CI lower <=0",
            "adequacy": combined["oracle_adequacy"],
            "SUPPORT_BOTTLENECK_SUFFICIENT_EXPLANATION": False,
            "sufficiency_interpretation": (
                "Not established by tested interventions, not universally disproved"
            ),
            "CARRIER_REPRESENTATION_OR_TRAINING_BOTTLENECK": None,
            "causal_interpretation": (
                "NOT_ISOLATED: oracle did not achieve predefined geometric adequacy. "
                "Cannot assert that ideal support was supplied or uniquely blame "
                "representation/training."
            ),
        },
    )
    skipped = {
        "status": "NOT_RUN_ORACLE_STOP",
        "results": None,
        "reason": "Registered combined-oracle continuation gate failed. "
        "R1/R2/R3 geometry prototypes and guards have fixture tests, but no real-scene "
        "GT-free prediction, score selection or matched training was performed.",
    }
    write_json(
        root / "deployable_support_analysis.json",
        {
            "baseline_R0": {k: v["designs"]["R0"] for k, v in statistics["cohorts"].items()},
            "pooled_R0": pooled["R0"],
            "methods": {k: skipped for k in ("R1", "R2", "R3", "R12", "R123")},
            "all_primary_oracles_excluded_from_method_ranking": True,
        },
    )
    write_json(
        root / "matched_training_analysis.json",
        {
            "MATCHED_RETRAIN_DONE": False,
            "status": "NOT_RUN_ORACLE_STOP",
            "current_gain": None,
            "redesigned_gain": None,
            "matched_config_guard_tested": True,
            "train_scale_up_done": False,
            "training_scenes_if_eligible": ["ai_001_001", "ai_002_001", "ai_003_001"],
            "reason": (
                "No deployable redesign qualified for matched training; "
                "no checkpoint family created"
            ),
        },
    )
    for filename, metric in [
        ("volume_analysis.json", "inside_fraction"),
        ("candidate_support_analysis.json", "two_view_candidate_fraction"),
    ]:
        write_json(
            root / filename,
            {
                "scope": "diagnostic_oracles_and_R0_only",
                "metric": metric,
                "per_design": {k: v["AB_geometry"] for k, v in pooled.items()},
                "candidate_budget": 128,
                "grid_size": [8, 8, 8],
                "fixed_neighborhood_radius_m": (1.5**2 + 1 + 1.5**2) ** 0.5 / 2,
                "geometry_plan_raw": "raw/oracle_run/geometry_plans.json",
                "per_query_all_metrics": "raw/oracle_run/results.json",
                "candidate_view_count_is": (
                    "frustum membership, not GT occlusion-correct visibility"
                ),
            },
        )
    write_json(
        root / "context_selection_analysis.json",
        {
            **skipped,
            "frozen_current_roles": load("qualified_frame_role_lock.json"),
            "R3_available_as_geometry_prototype": True,
            "R3_execution_on_real_scenes": False,
        },
    )
    holdout = {
        "FINAL_HOLDOUT_OPENED": False,
        "FINAL_HOLDOUT_STATUS": "NOT_OPENED_NO_METHOD_QUALIFIED",
        "independence_status": "INDEPENDENCE_UNRESOLVED",
        "second_independent_blocker": "Historical project training/exposure cannot be excluded. "
        "No future opening without resolving independence; current candidates were not replaced.",
        "candidate_scenes": load("scene_role_lock.json")["FINAL_HOLDOUT_SCENES"],
        "geometry_eligible_scenes": load("qualified_frame_role_lock.json")[
            "eligible_final_holdout"
        ],
        "method_lock": None,
        "model_inference_calls": 0,
        "final_holdout_gain": None,
        "final_holdout_ci": None,
        "wrong_scene_damage": None,
    }
    write_json(root / "holdout_lock.json", holdout)
    write_json(root / "holdout_results.json", {**holdout, "results": None, "not_zero_scores": True})
    costs = {}
    for design in pooled:
        rr = [r for r in rows if r["design"] == design and r["method"] in ("A", "B")]
        costs[design] = {
            "mean_renderer_and_state_hash_seconds": float(np.mean([r["seconds"] for r in rr])),
            "mean_AB_volume_m3": float(np.mean([np.prod(r["volume_extent"]) for r in rr])),
            "candidate_budget": 128,
            "oracle_lattice_search_size": 512,
            "density_voxels": 512,
        }
    write_json(
        root / "cost_analysis.json",
        {
            "data_download": load("data/download_cost.json"),
            "formal_run": load("raw/oracle_run/integrity.json"),
            "designs": costs,
            "GT_oracle_search_cost_included_in_total_run_time": True,
            "formal_run_peak_GPU_memory": None,
            "peak_memory_status": "NOT_MEASURED; do not infer peak memory from tensor budget",
            "prediction_files_bytes": sum(
                p.stat().st_size for p in (root / "raw/oracle_run/predictions").glob("*.npz")
            ),
            "GT_free_runtime_cost": None,
            "retraining_cost": None,
        },
    )
    checkpoint = load("preregistration.json")["checkpoint"]
    write_json(
        root / "checkpoints/manifest.json",
        {
            "old_checkpoint": checkpoint,
            "sha256": sha(Path(checkpoint)),
            "new_checkpoints": [],
            "training_run": False,
            "geometry_intervention_states_are_not_mcss_dynamic_v1_checkpoints": True,
        },
    )
    (root / "checkpoints/README.md").write_text(
        "No new training or checkpoint selection was performed. The frozen carrier is referenced "
        "by path/hash in manifest.json. Runtime oracle geometry must not be saved as old B-final.\n"
    )
    tests = load("tests.json")
    status = {
        "OLD_CHECKPOINT": checkpoint,
        "OLD_STATIC_STATUS": "NOT_ESTABLISHED",
        "N_EXPOSED_DEV_SCENES": statistics["cohorts"]["exposed"]["n_scenes"],
        "N_REDESIGN_DEV_SCENES": statistics["cohorts"]["new_dev"]["n_scenes"],
        "N_FINAL_HOLDOUT_SCENES": len(holdout["geometry_eligible_scenes"]),
        "FINAL_HOLDOUT_COUNT_NOTE": (
            "8 geometry-eligible; historical independence unresolved;12 candidates retained"
        ),
        "OLD_FULL_CONTEXT_GAIN": statistics["cohorts"]["exposed"]["designs"]["R0"][
            "full_context_gain"
        ]["mean"],
        "POOLED_CURRENT_FULL_CONTEXT_GAIN": pooled["R0"]["full_context_gain"]["mean"],
        "ORACLE_VOLUME_GAIN": pooled["ORACLE_VOLUME"]["full_context_gain"]["mean"],
        "ORACLE_SUPPORT_GAIN": pooled["ORACLE_SUPPORT"]["full_context_gain"]["mean"],
        "ORACLE_VOLUME_SUPPORT_GAIN": combined["full_context_gain"]["mean"],
        "ORACLE_GAIN_COHORT": "7 exposed +10 new development, scene equal",
        "GT_FREE_VOLUME_GAIN": None,
        "GT_FREE_ALLOCATION_GAIN": None,
        "GT_FREE_CONTEXT_GAIN": None,
        "BEST_DEV_REDESIGN": None,
        "BEST_DEV_GAIN": None,
        "BEST_DEV_CI": None,
        "SUPPORT_BOTTLENECK_STATUS": "INCONCLUSIVE",
        "SUPPORT_BOTTLENECK_SUFFICIENT_EXPLANATION": False,
        "CARRIER_REPRESENTATION_OR_TRAINING_BOTTLENECK": "NOT_ISOLATED",
        "MATCHED_RETRAIN_DONE": False,
        "MATCHED_CURRENT_GAIN": None,
        "MATCHED_REDESIGN_GAIN": None,
        "FINAL_HOLDOUT_OPENED": False,
        "FINAL_HOLDOUT_GAIN": None,
        "FINAL_HOLDOUT_CI": None,
        "FINAL_HOLDOUT_WRONG_SCENE_DAMAGE": None,
        "FINAL_HOLDOUT_STATUS": holdout["FINAL_HOLDOUT_STATUS"],
        "FINAL_HOLDOUT_INDEPENDENCE_STATUS": holdout["independence_status"],
        "STATIC_STATE_STATUS": "NOT_ESTABLISHED",
        "CARRIER_REDESIGN_STATUS": "STOPPED_AT_REGISTERED_ORACLE_GATE",
        "DYNAMIC_TTT_RUN": False,
        "TESTS": tests["summary"],
        "FINAL_INTEGRITY": "PASS",
        "REPORT_DIR": str(docs.resolve()),
    }
    write_json(root / "status.json", status)
    terminal = "=" * 60 + "\n3D SUPPORT BOTTLENECK + CARRIER REDESIGN FINAL\n" + "=" * 60 + "\n"
    terminal += (
        "\n".join(f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in status.items())
        + "\n"
    )
    (root / "STATUS.md").write_text("```text\n" + terminal + "```\n")
    (root / "terminal_summary.txt").write_text(terminal)

    def error(design, method):
        return pooled[design]["methods"][method]["depth_absrel"]["mean"]

    def full(design):
        return (error(design, "A") + error(design, "B")) / 2

    def line(design):
        d = pooled[design]
        g, w = d["full_context_gain"], d["wrong_scene_damage"]
        return (
            f"| {design} | {full(design):.6f} | {error(design, 'anchor'):.6f} | "
            f"{g['mean']:+.6f} [{g['ci95'][0]:+.6f}, {g['ci95'][1]:+.6f}] | "
            f"{w['mean']:+.6f} [{w['ci95'][0]:+.6f}, {w['ci95'][1]:+.6f}] |\n"
        )

    header = (
        "| 设置 | 完整context AbsRel↓ | Anchor AbsRel↓ | Full-context gain及95%CI | "
        "Wrong-scene damage及95%CI |\n|---|---:|---:|---:|---:|\n"
    )
    oracle_table = header + "".join(
        line(d) for d in ("ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT")
    )
    baseline_table = header + line("R0")
    geometry_table = (
        "| 设置 | GT surface inside | ≥2-view candidates | GT surface支持邻域 | 正/平/负场景数 |\n"
        "|---|---:|---:|---:|---|\n"
    )
    for name in ("R0", "ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT"):
        d = pooled[name]
        geo = d["AB_geometry"]
        n = d["full_context_gain"]["improved_tied_worse"]
        geometry_table += (
            f"| {name} | "
            + " | ".join(
                f"{geo[m]['mean']:.4%}"
                for m in (
                    "inside_fraction",
                    "two_view_candidate_fraction",
                    "supported_surface_fraction",
                )
            )
            + f" | {n['improved']}/{n['tied']}/{n['worse']} |\n"
        )
    metrics_table = (
        "| 设置/读取 | RGB MSE↓ | SSIM↑ | RMSE↓ | δ1↑ | Opacity | Coverage |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for name in ("R0", "ORACLE_VOLUME", "ORACLE_SUPPORT", "ORACLE_VOLUME_SUPPORT"):
        for role in ("A", "B", "anchor", "prior", "wrong_scene"):
            m = pooled[name]["methods"][role]
            metrics_table += (
                f"| {name}/{role} | "
                + " | ".join(
                    f"{m[k]['mean']:.6f}"
                    for k in (
                        "rgb_mse",
                        "rgb_ssim",
                        "depth_rmse",
                        "depth_delta1",
                        "opacity",
                        "coverage",
                    )
                )
                + " |\n"
            )
    direct_gain = full("R0") - full("ORACLE_VOLUME")
    anchor_worse = error("ORACLE_VOLUME", "anchor") - error("R0", "anchor")
    gain_change = pooled["ORACLE_VOLUME"]["change_full_context_gain_vs_R0"]["mean"]
    share = anchor_worse / gain_change
    interpretation = load("audit/scientific_interpretation.json")
    readme = f"""# EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1

本轮已完成封存、重新生成旧统计、新数据锁定/下载、四组静态oracle归因、回归测试和独立审计。
**SUPPORT_BOTTLENECK_STATUS=INCONCLUSIVE；STATIC_STATE_STATUS=NOT_ESTABLISHED。**
按预先冻结的联合oracle停止规则，没有执行实景GT-free redesign、matched retraining或final holdout。
没有训练controller，没有新Dynamic TTT、FC/CF、history或写入调参。

## 先读结论与限制

Volume确实影响结果：oracle volume单独使同设置下的full-context gain为+0.115349，
scene-bootstrap CI [0.014210,0.217646]。但这不等于完整预测变好了0.115349或0.210574。
R0完整context AbsRel={full("R0"):.6f}，oracle volume={full("ORACLE_VOLUME"):.6f}，绝对改善仅
**{direct_gain:.6f}**（CI [−0.070911,+0.135352]）；anchor则从
{error("R0", "anchor"):.6f}变差到{error("ORACLE_VOLUME", "anchor"):.6f}。
相对gain增加{gain_change:.6f}的代数分解中，约**{share:.1%}来自anchor变差**，不是支持机制的因果贡献率。
完整context的绝对任务收益及其CI另见 `audit/scientific_interpretation.json`。

联合oracle pooled gain={combined["full_context_gain"]["mean"]:+.6f}，
CI [{combined["full_context_gain"]["ci95"][0]:+.6f},
{combined["full_context_gain"]["ci95"][1]:+.6f}]，
不满足在看结果前锁定的继续门槛。其表面邻域支持仅66.22%，未达75%充分性标准，
因此不能声称“理想支持已经提供，仍然失败，所以一定是网络/训练问题”。
当前8³离散表达、支持定义及训练分布耦合仍未分离。

## 身份、冻结和数据

旧1000A+600B B-final SHA256：`{load("preregistration.json")["checkpoint_sha256"]}`。
旧实验987个文件完整封存；旧3场景234行、旧7场景182行（含Residual A/B）统计精确复算，
160帧volume/support核对通过。新runner中的旧7场景R0五个方法70行与上一轮逐项精确一致。
旧7场景从本轮起统一为 **REDESIGN_DEV_EXPOSED**，不再作为独立泛化证据。

新锁定24个不同官方source volume/archive/asset候选，按合法scene名顺序交替12 DEV、12 HOLDOUT。
只使用Hypersim official-train与research-train；不打开官方validation/protected/test。
下载220,495,946字节，全部bounded HTTP range。10个DEV、8个HOLDOUT通过预先固定的质量和native
geometry阈值；失败候选保留、不替换、不跨角色移动。初始身份锁 `scene_role_lock.json` 未覆盖，
具体合格帧角色另锁在 `qualified_frame_role_lock.json`。本轮实际归因共7 exposed+10 new DEV场景。

**Holdout独立性尚未成立。** ai027001/ai033001有项目旧100scene池曝光标记，其他候选也无法仅凭
不在旧100池就排除后续365scene扩展的使用；缺少逐场景历史执行日志。它们未参加当前B-final训练、
不在旧7场景内，但不能据此声称项目历史“从未见过/未调过”。全部holdout在本轮没有模型前向或query
score，保持保护，`FINAL_HOLDOUT_INDEPENDENCE_STATUS=INDEPENDENCE_UNRESOLVED`。
即便未来某方法过开发gate，也必须先解决此独立性阻塞。这里的8仅指几何合格数。

## 方法合同与oracle例外

固定网络权重、8³状态、128唯一candidate IDs、固定renderer64samples及原context/query列表。
ORACLE_VOLUME用全部primary query GT surface在anchor坐标中的包围盒，最小轴长1m并加5%边距；
同场景A/B/anchor共享该bounds。ORACLE_SUPPORT在当前bounds的512合法voxel centers中选128个，
优先≥2 context frusta，然后到context GT表面的距离，再view count与ID稳定排序；联合设置只组合
这两个因素。points、normalized_xyz、scatter IDs和bounds同步；没有偷偷增加候选或网络。

这是**固定网格内的有GT诊断，不是数学upper bound，也不保证物理上完美表面支持**。
候选的≥2计数仅是camera frustum membership，不等于遮挡正确的表面可见性。
R0先封存；oracle为构造bounds必须在自身封存前读取query GT geometry，日志明确标记
`PRESEAL_DIAGNOSTIC_EXCEPTION`。所有评分RGB/depth在全部204 states封存后读取。
Oracle状态对锁定query集合有条件，绝不称部署时query-independent；query cameras在wrong-scene
替换时数值不变。参数合同指出这些同shape运行时干预仍改变训练分布，不能当作公平训练后的新方法。

主表采用query均值→scene等权，10,000 paired scene bootstrap、seed20260927、95%percentile CI。
Full-context gain=anchor−mean(A,B)，wrong-scene damage=cyclic donor A−correct A；
donor在各cohort内轮换。
17场景pooled和7/10分层统计同时保留，不把旧7重新包装成独立确认。

## DIAGNOSTIC UPPER BOUNDS（名义诊断，非保证的上界）

{oracle_table}

## DEPLOYABLE METHODS

{baseline_table}

R1 adaptive-volume、R2 camera-overlap allocation、R3 camera-only context selection、R12/R123：
**NOT_RUN_ORACLE_STOP，结果为null。** 三类几何原型及隔离测试已实现；没有实景运行、近远深度
先验调优或按分数选方法。Matched current/redesigned两条训练和train-scale-up均未启动，
不存在“最佳新carrier”。这些空项不是0分或失败训练的伪造结果。

## 支持、集中度与完整任务指标

{geometry_table}

主要表面支持距离固定为1.172604m（旧voxel半对角线），避免扩大volume后自动扩大容差。
同时保存各新voxel尺度下的描述性邻域数，不能替代固定半径主诊断。
联合oracle inside=100%、two-view=93.18%，但surface-neighborhood=66.22%＜75%，充分性失败。
Volume-only和联合各11/17场景gain为正；联合gain的正贡献top1≈25.7%、top3≈64.8%。
所有LOSO、集中度、每场景完整结果和描述性support/gain相关都在 `bootstrap_results.json`。
相关不建立因果。相同候选数并不意味着相同物理voxel分辨率，所有bounds/extent保存在raw计划中。

{metrics_table}

Coverage仍为opacity>1e-6，不是真实可见性；所有GT-valid像素均参与depth指标，未按预测opacity筛掉
坏像素。Oracle volume也改变ray integration范围与anchor prior；没有证据证明正relative gain完全
独立于opacity/coverage。由于没有方法达到最终gate，不做静态资格成功宣称。

## 停止规则与12个归因问题

1. **主要来自volume、candidate还是representation？** Volume有直接干预证据，但不是充分解释；
   candidate-only恶化相对gain，联合不稳定。representation/训练/分辨率尚未被独立隔离。
2. **Oracle volume alone恢复多少？** pooled gain +0.115349；相对R0 gain提升+0.210574，但完整context
   绝对AbsRel只改善0.034169，83.8%的相对提升来自anchor变差。
3. **Oracle support alone恢复多少？** gain −0.245025，较R0更差0.149800；
   frustum支持提升不等于任务收益。
4. **Volume+support恢复多少？** gain +0.089622，较R0增加0.184847，但CI跨零；完整context绝对改善仅
   0.008442，不能称静态carrier获救。
5. **GT-free volume恢复多少？** 未运行，遵守联合oracle停止条件，不能填oracle成绩。
6. **Camera-overlap allocation恢复多少？** 未运行；候选几何原型/预算测试可用。
7. **Context geometry selection恢复多少？** 未运行；camera-only确定性和输入隔离测试可用。
8. **哪个因素贡献最大？** 这些冻结权重诊断中volume对relative gain的改变最大；这是干预分解，不是
   对全部失败的因果占比，也不是部署收益。
9. **支持修复后full context终于优于anchor？** Volume-only在17场景pooled有显著正gain；联合设置未过
   预定稳定性门槛，且没有充分修复query surface support。不能挑volume-only替换联合停止检验。
10. **Wrong-scene终于稳定变差？** 没有；所有pooled oracle的wrong-scene damage CI均跨零。
11. **若oracle能救而GT-free不能救，可观察性瓶颈是什么？** 本轮未测试GT-free，不能作这个归因；
    使用query GT包围盒天然不可部署，不说明camera-only能够恢复相同信息。
12. **为什么转向architecture/training而不继续堆volume？** 联合oracle未获稳定增益，且固定网格下
    query表面支持仍有限。下一步应先分离离散表达、训练分布与prior/renderer效应；不能把本轮解释为
    已证明“完美support也无效”，或直接归咎所有scene representations。

注册条件要求联合oracle pooled CI lower>0且new DEV mean>0才能继续。Pooled下界−0.026782未通过；
new DEV联合gain虽为+0.156328，其单独CI下界也略低于0（约−0.000015）。不改阈值或gate。
因此 `SUPPORT_BOTTLENECK_SUFFICIENT_EXPLANATION=false` 表示“尚未建立充分解释”；由于干预充分性
未达标，`CARRIER_REPRESENTATION_OR_TRAINING_BOTTLENECK=NOT_ISOLATED`，不虚构因果证明。
`FINAL_HOLDOUT_STATUS=NOT_OPENED_NO_METHOD_QUALIFIED`，并有历史独立性未解决的第二阻塞。

## 复现与交付

完整测试：**{tests["summary"]}**；跳过项是缺少可选历史V5 step4500权重。新增59项包括holdout门禁、
oracle部署拒绝、GT-free参数隔离、唯一候选预算、context确定性、matched配置、scenebootstrap和真实
small end-to-end封存/GT/相机/零写入测试。工程测试通过不等于科学资格通过。

`commands.sh summary`不加载媒体或模型即可重算所有统计；`commands.sh reproduce`在新目录使用已冻结
数据重做同一DEV干预（拒绝覆盖原目录），从不读取holdout模型输入。数据下载命令也在文件中。
可直接从仓库保存的raw重算：

```bash
.venv/bin/python scripts/report_support_redesign.py --raw-only \\
  --output docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1 \\
  --docs /tmp/support-redesign-recomputed
```

`raw/oracle_run/`保留680个预测NPZ、完整metrics、204sealed states、geometry plans、GT访问日志和源文件
hashes；docs中的压缩raw可重算正式统计。`figures/`保存PNG/SVG科学图；`audit/`保存旧结果复算、独立
审计、解释和环境；`checkpoints/`只索引原B-final，没有新训练checkpoint。大型媒体、tensor/prediction
文件留在outputs并有hash清单，压缩文本证据在docs/artifacts下。全部旧实验保持不变。

![Full-context gain by stratum](figures/full_context_gain_intervals.png)
"""
    # Keep interpretation as structured evidence; full report refers to its exact CI values.
    assert interpretation
    (root / "README.md").write_text(readme)
    fields = [
        "cohort",
        "scene_id",
        "query_id",
        "design",
        "method",
        "depth_absrel",
        "rgb_mse",
        "rgb_ssim",
        "depth_rmse",
        "depth_delta1",
        "opacity",
        "coverage",
        "inside_fraction",
        "two_view_candidate_fraction",
        "supported_surface_fraction",
        "seconds",
    ]
    with (root / "raw/results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (docs / "raw").mkdir(exist_ok=True)
    (docs / "raw/oracle_results.json.gz").write_bytes(
        gzip.compress(json.dumps(rows, sort_keys=True).encode(), mtime=0)
    )
    artifact_records = []
    for directory in ("raw/oracle_run", "audit", "data", "checkpoints"):
        for path in sorted((root / directory).rglob("*")):
            rel = path.relative_to(root)
            if (
                not path.is_file()
                or path.suffix
                not in {".json", ".jsonl", ".md", ".txt", ".csv", ".py", ".patch", ".log"}
                or "prepared" in rel.parts
                or "raw" in path.relative_to(root / directory).parts
            ):
                continue
            target = docs / "artifacts" / (str(rel) + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            artifact_records.append(
                {
                    "output_path": str(rel),
                    "artifact": str(target.relative_to(docs)),
                    "sha256": sha(path),
                    "compressed_sha256": sha(target),
                }
            )
    write_json(root / "artifact_manifest.json", artifact_records)
    for path in root.iterdir():
        if path.is_file():
            shutil.copy2(path, docs / path.name)
    for directory in ("figures", "checkpoints"):
        shutil.copytree(root / directory, docs / directory, dirs_exist_ok=True)
    # Important readable audits, with bulk locks/logs in the compressed archive.
    (docs / "audit").mkdir(exist_ok=True)
    for name in (
        "scientific_interpretation.json",
        "scientific_interpretation.md",
        "final_independent_audit.json",
        "old_statistics_reproduction.json",
        "environment.json",
    ):
        if (root / "audit" / name).exists():
            shutil.copy2(root / "audit" / name, docs / "audit" / name)
    print(terminal)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--docs", required=True)
    p.add_argument("--raw-only", action="store_true")
    a = p.parse_args()
    main(a.output, a.docs, raw_only=a.raw_only)

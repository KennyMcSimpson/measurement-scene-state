# Measurement Scene State / Inverse-JEPA + Dynamic-TTT research archive

2026-09-30 [V16](outputs/EXP-3D-RGBD-VOLUME-DISJOINT-V16/README.md)：在与训练集 volume 不相交的 FRESH-V2 上，TRAIN72 carrier 的补全优势平均 +0.185 但极不一致（8 好 6 差，CI 跨零），未能证实补全依赖对训练 volume 的熟悉（分支 D）。

2026-09-30 [V15](docs/experiments/EXP-3D-RGBD-WIDTH-DATA-V15/README.md)：在 127 个训练场景上把宽度从 32 加到 64，补全仍没有改善（−0.003，CI 跨零），组合与经典补洞打平（分支 C）；宽度 × 数据的 2×2 中两个因素都不改善补全。

2026-09-30 [V14](docs/experiments/EXP-3D-RGBD-DATA-SCALE-V14/README.md)：训练场景从 72 增加到 127，改善了被看到区域的精度（NEAR 0.288→0.246），但补全略差（FAR 0.421→0.443）；组合方法与经典补洞打平（分支 C）。

2026-09-30 [V13](outputs/EXP-3D-RGBD-VOLUME-V13/README.md)：状态体积之外的弃权回退（不透明度 < 0.5 改用调和插值补洞）在 79 个场景的远处 query 上稳定改善组合方法（+0.011，CI 下界 +0.005），但仍不能确立优于经典补洞（+0.016，CI 跨零，分支 B）。

2026-09-30 [Replica 跨数据集迁移](outputs/EXP-3D-RGBD-REPLICA-TRANSFER-V1/README.md)：冻结的方法显著差于经典几何补洞（−0.225，分支 C），学到的状态没有迁移；[V12](docs/experiments/EXP-3D-RGBD-WIDTH-V12/README.md)：宽度 64 没有改善补全（分支 C）。

2026-09-29 [EVAL-V4 复现](outputs/EXP-3D-RGBD-REPLICATION-EVAL-V4/README.md)：26 个新场景上，组合方法相对经典补洞 +0.010，CI 跨零，与 EVAL-V3 一致（分支 B）。[V10](docs/experiments/EXP-3D-RGB-PRIOR-BOUNDS-V10/README.md)：放宽 RGB-only 的 bounds 让 DEV 误差 0.603→0.461（CI 下界 +0.022），但仍与常数无法区分（分支 B）。

2026-09-29 [V11](docs/experiments/EXP-3D-RGBD-COMPLETION-V11/README.md) 与 [INPAINT-V1](outputs/EXP-3D-RGBD-INPAINT-BASELINES-V1/README.md)：加宽 carrier 没有稳定改善补全（分支 C）；"调和插值补洞 + carrier 补看不到的区域"在 EVAL-V3 上 AbsRel 0.2325，数值上最好，但相对经典补洞的优势（+0.012）CI 跨零（分支 B）。

2026-09-29 [Core B 第二轮](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V2/README.md)：在可变视角训练的 carrier 上，直接缓存新帧让 DEV AbsRel 从 0.258 降到 0.205；学习写入不再有额外收益，也不带场景信息，分支 C。

2026-09-29 [训练视角数 V9](docs/experiments/EXP-3D-RGBD-VIEWCOUNT-V9/README.md)：DEV AbsRel 0.259 / 0.257 / 0.275（固定 3 / 可变 3–7 / 固定 7 视角），VIEWCOUNT_GAIN +0.002，CI 跨零，分支 B。探索性诊断表明，让学到的状态超过几何的关键是补全质量；加宽 carrier 在未见场景上显著改善了补全。

2026-09-29 [强几何基线对照](outputs/EXP-3D-RGBD-GEOMETRIC-BASELINES-V1/README.md)：已合格的 carrier 与直接深度重投影在整体上无法区分（分支 B）。重投影在被看到的区域准得多，carrier 在未被看到的区域更好，并显著优于同表示的非学习融合。

2026-09-29 [Core B 第一轮](docs/experiments/EXP-3D-RGBD-DYNAMIC-WRITE-V1/README.md)：在冻结的合格 carrier 上学习写入规则。写入相对只缓存新帧 +0.061，但不优于静态状态和非学习的视角计数截断，也不是场景专属，分支 B。下一步用可变视角数训练 carrier（V9）。

2026-09-29 [分辨率 V8](docs/experiments/EXP-3D-RGBD-RESOLUTION-V8/README.md)：16³→32³，DEV AbsRel 0.280→0.259（24³ 为 0.244），RESOLUTION_GAIN +0.021，CI 跨零，分支 B。不做 checkpoint 选择时 32³ 好 0.061。

2026-09-29 [第二次独立资格验证](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V2/README.md)：V7 的 RGB-D carrier 在 17 个独立场景上 `QUALIFIED`。C1 AbsRel 0.290，BOUNDS_GAIN +0.145，相对常数 +0.300，四个冻结 gate 全部成立。Core A 静态 carrier 在 RGB-D 轨道上通过资格验证。

2026-09-29 [bounds 规则 V7](docs/experiments/EXP-3D-RGBD-DEPTH-BOUNDS-V7/README.md)：改用测得上下文深度确定 bounds，DEV AbsRel 0.459→0.280（BOUNDS_GAIN +0.179，CI [+0.084, +0.279]）。C1 首次在全部 DEV 场景上显著优于常数深度，预注册分支 A。下一步在 FRESH-V2（17 个独立场景）上做资格验证。

2026-09-29 [训练规模 V6](docs/experiments/EXP-3D-RGBD-TRAIN-SCALE-V6/README.md)：训练场景 24→72（CPU），DEV AbsRel 0.492→0.459。SCALE_GAIN +0.033，CI 跨零，分支 C。排除 NO_HIT 场景后，C1 比常数好 +0.111（CI >0），但只有 5/7 个场景不差于常数。下一步检验由测得深度确定的 bounds（V7）。

2026-09-29 [非学习 RGB-D 融合对照](docs/experiments/EXP-3D-RGBD-NONLEARNED-FUSION-V1/README.md)：把测得深度直接体素化进同一 16³ 网格和 renderer，DEV 上为 0.596，比常数（0.507）还差。学到的 RGB-D carrier（0.503）显著优于它，说明学习有价值，瓶颈更可能在表示与渲染这一环。

2026-09-29 [第一次独立资格验证](docs/experiments/EXP-3D-RGBD-FRESH-QUALIFICATION-V1/README.md)：V5 的 RGB-D carrier 在 5 个独立 Hypersim 场景上 `NOT_QUALIFIED`。深度收益没有复现（+0.006，CI 跨零；DEV 上为 +0.114），场景专属性与多视角融合则复现。DEV 选择偏差使 V5 的收益偏乐观，下一步检验训练规模（V6，24→72 个场景，CPU）。

2026-09-29 [RGB-D 证据 carrier V5](docs/experiments/EXP-3D-RGBD-EVIDENCE-CARRIER-V5/README.md)：上下文深度经零初始化旁路进入状态（单因素），DEPTH_GAIN +0.114（CI [+0.065, +0.166]）。`DEPTH_STATUS`、`SCENE_SPECIFICITY_STATUS`、`STATIC_DEV_STATUS` 首次同时为 `SUPPORTED`。但 C1 仍未显著优于常数深度：排除 NO_HIT 场景后为 0.432，常数约 0.493。预注册分支 B。

2026-09-29 [无几何参考再分析](docs/experiments/EXP-3D-READOUT-REFERENCE-REANALYSIS-V1/README.md)：预注册事后分析 V2–V4 全部 9 个 RGB-only carrier 的已封存 DEV 预测，无一优于只由 TRAIN 拟合的常数深度。2.37 m 常数的 DEV AbsRel≈0.507，carrier 为 0.61–0.65，全部显著更差；给定 GT 尺度后，深度形状也不如平面常数。RGB-only carrier 在 DEV 上没有携带可迁移的几何。

2026-09-29 [plane-sweep carrier V4](docs/experiments/EXP-3D-PLANE-SWEEP-CARRIER-V4/README.md)：单因素加入零初始化的光度 plane-sweep 旁路，SWEEP_GAIN +0.0072，CI [−0.0041, +0.0168]，`NOT_ESTABLISHED`，分支 C。在多视角可观测区域有描述性改善：OBS2PLUS +0.040，CI [+0.004, +0.087]。TRAIN-only 诊断显示 32×40 光度线索在真实上下文上几乎不含深度信息。下一轮进入 RGB-D 轨道（V5）。

2026-09-29 [稠密证据carrier V3](docs/experiments/EXP-3D-DENSE-EVIDENCE-CARRIER-V3/README.md)：证据候选128→4096（单因素），DEV收益+0.0101，CI跨零，`NOT_ESTABLISHED`；训练集深度AbsRel 0.490→0.339但DEV几乎不变，稠密模型均在500步最好，之后过拟合。预注册分支C：候选稀疏不是主因，下一轮研究几何推理式融合。

2026-09-29 [16³ geometry-aware learned carrier V2](docs/experiments/EXP-3D-GEOMETRY-AWARE-CARRIER-V2/README.md)：首次训练16³ learned carrier，C0/C1 matched（3 seeds）。表面监督收益+0.0195，95% CI [−0.025, +0.077]跨零，`NOT_ESTABLISHED`；wrong-scene与空间打乱不造成损伤，状态近似场景无关先验，full context未胜anchor。Dynamic TTT仍关闭。此前三轮direct-state归因（容量、优化与bounds、可观测性监督）已补入[EXPERIMENTS.md](EXPERIMENTS.md)。

2026-09-28 [Support bottleneck归因与redesign gate](docs/experiments/EXP-3D-SUPPORT-BOTTLENECK-REDESIGN-V1/README.md)：7个旧曝光＋10个新开发场景完成oracle诊断。Volume-only相对gain为正，但完整context绝对改善CI跨零，约83.8%的gain增幅来自anchor变差；联合oracle未过门槛且支持不充分，结论`INCONCLUSIVE`。按协议未继续实景GT-free／matched retraining，最终holdout未开启，历史独立性另有阻塞。

2026-09-27 [静态 carrier 归因与未见场景资格](docs/experiments/EXP-3D-STATIC-CARRIER-ATTRIBUTION-V1/README.md)：旧90条记录精确复现；7个未见场景 full-context AbsRel 0.7972，差于 anchor 0.6076，收益95% CI为[-0.2910, -0.0913]。ai003原始JPEG已全黑且A双视图支持为零；Residual完成训练但未建立通用改善。`STATIC_STATE_STATUS=NOT_ESTABLISHED`，未运行新Dynamic TTT。
协作入口：先看 [当前结果与复现说明](docs/handoff/SHARED_WORKSPACE.md)。[三维从零复现](docs/handoff/3D_REPRODUCTION.md) 提供便携脚本；大报告先运行 `python scripts/restore_report_artifacts.py` 恢复，权重与数据不随 Git 分发。

2026-09-27 [已训练权重机制补测](docs/experiments/EXP-3D-20260927-trained-state-history-v1/README.md)：同训练场景留出帧上，写入oracle AbsRel收益约0.000756，但有益历史动作翻转0/3、history额外空间0；静态context A AbsRel2.214，资格仍未建立。原训练query的1次翻转单列，不当独立证据。

2026-09-27 已完成 [1000A＋600B 三维训练](docs/experiments/EXP-3D-20260927-centered-training-1000a-600b-v1/README.md)：耗时112秒，TRAIN OFF AbsRel 为 A结束0.287、B结束0.332（短程1.229）；B相对A回退，未验证写入收益或泛化。

2026-09-27 三维修复：[锚点中心体积实验](docs/experiments/EXP-3D-20260927-centered-support-v1/README.md) 已恢复第三场景写入梯度（0/10 → 10/10），但重建指标变差；仅工程修复，保留原权重。

2026-09-27 后续探索：[折外全局轨迹与 cycle 诊断](docs/experiments/EXP-2D-followup-20260927/README.md)。12个discovery外层折都选择ALL/B；CycleGate仍低约0.209个百分点。仅属事后discovery分析，独立确认仍受阻。

V2 最新进展（2026-09-26）：[独立确认报告](docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md) 已完成旧 raw 的 global16 诊断、discovery nested LOSO、实现与测试；独立确认因外部原视频身份/标注兼容性审计未通过而标为 `BLOCKED_NO_INDEPENDENT_DATA`。discovery 的正值不能当成新测试成功。

最新进展（2026-09-26）：[Opportunity + visible selector v1](docs/experiments/EXP-2D-20260926-opportunity-selector-v1/README.md) 已完成。高分辨率评价支持小幅跨序列oracle机会，但简单部署选择器没有净收益。

这是 Kenny 的 CV 研究协作仓库。仓库把三条内容放在一起，但明确区分证据等级：

1. 原来的 `Measurement Scene State` 三维主线与 V5 比较基线；
2. 逆向 JEPA 与动态 TTT 的原始方案、ICLR 初稿和讨论记录；
3. 2026-09-21 用 DAVIS + 冻结 DINOv2 做的二维机制筛查。

当前公开内容是**可复核的研究档案和工程代码**。它还没有证明一个完整的逆向 JEPA + 动态 TTT 方法，也没有官方 DAVIS、DL3DV、Hypersim 或 CVPR 结果。请先阅读 [STATUS.md](STATUS.md) 和 [EXPERIMENTS.md](EXPERIMENTS.md)，再运行代码或提出下一步。

## 先看什么

- [STATUS.md](STATUS.md)：哪些东西已经验证，哪些只是开发或诊断证据。
- [EXPERIMENTS.md](EXPERIMENTS.md)：每个实验的目的、划分、命令、结果和不能声称的内容。
- [ROADMAP.md](ROADMAP.md)：下一步的门槛和停止条件。
- [docs/handoff/REPOSITORY_MAP.md](docs/handoff/REPOSITORY_MAP.md)：给合作者或 AI 的文件导航。
- [research/ideas/](research/ideas/)：原始中文方案、二维切口分析和首轮结果报告。
- [docs/provenance/CLAIMS.md](docs/provenance/CLAIMS.md)：`CONTRACT`、`ENGINEERING`、`EXPLORATORY`、`DIAGNOSTIC_ORACLE` 等证据标签的含义。

## 当前最重要的二维结果

二维实验使用 DAVIS-2017 train 的 24/12/16/8 序列划分，官方 val 30 段封存；冻结 DINOv2-S/14 输出 16×16×384 token。评价是 16×16 token 网格上的对象 mask transfer IoU，不是 DAVIS 官方 J&F。

| 方法 | validation object IoU | RGB MSE |
|---|---:|---:|
| OFF/OFF | 44.34% | 0.03022 |
| discovery 选出的固定 OFF/ALL | 45.08% | 0.01770 |
| RGB residual 动态 selector | 44.03% | 0.01250 |
| 答案可见的有限轨迹 oracle（诊断） | 45.51% | 0.02407 |

动态 selector 让 RGB MSE 下降约 58.6%，但对象对应没有改善。这是当前方向需要解决的具体问题：**能解释观测，不等于能保留对任务有用的状态**。完整数据与核验文件在本地 `outputs/vision_2d_probe_20260921_corrected/`，公开仓库的精简 bundle 在 `docs/experiments/EXP-2D-20260921-corrected/`。

## 安装和最小检查

需要 Python 3.11–3.12。锁文件 `uv.lock` 已纳入仓库；GPU 运行需要匹配的 PyTorch/CUDA 环境。原始数据和权重不随仓库分发，下载说明与本地完整性记录见 [docs/provenance/DATA.md](docs/provenance/DATA.md)。

```powershell
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

二维 probe 的 focused checks：

```powershell
uv run pytest tests/test_vision_probe_adaptation.py tests/test_vision_probe_geometry.py tests/test_vision_probe_runner.py -q
uv run ruff check scripts/run_vision_probe.py scripts/analyze_vision_probe.py scripts/inspect_vision_probe.py src/mcss/vision_probe tests/test_vision_probe_adaptation.py tests/test_vision_probe_geometry.py tests/test_vision_probe_runner.py
```

下载数据后，使用新的输出目录运行 probe；不要覆盖已封存的结果目录：

```powershell
uv run python scripts/prepare_2d_probe_assets.py
uv run python scripts/run_vision_probe.py `
  --data-root data/vision_2d_probe/davis2017_trainval_480p/DAVIS `
  --weights data/vision_2d_probe/weights/dinov2_vits14_pretrain.pth `
  --backbone-repo third_party/dinov2 `
  --output outputs/vision_2d_probe_reproduction `
  --device cuda --eta 0.5 2 5
uv run python scripts/analyze_vision_probe.py outputs/vision_2d_probe_reproduction
```

## 公开范围

仓库包含源代码、测试、协议、精简机器可读结果、原始研究笔记和 ICLR 初稿。以下内容保留在本机但不上传：DAVIS/Hypersim/DL3DV 数据、DINOv2 和其他模型权重、feature cache、checkpoint、`.venv`、大型生成日志以及下载包。这样合作者可以按来源和 hash 重新取得输入，也不会把数据许可证或 GitHub 单文件限制混进研究代码。

`third_party/` 中的外部依赖以固定 Git 链接记录，不把上游代码重新许可为本项目代码；克隆后运行 `git submodule update --init --recursive`。项目自己的代码适用 [LICENSE](LICENSE)，第三方代码和论文草稿不因该文件自动获得同一许可证。详细边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 贡献方式

先看 [CONTRIBUTING.md](CONTRIBUTING.md)。每次实验都要留下：固定协议、数据/代码 hash、精确命令、测试退出码、原始记录、结果解释和不能声称的内容。不要把某个开发集或 oracle 数字写成最终 benchmark。

```powershell
cd <workspace-root>
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run mcss train --config configs\synthetic_smoke.yaml
uv run mcss evaluate --config configs\synthetic_smoke.yaml `
  --checkpoint outputs\synthetic_smoke\checkpoints\step_000008.pt
uv run python scripts\cuda_smoke.py
```

The synthetic command is self-contained and does not download data. It writes checkpoints,
JSONL training logs, and evaluation JSON below `outputs/`, which is ignored by Git.
The CUDA smoke checks forward and backward execution for all three model modes under both the
legacy and evidence-residual state architectures. It exits with a clear message when the installed
PyTorch runtime cannot see a CUDA device.

## Commands

```text
mcss train --config CONFIG.yaml
mcss evaluate --config CONFIG.yaml --checkpoint CHECKPOINT.pt
mcss diagnose --config CONFIG.yaml --checkpoint CHECKPOINT.pt
mcss download replica --root data\replica_raw --dry-run
mcss download hypersim --root data\hypersim_raw --scenes ai_001_001 ai_001_002 --dry-run
mcss prepare replica --raw-root data\replica_rendered --output-root data\replica_prepared
mcss prepare hypersim --raw-root data\hypersim_raw --output-root data\hypersim_prepared --scenes ai_001_001
```

`docs/DATASETS.md` contains the official sources, disk estimates, subset workflow, and the
Replica rendering boundary. `docs/PROTOCOL.md` defines matched comparisons, held-out measurement
tests, metrics, and mechanism diagnostics.

The old verified 60/20/20 Hypersim subset and A+ configs remain diagnostic provenance. The new
protocol plans all 365 official train scenes, all 46 validation scenes, the 20 previously observed
test scenes as diagnostic-only, and the remaining 26 test scenes as a sealed final holdout. Inspect
and materialize the expansion with:

```powershell
uv run python scripts\plan_hypersim_expansion.py --dry-run
uv run python scripts\plan_hypersim_expansion.py
uv run python scripts\materialize_hypersim_expansion.py `
  --stage all --partitions train val diagnostic_test --workers 2
```

The materializer is resumable, preserves at least 100 GiB free by default, and refuses to touch
`final_holdout` without the explicit `--allow-final-holdout` release flag.

Before a main run, execute the real-data plumbing smoke and one-window overfit gate:

```powershell
uv run mcss train --config configs\hypersim_er_smoke.yaml
uv run mcss train --config configs\hypersim_er_mainres_smoke.yaml
uv run mcss train --config configs\hypersim_er_overfit.yaml
```

New JSONL logs include current, log-window mean, and EMA losses together with epoch progress,
learning rate, gradient norm, AMP scale/update status, scene IDs, modality validity, step time,
peak CUDA memory, evidence/unknown means, completion-gate use, and density/color residual energy.
Only after the overfit loss shows sustained reduction should the 150,000-microstep main config be
started with `configs\hypersim_er_train.yaml`.

To locate an RGB representation ceiling before changing the production model, run the
diagnostic-only appearance oracle against the accepted one-window checkpoint:

```powershell
uv run python scripts\run_appearance_oracle.py `
  --config configs\hypersim_er_overfit.yaml `
  --checkpoint outputs\hypersim_er_v4_scale4_overfit\checkpoints\step_005000.pt `
  --output-dir outputs\appearance_oracle_v1 `
  --steps 1000 `
  --learning-rate 0.05
```

This command intentionally optimizes hidden target RGB labels to estimate three unattainable
capacity ceilings: native shared color, doubled-resolution shared color, and one native color
volume per target view. Its outputs are diagnostics only. They are not model results, must not be
compared with baselines, and cannot be used on validation or test partitions.

The converged oracle selected doubled spatial appearance resolution: native shared color reached
19.04 dB, native per-view color 19.16 dB, and doubled-resolution shared color 22.17 dB. V5 therefore
adds only a typed high-resolution evidence-residual appearance field; it does not add a directional
or target-view-specific state. Run its gates in order:

```powershell
uv run mcss train --config configs\hypersim_er_v5_appearance2x_smoke.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_mainres_smoke.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_probe.yaml
uv run mcss train --config configs\hypersim_er_v5_appearance2x_overfit.yaml
uv run mcss evaluate --config configs\hypersim_er_v5_appearance2x_overfit.yaml `
  --checkpoint outputs\hypersim_er_v5_appearance2x_overfit\checkpoints\step_005000.pt
```

Do not start `configs\hypersim_er_v5_appearance2x_train.yaml` until the fresh 5,000-step V5
overfit result is evaluated. Its validation companion is
`configs\hypersim_er_v5_appearance2x_val.yaml`.

### V5 context-subset geometry consistency

The approved next direction is an opt-in training-only consistency path attached directly to the
unchanged V5 evidence-residual state. It removes one of the four context views, reuses the same
encoder and fixed renderer, and compares only counterfactual depth and visibility against a
detached full-context prediction on evidence-supported overlap. It does not add a model module,
state-dict key, target-label input, RGB consistency term, or inference branch. The strict design and
gates are in
[`docs/superpowers/specs/2026-08-14-v5-context-subset-geometry-consistency-design.md`](docs/superpowers/specs/2026-08-14-v5-context-subset-geometry-consistency-design.md).

The default `context_subset_geometry_weight` is zero, so existing V5 configs retain the original
training path. Run only the self-contained CPU software smoke while validating the implementation:

```powershell
uv run mcss train --config configs\synthetic_er_v5_context_subset_geometry_smoke.yaml
```

`configs\hypersim_er_v5_context_subset_geometry_train_v1.yaml` is a frozen, unexecuted **10,000
micro-step diagnostic pilot**, not the 150,000-step main run. It matches the V5
four-context/two-target Hypersim protocol with weight `0.1` and the existing 3,000-update warmup.
Do not start it until the design's one-window capacity, held-out context-drop, and eight-scene
validation gates are predeclared and evaluated. A lower consistency loss alone is not evidence
that it should replace V5.

## Current evidence boundary

The new geometry-certificate frontend has completed its pre-registered zero-training audit and is
sealed as `STOP_CERTIFICATE_FRONTEND`. Across 32 hash-selected Hypersim development-train scenes,
its certified source-grid coverage was 2.60%, median post-hoc 3D error was 1.52 m, p90 error was
12.24 m, and view-permutation point delta was 1.10 cm. These fail the frozen 15%, 0.25 m, 0.50 m,
and numerical-invariance gates. Do not attach this exact LoFTR plus reciprocal/cycle/DLT frontend
to the state or train it. The result and full report are under
`outputs/geometry_certificate_audit_v1/`; the failure seal is
`docs/superpowers/specs/2026-08-12-geometry-certificate-frontend-failure-seal.md`.

This failure does not alter the original context-to-typed-state-to-fixed-renderer backbone. It
rules out the current sparse certificate inlet. It blocked further code changes until a new
direction was discussed and approved; the 2026-08-14 V5 context-subset design discharges that
approval checkpoint without reopening the failed certificate branch.

The approved V7 branch qualifies a frozen MapAnything model as an external observation supplier.
It reads only context RGB and known cameras, seals structured supplier and certificate caches,
and opens context depth only for post-hoc evaluation. It does not modify the typed state, renderer,
or training loop. Both MapAnything and its DINOv2 architecture-code dependency are pinned and
verified locally; independent DINOv2 weights and remote hub loading are forbidden. Run the pinned
audit with:

```powershell
uv sync --extra dev --extra mapanything
uv run python scripts\run_mapanything_geometry_audit.py `
  --config configs\hypersim_mapanything_geometry_audit.yaml
```

The frozen contract is in
`docs/superpowers/specs/2026-08-12-v7-mapanything-certified-observation-design.md`.

The accepted V4 geometry checkpoint reaches depth AbsRel 0.00918, mean normal error 6.39 degrees,
point F-score@0.25 m 0.9934, and RGB PSNR 18.19 dB. The V5 implementation keeps that geometry and
adds a doubled-resolution typed appearance field selected by the training-only oracle. Its 200-step
real-data probe reduces the RGB training term from 0.1776 at step 10 to 0.0795 at step 200, versus
0.0932 for V4 at the same step, while keeping native and appearance evidence exactly invariant.
This is a capacity probe, not a final metric. The fresh V5 5,000-step overfit remains the next gate.
The full 365-scene train, 46-scene validation, and 20-scene diagnostic-test partitions are prepared;
the 26-scene final holdout remains unmaterialized and sealed. No main-run, matched-baseline win, or
CVPR-level empirical claim has been established yet.


## 2026-09-27 三维小规模工程训练

已完成3个train场景上的100步静态+30步写入训练，生成新的 `mcss.dynamic.v1` 权重。严格重载与写入重放通过；全套测试561 passed、1 skipped。一个场景因固定体积缺乏多视图候选支持而全部写入梯度为零，保留负例；当前仅工程可用，未取得独立静态/泛化资格。

详见 [训练报告](docs/experiments/EXP-3D-20260927-small-training-v1/README.md) 和 [权重索引](docs/experiments/EXP-3D-20260927-small-training-v1/checkpoint_index.json)。

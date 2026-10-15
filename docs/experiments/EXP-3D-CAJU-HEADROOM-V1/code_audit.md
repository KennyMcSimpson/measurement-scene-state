# 代码审计：Core B 实际优化什么（CAJU-V1 第一阶段）

审计时间：2026-10-14。地点：hit，`~/zonghan_o_mirror/measurement-scene-state`，main 4f0c0f6，工作区干净。审计时没有发现其他 agent 的 CAJU 分支、目录或进程。下文中，`V1` 指 `outputs/EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1/code/`，`V3` 指 `outputs/EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V3/code/`。

## 1. 执行路径

- **GPU 部分**：Core B 最近一次正式执行（V3，ARKIT_SYSTEM3）的 GPU 部分是 `V3/sys3_jobs.sh` 第 17–21 行。它调用 V1 锁定的 `ds_eval.py`（sha256 `36b05c497cae…`，与 ABLATION-V1 的副本相同），参数为 `--model ZS --scale-max 0.1 --arms CACHE TTT --ttt-mult 1 --ttt-steps 100`，分 4 个分片。
- **读出**：在 `V3/sys3_score.py` 第 74–97 行。

## 2. 模型和权重

- **权重**：发布版 DepthSplat，`depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth`（`ds_predict.py` 第 35–38 行）。
- **配置**：hydra 覆盖项为 `+experiment=dl3dv`，`num_scales=2`，`upsample_factor=4`，`lowest_feature_resolution=8`，`monodepth_vit_type=vitb`，`gaussian_scale_max=0.1`（`ds_predict.py` 第 39–50 行）。
- **单目 ViT**：用本地 DINOv2 代码构建，权重全部来自 DepthSplat 检查点，不下载 ImageNet 权重（第 56–66 行）。
- **测量深度注入**：`inject_measured_depth`（第 167–190 行）把测量的平面深度（由射线距离换算）在有效处替换为 DepthSplat 的最终上下文深度。这个深度用来放置高斯；匹配概率和特征仍是 DepthSplat 自己的。`INJECTED["align"]=False`，不做尺度对齐。

## 3. 高斯场景状态

- **构造**：`ds_eval.encode`（第 68–75 行）把到达帧输入编码器，得到 `Gaussians(means, covariances, harmonics, opacities)`，形状为 [1, N, ·]。CACHE 用 7 个视角（3 个上下文 + 4 个流帧），在 256×320 输入下每个视角每个像素一个高斯。
- **重置**：每个片段都重新编码，不保留跨片段状态。唯一的全局量 `INJECTED` 会在每次编码时重新赋值。

## 4. TTT（`ds_eval.ttt`，第 85–144 行）

- **优化的张量**：只有高斯状态的增量。
  - `d_means`：位置；
  - `d_logs`：每个高斯在协方差特征基内的逐轴对数尺度，特征向量（即旋转）固定；
  - `d_logit`：不透明度 logit；
  - `d_dc`：SH DC；
  - `d_rest`：高阶 SH。
  - 编码器和解码器的参数不在优化器里。
- **优化器**：Adam，eps 1e-15，每次调用新建，所以不同片段之间、同一片段的不同动作之间都没有状态共享。学习率是 m × 3DGS 默认值，m = 1：

  | 参数 | 学习率 |
  |---|---|
  | means | 1.6e-4 × 7 帧测量平面深度的中位数 |
  | log-scales | 5e-3 |
  | opacity logits | 2.5e-2 |
  | SH DC | 2.5e-3 |
  | 高阶 SH | 1.25e-4 |
- **损失**：`train_jepa.grounded_loss`（`train_jepa.py` 第 71–75 行）= 7 个到达帧有效测量像素上原始渲染深度（零背景上的 Σ α·T·z）的相对 L1，加 0.5 × RGB L1。
- **步数与失败处理**：100 步。损失非有限时，增量清零（回到 CACHE），并记录失败。
- **随机性**：TTT 不使用随机数。`torch.manual_seed(20261008)`，TF32 关闭。光栅化反向传播用原子加法，在 GPU 上可能不完全确定。项目以前的记录是"CPU 逐位确定，GPU 不确定"，本实验在试点中实测运行间的差异。

## 5. R7、F7 与 Core B

- **R7（REPROJ7_HARMONIC）**：在 `V1/geometry_seal.py` 第 34–57 行。7 帧测量深度用冻结函数处理：`measurement_planning.geometry` 的 `frame_points`，1 像素 z-buffer `reproject`，取 `union`（逐像素最小），再做冻结的调和填充 `harmonic`。ARKit 队列的填充常数是 2.3736329 m（来自臂评测封存）。NEAR7 是距离命中像素 8 像素以内的区域。
- **F7 与 CORE_B**：在 `V3/sys3_score.py` 第 85–97 行。
  - ray = depth_z / max(α, 1e-12) / cos；
  - F7 = α ≥ 0.5 处取 ray，否则取 R7；
  - CORE_B = sqrt(max(R7, 1e-6) × max(F7, 1e-6))。
- **opacity 阈值**：0.5，在所有 Core B 实验中固定。

## 6. 成本日志

`ds_eval.py` 只记录每个分片的总秒数和最大显存，没有逐片段、逐步的成本。本实验的回放用 CUDA 同步，对每个片段分段计时：编码、各动作读出、第 1–10 步、第 11–100 步。

## 7. 不是 Core B 执行路径的接口

- **旧快权重接口**：`fuse_down` / `complete_down` 只出现在旧的 carrier 模块里，包括 `src/mcss/dynamic/carrier.py`、`src/mcss/mechanism_pilot/*carrier*.py` 和 `rgbd_stream_write*.py`。这是早期项目对网络做快权重写入的接口，Core B 的 TTT 不经过它们。本实验也不使用它们。
- **`src/mcss` 里的 "ema"**：`engine.py` 中的 `_ScalarTracker(ema_decay)` 是对损失做平滑的统计量，不是 EMA target encoder。

## 8. 逆向 JEPA

见 `jepa_feature_contract.md`。结论：没有真正的 JEPA 潜空间预测器，`JEPA_FEATURE_STATUS=NOT_IMPLEMENTED`。

## 9. 数据审计

| 队列 | 规模 | 历史使用 | 本实验用途 |
|---|---|---|---|
| Hypersim FRESH78 / EVAL34 | 78 / 34 场景 | V1、ABLATION、ENSEMBLE、TTT-ENSEMBLE 等多次评测；EVAL34 反复使用 | 不使用 |
| ARKIT_VAL | 80 个场所 | ENSEMBLES、JEPA-ARKIT、JBT、OVERLAP、JC、OBJECTIVE、AT、CA、CW、NS、OS 都用过 | 不使用 |
| ARKIT_SYSTEM / SYSTEM2 | 各 80 个场所 | FULL-SYSTEM V1/V2 的评测，AT / CA / CW 的开发集；后来并入 NS / OS 的 502 扫描训练集 | 不使用 |
| ARKIT_SYSTEM3 | 160 个场所 | V3、NF、NS、AT、CA、CW 的确认集 | 只有 CAJU_COMPAT 的 2 个扫描用于引擎对照，不读真值 |
| CONFIRM / SELECT | 各 80 个场所 | 评测后并入 502 扫描训练集 | 不使用 |
| ARKit TRAIN | 182 个场所 | 逆向 JEPA 变体和 GA carrier 的训练数据；从未用 Core B 评测 | CAJU_SEL 16 + CAJU_EVAL 34；其余 132 个不动 |
| ARKit DEV | 8 个场所 | JEPA 训练的开发监控（CTX 臂）；从未用 Core B 评测 | CAJU_PILOT |

- **独立性**：所有 ARKit 队列都在场所级别互不重叠，各数据脚本中有断言。
- **ds_eval 跑过的队列**：从各实验的任务脚本统计，只有 ARKIT160、ARKIT160_OVERLAP、ARKIT_SELECT、ARKIT_SYSTEM、ARKIT_SYSTEM2、ARKIT_SYSTEM3、ARKIT_VAL、EVAL34、FRESH78、REPLICA，没有 TRAIN 或 DEV。
- **可信的独立确认资源**：Training 折中按种子顺序排在 1360–1477 位的约 118 个场所，从未下载。下载属于"下载大型新数据"，需要用户授权。所以本轮没有独立确认集。

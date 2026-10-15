# JEPA 特征契约（CAJU-V1）

## 结论

`JEPA_FEATURE_STATUS=NOT_IMPLEMENTED`

项目里没有满足任务书定义的 JEPA 预测表征。所谓"逆向 JEPA"，其实是用目标视角在像素空间的渲染损失，去微调 DepthSplat 编码器。

## 审计依据

我检查了全部 7 个训练脚本：
- `train_jepa.py`（V1 和 ABLATION）
- `train_jepa_arkit.py`（JA）
- `train_jepa_arkit_overlap.py`（JO、JC）
- `train_jepa_arkit_norm.py`（JN）
- `train_jepa_arkit_cache.py`（JK）
- `train_jepa_arkit_scale.py`（JL）
- `train_jepa_arkit_overlap_scale.py`（JOL）

以及它们的 `derive_*.py` 派生脚本。

| 任务书要求 | 现状 |
|---|---|
| context predictive encoder | 有一个编码器（DepthSplat），但它输出的是高斯状态，不是潜空间预测 |
| target encoder | 没有（找不到 target_encoder、EMA、momentum 或 stop-gradient 目标） |
| latent prediction | 没有：预测的是目标视角渲染出的深度和 RGB |
| predicted latent / target latent | 没有 |
| 可提取的 predictive innovation | 没有潜空间创新；唯一的"创新"是像素空间的渲染残差，与 Core B 的测量残差同类 |

各版本的损失：
- **V1 / JA / JO / JC / JL / JOL**：目标视角原始渲染深度的相对 L1 加 0.5 × RGB L1（`grounded_loss`）。
- **JN / JK**：同样的量，只在 α ≥ 0.5 的有效像素上计算，深度先除以 α。

更新的是整个 DepthSplat 编码器，单目 ViT 用十分之一学习率。它们训练出来的是另一组 DepthSplat 权重，不是独立的预测器。所以：
- 不能把 JN 等模型的 DepthSplat 特征称为 JEPA 特征；
- 它们最多是"另一组 backbone 特征"，只能作为 non-JEPA 对照（类似任务书中 B4 的一种变体）。

## 可以合法提取的 non-JEPA 特征（本实验 Gate 0 已记录）

只用 7 个到达帧和 CACHE 状态，见 `caju_replay.py` 的 `pre_update_features`：
- **残差**：
  - 原始深度残差（第 0 步的 TTT 深度项）的均值和 p50 / p90 / p99；
  - α ≥ 0.5 覆盖像素上归一化深度残差的均值和分位数；
  - RGB L1；
  - 残差最大的 10% 像素占残差总量的比例（空间集中度）；
  - 逐视角残差的最大、最小、标准差，以及上下文视角与流视角的残差。
- **测量与覆盖**：覆盖率（有效像素中 α ≥ 0.5 的比例）、有效测量比例、测量深度的 p10 / p50 / p90 和展布。
- **几何**：相机最大基线、最大视角差、高斯数量、尺度（深度中位数）。
- **DepthSplat 单目深度与测量的差异**：中位比值和 AbsRel。这是 backbone 层面的信号，属于 non-JEPA。
- **TTT 历史**：本轮是单次决策，只有逐步损失曲线可作为事后描述，不作为决策输入。

可以进一步提取、但本轮没有提取的：DepthSplat 多视角 Transformer 特征和匹配置信度（B4 的 backbone 特征）。

## 最小 JEPA 特征提取器设计（供用户决定，本轮不执行）

- **目标表征 z_t**：冻结的 DINOv2-B 或 DepthSplat 的单目 ViT，作用在新到达帧 o_t（RGB；也可以加深度通道）上，得到 patch token，256×320 下约 16×20×768。冻结目标编码器可以避免表征坍缩，不需要 EMA。
- **上下文 h_{<t}**：冻结 DepthSplat 编码器对已到达帧的多视角特征，池化成约 64 个 token；也可以用当前高斯状态的统计 token。
- **预测器 P**：4 层 Transformer，宽度 384，约 1000 万参数。输入 h_{<t} 和目标相机的 Plücker 射线嵌入 c_t，输出预测 ẑ_t。
- **损失**：对 stop-gradient 的 z_t 做 smooth-L1 或余弦损失。
- **创新残差**：r^J_t = z_t − ẑ_t，取逐 patch 范数的均值和分位数、空间集中度，以及与测量残差的相关性，作为效用 MLP 的 JEPA 输入。
- **训练数据**：ARKit TRAIN 中未进入 CAJU 的 132 个场所，每个场所 16 帧，用"3 个上下文 → 其余帧"组成的帧对，约 1.6 万对。
- **训练成本**：约 2 万步 × 0.3 s，在 GPU 4 上与其他进程分时约 2–3 小时。特征提取每个片段不到 0.1 s。
- **必要对照**：
  - 同容量的 non-JEPA 预测器：用相同的 DepthSplat 特征直接回归效用，不经过潜空间预测（B4）；
  - 打乱或替换 JEPA 输入的对照（B7）。
- **风险**：现有证据显示，Core B 的 TTT 增益集中在近处可见区域（NEAR7），而 JEPA 类的前馈改进主要体现为有选择的弃权。所以 JEPA 创新能否预测更新效用，没有先验理由乐观，必须由 Gate 1 检验。

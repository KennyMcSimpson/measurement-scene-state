# EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1：TTT 集成加上按深度范围选择的补全

本文件即预注册。分两步锁定：

1. 写定本文件后、下载任何新数据之前，把哈希记入 `audit/plan_locked_at.txt`。
2. 新队列准备好、评估根目录建好之后正式锁定（`preregistration.json`），之后才运行模型、读取真值。

## 1. 问题

两条线各自在真实数据（ARKit）上得到了确认：

- **Core B（TTT 集成）：** 7 帧状态（上下文 3 帧 + 流式帧 4 帧）用发布权重（尺度上限 0.1）做 TTT，读出与 7 帧重投影调和插值（R7）取几何平均。在 ARKIT_VAL 上，它稳定胜过 R7（EXP-3D-RGBD-ENSEMBLES-ARKIT-V1，B1：71/9）。
    - 在 7 帧都看不到的区域（FAR7），这个系统只能依靠 R7 的调和插值。
- **Core A（按深度范围选择的补全）：** 上下文深度范围 S ≥ 1.2 时用学到的补全载体（GA），否则用调和插值。它在全新扫描的近处和远处视角上都通过了 V2 gate（SELECTOR-V1、SELECTOR-FAR-V1）。

本实验在一批全新的扫描上回答：**把选择后的补全接进 Core B 的 FAR7 区域，完整系统是否稳定好于 Core B 本身？**

## 2. 数据

- **新队列 ARKIT_SYSTEM：** EXP-3D-RGBD-ARKIT-SYSTEM-DATA-V1，由 `scripts/prepare_rgbd_arkit_system_cohort.py` 准备。
    - 场馆取 ARKitScenes 训练折按试点种子顺序排列的第 680–839 位；每个场馆取 video id 最小的扫描。
    - 准备规则与 CONFIRM、SELECT 队列相同，按顺序准备到 80 个合格扫描为止。
- **不相交：** 新队列与以下数据都不相交，此前从未被使用：
    - 第 0–359 位（训练）；
    - 第 360–519 位（CONFIRM）；
    - 第 520–679 位（SELECT）；
    - Validation 折。
- **episode：** 记录 × 角色（A、B）。
    - 上下文：角色的 3 帧。
    - 流式帧：Core B 规则（`stream_frame_ids`），即 4 个平均分布的空闲帧，两个角色共用。
    - 查询：primary_query（第 14、15 帧）。
    - 规则与 ENSEMBLES-ARKIT 相同。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 组成部分（全部冻结，本实验不训练任何模型）

| 组成 | 来源 | 运行位置 |
| --- | --- | --- |
| R7（7 帧重投影 + 调和插值）、NEAR7、射线余弦 | V1 的 `geometry_seal.py`，并对 arms 封存回放核对 | CPU |
| ZS（发布权重，上限 0.1）：CACHE 与 TTT_m1（100 步） | V1 的 `ds_eval.py`，4 个分片 | GPU |
| GA、G1（3 个上下文帧，第 24000 步，各 3 个种子） | arms 评估器，合约同 SELECTOR-V1 | CPU |
| S（3 个上下文帧有效深度的 (p90 − p10) / 中位数） | 读取真值之前封存 | CPU |

## 4. 读出

- **ZS_TTT_F7：** α ≥ 0.5 处取 TTT 射线深度，其余取 R7。
- **CORE_B：** sqrt(R7 × ZS_TTT_F7)，即目前最好的 Core B 配置。
- **GAN7：** 每个 GA 种子在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余取 R7，再对种子平均。
- **R7S：** 在被选中的 episode（S ≥ 1.2）中，FAR7 像素取 GAN7；其余像素取 R7。
- **TTT_F7S：** α ≥ 0.5 处取 TTT 射线深度，其余取 R7S。
- **FULL：** sqrt(R7S × TTT_F7S)。
- **对照：**
    - FULL_G1：与 FULL 相同，但用 G1 代替 GA；
    - FULL_ALL：不做选择，所有 episode 都在 FAR7 中用 GAN7。
- **区域：**
    - ALL：所有有效像素；
    - FAR7：距 7 帧重投影命中 8 像素以外的有效像素；
    - NEAR7。
- **指标：** AbsRel。场景值 = 各角色的查询平均后，再对角色平均。

## 5. 主检验

主检验使用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。平局按 V2 gate 的规则计为不变差。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | CORE_B − FULL | ALL | FULL_SYSTEM_IMPROVES（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| P2 | R7 − FULL | ALL | FULL_SYSTEM_BEATS_MODEL_FREE（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | CORE_B − FULL | FAR7 | |
| S2 | R7 − CORE_B | ALL | 在全新扫描上复现 Core B |
| S3 | ZS_CACHE_F7 − ZS_TTT_F7 | ALL | TTT 的增益 |
| S4 | CORE_B − FULL_G1 | ALL | 改用 Hypersim 小载体 |
| S5 | CORE_B − FULL_ALL | ALL | 不做选择 |
| S6 | FULL_ALL − FULL | ALL | 选择本身的作用 |

另外报告：被选中的 episode 比例，以及 FAR7 占有效像素的比例。

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，arms 封存、几何封存（回放核对通过）、ZS 封存和 S 封存都必须完整，哈希不变。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只有 80 个扫描。
    - FAR7 只占一部分像素，ALL 上的变化预计较小。
    - 真值是 iPad LiDAR 的测量。

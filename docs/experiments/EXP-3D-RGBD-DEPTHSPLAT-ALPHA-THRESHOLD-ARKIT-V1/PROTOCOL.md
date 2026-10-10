# EXP-3D-RGBD-DEPTHSPLAT-ALPHA-THRESHOLD-ARKIT-V1：只调高读出阈值，能否得到逆向 JEPA 的前馈改进，能否改进 Core B

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

- **已有结论：** NF 与 NS 表明，只奖励精度的逆向 JEPA（JN、JL）让 7 帧前馈状态稳定地小幅变好，约 +0.004，PARTIAL。
- **推测的机理是弃权：**
    - 覆盖率从 0.72 降到约 0.55；
    - F7 读出在未覆盖处退回 R7；
    - 好处都在 NEAR7。
- **关键对照：** 如果只是弃权，那么不训练、只把发布权重的读出阈值从 α ≥ 0.5 调高到 α ≥ τ，可能得到同样的效果。这是对逆向 JEPA 这一正面结果的关键对照。
- **对 Core B 的可能改进：** Core B 的 TTT 读出也用 α ≥ 0.5，这个值是在 Hypersim DEV 上定的。在真实数据上调高阈值，也可能改进 Core B。

## 2. 数据

全部复用已有封存，不运行任何模型。

- **开发集（只用于选阈值）：** ARKIT_SYSTEM（FULL-SYSTEM-V1）与 ARKIT_SYSTEM2（FULL-SYSTEM-V2），共 160 个扫描、320 个 episode、640 个查询。
    - 复用两者的几何封存，以及发布权重（上限 0.1）的 CACHE + TTT_m1 封存。
    - 这两批扫描的真值已在各自的实验中读过。
- **确认集：**
    - **ARKIT_SYSTEM3**（160 个扫描）：V3 的几何封存与 ZS 的 TTT 封存，NF 的 ZS 与 JN1–JN3 CACHE 封存，NS 的 JL1–JL3 CACHE 封存。
    - **ARKIT_VAL**（80 个扫描）：ENSEMBLES-ARKIT 的几何封存与 ZS 封存，OBJECTIVE-TTT 的 JN1 封存，NS 的 JN2、JN3、JL1–JL3 CACHE 封存。
    - 确认集上，发布权重的读出此前只在 τ = 0.5 下计算过；任何 τ ≠ 0.5 的读出都没有在确认集上计算过。

## 3. 读出

- **ZS_CACHE(τ)：** α_CACHE ≥ τ 处取射线深度 (depth_z / α) / cos，其余取 R7。
- **ZS_TTT(τ)：** 同上，用 TTT_m1 的 α 与深度。
- **CORE_B(τ)：** sqrt(R7 × ZS_TTT(τ))。
- **JNS、JLS：** τ = 0.5（各自实验预注册的读出），三个种子 AbsRel 的平均。
- **指标：** AbsRel，区域为 ALL、NEAR7、FAR7。场景值 = 各角色查询平均后再对角色平均。

## 4. 阈值选择

在开发集上进行，读确认集的真值之前完成。

- **网格：** τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}。
- **选择规则：**
    - τ_C*：使开发集上 ZS_CACHE(τ)|ALL 的场景均值最小的 τ；
    - τ_T*：使开发集上 CORE_B(τ)|ALL 的场景均值最小的 τ；
    - 并列时取较小的 τ。
- **记录：** 选择结果写入 `selection.json`，并在 `audit/selection_locked_at.txt` 记录哈希，之后才读确认集的真值。

## 5. 主检验（ARKIT_SYSTEM3）

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | ZS_CACHE(τ_C*) − JNS_CACHE | ALL | JEPA_BEYOND_THRESHOLD（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / THRESHOLD_BEATS_JEPA |
| P2 | CORE_B(0.5) − CORE_B(τ_T*) | ALL | THRESHOLD_IMPROVES_CORE_B（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / THRESHOLD_HARMS_CORE_B |

若 τ_T* = 0.5，则 P2 恒为 0，记为 NOT_ESTABLISHED，即 Core B 不需要改变。

## 6. 次要分析（只给标签）

| 编号 | 对比 | 队列与区域 | 含义 |
| --- | --- | --- | --- |
| S1 | ZS_CACHE(0.5) − ZS_CACHE(τ_C*) | SYSTEM3，ALL | 阈值本身对前馈的作用 |
| S2 | ZS_CACHE(τ_C*) − JLS_CACHE | SYSTEM3，ALL | 与扩数据后的配方比较 |
| S3 | ZS_CACHE(τ_C*) − JNS_CACHE | VAL，ALL | P1 在第二个队列上 |
| S4 | CORE_B(0.5) − CORE_B(τ_T*) | VAL，ALL | P2 在第二个队列上 |
| S5 | ZS_CACHE(0.5) − ZS_CACHE(τ_C*) | VAL，ALL | |
| S6 | ZS_CACHE(τ_C*) − JNS_CACHE | SYSTEM3，NEAR7 | |
| S7 | ZS_TTT(0.5) − ZS_TTT(τ_T*) | SYSTEM3，ALL | TTT 单独 |

另外报告开发集上的整条曲线，以及确认集上 τ* 处的覆盖率。

## 7. 完整性与局限

- **完整性：**
    - 所有复用封存都核对哈希。
    - 选择结果在读确认集真值之前写定并记录哈希。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 阈值是全局常数，不随场景变化。
    - 开发集与确认集的场馆互不重叠：开发集与 SYSTEM3 来自训练折，VAL 来自 Validation 折。
    - 确认集上其他读出的结果此前已经读过。

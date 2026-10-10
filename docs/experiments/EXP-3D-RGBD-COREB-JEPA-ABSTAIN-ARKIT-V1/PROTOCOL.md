# EXP-3D-RGBD-COREB-JEPA-ABSTAIN-ARKIT-V1：用逆向 JEPA 学到的有选择弃权改进 Core B

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

- **已知一：** 只奖励精度的逆向 JEPA（JN）学到了有选择的弃权。
    - 它的 7 帧前馈状态在自己不可靠的像素上降低不透明度，这些像素交给 R7。
    - 这让前馈状态稳定地小幅变好，约 +0.004（NF、NS、AT）。
    - 全局阈值做不到这一点（AT：发布权重随阈值升高单调变差）。
- **已知二：** Core B（发布权重 + TTT + 与 R7 取几何平均）是目前最好的实用系统。
- **事后观察：** 在 SYSTEM3 上，JN 的前馈与 Core B 均值相同（0.1567 对 0.1569），但逐场景 Core B 更好（113/160）。两者的误差来源不同。

本实验检验：把 JN 的弃权信息用到 Core B 里，能否改进 Core B。

## 2. 候选

三种接法事先固定，都只用已有的状态，不训练任何模型。

- **MASK：** TTT 读出在 JN 认为不可靠的像素上也退回 R7。
    - T' = 在 α_TTT ≥ 0.5 且 α_JN ≥ 0.5 处取 TTT 射线深度，其余取 R7；
    - SYS_MASK = sqrt(R7 × T')。
- **TRIPLE：** (R7 × ZS_TTT_F7 × JN_F7)^(1/3)。
- **PAIR：** sqrt(ZS_TTT_F7 × JN_F7)。

其中：

- ZS_TTT_F7 = 发布权重 TTT_m1 状态的 F7 读出；
- JN_F7 = JN 前馈状态的 F7 读出；
- F7 读出：α ≥ 0.5 处取 (depth_z / α) / cos，其余取 R7；
- CORE_B = sqrt(R7 × ZS_TTT_F7)。

## 3. 数据

- **开发集（只用于选择候选）：** ARKIT_SYSTEM（FULL-SYSTEM-V1）与 ARKIT_SYSTEM2（FULL-SYSTEM-V2），共 160 个扫描、640 个查询。
    - 复用几何封存与发布权重的 CACHE + TTT_m1 封存。
    - JN1 在这两个队列上的 CACHE 由本实验评估（V1 的 `ds_eval.py`，只跑 CACHE 臂，上限 0.1）。JN1 只用 182 个 TRAIN 扫描训练，这两个队列对它是留出数据。
- **确认集：**
    - **ARKIT_SYSTEM3**（160 个扫描）：V3 的几何与 ZS（CACHE + TTT）封存，NF 的 JN1–JN3 封存。
    - **ARKIT_VAL**（80 个扫描）：ENSEMBLES-ARKIT 的几何与 ZS 封存，OBJECTIVE-TTT 的 JN1 封存，NS 的 JN2、JN3 封存。
    - 三种候选在任何队列上都没有计算过。
- **系统所用模型：** 用 JN1（单个模型）；JN2、JN3 只用于次要分析，检验对种子是否稳健。

## 4. 选择与停止规则

在开发集上进行，读确认集真值之前完成。

- **选择：** 选出开发集 ALL AbsRel 场景均值最小的候选。并列时按 MASK、TRIPLE、PAIR 的顺序取前者。
- **停止规则：** 若选出的候选在开发集上的均值不低于 CORE_B，则停止，不做确认，结论为 NO_CANDIDATE_ON_DEV。
- **记录：** 选择结果写入 `selection.json`，并在 `audit/selection_locked_at.txt` 记录哈希。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。SYS* 为选出的候选（用 JN1）。

| 编号 | 对比（> 0 表示右边更好） | 队列与区域 | 分支 |
| --- | --- | --- | --- |
| P1 | CORE_B − SYS* | SYSTEM3，ALL | JEPA_ABSTAIN_IMPROVES_CORE_B（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMS_CORE_B |
| P2 | CORE_B − SYS* | VAL，ALL | 同上 |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 队列与区域 | 含义 |
| --- | --- | --- | --- |
| S1 | CORE_B − SYS*（用 JN2） | SYSTEM3，ALL | 对种子是否稳健 |
| S2 | CORE_B − SYS*（用 JN3） | SYSTEM3，ALL | 同上 |
| S3 | CORE_B − SYS* | SYSTEM3，NEAR7 | |
| S4 | CORE_B − SYS* | SYSTEM3，FAR7 | |
| S5 | R7 − SYS* | SYSTEM3，ALL | 对比无模型方法 |
| S6 | ZS_TTT_F7 − SYS* | SYSTEM3，ALL | 对比 TTT 单独 |
| S7 | JN_F7 − SYS* | SYSTEM3，ALL | 对比 JN 前馈单独 |

另外报告开发集上三个候选与 CORE_B 的均值。

## 7. 完整性与局限

- **完整性：**
    - 所有复用封存都核对哈希。
    - 新封存核对 JN1 的检查点哈希。
    - 选择结果在读确认集真值之前写定并记录哈希。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 候选只有三种，都是简单的读出组合。
    - JN 的改进本身很小，组合的增益预计也很小。
    - 确认集上其他读出的结果此前已经读过。

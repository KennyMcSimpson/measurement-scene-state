# EXP-3D-RGBD-COREB-WEIGHT-ARKIT-V1：Core B 中 TTT 与 R7 的集成权重

本文件即预注册。锁定时间与哈希见 `preregistration.json`。

## 1. 问题

- **已知：** 在真实数据上，TTT 单独（ZS_TTT_F7）的均值一直比 Core B 低，例如 SYSTEM3 上 0.1544 对 0.1569。
    - Core B = sqrt(R7 × ZS_TTT_F7)，即等权几何平均。
    - Core B 对 R7 最一致（149/0/11）；TTT 单独的一致性较低（例如 ARKIT_VAL 上 62/18）。
- **本实验检验：** 在 R7^(1−w) × ZS_TTT_F7^w 中，把权重 w 调到 0.5 以上，能否在不损失太多一致性的前提下降低误差。
- **用途：** 结果决定推荐的实用读出。

## 2. 数据

全部复用已有封存，不运行任何模型。

- **开发集（只用于选 w）：** ARKIT_SYSTEM（FULL-SYSTEM-V1）与 ARKIT_SYSTEM2（FULL-SYSTEM-V2），共 160 个扫描。复用几何封存与发布权重的 TTT_m1 封存。
- **确认集：**
    - ARKIT_SYSTEM3（160 个扫描）：V3 的几何与 ZS 封存；
    - ARKIT_VAL（80 个扫描）：ENSEMBLES-ARKIT 的几何与 ZS 封存。
    - 任何 w ∉ {0.5, 1.0} 的读出都没有在确认集上计算过；w = 0.5（Core B）与 w = 1.0（TTT 单独）的均值已知。

## 3. 读出与选择

- **读出：** GEO_w = R7^(1−w) × ZS_TTT_F7^w，其中 ZS_TTT_F7 为 α ≥ 0.5 处的 TTT 射线深度，其余为 R7。
- **网格：** w ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}。
- **选择规则：** w* = 使开发集 ALL AbsRel 场景均值最小的 w；并列时取较小的 w（更接近 Core B）。
- **记录：** 选择结果写入 `selection.json`，并在 `audit/selection_locked_at.txt` 记录哈希，之后才读确认集真值。

## 4. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 队列与区域 | 分支 |
| --- | --- | --- | --- |
| P1 | CORE_B − GEO_w* | SYSTEM3，ALL | WEIGHT_IMPROVES_CORE_B（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / WEIGHT_HARMS_CORE_B |
| P2 | CORE_B − GEO_w* | VAL，ALL | 同上 |

若 w* = 0.5，则两项都恒为 0，记为 NOT_ESTABLISHED，即 Core B 不需要改变。

## 5. 次要分析（只给标签）

| 编号 | 对比 | 队列与区域 | 含义 |
| --- | --- | --- | --- |
| S1 | R7 − GEO_w* | SYSTEM3，ALL | w* 对无模型方法的一致性 |
| S2 | R7 − GEO_w* | VAL，ALL | 同上 |
| S3 | R7 − ZS_TTT_F7 | SYSTEM3，ALL | TTT 单独对无模型方法的一致性 |
| S4 | CORE_B − ZS_TTT_F7 | SYSTEM3，ALL | TTT 单独对比 Core B |
| S5 | CORE_B − GEO_w* | SYSTEM3，NEAR7 | |
| S6 | CORE_B − GEO_w* | SYSTEM3，FAR7 | |

另外报告开发集上的整条曲线。

## 6. 完整性与局限

- **完整性：**
    - 所有复用封存都核对哈希。
    - 选择结果在读确认集真值之前写定并记录哈希。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - w 是全局常数。
    - 确认集上 w = 0.5 与 w = 1.0 的均值已知，所以 S3、S4 不是全新的检验。

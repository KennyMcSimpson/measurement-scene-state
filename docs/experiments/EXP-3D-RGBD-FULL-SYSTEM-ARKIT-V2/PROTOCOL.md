# EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V2：TTT 集成加上 7 帧小载体的选择性集成

本文件即预注册。分两步锁定：

1. 写定本文件后、下载任何新数据之前，把哈希记入 `audit/plan_locked_at.txt`。
2. 新队列准备好、评估根目录建好之后正式锁定（`preregistration.json`），之后才运行模型、读取真值。

## 1. 问题

EXP-3D-RGBD-FULL-SYSTEM-ARKIT-V1 的结果：

- 把只看 3 个上下文帧的补全接进 Core B，没有改进。
- 事后探索（样本内，从 9 种变体里挑出）发现一种接法有效：
    - 补全载体 GA 用全部 7 帧（上下文 + 流式帧）建状态；
    - 在被选中的 episode（S ≥ 1.2）中，Core B 在 FAR7 区域的基底由 R7 改为 sqrt(R7 × GA7)；
    - 这让 ALL AbsRel 从 0.1475 降到 0.1451，好/平/差 20/50/10。

本实验在全新扫描上检验这种接法，所有设置事先固定。

## 2. 数据

- **新队列 ARKIT_SYSTEM2：** EXP-3D-RGBD-ARKIT-SYSTEM2-DATA-V1，由 `scripts/prepare_rgbd_arkit_system2_cohort.py` 准备。
    - 取 ARKitScenes 训练折、试点种子顺序第 840–999 位的场馆，每个场馆取 video id 最小的扫描。
    - 规则与之前的确认队列相同，按顺序准备到 80 个合格扫描为止。
- **与此前所有数据不相交：** 第 0–359 位（训练）、第 360–519 位（CONFIRM）、第 520–679 位（SELECT）、第 680–839 位（SYSTEM），以及 Validation 折。新队列此前从未被使用。
- **episode：** 与 V1 相同。
    - 记录 × 角色（A、B）；
    - 上下文 = 角色的 3 帧；
    - 流式帧 = Core B 规则（`stream_frame_ids`，4 个平均分布的空闲帧，两个角色共用）；
    - 查询 = primary_query（第 14、15 帧）。
- **许可：** 只用于非商业研究，媒体在 git 之外。

## 3. 组成部分

全部冻结，本实验不训练任何模型。

| 组成 | 来源 | 运行位置 |
| --- | --- | --- |
| R7、NEAR7、射线余弦 | V1 的 `geometry_seal.py`，对 arms 封存回放核对 | CPU |
| ZS（发布权重，上限 0.1）：CACHE 与 TTT_m1（100 步） | V1 的 `ds_eval.py`，4 个分片 | GPU |
| G1、GA 的 3 帧预测（用于几何回放） | arms 评估器，合约同 SELECTOR-V1 | CPU |
| **G1、GA 的 7 帧预测** | `code/eval_carrier_7view.py`（与 V1 事后探索的脚本逐字节相同） | CPU |
| S（3 个上下文帧有效深度的 (p90 − p10) / 中位数） | 读真值之前封存 | CPU |

## 4. 读出

- **ZS_TTT_F7：** α ≥ 0.5 处取 TTT 射线深度，其余取 R7。
- **CORE_B：** sqrt(R7 × ZS_TTT_F7)。
- **GA7：** 每个 GA 种子（7 帧状态）在不透明度 ≥ 0.5 处取 深度 / 不透明度，其余取 R7，再对种子平均。G17 同理。
- **SYS：**
    - 基底 B：在被选中的 episode（S ≥ 1.2）中，FAR7 像素取 sqrt(R7 × GA7)，其余像素取 R7；
    - SYS = sqrt(B × T)，其中 T 在 α ≥ 0.5 处取 TTT 射线深度，其余取 B。
- **对照：**
    - SYS_ALL：所有 episode 都被选中；
    - SYS_G1：用 G17 代替 GA7。
- **区域：** ALL（全部有效像素）、FAR7（NEAR7 之外的有效像素）。
- **指标：** AbsRel。场景值 = 各角色查询平均后再对角色平均。

## 5. 主检验

主检验用 V2 gate；统计用场景 bootstrap，10,000 次，种子 20260928。平局按 V2 gate 的规则计为不变差。

| 编号 | 对比（> 0 表示右边更好） | 区域 | 分支 |
| --- | --- | --- | --- |
| P1 | CORE_B − SYS | ALL | SYSTEM_IMPROVES（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |
| P2 | R7 − SYS | ALL | SYSTEM_BEATS_MODEL_FREE（SUPPORTED）/ _PARTIAL / _NOT_ESTABLISHED / _HARMFUL |

## 6. 次要分析（只给标签）

| 编号 | 对比 | 区域 | 含义 |
| --- | --- | --- | --- |
| S1 | CORE_B − SYS | FAR7 | |
| S2 | R7 − CORE_B | ALL | 第四次复现 Core B |
| S3 | ZS_CACHE_F7 − ZS_TTT_F7 | ALL | TTT 的增益 |
| S4 | CORE_B − SYS_ALL | ALL | 不做选择 |
| S5 | CORE_B − SYS_G1 | ALL | 改用 Hypersim 小载体 |
| S6 | R7 − GA7 | FAR7 | 7 帧载体单独对比 R7 |

## 7. 完整性与局限

- **完整性：**
    - 读真值之前，所有封存都必须完整、哈希不变：arms、几何（回放通过）、7 帧、ZS 和 S。
    - 分析跑两次，结果文件必须逐字节相同。
- **局限：**
    - 只有 80 个扫描。
    - 预期效应很小（样本内 ALL 约 −1.7%）。
    - 这条接法是在 V1 的数据上从 9 种变体中挑出的，本实验是它的第一次独立检验。

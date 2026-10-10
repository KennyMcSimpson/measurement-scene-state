# EXP-MEASUREMENT-PLANNING-ORACLE-V1：预算化深度采集的 oracle 余量审计

**结论：HEADROOM_STATUS = NOT_ESTABLISHED，POLICY_TRAINING_ALLOWED = false。**

- **测量本身价值很大：** 只请求 1 帧深度，ALL AbsRel 就从 0.289 降到 0.172 到 0.197 之间。
- **"测哪一帧"几乎被"测最新的帧"（LATEST）解决：** 样本内 oracle 比 LATEST 好 0.025。但 oracle 一旦必须用另一个查询来挑帧，优势只剩 0.005，且不显著，场景好坏数为 35/16/32。
- 按预注册的停止规则，本方向停止方法开发。理由见 [negative_result.md](negative_result.md)。

**说明：**

- 协议：[PROTOCOL.md](PROTOCOL.md)；预注册：[preregistration.md](preregistration.md)。2026-10-06 23:08:59 +0800 锁定，`preregistration.json` 的 sha256 为 `140364ae…`。
- 交接审计：[HANDOFF_AUDIT.md](HANDOFF_AUDIT.md)；相关工作：`docs/measurement_planning_related_work.md`。
- 本阶段没有训练任何模型。
- 统计单位是场景，bootstrap 10,000 次，种子 20260928。
- `FINAL_HOLDOUT_TOUCHED=false`，`NEW_DATA_DOWNLOADED=false`，`GITHUB_PUSHED=false`。

## 主判定（EVAL34 = EVAL-V3 ∪ EVAL-V4，83 个场景，166 个 episode，每条记录 9 个候选帧）

| K | 不加深度 | 随机期望 | UNIFORM | 最强启发式 | ORACLE | XORACLE | QUERY_NEAREST |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.2890 | — | — | — | — | — | — |
| 1 | — | 0.2380 | 0.2552 | LATEST 0.1974 | 0.1724 | 0.1920 | 0.1922 |
| 2 | — | 0.2136 | 0.2088 | LATEST 0.1774 | 0.1565 | 0.1730 | 0.1729 |
| 4 | — | 0.1846 | 0.1872 | LATEST 0.1605 | 0.1471 | 0.1611 | 0.1578 |

表中数值都是 ALL AbsRel 的场景平均。

| K | 样本内余量 D = 最强 − ORACLE | 余量比例 F | 交叉查询余量 X = 最强 − XORACLE | X 的 V2 gate | 判定 |
| ---: | --- | ---: | --- | --- | --- |
| 1 | +0.0250 [+0.0159, +0.0361]，60/23/0 | 0.215 | +0.0054 [−0.0043, +0.0157]，35/16/32 | NOT_ESTABLISHED | 不成立 |
| 2 | +0.0208 [+0.0136, +0.0308]，74/9/0 | 0.157 | +0.0044 [−0.0037, +0.0129]，46/6/31 | NOT_ESTABLISHED | 不成立 |
| 4 | +0.0134 [+0.0089, +0.0184]，77/6/0 | 0.094 | −0.0006 [−0.0083, +0.0065]，41/1/41 | NOT_ESTABLISHED | 不成立 |

**读法：**

- K = 1 时余量比例 0.215，略过 0.20 的门槛，但交叉查询余量的区间跨 0，所以不成立。
- 样本内 oracle 多出来的那部分，是针对被打分的那个具体查询挑出来的，换一个查询就不再成立。
- 即使直接知道查询相机（QUERY_NEAREST，不可部署），也只比 LATEST 好约 0.005。

## 次要分析

| 分析 | K = 1 | K = 2 | K = 4 |
| --- | --- | --- | --- |
| EVAL34_GAP3（只留帧序距查询 ≥ 3 的候选帧，每条记录 4–9 个） | LATEST 0.2191，ORACLE 0.2027，F 0.191，X +0.0039 [−0.0052, +0.0133] | LATEST 0.2103，ORACLE 0.1905，F 0.200，X +0.0064 [−0.0018, +0.0153] | GREEDY_FILL_DISTANCE 0.1966，ORACLE 0.1860，F 0.103，X +0.0009 [−0.0046, +0.0062] |
| EVAL_FAR（远处查询，79 个场景） | GREEDY_COVERAGE 0.2084，ORACLE 0.1884，F 0.275，X −0.0104 [−0.0241, +0.0015] | GREEDY_FILL_DISTANCE 0.1948，ORACLE 0.1748，X −0.0117 [−0.0224, −0.0019]（HARMFUL） | GREEDY_FILL_DISTANCE 0.1809，ORACLE 0.1677，X −0.0111 [−0.0210, −0.0030]（HARMFUL） |
| REPLICA（8 个场景，描述） | LATEST 0.0717，ORACLE 0.0690 | LATEST 0.0701，ORACLE 0.0600 | GREEDY_COVERAGE 0.0612，ORACLE 0.0589 |

**读法：**

- 去掉相邻帧之后，仍然没有可推广的余量。
- 远处查询上最强的规则换成了覆盖类贪心；交叉查询 oracle 反而比它差：选帧过度拟合了一个查询，到另一个查询上不如覆盖规则。
- G1 混合后端（已测区域用调和补洞，其余用 G1）的结论相同：
    - K = 1：不加深度 0.2559、LATEST 0.1947、ORACLE 0.1712；
    - K = 4：LATEST 0.1560、ORACLE 0.1443。

## 逐条回答第一阶段的问题

1. **额外深度有多大 oracle 余量？** 很大。ALL AbsRel 从 0.289 降到 0.172（1 帧）、0.157（2 帧）、0.147（4 帧）。FAR 区域从 0.493 降到 0.261（1 帧）、0.228（4 帧）。
2. **1、2、4 次测量各带来多少收益？**
    - 最强可部署规则（LATEST）：0.092、0.112、0.129；
    - oracle：0.117、0.133、0.142。
3. **是否很快饱和？** 是。按已测帧数 0、1、2、3 分组，加一帧的平均边际效用依次为 0.051、0.024、0.017、0.012。oracle 曲线在 K = 5–6 时最低（0.146），9 帧全测反而回升到 0.150：在这个无模型后端里，更多深度不一定更好。
4. **哪个简单规则最强？**
    - 主分析中 K = 1、2、4 都是 LATEST（最新的 k 帧）；
    - 远处查询与 GAP3 的 K = 4 是覆盖或距离类的贪心。
5. **简单规则离 oracle 多远？** 样本内 0.025、0.021、0.013（占 oracle 收益的 21%、16%、9%）。能推广到别的查询的部分（交叉查询）只有 0.005、0.004、−0.001，都不显著。
6. **不同场景的最优测量帧是否不同？** 是。best-1 帧最常见的位置是池中最新的一帧，但也只占 31%（52/166），其余分散在整个池中（图 3）。不过这种场景差异无法推广到别的查询视角。
7. **覆盖能否解释 oracle？** 只能部分解释。在主分析中，覆盖类规则（K = 1 时 0.213）不如 LATEST；在远处查询上覆盖类最强，但离 oracle 仍有 0.020 的样本内差距，这部分同样不可推广。
8. **位姿新颖性够不够？** 不够。POSE_FARTHEST 在 K = 1 时排第 2（0.2095），比 LATEST 差 0.012。
9. **是否存在"覆盖高但效用低"或"覆盖低但效用高"的帧？** 存在，各约 8%。在 episode 内按三分位划分，1494 个候选帧中分别有 121 个和 125 个。
10. **效用是否有可学习的结构？** 有中等强度的结构，但主要就是时间新近性。单步效用与各特征在 episode 内的平均 Spearman：
    - 池中位置（越新越大）：+0.385；
    - 三维新颖性：+0.257；
    - 填补距离：+0.223；
    - 未覆盖比例：+0.222；
    - 光度残差：+0.199；
    - 位姿距离：+0.182；
    - 旋转：+0.166；
    - 平移：+0.155。

    远处查询上最强的特征是三维新颖性，相关只有 +0.274（WEAK）。
11. **值得训练效用预测器吗？** 不值得。预注册判定不成立：可推广的余量只有约 0.005，而且不显著。交叉查询 oracle 用的是邻近真实查询的效用，这是部署时拿不到的特权信息，可它也只比 LATEST 好约 0.005。这为学习式预测器能达到的收益给出了一个很低的参照。
12. **与 NBV / active mapping 的边界？** 见相关工作综述。在"轨迹给定、RGB 免费、深度按预算请求、在留出视角上评价"这个设定上，没有找到同构的已发表工作。但"用 RGB 决定何时开深度传感器"已有先例：Noraky & Sze（TCSVT 2020）、ADI 与 Ericsson 的专利、NeuroLiDAR（ICRA 2026）。而且均匀与覆盖策略在相关文献中本来就很强。本实验的结果与这一点一致。

## 完整性

- **K = 0 复核：** EVAL34 与 GAP3 各做了 332 次，与封存的 REPROJ_HARMONIC 最大差为 0，NEAR 掩码全部相同。
- **读取审计：** 4 个队列中，每个 episode、每个规则、每个预算的读取全部合规。教师路径只出现三种用途：`ORACLE_TEACHER_ONLY`、`QUERY_CAMERA_TEACHER_ONLY`、`SEALED_QUERY_EVALUATION`。
- **确定性：** 分析运行两次，结果逐字节相同。
- **测试：** 41 项通过（`tests.json`）。`FINAL_INTEGRITY=PASS`。

## 文件

- **契约：** `task_contract.json`、`information_contract.json`、`baseline_contract.json`、`scene_manifest.json`、`frame_role_lock.json`、`preregistration.json`。
- **结果：** `oracle_results.json`、`heuristic_results.json`、`budget_curve.json`、`bootstrap_results.json`、`final_summary.json` 与 `.txt`、`integrity.json`、`tests.json`。
- **原始数据：** `raw/<队列>/<场景>_<角色>.npz`（全部子集的表）与 `.json`（选择、特征、复核、审计）。
- **图：** `figures/<队列>/`，共 7 张：
    1. 预算曲线；
    2. 逐场景的 oracle 余量；
    3. best-1 帧位置；
    4. 效用与覆盖；
    5. 效用与位姿；
    6. 按历史分组的边际效用；
    7. FAR 覆盖增益与 FAR 改善。
- **起点快照与复现命令：** `audit/start_state/`、`commands.sh`。

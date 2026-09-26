# 3D 共享状态与历史动作价值资格实验

本轮正式结果为 **BLOCKED_CARRIER_NOT_READY**。没有可用的动态 checkpoint，也没有现存的 RGB/camera/depth 三维 cohort；真实 train/dev/eval scene 数均为 0。已经完成审计、独立运行接口、指标和统计、35 项新增测试、三组合成场景工程 smoke，以及二维公平基线复核。不能将工程检查解释为核心 A/B 成立或失败。

完整机器可读交付位于 `outputs/EXP-3D-20260927-state-history-pilot-v1/`。正式 `raw/` 与 `analysis/` 明确记载未运行，数值为 null；合成数字只在 `smoke/`。首次 smoke、首次失败测试日志都保留。`FINAL_INTEGRITY=PASS` 只指工程及文件完整性，不代表研究假设成立。

## 载体与远程恢复审计

现有 `DynamicSceneCarrier` 使用小型卷积图像 encoder，真实写入位置为 `fuse_down.weight + delta_fuse` 与 `complete_down.weight + delta_complete`。其场景 materialize 不接受 query，相同 typed state 可交给参数为零的固定 renderer 输出 RGB、ray-distance depth、opacity。它不是二维 DINOv2 A/B probe。

V5 配置、训练日志及源码仍在，但 `checkpoint/step_004500.pt` 缺失；V5 本身是 `ai_001_001` 单窗口 overfit，不能当多场景底座。动态 9/19、9/20 的 checkpoint、episode index 与训练输出目录均不存在。历史文档中的 3 train + 3 dev 工程 cohort、默认训练步数、CSV 分区数量都不算本轮实际训练或可用数据证据。

按用户追加要求检查了 [GitHub 仓库](https://github.com/KennyMcSimpson/measurement-scene-state)：远程 main 与本地 HEAD 同为 `70be8e0ca4b0953a865fc83707fbf511f3268984`；1 个分支、0 tags、0 Releases、0 Actions artifacts。递归树完整返回 398 项，本地没有缺失的远程普通文件。完整 Git 历史无权重对象，无 LFS pointer；三个 ZIP 只包含源码，没有权重或三维样本。没有发现动态 checkpoint 下载外链。已下载保存 API 元数据及 SHA256，见 `audit/remote_repository/`，没有可恢复的 carrier 文件可供下载。未打开保护集，也未自动下载超大数据。

## 固定协议与实现

预实现文件先于新增 Python 模块建立；`implementation_audit.json` 保存已有文件 hashes，`dirty.patch` 保存开始时 diff。实验参数为 seed 20260927、10,000 次 scene bootstrap、95% percentile CI、float32 tie tolerance 1e-8。主要指标保持 scene-macro depth AbsRel gain，相对于同一历史且同样吸收当前图的 OFF；不能回退成 RGB 主指标。

合成 fixture 为三个解析纹理平面，16×16、已知 pinhole calibration 和精确 ray-distance GT。没有训练，没有真实独立场景。carrier/write 使用原默认参数；同一个随机初始化 W0 用于所有 fixture。context A=[0,1,2]、B=[0,3,4]，共同 anchor=0；stream=[5,6]，continuation=7，query=[8,9]。正式 cohort 尚未选择或锁定；合成分配不能冒充正式数据 split。

- `runtime.py`：独立 cache/fast/state/history；FC/CF 相同观测、固定 candidate pack、两次写入和相同动作 multiset；下一图的反馈在 append 前计算。OFF 正常 append、fast 不变、重新 materialize。ALL 使用一次旧权重 trace 产生两 proposal，同步 commit。
- `contracts.py`：物理地点跨 split 检查；所有预定候选封存前禁止 query loader；策略字段白名单；不能通过残差 head 修改共享状态；跨场景替换保持 query camera。
- `statistics.py`：逐 query 的 PSNR/明确实现定义的 SSIM、AbsRel/RMSE/δ1、GT valid fraction、opacity 与预测 coverage；完整 GT-valid 区域不按预测 opacity 裁剪。LPIPS 未实现，不填造数值。
- `ResidualReadoutControl`：可通过正常 optimizer 训练的 query-conditioned residual，前向/反向不修改共享 state。无训练数据，因此训练状态 `UNTRAINED_UNEVALUATED`；没有把随机 head 当公平 R-Residual 结果。

合成 evaluator 在 33 个 state 全部封存之后才读取六组 query camera/GT。几何 common-visible 区域通过解析平面上的 query 点投影回各 context 计算，不以预测 opacity 冒充真实可见性。查询坐标使用本 fixture 的共同 identity anchor；本脚本不是任意真实场景的数据运行入口，真实 cohort 需独立适配并核对坐标、ray-depth 定义和物理地点。

scene 是统计单位。Gamma 的六组 action pairs、oracle 与 harmful rate 都保存到 scene/continuation 层。bootstrap 和 LOSO 每个子集都重新选择 hindsight-global action。winner flip 使用容差内最优集合不相交且至少一侧 reward>1e-8 的规则，OFF 包含在候选中；两个坏动作互换名次不算 beneficial flip。oracle 均为离线诊断。

## 工程 smoke 与限制

完成 30 行静态 query 记录（A/B/anchor/prior/wrong-scene）和 48 行动态 query 记录（3 fixtures × 2 histories × 4 actions × 2 queries）。静态 prior 明确固定 RGB=.5、depth=5m，不拟合 query。保存 pre/post 预测差异、完整动作记录、封存 hashes、读 GT 日志和成本。随机载体/写入器分别有 5,125 / 528 参数，固定 renderer 0 参数。实测耗时见 `smoke/summary.json`；24 个 continuation 候选步骤与 12 个历史步骤均计入，未训练策略，无双向匹配。

合成运行中，FC/CF fast-weight 距离约 4.78e-4，但 pre RGB/depth RMS 差异仅约 6.74e-7 / 3.27e-6；ALL 在所有历史最优。两个额外 oracle gap 均为 0，beneficial flip=0，Gamma CI 均跨 0。**这些只是随机未训练载体的工程记录，不能推断真实场景无历史作用，也不能凭数值不同宣称 trajectory mechanism。**

自然历史与 Phase C 没有启动。政策比较 P0/P1/P2、净收益与 regret 均 NA。正式 WRITE/HISTORY 状态栏使用 NOT_ESTABLISHED，并且绑定 `EVALUATION_STATUS=NOT_EVALUATED_CARRIER_BLOCKED`：意思是没有证据，绝非已经做出负实验。

## 二维分支封存

仅复用 discovery raw 重算 outer-fold fixed16 的决策，逐折排除 heldout sequence，重算决策与既有 followup 一致。`POSTHOC_FAIR_BASELINE=true`，不重新包装为预注册结论。

| OOF 方法 | sequence 等权 J |
|---|---:|
| OFF | 0.4397619832 |
| Outer-fold fixed16 | 0.4430925043 |
| GateOnly | 0.4409180983 |
| CycleGate | 0.4410062694 |

12 折固定轨迹均为 ALL/B。CycleGate − fixed16 = −0.00208623，95% CI [−0.00486055, −0.00013578]；GateOnly − fixed16 = −0.00217441，CI [−0.00492138, −0.00020132]。这是既有 OOF 的条件性 post-hoc 描述，不是独立确认，也没有在 bootstrap 中重新训练 selector。旧 V1/V2 文件保持原样；`2D_BRANCH_STATUS=SEALED_PENDING_INDEPENDENT_CONFIRMATION`。

## 必须回答的 16 个问题

| 问题 | 本轮答案 |
|---|---|
| 1. 三维载体是否合格？ | 否：可用多场景动态 checkpoint 和数据均缺失，资格阻塞。 |
| 2. state 是否使用当前观测？ | 源码具有观测路径；真实任务中的贡献尚未评价。 |
| 3. 一个封存 state 能否服务多 query？ | 工程接口已验证；科学预测质量未建立。 |
| 4. R-Fixed 与 R-Residual 差异？ | 前者固定测量；后者增加可训练 query-conditioned residual。公平实测需充分训练 head，本轮未做。 |
| 5. 错场景是否显著破坏预测？ | 替换接口与相机不变测试通过；真实多场景效应 NA。 |
| 6. 写入超过仅 append 的收益？ | 正式 NA；合成值不能作研究证据。 |
| 7. 哪些动作有害？ | 正式 NA；统计模块保存各动作有害率，未删负例。 |
| 8. FC/CF 是否不同？ | 合成 fast/state 有微小数值差异；真实数据未测。 |
| 9. 差异影响留出任务吗？ | 正式未知。不能把合成数值误差当任务证据。 |
| 10. history 改变动作排名吗？ | 正式未知；本次合成最优均 ALL，无有益翻转。 |
| 11. 多少 scene 有 beneficial flip？ | 真实 NA（评价 scene=0）；合成独立列出 0/3。 |
| 12. per-state 超过 global fixed 吗？ | 正式未知；合成 gap=0，不外推。 |
| 13. feedback 能识别差异吗？ | SKIPPED：正式 Phase B 前提未建立。 |
| 14. 显式 history 的额外价值？ | 未测，P0/P1/P2 没有训练。 |
| 15. 当前主要瓶颈？ | carrier 及数据恢复，尚不能归因于 write/history/feedback/policy。 |
| 16. 最小下一步？ | 恢复兼容 `mcss.dynamic.v1` 的 checkpoint及训练来源；若无法恢复，在少量已审计的多 scene RGB/camera/depth 训练/开发 cohort 重建原 carrier，充分训练 residual 对照，先做静态资格。必须先锁定物理地点 split、上下文、query 和指标，再评价；不能直接增加 controller。 |

## 测试、复现与审计

完整 suite 首次 native segmentation fault（旧 DL3DV 测试处），原因未诊断；原日志保存为 `audit/full_suite_attempt1.log`，未改测试规避。第二次 **537 passed, 1 skipped**，缺失 V5 权重的兼容性检查跳过。新增 35 项测试覆盖用户要求的边界，既有二维 outer-fold 隔离测试继续通过。最后仅增加 post 差异输出和拆开 camera fixture 后，4 项集成测试再次通过。Ruff PASS。

`commands.sh` 给出复现命令，默认输出使用独立目录，运行器拒绝覆盖已有 smoke。集成测试验证两次预测 hashes/数值完全一致，统计能从保存 raw 重算；耗时不要求相同。锁与信息合同、环境、原始 diff、原始测试日志均保留。ledger 是受控 loader API，不是操作系统沙箱；不能检测恶意把 GT 改名塞进合法字段。真实数据接入仍须审计提供者，不能仅凭字段白名单宣称绝无泄漏。

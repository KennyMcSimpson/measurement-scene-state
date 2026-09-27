# 当前状态（2026-09-26）

2026-09-27 后续探索：[折外全局轨迹与 cycle 诊断](docs/experiments/EXP-2D-followup-20260927/README.md)。12个discovery外层折都选择ALL/B；CycleGate仍低约0.209个百分点。仅属事后discovery分析，独立确认仍受阻。

V2 当前状态：`BLOCKED_NO_INDEPENDENT_DATA`。旧 raw 真正逐样本空间约 +0.2055 个百分点；CycleGate discovery 外层 OOF 约 +0.1244 个百分点，但尚无独立测试结果。完整测试重试通过（500 passed、1 skipped），首轮原生崩溃已留档。[V2 报告](docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md)。

最新机制实验已完成：448像素J下有小幅、跨序列的oracle机会（H1=STRONG），但部署可见选择器净收益未建立（H2=NOT_ESTABLISHED）。详见 [新实验报告](docs/experiments/EXP-2D-20260926-opportunity-selector-v1/README.md)。以下保留历史证据状态。

| ID | 状态 | 证据等级 | 已验证 | 当前可以说什么 | 当前不能说什么 |
|---|---|---|---|---|---|
| `BASELINE-V5-4500` | `FROZEN_REFERENCE` | `DEVELOPMENT` | V5 typed appearance 与旧工程记录保留 | 不能当 validation/test 或 CVPR 结果 |
| `EXP-3D-20260919-engineering` | `COMPLETE_ENGINEERING` | `ENGINEERING` | runner、CPU contract、CUDA smoke 和保存恢复检查 | 不能当三维方法胜出证据 |
| `EXP-3D-20260920-method-dev` | `COMPLETE_DEVELOPMENT` | `DEVELOPMENT` | carrier/policy 开发记录和失败修复被保留 | 不能当最终方法或外部 benchmark |
| `EXP-2D-20260921-original` | `SUPERSEDED` | `ENGINEERING` | 原始运行保留；已记录 pair 汇总 bug | 不能引用旧汇总数字 |
| `EXP-2D-20260921-corrected` | `EXPLORATORY` | `EXPLORATORY` | 2424 raw rows、32 validation pairs、10 focused tests、completion verification | 不能叫完整逆向 JEPA、streaming TTT 或 DAVIS J&F |
| `EXP-GEOMETRY-CERT-V1` | `STOPPED` | `DIAGNOSTIC` | certificate frontend 失败门槛已封存 | 不能继续把该 frontend 接入主模型 |
| `EXP-MAPANYTHING-V1` | `STOPPED` | `DIAGNOSTIC` | supplier qualification 失败记录保留 | 不能当外部监督或方法贡献 |
| `OFFICIAL-DAVIS-VAL` | `SEALED` | `UNRUN` | 30 段官方 val 未读取 | 不能暗示已经完成官方 benchmark |
| `DL3DV/Hypersim final holdout` | `SEALED` | `UNRUN` | 旧本地数据已按授权清理，结果和来源清单保留 | 不能声称有 final holdout 结果 |

## 当前研究判断

二维 probe 显示观测 RGB 反馈可以显著降低 RGB 重建误差，但当前 selector 没有把它转化成对象对应收益。下一步应先测试更可靠的高分辨率对应评价和允许 `OFF` 的 task-grounded gate，再决定是否扩大模型。不要直接扩大 RGB 重建训练规模。

## 存储和协作边界

旧三维数据删除记录和二维资产下载记录仍在 `outputs/` 本地。公开仓库只提交小型 provenance 和结果 bundle；数据、权重、缓存和 checkpoint 按 `.gitignore` 排除。项目记忆的持久状态写入中央 CV 分支，不以仓库文件替代。

## 2026-09-27 三维小规模工程训练

已完成3个train场景上的100步静态+30步写入训练，生成新的 `mcss.dynamic.v1` 权重。严格重载与写入重放通过；全套测试561 passed、1 skipped。一个场景因固定体积缺乏多视图候选支持而全部写入梯度为零，保留负例；当前仅工程可用，未取得独立静态/泛化资格。

详见 [训练报告](docs/experiments/EXP-3D-20260927-small-training-v1/README.md) 和 [权重索引](docs/experiments/EXP-3D-20260927-small-training-v1/checkpoint_index.json)。

## EXP-3D-20260927-centered-support-v1

第三场景零写入支持已通过统一锚点中心体积修复：支持点 0→4，B 阶段非零 writer 梯度 0/10→10/10。100+30 步同种子重训、checkpoint 审计和 573 passed/1 skipped 全套测试通过；最终 TRAIN OFF 场景均值 AbsRel 0.919→1.229，未获得质量提升。旧实验和权重保留，新模式显式 opt-in，不构成独立泛化或写入收益证据。

[报告与复现](docs/experiments/EXP-3D-20260927-centered-support-v1/README.md)。

## EXP-3D-20260927-centered-training-1000a-600b-v1

2026-09-27 已完成 [1000A＋600B 三维训练](docs/experiments/EXP-3D-20260927-centered-training-1000a-600b-v1/README.md)：耗时112秒，TRAIN OFF AbsRel 为 A结束0.287、B结束0.332（短程1.229）；B相对A回退，未验证写入收益或泛化。 两阶段权重均保留，三个场景各200/200步非零writer梯度，15项回归测试及独立审计通过。

## EXP-3D-20260927-trained-state-history-v1

2026-09-27 [已训练权重机制补测](docs/experiments/EXP-3D-20260927-trained-state-history-v1/README.md)：同训练场景留出帧上，写入oracle AbsRel收益约0.000756，但有益历史动作翻转0/3、history额外空间0；静态context A AbsRel2.214，资格仍未建立。原训练query的1次翻转单列，不当独立证据。 581 passed、1 skipped；90条static和144条dynamic raw完整，R-Residual缺已训练head，自然历史及P0/P1/P2按原资格门槛跳过。

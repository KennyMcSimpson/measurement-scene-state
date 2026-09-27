# 实验注册表

## EXP-2D-followup-20260927

- **状态**：POST_HOC_DISCOVERY_ONLY；保留V2全部冻结产物。
- **新增诊断**：仅用其余11序列选择全局16轨迹，12折都选ALL/B；CycleGate−该折外全局基线为−0.208623个百分点。f_cycle改善的168条候选中33条任务收益为负。
- **边界**：不改V2主要比较，不构成独立确认，不据此断言所有可见信号都无效。
- **验证**：2项新增测试通过；63份V2文件哈希无变化；重生成逐字节一致。
- **报告**：[结果与资源审计](docs/experiments/EXP-2D-followup-20260927/README.md)。

## EXP-2D-opportunity-selector-v2-20260926T015728+0800

- **状态**：`BLOCKED_NO_INDEPENDENT_DATA`；已完成实现、旧 raw 后处理、discovery nested LOSO、全套测试与审计。
- **旧 raw 诊断**：hindsight-global16=A/ALL；dynamic−hindsight-global16=+0.205456 个百分点，重采样/删一均重新求全局winner。
- **Discovery**：CycleGate 外层 OOF ΔJ=+0.124429 个百分点，GateOnly=+0.115612；Cycle−Gate 95%区间跨0。只属 discovery 过程评估。
- **独立数据**：VOST片段缺原视频映射；FBMS原视频身份及mask编码不合格。未打开旧保护集，独立序列0，未生成确认结果。
- **验证**：500 passed、1 skipped（重试）；原生崩溃日志保留，674项V1产物未变。
- **产物**：[完整报告](docs/experiments/EXP-2D-opportunity-selector-v2-20260926T015728+0800/README.md)；协议/锁/raw/统计/复现命令均保存在独立目录。

这里是唯一的人类可读实验索引。机器可读的细节放在各 artifact bundle 和本地 `outputs/` 中。

## EXP-2D-20260926-opportunity-selector-v1

- **状态**：已完成锁定验证与独立审计；按预注册门槛 H1=STRONG、H2=NOT_ESTABLISHED。
- **主评价**：448 输入、真实32×32 DINO patch、原始像素空间对象 J；16个历史内部validation序列、32 pairs。不是新holdout或官方DAVIS J&F。
- **结果**：OFF J=55.2677%；dynamic oracle=55.5129%（+0.2451个百分点，sequence-bootstrap95%CI[+0.1173,+0.3934]）；visible selector=55.2207%（−0.0470个百分点）。
- **解释**：小幅答案可见机会分布于14个序列；非恒定轨迹机会比constant oracle多0.0750个百分点（不证明顺序因果机制）。部署选择器尚未获得净收益，29次写入有16次有害。
- **验证**：458 tests passed、1项缺失历史checkpoint明确skip；1792 raw rows完整、输入hash和封存边界通过审计，全部分析可逐字节重建。reserve及official val保持封存。
- **artifact**：[实验报告](docs/experiments/EXP-2D-20260926-opportunity-selector-v1/README.md)。

## EXP-2D-20260921-corrected

- **状态 / 证据**：`EXPLORATORY`；修正版，替代 `EXP-2D-20260921-original`。
- **问题**：可见观测残差产生的私有 fast-weight 写入，能否改善后续对象对应？RGB probe 能否选到有益动作？
- **数据**：DAVIS-2017 train 60 段；fit/discovery/validation/reserve = 24/12/16/8；官方 val 30 sealed。
- **模型**：冻结 DINOv2-S/14，224 输入、16×16×384 token；fit-only PCA whitening 到 64 维；冻结 RGB ridge readout；每个目标图像两份私有 64×64 状态矩阵。
- **动作**：两步 `OFF/A/B/ALL`，共 16 条路径；selector 每步以可见 probe RGB MSE 选四个候选。selector 的候选计算额外计账，oracle 看答案且仅作诊断。
- **主指标**：16×16 token-grid mask transfer 的 mean object IoU；不是 DAVIS J&F。
- **验证结果**：OFF/OFF 44.34%；发现集选定固定 OFF/ALL 45.08%；RGB selector 44.03%；有限答案 oracle 45.51%。selector RGB MSE 从 0.03022 降到 0.01250，但 IoU 下降 0.31 个百分点。
- **不确定性**：按序列的描述性 bootstrap 区间宽，未做多重比较校正；tiny-object token 分辨率是明显限制。
- **可声称**：当前启发式反馈通道能改变特征并降低重建误差；RGB 重建和对象对应存在失配。
- **不可声称**：完整逆向 JEPA、自然 streaming TTT、动态策略成功、官方 DAVIS 排名或 CVPR 级结果。
- **artifact**：[curated bundle](docs/experiments/EXP-2D-20260921-corrected/README.md)。

## EXP-2D-20260921-original

- **状态**：`SUPERSEDED`。
- **原因**：`bestgloballyfixed` 汇总时每段序列只复制了第一个 target slot；原始 raw records 保留在本机 `outputs/vision_2d_probe_20260921/`，并有 `CORRECTION_REQUIRED.json`。
- **修复规则**：同一协议、同一缓存、同一参数选择；只修复记录覆盖并重跑。修正版的未受影响指标逐条精确复现。

## 既有三维记录

`BASELINE-V5-4500`、`EXP-3D-20260919-engineering`、`EXP-3D-20260920-method-dev`、`EXP-GEOMETRY-CERT-V1` 和 `EXP-MAPANYTHING-V1` 的状态与证据边界见 [STATUS.md](STATUS.md)。历史输出没有删除；公开仓库只提交代码、说明和小型 provenance，避免误把数 GB 的 checkpoint/log 当成可复核结果。

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

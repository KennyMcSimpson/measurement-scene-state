# 用已训练权重补做静态状态与受控历史实验

使用用户指定的最新 **1000A+600B 的 B-final** 权重，已补做此前缺权重而未完成的真实数据静态对照与 FC/CF 受控历史实验。没有根据 A-final 更好的训练分数换权重，没有训练新模型或调动作参数。

本轮结论：**能测到小幅写入机会和历史数值/预测差异，但主要留出帧上没有 history-dependent 最优动作翻转；静态底座资格仍未建立。**

## 范围与锁定

- 三个 Hypersim TRAIN 场景 ai_001_001/ai_002_001/ai_003_001；未访问 dev/test 或保护集。
- primary：第8、9帧，沿用旧 synthetic smoke 的角色编号，属于本次新锁定的真实 cohort。未作为1000/600训练的优化目标，但场景已经训练，因此仅是同场景帧留出诊断，不能当未见场景确认。
- secondary：第12–15帧，已用于训练，单独报告。两组不混合统计。
- context A=0/1/2、context B=0/3/4，只共享anchor0；stream=5/6；同一continuation=7。
- 448二维分支不动。本次3D沿用128×160、8³体积、128候选、原写入率/裁剪和64采样renderer。
- 主要reward为同history下、OFF也加入continuation后的 depth AbsRel(OFF)−AbsRel(action)。统计先query平均，再history/continuation，最后scene等权。seed20260927、10000次scene paired bootstrap、95%percentile CI、tie tolerance1e-8；bootstrap/LOSO都重新求hindsight-global winner。
- metadata manifest包含query相机，完整性检查读取文件bytes；本轮保证的是封存前不构造/使用query相机张量、不解码query标签、不向runtime提供query或depth，而非操作系统级文件零读取。

冻结记录见 preregistration.json 与本地 run/run_lock.json。checkpoint SHA256：`be7b8b6d2ef366cad245732b9ff227802db596da65082ac0e160f24541579e42`。

## 静态状态：主要留出帧

每个状态先封存，再用于全部查询。相机只进入固定renderer；wrong_scene按锁定顺序使用下一个场景的A状态，目标query相机数值保持不变，不拟合坐标配准。

| 方法 | 深度 AbsRel↓ | RGB MSE↓ | SSIM↑ | opacity | 近似共同可见区 AbsRel↓ |
|---|---:|---:|---:|---:|---:|
| A | 2.213562 | 0.064667 | 0.197974 | 0.799395 | 4.068888 |
| B | 1.210011 | 0.041768 | 0.254105 | 0.885932 | 1.771183 |
| anchor | 2.233817 | 0.065750 | 0.241269 | 0.582869 | 4.124200 |
| prior | 3.473366 | 0.132712 | 0.229379 | 1.000000 | 5.390804 |
| wrong_scene | 2.699125 | 0.100404 | 0.244563 | 0.763684 | 4.549744 |

所有方法coverage均为1（定义：预测opacity>1e-6），这不是实际表面覆盖或正确几何的证明。prior沿用旧smoke的RGB0.5、depth5m常数，仅是弱工程参照，不能代替充分训练的先验基线。共同区域采用query GT反投影和两组context depth一致性（nearest像素、0.05m+1%容差），是近似可见性；另有纯共同视锥指标。所有6个primary query都有非空共同区域，未剔除困难帧。

A相对anchor的平均AbsRel优势仅0.020255，95% CI [-0.125139,0.185904]，跨场景不稳定。wrong_scene替换导致AbsRel增加0.485563，CI [0.147965,1.122819]，说明观察内容在部分场景中影响预测，但不能单独证明共享状态合格。

ai003尤其不合格：context A的AbsRel=5.782568，context B=2.592783；近似共同可见区域A=11.618104。A与anchor预测相同，与此前context A无双视图支持一致；stream恢复写入并不自动修复静态context A。该场景所评六张RGB查询均为全黑，保留原图与全部记录，详见query_quality.json；因此不能靠RGB好看宣称几何成立。未据此换数据或改指标。

`STATIC_STATE_STATUS=NOT_ESTABLISHED`，`CARRIER_STATUS=PARTIAL`。除了实测质量问题，仍没有未见开发场景、已审计物理地点划分或充分训练的R-Residual，不能升级正式静态资格。

## 写入与历史：主要留出帧

FC与CF使用相同stream、候选pack、动作multiset和写入次数，各自独立缓存/fast/state。每条历史在同一个第7帧到达后分成OFF/FUSE/COMPLETE/ALL，OFF正常append，ALL两proposal来自同一旧trace。对比只有本次新增write不同。

| 场景 | FC最优动作集合 | CF最优动作集合 | per-state oracle gain | 相对observation-only oracle额外空间 |
|---|---|---|---:|---:|
| ai_001_001 | COMPLETE | COMPLETE | 0.000535117 | 0.000000000 |
| ai_002_001 | ALL | ALL | 0.001633588 | 0.000000000 |
| ai_003_001 | FUSE/ALL | FUSE/ALL | 0.000098579 | 0.000000000 |

- 写入oracle平均AbsRel收益 **0.000755762**，描述性95% CI [0.000098579,0.001633588]。三场景都有正机会，但绝对幅度很小；oracle含OFF，其非负不能证明合法策略有效。
- 事后全局动作是COMPLETE，平均收益0.000190360；per-state oracle相对hindsight-global差0.000565402，CI [0,0.001065084]。这是跨场景/观测选择机会，不等于需要history。
- observation-only oracle与per-state oracle完全相同，**history额外空间=0，CI=[0,0]；beneficial flip=0/3**。最优集合考虑并列，未强行破tie。
- FUSE、ALL在2/6个history状态（1/3场景）上有害，COMPLETE为0/6；有害率按query平均后的state定义，非逐pixel。FUSE平均收益−0.002357568，ALL−0.002163924，不能把oracle小正值当成任意写入都好。
- 完整RGB/depth/opacity/coverage的配对变化与CI在action_metric_deltas.json。文件统一使用action−OFF，误差指标改善为负号。

历史四级证据：

1. fast差异L2为0.056255、0.003149、0.002316，三场景均非零。
2. 相同continuation预更新深度RMS差为0.053103、0.002879、0.000420m；所有query的post差异另存raw。
3. 动作相对价值的Gamma存在数值差异，主要由ai001贡献；全六对如下，逐场景和LOSO全部保留。
4. 主要查询的动作排序/最优动作没有出现有益翻转。因此不能从数值不等推导决策相关的历史机制。

| Gamma动作对 | scene mean | 95% scene-bootstrap CI |
|---|---:|---|
| COMPLETE__ALL | -0.004257398 | [-0.012698654, 0.000002456] |
| FUSE__ALL | 0.000048715 | [0.000000000, 0.000146089] |
| FUSE__COMPLETE | 0.004306113 | [-0.000002400, 0.012844743] |
| OFF__ALL | -0.004221805 | [-0.012591854, 0.000002436] |
| OFF__COMPLETE | 0.000035594 | [-0.000000020, 0.000106801] |
| OFF__FUSE | -0.004270520 | [-0.012737943, 0.000002380] |

`WRITE_OPPORTUNITY_STATUS=PARTIAL`（仅同场景帧留出微小oracle机会），`HISTORY_ACTION_STATUS=NOT_ESTABLISHED`（本次主要查询没有beneficial flip，不代表所有历史/信号都不可能有效）。

## 原训练查询：仅次要对照

第12–15帧的context A AbsRel **0.331527959** 精确复现上轮最终训练评价，校验了权重、数据与坐标链路。其write oracle gain=0.001240973，history额外空间0.000359326，出现 **1/3** 场景有益翻转（ai001 FC=ALL、CF=COMPLETE）。该额外空间95% CI为[0,0.001077977]，全部来自ai001，不能替代primary的0翻转，更不能用已优化标签证明独立机制。

## 未启动的部分及原因

- R-Residual：代码接口存在，但没有充分训练的residual head。新carrier权重不能替代它，未用随机head制造弱对照。
- 自然历史：原协议要求基本静态资格后才启动；此项仍不满足，因此保留SKIPPED。
- P0/P1/P2、反馈可识别性、净policy gain/regret：原协议要求先建立值得学习的历史动作差异；primary没有beneficial flip且静态资格不足，故未训练controller。`FEEDBACK_IDENTIFIABILITY_STATUS=SKIPPED`，不是证明feedback无效。
- 独立场景验证：仍缺新的、身份审计通过的开发cohort，不打开保护集。
- 历史V5兼容性：新checkpoint属于mcss.dynamic.v1，不能替代缺失的V5 step4500文件，该测试仍skip。

## 成本、访问边界与复现

真实运行 5.008 秒（状态构建含检查点加载/hash校验约 1.860 秒）；峰值CUDA约 200.6 MiB。carrier5125参数，writer528，fixed renderer0参数。

共33个封存状态、90条static raw、144条dynamic raw。构造12个历史写入步，24个continuation候选步，其中18步写入；48次materialize、30次trace/proposal、66次图像encode。渲染总计258次：query216、历史feedback12、候选feedback24、额外pre比较6。prior18次常数预测另计。没有双向匹配、policy训练或隐藏搜索。

39项专项测试通过；完整suite **581 passed、1 skipped**。首轮fixture发现episode ID含斜线违反既有封存接口，改为简单ID后重跑通过，失败日志保留；真实评估只有一次，没有按结果调参。全部33个state先封存才调用query相机/标签和context depth evaluator。checkpoint/source/data哈希未变；原实验完整保留。独立复算审计见independent_review.json。

本地run/保存全部raw、sealed_states.pt、state_manifest、history traces、prediction hashes、query与label访问日志和source lock；本目录保存精简报告与可复算raw。commands.sh提供新目录复现命令，CHECKSUMS.sha256用于完整性核对。

## 对原16个问题的回答

1. 多场景研究底座是否合格：尚未合格；可运行，但静态质量和独立验证不足。
2. state是否使用场景观测：wrong-scene会损害预测，但context优势不稳定。
3. 同一个state能否服务多个query：可以，封存后六个query由同一renderer读取。
4. Fixed/Residual差异：Residual尚无训练权重，不能比较。
5. wrong-scene是否损害：primary三场景均增加AbsRel，均值+0.485563，仍仅诊断。
6. write是否超出append收益：存在微小oracle机会，合法策略尚未验证。
7. 哪些action有害：primary FUSE/ALL各2/6状态，COMPLETE0/6。
8. FC/CF是否不同：fast和预测都不同。
9. 是否影响留出任务：reward magnitude可变，但主要最优动作不变。
10. 是否改变rank：primary未发现最优排序改变；secondary ai001改变。
11. beneficial flip场景数：primary0，secondary1。
12. oracle超过global：差0.000565402，CI含0；不证明history必要。
13. feedback能否识别：未做policy，未知。
14. 显式history额外价值：primary oracle空间为0，当前不支持投入controller。
15. 主要瓶颈：首先是carrier静态/跨帧质量，随后是历史决策空间很弱；不是已证明policy失败。
16. 最小下一步：先审计ai003全黑RGB及context支持/深度读出问题，并在预先锁定、身份清楚的未见开发场景上做静态资格和充分训练Residual对照；当前不应直接扩大controller。

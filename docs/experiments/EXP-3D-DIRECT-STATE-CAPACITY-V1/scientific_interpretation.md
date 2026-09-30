## 十二个研究问题的回答

结论是：**检测到 RGB+D 的 8³→16³ 分辨率瓶颈，但尚未建立“当前 shared state 已充分、只需重训 learner”的资格。** Bounds、context 恢复和优化充分性仍影响解释。下列判断使用已冻结的规则；没有把更有利的 bounds 或高 grid 事后改成主要配置。

1. **8³ + fixed renderer 能否表示正确 geometry？** 在主配置 CURRENT_BOUNDS 下尚未建立充分性。8³ context RGB+D 的 query AbsRel 为 0.460804，未达到预注册工程门槛。明确特权的 query-supervised oracle 在相同主配置下也未过完整门槛，但它不是数学最优解，不能据此判定表示天生无能。后述预定 bounds 敏感性说明这个判断有条件性。

2. **Query-supervised direct state 能做到什么水平？——仅诊断。** CURRENT 的 8³/16³/32³ oracle AbsRel 分别为 0.289314/0.246014/0.221586；三者 ≤0.35 的场景数仅 11/17、11/17、12/17，因此均未通过“mean≤.25 且至少75%场景≤.35”的完整条件。预先定义的替代 bounds 下，8³ oracle 为 0.202158，15/17 场景≤.35，达到该工程条件。这说明不能脱离 bounds 笼统否定8³表示；该组仍是用了query GT的诊断，不是泛化结果。

3. **Context RGB+D 达到什么水平？** 8³为0.460804，16³为0.402736，32³为0.389382。相对旧carrier确有改善，但绝对query误差仍高；不能把“比差基线好”直接写成容量已经充分。

4. **RGB-only 达到什么水平？** 8³为0.609175；16³/32³为0.617032/0.622991，没有随着grid增大而改善。8³相对carrier改善的CI跨零，也未稳定优于anchor。它在context上的RGB MSE很低，但depth AbsRel仍为0.570093，说明本配置下颜色拟合不能替代几何恢复。这不证明所有RGB-only信号或优化方法都不可能有效。

5. **Carrier与direct gap多大？** 主配置8³，RGB+D capacity gap为 **+0.219022 [0.142366, 0.292736]**；RGB-only为 **+0.070651 [−0.011010, 0.151245]**。前者有强的可达改善证据，但使用额外context depth，不能把整个gap全归于RGB-only learner没学会。RGB+D相对旧anchor的gap为+0.123797 [0.066338, 0.179595]。

6. **改善是否跨多个场景？** RGB+D对carrier为15改善/1持平/1更差；LOSO均值区间[0.198443,0.241792]全正。最大单场景占正收益14.17%，前三占35.16%，不是一个scene独占。RGB-only为10改善/1持平/6更差，稳定性明显不足。全部17场景保留，没有移除零ray-hit或不利场景。

7. **分辨率提高是否稳定改善？** RGB+D 8→16收益 **+0.058068 [0.028670,0.091763]**，13改善/1持平/3更差，满足冻结稳定性门槛。16→32只有+0.013353 [−0.004395,0.032370]，不满足。`RESOLUTION_LIMITED`来自前一项证据，不表示32³必要，也不表示分辨率是唯一原因。

8. **Renderer samples重要吗？** 在预定64/128范围内没有实用收益：8³的64→128 gain为−0.000304，16³为−0.000099，CI均跨零；没有支持把ray采样数视为主要瓶颈。这个结论只覆盖该采样对照，不能排除renderer表示语义或其他结构性限制。

9. **是否只是记住context？** RGB+D 8³的context AbsRel=0.188866，query=0.460804，gap=+0.271937 [0.168836,0.381501]，明显存在context→query缺口，不能宣称已经形成充分可复用的状态。但它也有跨场景的query改善和wrong-scene损害证据，不能简化为完全只有记忆。RGB-only的depth gap虽小，却是context与query的depth都差，不是几何泛化成功。

10. **Wrong-scene是否损害direct？** RGB+D 8³的损害为+0.100210 [0.043297,0.160697]，说明同一query camera下，正确场景状态携带了有用信息。RGB-only损害为+0.022024 [−0.057422,0.094209]，未稳定建立。Shuffle/density-only/color-only/zero均实际渲染评分，原状态保持不变。

11. **主要瓶颈在哪里？** 预注册标签是`RESOLUTION_LIMITED`；纯learner失败、纯renderer-sampling失败和充分容量均未建立。8³的context RGB+D−oracle gap仍有+0.171489 [0.099124,0.254048]，但CURRENT oracle8未过good gate，所以不能将主标签事后改为CONTEXT_INFERENCE_LIMITED。该未触发标记也不是排除context信息/恢复困难。CURRENT中仅62.19%的query GT点在bounds内，37.81%的有效GT ray-distance超过AABB far；替代bounds后分别为71.21%和28.57%，仍不充分。全部425/425状态都选择第1000步，且最后三次目标仍下降；因此实验测到的是固定预算下的可达水平，不能当成表达上限；目标下降也可能来自正则，不能保证继续增加步骤会改善query。

12. **下一轮具体做什么？** 选择**16³作为下一阶段高分辨率carrier设计的候选**，先单独预注册优化充分性与固定GT-free bounds对照，确认context恢复的剩余差距，再决定是否进入matched carrier训练，以及是否必须显式采用depth/geometry inference。暂不直接升32³、增加renderer samples、扩大当前8³训练或停止整个representation路线。本轮没有执行新carrier训练，也没有开启final holdout。

以上是归因证据及下一阶段设计建议，不是严格因果分解，也不是对独立测试集的确认。

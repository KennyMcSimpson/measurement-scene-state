# V2 后续探索：更公平的全局基线与 cycle 信号诊断

状态：POST_HOC_DISCOVERY_ONLY；独立确认仍为 BLOCKED_NO_INDEPENDENT_DATA。

本轮只分析封存 discovery 的384行候选与既有外层OOF决策。没有改变V2 selector、特征、超参数、写入规则、评价或数据划分；没有新读取任何 target GT 文件，也没有打开保护集。

## 新发现：固定轨迹优势不是完整 discovery 选赢家造成的

此前 discovery-fixed16 是在完整 discovery 上选出的轨迹，与折外 selector 比较存在选择范围不对称。本轮增加一个后处理诊断：每次只用其余11个序列，以 sequence 等权平均J从16条轨迹选一个全局赢家，再应用于留出的完整序列。它只能用于本轮探索，不能替换预注册V2的正式主要基线。

12折均选择 ALL/B，因此这个严格按序列交叉验证的全局策略与原 discovery-fixed16 给出相同动作。

| 策略（均为折外评价） | mean J | ΔJ vs OFF（百分点） |
|---|---:|---:|
| LOSO-global16 | 0.443092504283 | +0.333052 |
| GateOnly | 0.440918098316 | +0.115612 |
| CycleGate | 0.441006269414 | +0.124429 |

CycleGate−LOSO-global16为 **−0.208623个百分点**，描述性paired sequence bootstrap95%CI为 **[−0.486055,−0.013578]个百分点**。

这表明当前discovery上，复杂的逐样本选择尚未优于稳定的简单全局动作。不能仅把落后归因于固定动作看过完整discovery；也不能据此认定ALL/B会在新域获胜。旧validation已有结果显示它相对OFF并未带来净收益，因此不能把discovery优势外推。

Bootstrap使用10,000次、seed=20260927，固定保存的折外决策并按sequence抽样；各折训练集重叠，区间仅为事后描述，不包含重新训练的变异，不是新独立确认。与前次seed不同造成的细微分位数差异不是性能变化。

## 新发现：cycle 改善不能直接当作任务收益

在24个pair的15条非OFF候选中，168条f_cycle>1e-12，其中33条J比OFF差（19.64%）。这里分母是cycle改善的候选，不是selector真正写入的pair，不能混称有害写入率。

19个pair同时有非退化cycle与reward变化，其中候选内Spearman相关11正、8负；其余5个记NA，没有用0替代。这只检查提前指定的f_cycle，没有搜索其他特征。12个外层模型的cycle标准化系数有11正、1负，但系数符号并不证明它有独立预测价值。

CycleGate−GateOnly的OOF差仍只有+0.008817个百分点，95%CI [−0.012120,+0.034290]个百分点，跨0。当前证据支持继续把cycle当待验证信号，不能宣称它已有效。

## 接下来应推进什么

1. 优先补齐可审计的独立原视频与标注语义，运行已冻结的V2确认；不能靠调整现有discovery方法代替独立数据。
2. 若另起方法开发实验，把交叉验证全局策略保留为简单比较对象。当前更需要解释selector为什么丢失全局动作收益，而不是增加模型复杂度。
3. 任何新数据adapter或任务定义变更另写协议，预先锁定，再处理新测试target GT。不能把本轮诊断用于修改V2后继续称原方案独立确认。

外部资源调查见 [resource_audit.md](resource_audit.md) 和机器可读resource_audit.json；本轮不根据任何新数据分数筛选数据。

## 文件与复现

- discovery_diagnostics.json：全部折外全局动作、每序列比较、cycle候选诊断与系数。
- verification.json：63份V2封存文件哈希均未变；新增2项测试通过；结果重生成逐字节一致。
- run.log：本轮实际计算输出。

在仓库根目录执行：

```bash
PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/explore_v2_discovery.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest tests/test_v2_followup.py -q
```

新增测试验证留出序列的标签变化不会改变该折选择的轨迹，以及缺少/重复候选时拒绝计算。本轮没有修改冻结实现，因此只运行相关新增测试；此前完整测试记录仍保留在V2目录，未将其改写为本轮全套测试。

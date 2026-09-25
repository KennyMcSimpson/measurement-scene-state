# 逆向 JEPA + 动态 TTT：第一阶段实现与调试

日期：2026-09-19。代码根目录：`D:/code/codepy/measurement_scene_state`。

本次完成的是可运行、可反向传播的第一阶段工程实现。它能够从多张已见图像构建场景状态，逐帧选择写入位置、更新场景私有快速权重，再用固定 renderer 读出结果。小模型的程序链路检查不能证明研究方法有效，也不代表完整动态策略已经训练完成。

## 两个核心思路如何落地

**逆向 JEPA** 体现在“图像上下文 → 查询无关的三维状态 → 固定测量 → 真实 RGB/深度监督”这条链路。训练时，从同一场景取两组互不重叠的上下文，各自构建状态，再面对同一组查询图像计算误差。这样需要由场景状态解释观测，读出器本身没有可学习参数。代码是 `training/grounded.py`、`dynamic/carrier.py` 与已有固定 renderer 的组合，当前并没有增加隐空间相似度损失来冒充这一思路。

**动态 TTT** 体现在流式更新。新图像到达时，先用旧状态在该相机位置做预测；比较预测与当前图像后，决定 `OFF / FUSE / COMPLETE / ALL`。随后缓存新观测，按动作向两个明确的矩阵写入偏移，重新构建场景。`OFF` 仍接收新图像，只是不写快速权重。

当前已经有四个固定动作对照，以及一个按预测误差阈值选动作的调试策略。阈值策略没有训练，不能当作论文中的学习控制器。离线动作教师、未来收益标签与策略学习仍需后续实现。

## 文件职责

| 文件/目录（相对代码根目录） | 职责 |
| --- | --- |
| `src/mcss/data/episodes.py` | 逐帧读取已到达的 RGB/相机，不提供查询标签 |
| `src/mcss/data/pilot_provenance.py` | pilot 入口核对 split、scene、帧 ID 与实际资产路径 |
| `src/mcss/dynamic/types.py`、`cache.py` | 分离观测缓存、快速权重、写入提案和封存状态 |
| `src/mcss/dynamic/lifting.py`、`carrier.py` | 提取图像特征、投影并构建三维状态；提供两处可写矩阵 |
| `src/mcss/dynamic/write_rule.py` | 直接外积写入、支持权重、更新裁剪及同步提交 |
| `src/mcss/dynamic/feedback.py`、`policy.py` | 更新前反馈与动作选择 |
| `src/mcss/dynamic/budget.py`、`runner.py` | 预算、执行顺序与逐步记录 |
| `src/mcss/dynamic/checkpoint.py` | 独立动态 checkpoint 格式；不保存场景私有快速状态 |
| `src/mcss/training/grounded.py`、`write_unroll.py` | 静态测量监督及可微短轨迹展开 |
| `src/mcss/training/supervision.py`、`smoke.py` | 离线训练标签读取与短优化检查 |
| `src/mcss/evaluation/sealed_queries.py`、`streaming_report.py` | 封存后读取查询并生成工程报告 |
| `scripts/compile_dynamic_episodes.py` | 编译分离的 online/query 清单 |
| `scripts/run_dynamic_ttt_pilot.py` | 薄入口：运行多个动作对照 |
| `scripts/smoke_dynamic_training.py` | 薄入口：运行有限步数的真实数据训练检查 |

旧静态模型与训练入口继续保留。动态 checkpoint 使用 `mcss.dynamic.v1`，明确拒绝直接加载旧 V5 checkpoint。

## 已固定的执行边界

- 在线运行器不导入训练标签读取器或最终查询读取器；查询只在封存后由外层评估打开。
- 所有场景采用首张 warmup 相机作为坐标锚点，使用配置中的固定米制范围；范围不从查询深度估计。
- `ALL` 的两个更新来自同一个旧状态，不先改一个矩阵再计算另一个。
- 新场景重置快速权重；不同分支使用不同身份。提案还绑定精确源状态，防止同版本、不同内容的后继误用彼此的更新。
- 分支与状态身份不进入数值内容哈希，因此同样的数据与权重仍可以做可复现的哈希比较。
- 部署冻结慢参数，不执行反向传播；离线训练复用同一计算内核并保留梯度。
- 账单记录真实模块调用次数及全部已见视图的重处理量。`declared_operator_work_proxy` 是估计工作量，**不是实测 FLOPs**。

## 数据用途

本次下载的 26 个完整 Hypersim ZIP 属于 `final_holdout`，压缩包合计 **102,405,935,184 字节**。解压目标是 `data/hypersim_final_holdout_raw`；压缩包保留。CRC 检查属于文件完整性检查，不属于模型评测。

解压和独立目录复核均已完成：26 个场景、136,795 个实际文件、**102,370,486,124 字节**，缺失/额外文件及文件大小不符均为 0。下载 heartbeat `hypersim` 已暂停。

开发调试使用此前准备的六个场景，分辨率 128×160：

- train：`ai_001_001`、`ai_001_002`、`ai_001_003`。
- dev：`ai_003_010`、`ai_004_003`、`ai_004_004`。
- 每个流式 episode：4 张 warmup、8 张 stream、4 张独立 query。
- 优化检查仅用 `ai_001_001`：warmup 0–3，短 stream 4–5，query 12–15。

清单中的 `dev` 对应磁盘 `hypersim_er_prepared/val`，代码使用明确的固定映射核验；`train` 对应磁盘 `train`。

最终测试集不参与本轮代码调试、训练、阈值选择或模型选择。

## 验证产物

最终回归：**84 项测试通过，Ruff 检查通过，冻结 V5 的 7/7 个文件 SHA256 一致**。真实 CUDA 流式检查完成 30/30 次运行；训练检查完成 3 次优化，两次含动态写入的更新均有有限非零梯度，写入规则本身的参数哈希也发生变化。checkpoint 保存/重载后的同一展开结果哈希一致。

- `outputs/dynamic_ttt_pilot_20260919/runtime_smoke_verified/report.json`：六场景 × 五策略的完整 CUDA 工程运行记录；所有策略从同一随机初始化出发。
- `outputs/dynamic_ttt_pilot_20260919/training_smoke_verified/report.json`：一轮双上下文优化及两轮短 `ALL` 展开优化，分别记录载体与写入规则的梯度、更新前后参数哈希，以及 checkpoint 重载一致性。
- `outputs/dynamic_ttt_pilot_20260919/final_verification.json`：最终测试、代码检查、源码哈希与冻结 V5 核验结果。
- `outputs/hypersim_holdout_extraction_20260919/receipt.json`：解压完成与逐成员 CRC 检查记录。
- `outputs/hypersim_holdout_extraction_20260919/independent_inventory_verification.json`：独立核对解压目录的文件集合、每个文件大小及总字节数。

训练 smoke 的 `untrained-engineering-smoke` 表示“尚未完成正式训练的调试模型”，并不是说没有执行优化。三次优化用于验证梯度与保存/恢复链路；不能用几个 loss 数值论证收敛、泛化或动态 TTT 的收益。

## 复现命令

在代码根目录用项目虚拟环境执行。每次运行使用新的输出目录，避免覆盖已有证据。

```powershell
rtk proxy .\.venv\Scripts\python.exe -B scripts/run_dynamic_ttt_pilot.py --index outputs/dynamic_ttt_pilot_20260919/episodes/episode_index.json --output outputs/dynamic_ttt_pilot_20260919/runtime_rerun_01 --device cuda --policies OFF FUSE COMPLETE ALL RESIDUAL_THRESHOLD

rtk proxy .\.venv\Scripts\python.exe -B scripts/smoke_dynamic_training.py --device cuda --output-dir outputs/dynamic_ttt_pilot_20260919/training_rerun_01

rtk proxy .\.venv\Scripts\python.exe -B -m pytest -q tests/dynamic tests/test_episode_access.py tests/test_sealed_queries.py tests/test_dynamic_ttt_pilot.py tests/test_pilot_provenance.py tests/test_extract_hypersim_holdout_archives.py tests/test_types.py tests/test_geometry.py tests/test_measurements.py tests/test_v5_checkpoint_compatibility.py
```

## 下一阶段

先把载体及写入规则训练到有意义的开发集水平，再加入离线动作教师和只看前缀的学习策略。之后做匹配计算预算的固定动作/动态动作对照，以及逆向 JEPA 的匹配表示与读出对照。只有这些完成，才适合讨论两个创新的独立收益与结合收益。

当前载体使用 8×8×8 网格和小型 CNN，没有预训练骨干。它的尺寸服务于接口和梯度调试；大场景覆盖、高质量几何、正式训练日程、多种子统计以及公开基线比较仍未验证。固定坐标范围可能截掉场景，后续需要在训练/开发集上检查覆盖情况，不能用最终测试集反向调范围。

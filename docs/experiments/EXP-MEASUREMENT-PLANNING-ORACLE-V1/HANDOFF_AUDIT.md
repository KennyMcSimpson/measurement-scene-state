# HANDOFF_AUDIT：BUDGETED-MEASUREMENT-PLANNING 的交接审计（2026-10-06）

新研究线问的是：RGB 与相机免费，深度测量有预算，**应该在哪些帧上请求深度**。本审计记录它可以依赖的仓库状态、可复用组件、帧角色与数据污染情况。审计只读，没有运行任何实验。

## 1. 仓库当前状态

- **位置：** hit 上的 `~/zonghan_o_mirror/measurement-scene-state`。
- **提交：**
    - HEAD 为 `249becbbe7d72be2a40d5e3c2b3f84ec01d8d7d2`（main，等于 GitHub main，即 PR #3 合并之后）。
    - 工作区有 132 项未提交的改动：10-01 之后的全部工作（V17–V33、Core B V3–V11、V24–V30、DepthSplat 探针归档、STATUS 与 EXPERIMENTS 的更新）。这些改动从未推送。
- **新分支：** 本地分支 `research/measurement-planning`，基于 249becb，带着上述未提交改动，没有推送。
- **起点快照：** `outputs/EXP-MEASUREMENT-PLANNING-ORACLE-V1/audit/start_state/`，内含 `git_head.txt`、`git_status.txt`、`git_diff.patch`（797 行）、`git_branch_before.txt`、`captured_at.txt`。
- **版本控制范围：** `outputs/` 整体在 `.gitignore` 中，是本地实验记录；`docs/`、`scripts/`、`src/`、`tests/` 受版本控制。
- **权威记录：**
    - `STATUS.md` 与 `EXPERIMENTS.md`，最新条目在前；
    - 阶段总结 `docs/research_summary_20261003.md`；
    - 10-06 的项目总结报告（Claude Doc，见 STATUS 的 10-06 条目）。

## 2. 可复用的组件（逐字复用，不重写）

| 组件 | 位置 | 行为 |
| --- | --- | --- |
| 读帧与访问记录 | `src/mcss/mechanism_pilot/direct_capacity_contracts.py`：`_Media.read(ids, depth_allowed, purpose)` | RGB 读成 float32 后除以 255；内参与 c2w 为 float32；深度 `np.load` 后转 float32，无效值置 0；每次读取都把 scene、frame、purpose、channels 写入访问记录。`ContextOnlyLoader` 只允许读上下文角色，查询深度另走封存评价路径。 |
| 世界点 | `scripts/run_rgbd_geometric_baselines.py`：`context_points` | 所有有效深度样本的世界点，float64。 |
| 重投影 | 同上：`reproject` | 1 像素 z-buffer，每个像素取最小的光线距离；返回 `(depth, hit)`。 |
| 查询相机 | 同上：`query_camera` | 由清单的内参与 c2w 构造，float32。 |
| 调和补洞 | `scripts/run_rgbd_inpaint_baselines.py`：`harmonic_fill`、`ray_cosine` | 命中像素保留；空洞在逆平面深度上解 4 邻域 Laplace 方程；完全没有命中时用 TRAIN 常数 2.3736 m。 |
| NEAR 掩码 | 同上：`near_mask` | 到最近命中像素的欧氏距离不超过 8 像素。 |
| 多帧版本 | V24 `scripts/analyze_rgbd_stream_baselines_v24.py` | REPROJ7_HARMONIC 把上下文与 4 个流帧拼成一个 `RGBDContext`，再调用同样的函数。 |
| 统计模板 | `mcss.mechanism_pilot.direct_capacity_statistics.describe / paired`；`geometry_carrier_statistics._evidence / surface_status`；聚合见 `scripts/evaluate_rgbd_completion_eval_v3.py` 的 `scene_values` | 场景 bootstrap，10,000 次，种子 20260928；V2 gate 要求均值大于 0、95% 区间下界大于 0、至少 75% 的场景不变差、留一场景增益全为正、正贡献最大的单个场景占比不超过 0.5。场景值的算法：角色内先对查询取平均，再在角色之间平均。 |
| 已封存的基线 | `~/cv_work/depthsplat/probe_eval34`（EVAL-V3∪V4，83 个场景）、`~/cv_work/depthsplat/probe_dev`（DEV） | 都含 REPROJ_HARMONIC、NEAR 掩码与 G1 的预测；EVAL34 上 G1 1 个 seed，DEV 上 3 个 seed。 |
| 运行器 | `~/cv_tools/cv_host_run.py` | 把记录里的 `/home/zonghan/` 路径映射到镜像。只用 CPU 的命令要设 `CUDA_VISIBLE_DEVICES=""`。 |

**关键的等价关系：** z-buffer 对每个像素取最小值，所以"多帧一起重投影"严格等于"各帧分别重投影后逐像素取最小值"。新实验据此为每个查询、每一帧预先算一次重投影；任意子集的结果是这些图的最小值，再做调和补洞。K=0 时的结果必须与封存的 REPROJ_HARMONIC **逐位相同**，这一条作为完整性检查。

## 3. 帧角色（权威定义）

- **帧数：** 每条记录 16 帧。EVAL-FAR 的记录多 2 个远处查询帧，共 18 帧。
- **角色：**
    - `context_a`、`context_b`：各 3 帧，共享 anchor 帧 0；
    - `primary_query`：2 帧，与 `query` 相同；
    - EVAL-FAR 另有 `far_query`：2 帧，例如 32、49；
    - TRAIN72 的记录另有 `stream`（2 帧）与 `ignored`。
- **空闲帧：** 不属于任何角色的帧。DEV、EVAL-V3、EVAL-V4、EVAL-V5、TRAIN-EXT、EVAL-FAR、Replica 的每条记录都恰好有 9 个。
- **Core B 的流帧：** Core B（V4–V11、V24、V25）使用的 4 个流帧，就是 9 个空闲帧中按帧序等距选出的 4 个。例如 `[5, 8, 10, 13]`，见 `outputs/EXP-3D-RGBD-DYNAMIC-TTT-SOURCE-V8/stream_roles.json`。它相当于一种固定的 UNIFORM-4 规则。
- **查询的位置：**
    - EVAL-V3/V4 的 83 条记录里，69 条的查询是 14、15；
    - 有些记录的查询夹在空闲帧中间，例如 ai_001_005 的查询是 12、15，空闲帧里有 13、14。
- **相邻性：**
    - EVAL-V3/V4 的 83 条记录中，81 条的空闲帧里有一帧与某个查询帧序相差 1；
    - EVAL-FAR 的远处查询与所有空闲帧都相差很远；
    - Replica 的查询与最近的空闲帧相差一个采样间隔（50 帧）。

## 4. 评价队列的使用情况

在新方向里，下列队列只能作为开发与机制数据，不能称为独立测试。

| 队列 | 场景数 | 以前的用途 |
| --- | ---: | --- |
| TRAIN72 | 72 | 所有 carrier 的训练集 |
| SEG-V1 | 240 段 | V21 训练、DepthSplat 微调 |
| TRAIN-EXT | 55 | V14、V18、V20 的扩充训练；Core B V6、V9，V27，GPU 试点的评价 |
| DEV | 8 | 每一轮的开发与检查点选择 |
| EVAL-V3、EVAL-V4 | 57、26 | V11 以后几乎每一轮的主评价；DepthSplat 探针 |
| EVAL-FAR | 79 | V13、V22、V25 等 |
| FRESH-V1、FRESH-V2 | 5、17 | 资格检验，已用尽（FRESH-V2 用过 3 次） |
| EVAL-V5 | 23 | Core B V7、V27 |
| Replica | 8 | Replica 迁移、V28 |

## 5. 受保护的保留集

- 清单中的 `protected_scene_ids` 共 58 个场景（官方 test 划分）。它们与以上所有队列都不重叠，`_scene_guard` 禁止读取。
- 新方向不打开它：`FINAL_HOLDOUT_TOUCHED=false`。

## 6. 对新任务的含义

- **episode：** 一个场景加一个上下文角色（A 或 B）。
- **初始上下文：** 该角色的 3 个 RGB-D 帧。
- **候选池：** 该记录的 9 个空闲帧。它们不与查询帧、也不与两个角色的上下文帧重叠，两个角色共用同一个池。
- **查询：** `primary_query` 的 2 帧。EVAL-FAR 另有远处查询，可作为"远离所有候选帧"的次要查询集。
- **信息：**
    - 候选帧的 RGB 与相机可见；
    - 候选帧的深度只在请求之后读取，读取用途分开记录：可部署策略为 `CANDIDATE_DEPTH_REQUESTED`，oracle 为 `ORACLE_TEACHER_ONLY`；
    - 查询帧的 RGB、深度与相机都不给可部署策略。
- **主要风险：**
    - 几乎每个 episode 都有一个与查询相邻的候选帧，"测最后几帧"这类按时间新近选帧的规则可能就接近 oracle；
    - 预注册时需要用帧序距离池与远处查询，把相邻帧效应与真正的规划余量分开。
- **不需要训练：** 后端用无模型几何，不涉及 carrier；G1 只作次要分析，使用已封存的预测。

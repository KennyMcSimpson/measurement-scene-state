# EXP-3D-RGBD-DEPTHSPLAT-JEPA-TTT-V1：以 DepthSplat 为基座检验逆向 JEPA 与 TTT

本文件即预注册。锁定时间与哈希见 `preregistration.json` 和 `audit/locked_at.txt`。锁定之后不改任何设置；如需修订，只在文末追加带日期的记录。

## 1. 要回答的问题

用户要求（2026-10-07）："在新的基座上，验证一下逆向jepa和ttt"。新的基座是 DepthSplat：用户 10/5 指定以它为基座。

- **Core A（逆向 JEPA）：** 场景状态是 DepthSplat 由 3 帧 RGB-D 上下文生成的高斯，测得深度在有效处注入。训练目标是"预测其他视角的测量"，即在其他帧渲染深度和 RGB 并与测量比较。问题是：这样训练整个编码器之后，状态对上下文没看到的区域（FAR）的深度预测，能否稳定好于无模型的调和补全？
- **Core B（TTT）：** 新到达 4 帧 RGB-D 流帧。把它们作为额外视角一起前馈，称为 CACHE。问题是：在此基础上，再对高斯状态做测试时梯度优化（TTT），能否进一步改进查询深度？能否好于用同样 7 帧做的无模型重投影？

## 2. 数据

- **训练：** TRAIN72（V7 split）加 SEG-V1（TRAIN72 场景的 240 个轨迹片段），共 312 条记录。
- **DEV：** V7 DEV，8 个场景，16 个 episode。只用于开发：
    - 观察尺度上限；
    - 选择 TTT 步长；
    - 演练整个流程。

    训练过程中的 DEV 读数只记录，不用于任何选择。
- **评估：** FRESH78 = EVAL-V5（23 个场景）加 TRAIN-EXT（55 个场景），共 156 个 episode、312 个查询。
    - 没有被任何 DepthSplat 运行、本实验的训练数据或测量规划审计读取过。
    - 与 TRAIN72、DEV、保护集在场景和物理场景两个层面都不相交（`make_eval_root.py` 断言）。
    - 旧项目 Core B V6 和 V7 曾把它们当评估队列用过（用的是旧载体）。所以对新基座是新数据，但不是"从未被看过"。
    - 保护集（官方 test）不打开。
- **episode：** 一条记录配一个角色（A 或 B）。
    - 上下文 = 该角色的 3 帧。
    - 流 = Core B 帧号规则：在空闲帧中均匀取 4 帧，两个角色共享，只看帧号。该规则与 V6/V7 封存的流角色逐一一致（已校验）。
    - 查询 = primary_query。
    - 冻结在 `frame_role_lock.json`。

## 3. 状态构建器

所有构建器都注入测得的上下文（及流）深度：有效像素用测量的平面深度替换 DepthSplat 的最终深度。输入为 256×320，渲染为 128×160。

- **ZS：** 发布权重（re10k+dl3dv base，sha f8ddd845…），高斯尺度上限 0.1。这是 README 的推理设置，与 probe 相同。
- **ZS3：** 发布权重，尺度上限 3.0。3.0 是 DepthSplat 训练配置的默认值。ZS3 是与训练模型同设置的对照。
- **J1–J3（逆向 JEPA）：** 从发布权重出发训练整个编码器，用 `code/train_jepa.py`。
    - **种子：** 20261008、20261009、20261010。
    - **步数与学习率：** 4000 步；AdamW，学习率 5e-6（单目 ViT 为 5e-7）；weight decay 1e-4；预热 200 步后余弦衰减到 0；梯度裁剪 1.0；eval 模式；TF32 关闭。
    - **每步数据：** 1 条记录、1 个角色、2 个目标帧。目标帧从该记录的非上下文帧中随机抽取。
    - **损失：** 原始渲染深度（零背景上 Σα·T·z，未覆盖像素按全深度计误差）对目标测量平面深度的相对 L1（只算有效像素），加 0.5 × RGB L1。
    - 尺度上限 3.0，好让状态有能力覆盖未观测区域。
    - 使用最后一步，不做选择。

## 4. 臂与读出

**CPU 臂（冻结的项目函数）：**

- **REPROJ_HARMONIC：** 3 帧上下文的 1 像素 z-buffer 加调和补洞。
- **REPROJ7_HARMONIC：** 同一方法，用 3 帧上下文加 4 帧流。
- **G1：** V21 G1 载体（V17 L 配方，T24000），3 个种子，由 arms 评估器封存。
- **FAR_G1：** NEAR 用 REPROJ_HARMONIC，其余用 G1。

**NEAR 掩码：** 距 3 帧重投影命中 8 像素以内的区域记为 NEAR，其余有效像素为 FAR。用 7 帧计算的同样掩码记为 NEAR7 / FAR7。

**GPU 臂（每个构建器、每个 episode）：**

- **CTX：** 3 帧上下文。
- **CACHE：** 3 帧上下文加 4 帧流，作为 7 视角输入。
- **TTT：** 只用于 ZS。对 CACHE 的高斯在这 7 帧测量上做 100 步 Adam，损失与训练相同。
    - 优化对象：均值；在各高斯自身特征基下的逐轴对数尺度（旋转固定）；不透明度 logit；SH 系数。
    - 学习率为 m 乘以 3DGS 默认值：
        - 均值：1.6e-4 × 7 帧测得平面深度的中位数；
        - 对数尺度：5e-3；
        - 不透明度：2.5e-2；
        - SH DC：2.5e-3；
        - 高阶 SH：1.25e-4；
        - Adam eps：1e-15。
    - J 模型不做 TTT：每个 episode 的 TTT 约需 50 秒 GPU 时间（7 个视角、57 万个高斯）。这一点在开发阶段已确定。

**读出：** 深度 = (depth_z / α) / cos，只在 α ≥ 0.5 处使用；其余像素用回退值。

- `X_CTX_F`：回退到 REPROJ_HARMONIC。
- `X_CTX_G1`：NEAR 用 REPROJ_HARMONIC；FAR 中 α ≥ 0.5 处用 DepthSplat，其余用 G1（按 G1 种子平均）。
- `X_CACHE_F7` / `X_TTT_F7`：回退到 REPROJ7_HARMONIC。
- `X_CACHE_H7` / `X_TTT_H7`：混合读出。NEAR7 用 REPROJ7_HARMONIC；其余像素在 α ≥ 0.5 处用 DepthSplat，否则用 REPROJ7_HARMONIC。含义是"测量够得着的地方用测量，其余用状态"，与项目以往的 AHFILL8 类混合读出一致。

**指标：** AbsRel，分 ALL（有效真值）、NEAR、FAR、NEAR7、FAR7 五个区域；覆盖率 = α ≥ 0.5 的像素占比。

**汇总：** 先在每个角色内对查询求平均，再在每个场景内对两个角色求平均；空区域跳过。J 取 J1–J3 的场景值平均。

## 5. 主检验

主检验使用 V2 gate：均值 > 0，95% CI 下界 > 0，不变差的场景 ≥ 75%，所有留一场景均值 > 0，正向贡献最大的单个场景占比 ≤ 0.5。统计用场景 bootstrap，10,000 次，种子 20260928。

| 编号 | 对比（> 0 表示右边更好） | 区域 |
| --- | --- | --- |
| A1 | REPROJ_HARMONIC − J_CTX_F | FAR |
| B1 | ZS_CACHE_F7 − ZS_TTT_F7 | ALL |
| B2 | REPROJ7_HARMONIC − ZS_TTT_F7 | ALL |

**分支：**

- **Core A：** A1 通过 → A_SUPPORTED。只有均值 > 0 且 CI 下界 > 0 → A_PARTIAL。CI 上界 < 0 → A_HARMFUL。其余 → A_NOT_ESTABLISHED。
- **Core B：**
    - B1 和 B2 都通过 → B_SUPPORTED_PRACTICAL。
    - 只有 B1 通过 → B_MECHANISM_ONLY：写入有用，但不比无模型方法好。
    - 其余情况按 B1 的状态，记为 B_PARTIAL、B_HARMFUL 或 B_NOT_ESTABLISHED。

## 6. 次要分析

次要分析只给标签，不改变分支。

- **A2a：** ZS3_CTX_F − J_CTX_F（FAR），同一尺度上限下的训练收益。
- **A2b：** ZS_CTX_F − J_CTX_F（FAR），相对发布设置的收益。
- **A3：** FAR_G1 − J_CTX_G1（FAR），J 在 G1 之上有没有增量。
- **A4：** REPROJ_HARMONIC − J_CTX_F，分 ALL 与 NEAR。
- **A5：** REPROJ_HARMONIC − ZS_CTX_F（FAR）。
- **A6：** REPROJ_HARMONIC − FAR_G1（FAR）。
- **B3：** B1 在 FAR7 上的版本。
- **B4：** ZS_CTX_F − ZS_CACHE_F7（ALL），缓存流帧本身的收益。
- **B5：** REPROJ_HARMONIC − REPROJ7_HARMONIC（ALL）。
- **B6：** REPROJ7_HARMONIC − ZS_CACHE_H7（ALL）。
- **B7：** REPROJ7_HARMONIC − J_CACHE_H7（ALL）。
- **B8：** REPROJ7_HARMONIC − ZS_TTT_H7（ALL），混合读出的版本。
- **B9：** REPROJ7_HARMONIC − ZS_TTT_F7（FAR7）。
- 各读出、各区域的均值与覆盖率。

## 7. 开发阶段（锁定前、只用 DEV 与 TRAIN）已知的事

见 `development/`。

- **冒烟测试（DEV，零样本，注入深度）：**
    - 尺度上限 0.1：覆盖像素的 AbsRel 0.140，覆盖率 0.62。
    - 尺度上限 3.0：0.255，覆盖率 0.82。
    - 在 3.0 下训练 40 步：覆盖率上升到 0.88，覆盖像素的 AbsRel 变为 0.279。

    训练每步约 0.85 秒，显存 6.3 GB。
- **计时：** 7 视角渲染一次前向加反向约 0.5 秒。这决定了 TTT 只做 ZS。
- **DEV 演练：** 几何封存与 arms 封存的复核最大差 7.1e-15，NEAR 完全一致。
- **TTT 步长选择（DEV，ZS，m ∈ {1, 3, 10}，取 DEV ALL AbsRel 最低者）：** 选中 **m = 1**，见 `ttt_selection.json`。
    - ZS_TTT_F7 的 ALL 分别为 0.1103（m = 1）、0.1110（m = 3）、0.1120（m = 10）。作为参照，ZS_CACHE_F7 为 0.1294，REPROJ7_HARMONIC 为 0.1127。
    - 在这 7 帧上，TTT 把深度项从 0.150 降到 0.043（m = 1）、0.013（m = 10）。步长越大，对这 7 帧拟合得越好，查询反而略差。
    - 没有出现失败；协方差重建误差 ≤ 1e-6。
- **B2 读出的决定（基于 DEV，在任何 FRESH78 计算之前）：** 起草时 B2 用的是混合读出 H7。但 DEV 上 TTT 之后的状态在 NEAR7 比 REPROJ7 更准（0.0745 对 0.0766），而 H7 恰好把 NEAR7 换回 REPROJ7，丢掉了 TTT 的主要效果；剩下只比较 FAR7 中约 9% 的覆盖像素，近乎退化（DEV 上 H7 为 −0.0004，F7 为 +0.0023）。所以 B2 改用状态自身的读出 F7，H7 版本作为次要分析 B8。
- **检查点存取：** 训练 10 步并保存，再由 `ds_eval.py --checkpoint` 读回（DEV 2 个 episode，CTX、CACHE、TTT 5 步）。预测确实改变：α 最大差 0.045。
- **GPU 共用：** `nvidia-smi pmon` 显示 4 号卡上另一用户的进程实际占用约 49% 的 SM，而 `utilization.gpu` 在本机读数为 0，不可靠。正式作业因此把 ZS 评估分成 3 个分片，与 3 个训练并发。
- **分析阶段演练（DEV）：** 用发布权重（上限 3.0）作为 J1–J3 的替身，score、select、analyze 全部跑通。替身就是 ZS3，所以 A2a 恰好为 0。DEV 上的其他读数，例如上限 3.0 时 FAR 为 0.403、调和补全为 0.325，见 `development/DEV_results.json`。这些只是开发信息，不是本实验的结果。

## 8. 完整性

- **查询深度：** 只在 `score_analyze.py score` 中读取。读取前，所有封存必须完整且哈希不变：arms 封存、几何封存，以及每个构建器的 DepthSplat 封存。读取记录写入 `scores/FRESH78/score_info.json`。
- **复核：**
    - 几何封存对 arms 封存复核 REPROJ_HARMONIC 与 NEAR，最大差 ≤ 1e-9，NEAR 必须完全相同。
    - 流帧与 V6/V7 的封存一致。
- **确定性：** 分析运行两次，结果文件必须逐字节相同。
- **GPU 训练不可逐位复现：** 用 3 个种子，并且只做同设备比较。
- **训练失败：** 某个种子训练失败时如实报告；只要至少 2 个种子完成，就用完成的种子做分析，不改任何设置。
- **TTT 失败：** 某个 episode 的 TTT 出现非有限损失时，该 episode 保留 CACHE 状态。失败次数写入 DepthSplat 封存并报告。
- **其他约束：** 不打开保护集，不推送 GitHub，不删除旧输出，不下载新数据。

## 9. 已知局限（锁定前写明）

- **相机模型：** DepthSplat 使用针孔内参 (fx, fy, cx, cy)。部分 Hypersim 相机有斜切项（tilt-shift），这些项被忽略；测得深度转平面深度时也用同样的近似。无模型臂用的是完整内参。所以在这些场景上 DepthSplat 略吃亏。这对所有 DepthSplat 臂相同。
- **J 的设置与零样本不同：** J 用 3.0 的尺度上限训练和读出；ZS 用 0.1。两者的差异由 A2a（同上限）和 A2b（对发布设置）分别给出。

## 修订记录

- **修订 1（2026-10-08 01:2x，执行层面，不改任何设置）：**
    - 正式 GPU 作业的第一次尝试把 3 个训练与 3 个 ZS 分片、ZS3 同时调度，显存超过 47.4 GiB。三个训练在第一步因显存不足退出，没有完成任何优化步。
    - ZS、ZS3 评估不受影响，保留。
    - 训练与 J 评估改由 `code/formal_gpu_job_amend1.sh` 在第一次尝试释放 4 号卡后重跑，命令与设置逐字相同。
    - 详见 `audit/incident_attempt1_oom.md` 与 `audit/amendment_1.json`。
- **修订 2（2026-10-08 03:0x，读取真值的时机）：**
    - 用户要求先看 TTT 的结果，再测逆向 JEPA（"测试完ttt的效果后可以测一下逆向jepa"）。所以 Core B 分两阶段读取真值：
        - **第一阶段：** ZS、ZS3 封存完成后，并且在三个逆向 JEPA 训练（修订 1）已经运行时，用锁定的 `score_analyze.py` 只对 ZS、ZS3 与无模型臂打分，并运行探索性事后分析。由 `code/coreb_stage1.sh` 执行，结果在 `stage1_coreB/`。
        - **第二阶段：** J 的封存完成后，对全部构建器重新做正式评分与分析（`formal_cpu_jobs.sh score`）。
    - J 的流程是锁定且自动的：固定步数，用最后一步，不做任何选择。所以提前读取 Core B 的真值不会影响它。
    - 除此之外不改任何设置或判定规则。

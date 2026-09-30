# 接手修订记录（正式封存前）

日期：2026-09-29（+08）。

构建本实验的 Codex 会话于 2026-09-28 22:52（+08）因用量额度耗尽中断。中断时已完成数据锁定（24 TRAIN / 8 DEV，fresh 为 `BLOCKED_INDEPENDENCE_UNRESOLVED`），尚未封存 preregistration，**没有任何真实数据上的模型训练、checkpoint DEV 评价或 query 评价**。随后由新的代理（Claude）按用户授权接手。以下修订全部发生在正式封存之前，修订时不存在任何可供参考的本轮结果。

## 修订内容

1. **加入次要消融 C2 = FREE_SURFACE16**（C1 + 0.1 × free-space，沿用 direct-state S1/S3 的 free-space 定义与 λ_free=0.1）。协议 §4 允许该 secondary ablation。C2 与 C0/C1 使用相同 architecture、seed、初始化、场景顺序和射线采样流；只报告 `FREE_SPACE_EXTRA_GAIN = AbsRel(C1) − AbsRel(C2)`。C2 不进入 C0/C1 共同 opacity mask、OBS 区域诊断或任何资格 gate，`PRIMARY_METHOD` 固定为 C1，不能事后更改。
2. **C0/C1 主共 mask 与 OBS 区域只由 C0、C1 计算**，评价与 finalize 复算两处一致，保证 secondary 结果不改变主比较。
3. **STATIC_DEV_STATUS 的 opacity artifact 检查改为 C1 自身的静态检查**：每个 DEV scene 的 C1 coverage ≥ 0.99 × ray_hitfraction；在 anchor 与 direct 同时 opacity>1e-6 的 GT-valid 像素上，anchor−direct AbsRel 及其 depth/opacity 归一化版本的 scene bootstrap CI 下界均 > 0，且至少 75% scene 有非空 mask。原实现复用了 C0/C1 的 surface opacity 检查，与协议 §24.6 "geometry 不是 opacity artifact" 所指的 full-context-vs-anchor 比较不对应。`SURFACE_TRAINING_STATUS` 的所有检查保持不变。
4. **数值库单线程**：`num_threads=1`，`OMP/OPENBLAS/MKL_NUM_THREADS=1`。
5. **CPU 亲和性固定为 8–23 号核**（用户要求）。运行器启动时检查亲和性，不在该范围内即拒绝运行，所有子进程继承。封存前的诊断：机器为 Intel Core Ultra 9 285K（0–7 为 5.1–5.2 GHz 的 P 核，8–23 为 4.6 GHz 的 E 核）。仅运行两份统计测试文件，不绑核时 12 次中 3 次在 numpy `mean` 内原生段错误，绑定 8–23 后 12 次 0 次；绑核后的完整测试通过（847 passed，1 skipped）。项目早先多次记录的"原生段错误"很可能源于同一硬件不稳定，而非代码。
6. **预先规定基础设施故障处理**：被信号终止（负返回码）的子进程，或日志中出现固定 CUDA 资源分配错误特征（`CUDA out of memory`、`CUBLAS_STATUS_ALLOC_FAILED`、`CUSOLVER_STATUS_INTERNAL_ERROR` 等）的子进程，其日志与运行目录移入 `audit/failed_attempts/` 并记录 `audit/incidents.json`（含 `kind`），以完全相同的命令从头重跑；每个运行最多重跑 2 次，按已归档的失败尝试计数，跨运行器调用累计。失败尝试的 checkpoint 与 DEV 曲线永不使用。其他任何 Python 错误（包括 gate 拒绝）都不重试。已完成的运行（存在 `summary.json`）不重训。
6b. **次要消融不阻塞主比较**：任一 C0/C1 运行失败即停止实验；C2 任一 seed 未完成或没有合格 checkpoint 时，C2 整体从评价中排除，记录于 `evaluated_variants.json` 的 `secondary_status` 与 `audit/training_failures.json`，报告字段 `SECONDARY_C2_STATUS`，主比较与所有 gate 不受影响。
7. **执行方式**（封存尝试 2）：3 个 seed 依次执行，同一 seed 的 C0/C1/C2 同时训练（3 workers），保证三者资源竞争条件相同；每次（重新）启动前等待共享 GPU 至少有 3072 MiB 空闲显存，等待记录于 `audit/gpu_gate.jsonl`。GPU 同时被其他项目的进程共享，成本记录据此解释。

## 封存尝试 1：启动失败，未进行任何训练

第一次封存后按"9 个运行同时训练"启动。共享 GPU 上其他项目已占用约 26 GiB / 32 GiB，9 个进程同时初始化时全部在创建 cuBLAS/cuSOLVER 句柄处因显存分配失败退出（`CUBLAS_STATUS_ALLOC_FAILED` / `CUSOLVER_STATUS_INTERNAL_ERROR`）。核查确认：9 个运行均为 **0 个训练步**，没有任何 DEV 评价，只读取过首个训练场景的上下文 RGB 与相机，未读取任何深度或标签，没有产生或观察任何科学结果。首次封存的 preregistration、全部合同与失败运行的日志和目录原样移入 `audit/freeze_attempt1_launch_failed/`（含 `incident.json`），随后以第 7 条的执行方式和第 6 条的故障处理重新封存（`training_contract.json` 中 `freeze_attempt=2`）。所有科学定义（数据、帧角色、loss、预算、选择规则、gate）与尝试 1 完全相同。

## 未改变的内容

主比较 C0 vs C1 的定义、loss（RGB Charbonnier + masked depth AbsRel；C1 加 0.1×surface，τ=0.5×voxel diagonal）、16³ grid、64 samples、GT-free bounds、24/8 数据与帧角色、3 个 seed、3000 步、checkpoint 步点、DEV 选择规则、`SURFACE_TRAINING_STATUS` 判据、fresh qualification 保持关闭、不运行 Dynamic TTT。

## 核查记录

- 修订前原文件：`audit/pre_takeover_source/`。
- 独立只读代码审查（子代理）：未发现崩溃或 evaluate→statistics→finalize→report 字段不一致；逐项推演确认新旧测试预期正确，确认 C2 与主比较隔离、静态共 mask 的计算与符号、finalize 复算行数正确。采纳的建议：重试预算跨调用累计；C2 失败不阻塞主比较（6b）；静态 gate 使用独立键名 `static_opacity_artifact_gate`；静态行增加 `comparison` 与 GT-valid 分母一致性校验；报告披露原生崩溃重跑次数与 C2 状态；续跑时合并阶段耗时；评价完成标记在最终锁校验之后写入；并行训练失败时列出全部错误；校验 `primary_pair`；更新过时注释；测试中的崩溃子进程禁用 core 文件。
- 含 C2 的合成端到端流程（train → evaluate → analyze → statistics audit → finalize）：`audit/synthetic_full_pipeline_v3_c2.json`，仅工程验证。
- 封存前完整测试的每次尝试（含原生崩溃）：`audit/full_pytest_*_result.json`。

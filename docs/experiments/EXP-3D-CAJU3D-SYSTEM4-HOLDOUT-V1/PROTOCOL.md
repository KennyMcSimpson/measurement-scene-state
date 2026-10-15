# EXP-3D-CAJU3D-SYSTEM4-HOLDOUT-V1 预注册协议（全新保留队列上的读出检验）

计划写于任何 SYSTEM4 数据准备之前，也早于任何 SYSTEM4 预测或真值。

## 0. 背景与目的

- **已有的读出结果**：
  - 卡尔曼式分箱表 KF_BASE 约改善 2%，在 SYSTEM3、VAL 和 CAJU 场所上都成立。
  - 学到的读出 FUSE_BASE（下称 FUSE_V1）在这三批上把均值降低 15–27%。但它的收益偏重尾，每批有 13–26% 的场所变差。
- **缺口**：这些确认集以前都被反复使用过。
- **本实验的两个目的**：
  - 在一个从未准备、从未评测过的 ARKit 队列上，对冻结的读出做一次干净的外部检验；
  - 检验按固定配方训练的 V2 能否更稳。

## 1. 保留队列 ARKIT_SYSTEM4（规则锁定）

- **来源**：EXP-3D-RGBD-ARKIT-SYSTEM3-DATA-V1 的批次（训练折种子顺序第 1000–1359 位）。SYSTEM3 准备到第 160 个合格扫描（批内第 268 位）就停止了。批内第 269–359 位的 91 个扫描已经下载、相机元数据已锁定，但从未准备过。
- **准备**：
  - 用不变的基础规则（`scripts/prepare_rgbd_arkit_data.py` 的 prepare_scan，SYSTEM3 的 candidate_lock 及其源文件哈希）按顺序准备全部 91 个扫描，不设目标数；
  - 合格的扫描组成 ARKIT_SYSTEM4。
  - 不下载任何数据。
- **片段**：与 FULL-SYSTEM-V3 相同的规则：
  - 每个记录 × 角色 A、B；
  - 上下文为角色帧；
  - 流式帧按 Core B 的帧号规则取 4 帧，两个角色共用；
  - 查询为 primary_query。
- **Core B 预测**：与 V3 相同的流程：
  - 臂评估器封存；
  - V1 的 geometry_seal.py，给出 REPROJ7_HARMONIC、near7、cosine；
  - V1 的 ds_eval.py：发布版权重，上限 0.1，CACHE + TTT，m = 1，100 步，在 GPU 4 上运行。

## 2. 被检验的读出（锁定）

- **CORE_B**：sqrt(R7 × F7)。
- **KF_BASE**：KFUSE-V1 冻结的分箱表（sha256 `15100550…`）。
- **FUSE_V1**：FUSENET-V1 冻结的三个 BASE 网络，按其 audit/fit_log.txt 核对哈希，取 log 平均。
- **FUSE_V2**：按下面的固定配方训练，训练完成、哈希记录之后才读取 SYSTEM4 的真值。
  - 已覆盖像素（α ≥ 0.5）：使用 KF_BASE 的权重，不再学习；
  - 未覆盖像素：log 预测 = log R7 + r，r = 0.7·tanh(·)；
  - 网络与 FUSENET-V1 的主干相同，但膨胀率为 1、2、4、8、16、1，以看到更大的空洞上下文；输入为同样的 8 通道基础图像；输出头初始化为零。
  - 训练数据：除 SYSTEM4 以外所有有 Core B 预测的队列，共 458 个场所：
    - FIT_SYSTEM、FIT_SYSTEM2（KFUSE-V1 根目录）；
    - CONF_SYSTEM3、CONF_VAL（同上）；
    - CAJU_PILOT、CAJU_SEL、CAJU_EVAL（CAJU-V1 的 100 步回放与几何封存）。
  - 按种子 20261200 取 10% 的场所作为验证集，用于选 epoch。
  - 3 个种子（20261201–20261203），30 个 epoch，Adam lr 1e-3，权重衰减 1e-4，每批 8 个查询，损失为场所加权的 AbsRel。
  - 读出取三个种子的 log 平均。

## 3. 检验（锁定）

- **主检验**：
  - P1 = CORE_B − FUSE_V2（SYSTEM4 全部场所），用 V1 的配对场所 bootstrap 和 V2 判定。
  - P2 = CORE_B − FUSE_V1（冻结模型的外部检验），方法相同。
- **结论**：
  - P1 与 P2 都为 SUPPORTED → LEARNED_READOUT_CONFIRMED；
  - 任一为 SUPPORTED 或 PARTIAL → LEARNED_READOUT_PARTIAL；
  - 任一为 HARMFUL → LEARNED_READOUT_HARMFUL；
  - 其余 → LEARNED_READOUT_NOT_ESTABLISHED。
- **次要**：
  - FUSE_V1 − FUSE_V2；
  - CORE_B − KF_BASE；
  - KF_BASE − FUSE_V2；
  - 变差的场所数；
  - 已覆盖与未覆盖像素的 AbsRel；
  - 逐场所改善的分位数。

## 4. 完整性

- 所有预测（Core B、各读出）封存之后，才经 TeacherAccess 读取 SYSTEM4 的真值。
- 分析运行两次，结果必须逐字节一致。
- 只用 GPU 4；不下载数据；不打开受保护的保留集；数据与权重不进 git。
- **修订规则**：锁定后的修改只能以 `audit/amendment_N.json` 加本文件末尾追加的方式记录，而且必须在读取 SYSTEM4 的真值之前。

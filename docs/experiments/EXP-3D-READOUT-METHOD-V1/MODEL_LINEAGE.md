# MODEL_LINEAGE：模型血缘（截至 2026-10-15）

| 模型 | 文件 / 哈希 | 训练数据（场所） | 选择 / 验证 | 祖先 | 在哪些队列上是干净的 |
|---|---|---|---|---|---|
| DepthSplat 发布版 | `…-f8ddd845.pth` | RealEstate10K + DL3DV（作者训练），无 ARKit | — | — | 所有 ARKit 队列 |
| Core B | 无额外参数，每个场景做 100 步 TTT | — | TTT 学习率倍数 m = 1，在早期开发集上选定 | DepthSplat | 所有 ARKit 队列 |
| H-JEPA（JEPAGATE-V1） | `hjepa_final.pth`，sha256 `a3f77893…` | 试点 TRAIN 182 | 试点 DEV 8 | — | 除 CAJU 外的所有队列 |
| KF_BASE 表（KFUSE-V1） | `fusion_tables.json`，sha256 `15100550…` | SYSTEM + SYSTEM2（160） | — | Core B 的 F7 / α、R7 | SYSTEM3、VAL、CAJU、SYSTEM4 |
| FUSE V1 BASE（FUSENET-V1，3 个种子） | `BASE_s20261101/2/3.pt`，sha256 `5ed95d59…` / `17430a0c…` / `07e5a79d…` | SYSTEM + SYSTEM2 中的 128 个 | 其余 32 个 | Core B 输入 | SYSTEM3、VAL、CAJU、SYSTEM4 |
| FUSE V1 JEPA（3 个种子） | `JEPA_s*.pt` | 同上 | 同上 | 另加 H-JEPA 特征 | SYSTEM3、VAL、SYSTEM4（CAJU 的 H-JEPA 特征属于样本内） |
| FUSE V2（SYSTEM4-HOLDOUT-V1，3 个种子） | `V2_s20261201/2/3.pt`（见该实验的 `audit/v2_fit_log.txt`） | SYSTEM、SYSTEM2、SYSTEM3、VAL、CAJU 中的 412 个 | 其余 46 个 | KF_BASE 表 | **只有 SYSTEM4** |
| READOUT-LOCO-V1 的各折模型 | 未保存 | 每折为除留出队列外的全部队列 | 训练场所中的 10% | KF_BASE 表 | 各自的留出折 |
| 本实验（READOUT-METHOD-V1）的各折模型（B6、B6W、B7、SYS_A/B/C） | 未保存权重；保存了逐查询预测与训练日志 | 同上 | 同上 | KF_BASE 表（B 系列） | 各自的留出折 |
| 修正案 1 的 KF_FF 表 | `results/amend1/fusion_tables_ff.json`，sha256 `4c543e02…` | SYSTEM + SYSTEM2（160），前馈渲染 | — | 与 KF_BASE 同一拟合流程 | SYSTEM3、VAL、CAJU、SYSTEM4 |
| 修正案 1 的 B7_FF 各折模型 | 未保存权重；保存了逐查询预测与训练日志 | 同 B7 | 同 B7 | KF_FF 表 | 各自的留出折 |

- 每个模型的训练与验证场所列表见各实验的训练日志：FUSENET-V1 的 `results/train_log.json`、SYSTEM4-HOLDOUT-V1 的 `results/v2_train_log.json`、本实验的 `results/phase12/fold_*.json`。
- 所有用 ARKitScenes 训练的权重都不进 git，哈希记在各实验的 `audit/` 中。

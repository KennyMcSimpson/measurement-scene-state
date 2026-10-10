# EXP-3D-RGBD-COMPLETION-ENSEMBLE-REPLICA-DESC：Replica 上的描述性检查（事后，未预注册，无判定）

这是 EXP-3D-RGBD-COMPLETION-ENSEMBLE-V1 之后的附加检查：在不同的数据来源 Replica（8 个场景）上，看"调和插值与 G1 取平均"的表现。场景太少，不做 V2 gate。

| 读出 | FAR | ALL（NEAR 用调和） | FAR 比调和好的场景 |
| --- | ---: | ---: | ---: |
| H3（调和插值） | 0.385 | 0.331 | — |
| G1（3 种子平均） | 0.479 | 0.390 | 2/8 |
| HG = (H3 + G1) / 2 | 0.300 | 0.257 | 4/8 |

- **G1 迁移差：** 在 Hypersim 上训练的 G1，单独用于 Replica 比调和插值更差。
- **平均仍降低均值：** 与调和插值等权平均后，均值明显下降。两个误差部分独立的预测取平均，能压住大误差。
- **一致性没有达到 Hypersim 上的水平：** 只有一半场景变好。
- **方法：** G1 三种子、REPROJ_HARMONIC 与 NEAR 由 arms 评估器封存（`raw/eval/REPLICA/`，`query_depth_read = false`），之后才读真值。代码见 `code/replica_hg.py`，结果见 `replica_hg_results.json`。
- **数据许可：** Replica 只用于研究，不再分发。

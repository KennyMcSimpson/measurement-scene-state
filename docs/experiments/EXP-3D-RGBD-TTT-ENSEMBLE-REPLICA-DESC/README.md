# EXP-3D-RGBD-TTT-ENSEMBLE-REPLICA-DESC：Replica 上的描述性检查（事后，未预注册，无判定）

这是 EXP-3D-RGBD-TTT-ENSEMBLE-V1 之后的附加检查：在不同的数据来源 Replica（8 个场景、16 个 episode）上，看 TTT 集成的表现。流帧沿用 Core B V28，已校验与帧号规则一致。场景太少，不做 V2 gate。

| 读出（ALL） | 场景平均 AbsRel | 比 R7 好的场景 |
| --- | ---: | ---: |
| R7（7 帧无模型重投影） | 0.0615 | — |
| CACHE（只缓存） | 0.0662 | — |
| TTT | 0.0585 | 5/8 |
| √(R7 × CACHE) | 0.0629 | — |
| √(R7 × TTT) | 0.0591 | 7/8 |

- TTT 对比只缓存：8/8 个场景更好。
- √(R7 × TTT) 对比 √(R7 × CACHE)：8/8。

与 EVAL34 上的预注册结论方向一致：集成提高了一致性，TTT 是必要的。

- **方法：**
    - 几何封存对 Replica 的 arms 封存复核，最大差 0.0；
    - DepthSplat 推理用 V1 锁定的 `ds_eval.py`（发布权重，尺度上限 0.1，TTT 100 步，m = 1）；
    - 封存完成后才读真值。
- **代码与结果：** 代码见 `code/`，结果见 `replica_tttens_results.json`。
- **数据许可：** Replica 只用于研究，不再分发。

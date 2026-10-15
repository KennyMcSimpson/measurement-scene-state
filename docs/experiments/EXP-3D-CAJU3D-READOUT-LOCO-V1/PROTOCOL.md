# EXP-3D-CAJU3D-READOUT-LOCO-V1 协议（学到的读出的留一队列稳健性分析）

## 0. 目的

- **已有结果**：学到的未覆盖修正（FUSE_V2 的配方）在全新保留队列 SYSTEM4 上把 Core B 从 0.146 降到 0.092（47/0/5）。
- **缺口**：没有新的未用数据，除非新下载。
- **本分析的问题**：同一配方在其它每个队列上，用"除该队列外的全部数据"训练时，表现是否一致。本分析是描述性的稳健性检验，不做新的确认性结论；分析一律报告，不论结果。

## 1. 设计（锁定）

- **队列**：
  - FIT_SYSTEM、FIT_SYSTEM2：这两个队列拟合过 KF 表，所以只作训练，不作留出；
  - CONF_SYSTEM3；
  - CONF_VAL；
  - CAJU（PILOT + SEL + EVAL 合并）；
  - ARKIT_SYSTEM4。
- **留出折**：CONF_SYSTEM3、CONF_VAL、CAJU、ARKIT_SYSTEM4，共 4 折。每折用其余全部队列训练，覆盖 5 个队列中的另外 4 个以及 FIT。
- **配方**：与 SYSTEM4-HOLDOUT-V1 的 FUSE_V2 完全相同：
  - 已覆盖像素用冻结的 KF 表，只学未覆盖修正；
  - 主干膨胀率为 1、2、4、8、16、1；
  - 30 个 epoch，Adam lr 1e-3，权重衰减 1e-4，每批 8 个查询；
  - 训练场所中 10% 作验证集，种子为 20261300 + 折序号；
  - 3 个种子（20261301–20261303），读出取 log 平均。
- **报告**：
  - 每折的 CORE_B、KF_BASE、FUSE_LOCO 场所均值；
  - CORE_B − FUSE_LOCO 与 KF_BASE − FUSE_LOCO（V1 bootstrap + V2 判定）；
  - 中位改善、变差场所数；
  - 4 折合并（350 个场所）的同样统计。
- **只用 GPU 4**；不下载数据；权重不进 git。

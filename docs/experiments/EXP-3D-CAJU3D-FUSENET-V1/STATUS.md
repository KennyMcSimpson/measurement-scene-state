# EXP-3D-CAJU3D-FUSENET-V1 状态

**结束（2026-10-14 22:28）：JEPA_FUSION_NOT_ESTABLISHED。** JEPA 特征没有改进学到的读出（P1、P2 都不成立）。

次要结论：不用 JEPA 的学到的读出把 Core B 在 SYSTEM3 上降到 0.1143（PARTIAL），在 VAL 上降到 0.0984（SUPPORTED），收益几乎全部来自未覆盖区域。随后在 CAJU-V1 的 58 个场所上做了预注册复现，结果为 REPLICATED。

| 时间 | 事件 |
|---|---|
| 10/14 22:15–22:19 | 协议、代码；8 个测试通过；根目录文件从 KFUSE-V1 逐字节复制 |
| 22:20 | 特征排练：与 KFUSE-V1 的 log P 完全一致 |
| 22:20:46 | 锁定（sha256 `fda3cf90…`） |
| 22:21–22:22 | 查询格 H-JEPA 特征：800 个片段，GPU 4 |
| 22:22–22:24 | 训练 6 个 FuseNet（BASE、JEPA 各 3 个种子），GPU 4，136 秒；模型哈希写入 audit/fit_log.txt |
| 22:24–22:28 | 确认（CPU，两次逐字节一致）：JEPA_FUSION_NOT_ESTABLISHED |
| 22:30 | 描述性分析：每个种子、像素分区、修正量、逐场所分位数 |
| 22:34 | 冻结的 FUSE_BASE 在 58 个 CAJU 场所上预注册复现：REPLICATED |

只用 GPU 4；没有下载数据，没有打开受保护的保留集，没有推送。

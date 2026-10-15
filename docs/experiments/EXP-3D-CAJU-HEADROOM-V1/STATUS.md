# EXP-3D-CAJU-HEADROOM-V1 状态

**结束（2026-10-14 02:24）**：`GATE0_STATUS=NOT_ESTABLISHED`（`NO_USEFUL_DYNAMIC_ACTION_SPACE`），`GATE1_ALLOWED=false`，`JEPA_FEATURE_STATUS=NOT_IMPLEMENTED`，`INTEGRITY=PASS_WITH_AMENDMENT_1`。

| 时间（+0800） | 事件 |
|---|---|
| 10/14 00:45 前后 | 第一阶段审计：Core B 的 TTT 只优化高斯状态；逆向 JEPA 没有潜空间预测器；数据暴露情况已查清。见 `code_audit.md`、`jepa_feature_contract.md` |
| 01:0x | 建根目录；几何封存用时 19 s（COMPAT 与 V3 逐点一致）；17 个测试通过；根目录外的单片段排练（52 s，5.1 GB） |
| 01:07:17 | 锁定（`preregistration.json`，sha256 `f82179bf…`） |
| 01:07:54–01:27:49 | 试点：PILOT 16 个片段 + COMPAT 4 个片段 + V1 独立参考运行 |
| 01:27:55 | 预测层面的单像素最大差检查有两项未通过，自动链按设计停止，正式回放没有启动 |
| 01:31:37 | 修订 1：分布诊断显示引擎等价，差异属于 GPU 非确定性；引擎和 Gate 0 判定都不变；收尾脚本按修订更新 |
| 01:31:50–02:23 | 正式回放：SEL + EVAL 共 100 个片段，3 个分片，各约 50 分钟 |
| 02:24 | 打分：SEL 选出 FULL 并记录哈希，再读 EVAL 真值；分析两次逐字节一致；收尾 |

GPU 只用了 hit 的 4 号卡，与另一用户的常驻进程共用。没有训练，没有下载，没有推送。

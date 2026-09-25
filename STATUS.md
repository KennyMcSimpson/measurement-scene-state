# 当前状态（2026-09-25）

| ID | 状态 | 证据等级 | 已验证 | 当前可以说什么 | 当前不能说什么 |
|---|---|---|---|---|---|
| `BASELINE-V5-4500` | `FROZEN_REFERENCE` | `DEVELOPMENT` | V5 typed appearance 与旧工程记录保留 | 不能当 validation/test 或 CVPR 结果 |
| `EXP-3D-20260919-engineering` | `COMPLETE_ENGINEERING` | `ENGINEERING` | runner、CPU contract、CUDA smoke 和保存恢复检查 | 不能当三维方法胜出证据 |
| `EXP-3D-20260920-method-dev` | `COMPLETE_DEVELOPMENT` | `DEVELOPMENT` | carrier/policy 开发记录和失败修复被保留 | 不能当最终方法或外部 benchmark |
| `EXP-2D-20260921-original` | `SUPERSEDED` | `ENGINEERING` | 原始运行保留；已记录 pair 汇总 bug | 不能引用旧汇总数字 |
| `EXP-2D-20260921-corrected` | `EXPLORATORY` | `EXPLORATORY` | 2424 raw rows、32 validation pairs、10 focused tests、completion verification | 不能叫完整逆向 JEPA、streaming TTT 或 DAVIS J&F |
| `EXP-GEOMETRY-CERT-V1` | `STOPPED` | `DIAGNOSTIC` | certificate frontend 失败门槛已封存 | 不能继续把该 frontend 接入主模型 |
| `EXP-MAPANYTHING-V1` | `STOPPED` | `DIAGNOSTIC` | supplier qualification 失败记录保留 | 不能当外部监督或方法贡献 |
| `OFFICIAL-DAVIS-VAL` | `SEALED` | `UNRUN` | 30 段官方 val 未读取 | 不能暗示已经完成官方 benchmark |
| `DL3DV/Hypersim final holdout` | `SEALED` | `UNRUN` | 旧本地数据已按授权清理，结果和来源清单保留 | 不能声称有 final holdout 结果 |

## 当前研究判断

二维 probe 显示观测 RGB 反馈可以显著降低 RGB 重建误差，但当前 selector 没有把它转化成对象对应收益。下一步应先测试更可靠的高分辨率对应评价和允许 `OFF` 的 task-grounded gate，再决定是否扩大模型。不要直接扩大 RGB 重建训练规模。

## 存储和协作边界

旧三维数据删除记录和二维资产下载记录仍在 `outputs/` 本地。公开仓库只提交小型 provenance 和结果 bundle；数据、权重、缓存和 checkpoint 按 `.gitignore` 排除。项目记忆的持久状态写入中央 CV 分支，不以仓库文件替代。

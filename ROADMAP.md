# Roadmap and gates

这不是“把所有代码继续跑一遍”的清单，而是防止把错误反馈目标包装成方法提升的门槛。

## Gate 0：公开档案可复核

- 保持协议、split、代码 hash、命令和结果 bundle 一致。
- 公开仓库不含数据、权重、缓存、凭证或未授权的大文件。
- 合作者能从 README 找到原始思路、失败实验、当前结论和下一步。

## Gate 1：修正二维评价载体

- 在新的 development split 上提高 token-grid 分辨率，或采用经过验证的 dense readout。
- 量化小目标 token 数、最近邻改变率和对象标签改变率。
- 若收益主要由单一 tiny-object 视频造成，停止把它作为主方向证据。

## Gate 2：寻找合法的 task-grounded feedback

- 冻结当前 backbone、readout 和写入规则。
- 只比较部署时可取得的 RGB residual、跨观测一致性、cycle consistency、写入幅度和历史。
- 拟合允许 `OFF` 的简单 gate；与 OFF、固定 OFF/ALL 和有限 oracle 同时比较。
- gate 必须在新封存 validation 上有跨序列收益，不能只改善 RGB MSE。

## Gate 3：恢复三维主线的协议完整性

- 修复非刚性 frame-0 camera protocol、policy feature roll-in 和 matched-compute 对照。
- 先通过一场景 metadata/camera adapter gate，再谈 DL3DV 外部评估。
- 保持 V5 与 R0 的诊断边界，不把工程 smoke 当 scientific performance。

## Gate 4：完整方法和最终评测

- 明确定义共享状态、真实观测检查接口、动态动作空间、状态回滚和错误写入代价。
- 新建并封存 validation/holdout，至少多 seed、强 baseline、完整官方指标和成本核算。
- 只有通过这些门槛，才使用 `OFFICIAL_BENCHMARK` 或 `FINAL` 证据标签。

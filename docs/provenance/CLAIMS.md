# 证据标签

- `CONTRACT`：接口、输入输出、状态隔离或禁止标签泄漏由测试直接覆盖。
- `ENGINEERING`：代码、保存恢复、运行器、hash 或 CUDA smoke 已通过；不代表科学性能。
- `DEVELOPMENT`：固定开发划分上的结果，可用于选协议或发现问题。
- `EXPLORATORY`：低成本机制筛查，允许结果为负；不能当最终方法结论。
- `DIAGNOSTIC_ORACLE`：使用答案、目标标签或有限搜索暴露机会/上界；不可部署。
- `STOPPED`：门槛失败后停止的分支；失败本身是要保留的证据。
- `SEALED`：数据或 split 已声明保留但本次没有读取。
- `OFFICIAL_BENCHMARK`：官方协议、封存测试、强 baseline、完整指标和成本都满足。
- `FINAL`：多 seed、外部/封存测试、预注册主终点和复现包都满足。

本仓库目前没有 `FINAL` 或 `OFFICIAL_BENCHMARK` 的完整逆向 JEPA + 动态 TTT 结果。

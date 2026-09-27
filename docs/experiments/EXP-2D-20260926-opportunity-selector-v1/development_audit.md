# Validation 前开发审计

- 新实验开始前冻结 config.json 与 split_manifest.json。所有模型/阈值选择只使用 discovery。
- 运行前 review 将 ridge 的 raw J 目标修正为同 pair Jcandidate−Joff，并将 OFF 预测收益固定为零；加入对应回归测试。此时未开始 discovery 正式运行。
- 新增输入图片/mask哈希、权重和DINO源码指纹。所有 validation 决策先持久化，再打开对应 target mask。
- 完整测试首次 collection 发现历史脚本直接导入 Windows msvcrt，现用平台兼容文件锁；补 OpenCV/scikit-image 的显式测试依赖。
- 历史 V5 checkpoint 不在公开仓库；如缺失，其真实权重兼容性测试明确跳过，不用伪造权重替代。
- 固定维度64、eta5、两步16轨迹不改；没有依据 validation 调特征、阈值、指标或样本。
- 本轮使用历史内部 validation 身份，属于预先锁定的新评价协议，不是从未查看过的最终holdout。reserve与官方val持续封存。

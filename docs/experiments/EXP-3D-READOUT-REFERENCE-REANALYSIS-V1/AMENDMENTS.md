# EXP-3D-READOUT-REFERENCE-REANALYSIS-V1 修订记录

## Amendment 1（2026-09-29，在任何指标计算之前）

**起因**：第一次冻结后的运行在读取 DEV GT 之后、计算任何指标之前中止，因为预注册的 MEDIAN_SCALED 读出在部分视角上无定义：缩放系数 median(GT) / median(RAW) 的分母为 0。原始锁与 GT 访问记录保存在 `audit/attempt1_median_scale_undefined/`，那次尝试没有产生、写出或查看任何指标。

**诊断**（只读预测文件，不读 GT）：1728 个预测中有 216 个视角的预测深度处处为 0、opacity 处处为 0。它们全部来自 DEV 场景 ai_009_001：该场景的 query 射线全部落在由上下文相机导出的 bounds 之外（评价中记录的 NO_HIT 场景），renderer 不产生任何采样。V2、V3、V4 的每个变体、每种构造都一样，每组 12 个。

**修订**：
1. MEDIAN_SCALED 在预测中位数不为正时保持不缩放。全零预测乘以任何系数仍是全零，所以这是唯一有意义的定义；在原规则有定义的每个视角上，结果与原规则完全相同。
2. 新增次要敏感性分析：排除所有直接预测都为全零的场景，其余统计照旧。主分析仍按冻结规则使用全部 8 个场景。

代码与测试的修改已在新锁的 source_sha256 中重新冻结（`test_no_hit_views_stay_unscaled_and_feed_the_sensitivity`）。

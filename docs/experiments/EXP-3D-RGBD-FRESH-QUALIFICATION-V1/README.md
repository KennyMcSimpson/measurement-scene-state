# EXP-3D-RGBD-FRESH-QUALIFICATION-V1

`FRESH_QUALIFICATION_STATUS=NOT_QUALIFIED`。

对 V5 已封存的 RGB-D carrier（每个 seed 各自的 DEV 选择，不重新训练）在独立队列 FRESH-V1 上做预注册资格验证。FRESH-V1 共 5 个场景，均来自官方 val split、从未被观察、与 V2–V5 的 TRAIN/DEV volume 不重叠，每个 volume 一个场景。所有 fresh 状态封存之后才读取 query GT；统计沿用冻结的 V2 估计器与 V5 的 gate；受保护的官方 final holdout 未打开。

样本量说明：5 个场景低于数据锁的最低 6 个（修订 1 没有找到替代场景）。QUALIFIED 仍要求每一个冻结 gate；NOT_QUALIFIED 可能反映统计功效不足。

| 项目 | FRESH-V1 | V5 DEV（参照） |
|---|---|---|
| C0 只用 RGB，query AbsRel | 0.4788 | 0.617415 |
| C1 测得深度，query AbsRel | 0.4724 | 0.503044 |
| DEPTH_GAIN | +0.0063，95% CI [-0.1120, +0.1157] | 0.114371 |
| C1 wrong-scene 损伤 | +0.1314，95% CI [+0.0630, +0.1999] | 0.168744 |
| C1 shuffle 损伤 | +0.1973，95% CI [+0.1079, +0.3367] | 0.215871 |
| 3 视角相对 anchor | +0.2000，95% CI [+0.0712, +0.3441] | 0.152435 |
| DEPTH_STATUS | NOT_ESTABLISHED | SUPPORTED |
| SCENE_SPECIFICITY_STATUS | SUPPORTED | SUPPORTED |
| STATIC_STATUS | SUPPORTED | SUPPORTED |

## 逐场景（RAW AbsRel，seed 与角色平均）

| 场景 | 常数（TRAIN 拟合） | C0 | C1 | C2 |
|---|---:|---:|---:|---:|
| ai_015_004 | 0.471 | 0.315 | 0.383 | 0.475 |
| ai_022_010 | 0.289 | 0.257 | 0.478 | 0.443 |
| ai_041_003 | 0.596 | 0.922 | 0.858 | 0.887 |
| ai_051_004 | 0.516 | 0.373 | 0.198 | 0.221 |
| ai_052_003 | 0.595 | 0.527 | 0.445 | 0.445 |

## 与无几何常数相比（描述性）

CARRIER_REFERENCE_STATUS=NO_CARRIER_ABOVE_GEOMETRY_FREE_REFERENCE；C1 RAW AbsRel 相对 AbsRel 最优常数：+0.0211，95% CI [-0.1649, +0.2048]（NOT_DISTINGUISHABLE）；排除 NO_HIT 场景后标签：NO_NO_HIT_SCENE。

## 区域诊断（描述性）

C0−C1 条件 AbsRel：OBS0=+0.0114，95% CI [-0.1081, +0.1135]；OBS1=-0.0217，95% CI [-0.1596, +0.1075]；OBS2PLUS=-0.0205，95% CI [-0.2053, +0.1439]。

## 结论

冻结 gate 未在独立队列上全部复现：Core A 在 RGB-D 轨道上尚未通过资格验证，Core B 保持关闭。未通过的 gate 见上表；由于只有 5 个场景，需区分“效应不存在”与“功效不足”。

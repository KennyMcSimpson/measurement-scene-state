# 三维训练前准备

已完成仓库复用审计、官方校准资源下载、共同 anchor 训练损失、精确相机内参转换、配置/小 cohort 草案和 CUDA 资源检查。**TRAINING_READY=false**：尚未开始正式训练，未读取真实 query 图像/depth，未下载场景媒体。输出保存在 `outputs/EXP-3D-20260927-training-preparation-v1/`；旧实验文件未修改。

## 可以直接复用的内容

| 仓库组件 | 用途与限制 |
|---|---|
| `src/mcss/dynamic/carrier.py` | 当前三维 carrier、真实 fuse/complete 写入矩阵，保持架构不变 |
| `src/mcss/dynamic/write_rule.py` | 现有直接写入规则，无需重写 |
| `src/mcss/measurements.py` | 参数为零的固定 RGB/depth renderer |
| `src/mcss/data/download.py` | 官方 Hypersim ZIP 的 HTTP Range 局部下载；必须提供 scene/camera/frame 白名单；下载器自身不查 split |
| `configs/hypersim_er_partitions.csv` | 现有 train/val/diagnostic/final 分区名单，不能当作物理地点身份证明 |
| `scripts/compile_dynamic_training_episodes.py` | 数据检查、online/query 分离、窗口编译；函数 API 支持小规模 `expected_counts`，CLI 默认仍为旧 365/46 |
| `src/mcss/data/pilot_provenance.py` | 训练/开发身份、路径、manifest 哈希核验；不读媒体，必须传新 prepared_root |
| `src/mcss/training/experiment.py` | 两阶段优化、日志、断点、RNG/optimizer 恢复；旧 Phase A 使用不同 anchor，不能直接作为新协议启动入口 |
| `src/mcss/dynamic/checkpoint.py` | 严格保存/加载 `mcss.dynamic.v1`，同时保存 carrier 与 writer slow weights |
| `src/mcss/mechanism_pilot/` | 已有封存、四动作隔离、scene bootstrap、指标及残差对照接口 |

## 本轮补齐的工具

- `calibration.py`：从官方逐场景 `M_cam_from_uv` 构造相机矩阵，遵守 OpenGL→OpenCV、像素中心与横纵分别缩放。验证了 shifted/skew 情况与官方 ray 公式一致；无效参数直接报错，不回退到 60°。已对官方 482 行参数完成矩阵有效性检查，**尚未与真实图像/几何做配准核验**。
- `training.py`：`common_anchor_measurement_loss`，两组等量上下文只共享内容完全相同的 anchor，各自独立建立零 fast-offset 状态。两个状态都建完后 query 才进入 renderer。仅允许 train split；split 字符串不是数据来源证明。
- `scripts/prepare_3d_training.py`：不读媒体、不启动训练，生成校准检查、cohort 草案、A/B 配置和阻塞项。拒绝覆盖已有报告。
- `scripts/probe_3d_training_resources.py`：一次合成前向/反向，验证梯度和显存；不调用 optimizer、不保存 checkpoint，验证参数值未改变。

新校准与 anchor 模块是可复用适配器，**尚未接入旧预处理器和训练循环**。配置文件可被现有配置解析器读取，但不意味着旧训练入口已经满足新协议。不能直接照旧命令启动正式训练。

## 已下载的官方材料

来源固定为 apple-aiml-research/ml-hypersim commit `3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254`：逐场景相机 CSV（312,522 bytes）、说明、raycasting 示例 notebook、LICENSE。下载 URL、大小及 SHA256 见 `metadata/sources.json`。只读取 notebook JSON 中的算法说明，没有执行上游 notebook，也没有下载其中示例媒体。

官方说明每个 scene 的内参可能不同，不能假定统一 FOV；depth 是到光心的欧氏距离。官方还提示 preview JPEG 不适合一般下游训练。历史仓库使用 preview JPEG；重新训练前须明确保留该输入以复现，还是使用 HDR RGB 与固定 tone-mapping，不能静默切换。

参考：[官方相机说明](https://github.com/apple-aiml-research/ml-hypersim/tree/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/contrib/mikeroberts3000)、[数据说明](https://github.com/apple-aiml-research/ml-hypersim)。

## 小 cohort 草案，不是正式 split 锁

按已有 partition、frame_count>=16、不同 asset volume、字典序选出，不用图像或结果：

| 角色 | 候选 scene |
|---|---|
| train | ai_001_001、ai_002_001、ai_003_001 |
| dev | ai_004_003、ai_005_005、ai_006_007 |

不同 volume 只是保守筛选，不证明物理地点独立；`physical_scene_id=null`，状态明确 UNVERIFIED。每 scene 计划使用前 16 个实际公开有效帧，**具体帧号尚未锁定，不能假定就是 0–15**。这只是 3+3 工程/开发 pilot，不是统计功效足够的泛化确认，也不是此前未曝光场景的声明。没有选 diagnostic_test 或 final_holdout。

## 必须避免的旧默认

1. `hypersim.py` 旧内参用 60° FOV 在目标尺寸重新计算 fx=fy；128×160 非等比例 resize 会带来误差。应接入本轮官方矩阵，确保 RGB/depth 的重采样像素中心一致。
2. scene scale 缺失时旧函数回退 1.0。新流程必须要求官方 scale，核对 `_detail/metadata_scene.csv`，缺失即停止。
3. 旧转换器用已转换全部 depth 推 bounds。新训练只能用固定 context-local bounds，不得使用 query-derived bounds。
4. `materialize_hypersim_expansion.py` 默认包含 diagnostic_test、所有帧及 100GiB 保留空间，不能直接调用默认流程。应以锁定 train/dev 白名单使用局部 ZIP 下载，先检查压缩/解压大小。
5. `prepare_dynamic_ttt_pilot.py` 会打开 online 帧 depth/normal，且额外要求 normal。当前训练只需 RGB/camera 和离线 query depth；优先走 training compiler 的 metadata-only API，明确记录何时读取 train depth。

## 配置与验收顺序

草案：128×160、原 carrier/write 配置、固定 renderer 64 samples（与新机制 pilot 一致，**显式区别于历史训练的16**）；Stage A 1000 步、lr=1e-3，Stage B 600 步、lr=3e-4。它们是待核定工程预算，不是收敛保证。A 配置中 B steps=0，先检查静态资格；B 单独配置，需要显式传 A-final。尚未冻结真实数据上的窗口与采样规则。

正式启动前剩余：

1. 核实物理地点映射、scene/camera/具体帧号并冻结清单；只选合法 train/dev。
2. 按精确成员清单下载小规模 RGB、相机和 depth，保存来源/hash/体积；不全量下载数据集。
3. 将官方校准与严格米制 scale 接入独立预处理路径，核对图像射线/深度坐标，锁定 RGB 来源及重采样。
4. 将共同 anchor 损失显式接入训练器配置，测试日志、续训兼容性和状态隔离，避免通过临时 monkeypatch 启用。
5. 编译 train/dev online/query 清单；train 标签健康检查有访问日志；dev query 在评价 seal 前不读取。
6. 用真实训练输入测完整动态展开显存/耗时，再启动预先冻结的 A 阶段。充分训练 R-Residual 对照后评价静态资格，再决定 B 与后续机制实验。

## 验证与资源

14 项新增测试通过（共同 anchor 11、校准3）。本轮新模块及复用的编译、provenance、下载组件回归结果见 `regression_tests.log`；Ruff PASS。没有修改旧实现，未重复无关的全仓库测试。

RTX 5090 上，128×160、两组各3帧、单query、64 samples、chunk=2048 的合成 forward/backward 为 **0.462秒**；PyTorch peak allocated **257,635,328 bytes ≈246MiB**，reserved ≈276MiB。仅测静态单步，未包含真实 I/O、optimizer状态或 Phase B 八步展开，不据此估算总训练时间。optimizer steps=0，未生成伪训练 checkpoint。主盘当时约94GiB可用，内存/显存不是目前唯一门槛。

复现见 `commands.sh`；完整输出位于独立 outputs 目录。

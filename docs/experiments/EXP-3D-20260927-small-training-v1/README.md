# 三维小规模训练：已完成，工程载体仍未取得科学资格

本轮已用实际下载的 Hypersim 数据完成 **3 个 train 场景、100 步静态 A + 30 步写入 B**，生成可严格加载的 `mcss.dynamic.v1` checkpoint。这是重新训练得到的新权重，不是恢复了缺失的历史模型。Hypersim 是渲染数据，本轮未使用真实世界视频或此前的解析平面 fixture 作为训练数据。

最终权重：`outputs/EXP-3D-20260927-small-training-v1/training/phase_b_final.pt`。

SHA256：`4969b46a695bf12e7a9517b307401390bf89e9595876d8fe4fd03e2f2eebddc3`。

`CARRIER_STATUS=PARTIAL`：训练/保存/重载工程链路可用，但没有未见 dev/test、没有充分训练的 R-Residual 对照、没有正式静态资格或历史动作价值确认。旧实验的 BLOCKED 记录保留为当时状态，不覆盖或改写。

## 数据和协议

预先锁定 `ai_001_001、ai_002_001、ai_003_001`，均来自现有 train 分区，camera=`cam_00`。ZIP 目录锁定后选出每场景首 16 个共同有效 RGB/depth 帧，本次恰为 0–15。没有按训练分数选 scene 或帧；没有接触 dev、diagnostic_test、final_holdout。

物理地点映射仍未验证，因此仅在同一个 train split 内使用，不能声称 train/dev 物理独立。上下文 A=[0,1,2]、B=[0,3,4]，只共享 anchor0；写入阶段 warmup=A、stream=[5,6]，监督 query=[12,13,14,15]。7–11 明确未参与本轮优化。预处理读取的所有数据都是 train，访问日志完整记录；不能把这些 train query 称为未见独立测试。

输入是历史工程设置的 preview JPEG，128×160。使用官方逐场景 `M_cam_from_uv` 转换相机内参；OpenGL→OpenCV 坐标变换；米制 scale 与 archive 元数据严格交叉核对。RGB 与 depth 都按 half-pixel bilinear 缩放，depth 使用 finite-positive valid-weighted interpolation，无效值为0。固定 local bounds 来自原模型配置，未使用 query depth 推 bounds。

每个训练场景第一帧增加 position.hdf5，核对原生分辨率的 ray-depth、位姿与世界点：p95 误差为 1.62/6.53/6.28 mm。1 cm 是实现中的工程排错阈值，**原 calibration plan 只写“明显不匹配则阻塞”，没有直接预注册此数值**；不把这项检查当作研究结果或资格门槛。

新下载器要求精确206/Content-Range/ETag、单请求64MiB上限、共享2GiB响应数据预算，禁止整包回退。完成的下载尝试传输20,106,687 bytes，raw17,997,642 bytes；首次尝试为补充几何检查与全局预算主动中断，部分数据、原锁和日志保留在 `data_attempt1_interrupted/`，其额外网络流量不能混入“成功尝试20MB”的表述。官方资源来源/许可证见上一轮 preparation 的 `metadata/`。

## 实际训练

保持原 CarrierConfig/WriteConfig：小型卷积 encoder，8³网格、128候选、feature/hidden8、expansion16，无预训练 backbone。固定 renderer64 samples、chunk2048。seed20260927。

- A：100步，Adam lr=1e-3，clip1。两个共同 anchor 的独立 OFF 状态接受 train query RGB/depth 监督；writer不更新。
- B：30步，Adam lr=3e-4，clip1。carrier+writer联合优化两步 FUSE/FUSE、COMPLETE/COMPLETE、ALL/ALL。动作按每个 scene 的访问轮次轮换，不与 scene ID 固定绑定。ALL 两提案来自同一旧 trace，同步提交。
- 每步 scene=`step%3`，query=`(step//3)%4`；A 的场景步数34/33/33，B各10步。B每场景FUSE4、COMPLETE3、ALL3。保持预定预算，没有为追求正结果追加训练。
- B是短程可微写入工程检查，**不是已通过静态资格后的正式机制实验**。未训练 controller，未做 policy或oracle搜索。

训练时间约11.66秒（包含该训练入口内部保存/训练集评价，不含数据准备及测试）；峰值CUDA allocated267,310,080 bytes≈255MiB。模型约5,125参数、writer528参数，因此38KB量级推理权重是预期大小。不能将这个时间外推到更大场景状态或长期展开。

## 训练集结果：固定 context_A、OFF 重建

以下均为先逐query、再scene等权；每个 checkpoint 对相同3×4个训练 query 评价。**不是写入动作相对 OFF 的收益**，也不是独立泛化指标。

| 指标 | 随机初始 | A-final | B-final |
|---|---:|---:|---:|
| RGB+depth loss | 1.085769 | 1.067332 | 1.065914 |
| Depth AbsRel ↓ | 0.934244 | 0.917263 | 0.919360 |
| Depth RMSE ↓ | 1.425108 | 1.350935 | 1.342744 |
| Depth δ1 ↑ | 0.135046 | 0.169092 | 0.179195 |
| RGB MSE ↓ | 0.057127 | 0.059905 | 0.055796 |
| RGB SSIM ↑ | 0.548713 | 0.553986 | 0.554400 |
| Opacity | 0.451532 | 0.417369 | 0.440092 |
| Coverage | 0.722449 | 0.722449 | 0.722449 |

AbsRel 从初始到B略有下降，但 B比A更差，不能包装成写入效果改善。按scene看，ai_001_001为0.4532→0.4233、ai_002_001为0.4225→0.3648；ai_003_001反而1.9270→1.9700，平均覆盖率仅0.1673。没有删除该失败场景。

ai_003_001 的query14为全黑 RGB，预测也全黑，coverage=0、depth AbsRel=1。该 query 的 RGB MSE=0导致PSNR为+∞，原指标用null+perfect标记表达。因此总体PSNR不报告有限均值，保留全部行并用RGB MSE、SSIM及depth同时说明；不能丢掉该行或把全黑命中说成好预测。LPIPS未实现。

## 权重、梯度与恢复审计

A阶段carrier参数变化L2=2.301898，writer变化=0，符合关闭写入训练。B阶段carrier变化L2=0.263161，writer变化L2=0.049944，八组writer张量均改变。

30个B步骤中20个writer梯度非零，来自ai_001_001和ai_002_001各10步；ai_003_001全部10步为零。独立排查已经定位：该场景帧0/1/2/5/6分别看见69/0/0/0/0个固定候选；写入规则要求至少两帧支持，因此支持数=0、两层写入增量=0。其他两场景支持数59/61。该场景后续相机相对anchor向后，当前前向固定体积未覆盖这些视角；query14射线也完全不与状态边界相交。证据见 `independent_failure_diagnostics.json`，没有调整bounds或选帧来修饰本轮结果。

三份推理checkpoint均经过严格加载；carrier/writer state_dict 完全一致，OFF状态完全一致，非零ALL两步后的 fast weights 与场景状态也完全一致。独立审计再次读取三份checkpoint、复算参数变化、核对130行训练记录、36行评价记录、source/data hashes；结果见 `independent_audit.json`。

同时保存阶段末optimizer/RNG状态用于溯源；**本入口未实现 --resume，不能声称具备精确中断续训**。从头复现命令见 `commands.sh`，输出目录拒绝覆盖。锁记录算法/数据/源码，不保证跨设备训练数值逐bit重现；checkpoint重放的逐bit一致性已验证。

## 测试与失败记录

本轮新增数据/训练测试10项，加上校准、anchor和checkpoint测试，针对性回归26 passed；全仓库 **561 passed, 1 skipped**。跳过项仍是缺少旧V5权重的可选兼容性检查，与本轮新的动态权重不是同一架构。Ruff PASS。

独立报告导出第一次因PSNR null求均值报错，保留 `independent_audit_attempt1.log`；修复为传播null并标注perfect，无样本删除、无重训练、无指标替换。下载首次中断记录也保留。958条训练访问事件均为train；训练query可用于离线监督，但没有进入观测缓存或写入kernel。

## 接下来最小必要工作

现已拥有可用的小规模训练权重，可以作为工程初始化。下一步先解释第三个场景的零writer梯度、低coverage和黑帧，以及固定候选/体积与相机观测范围的关系；不能直接加controller或将当前权重认定为正式研究底座。任何改变数据选帧、bounds、候选规则或训练量，都应另建实验并记录理由，再做独立开发场景静态资格和充分训练的R-Residual比较。

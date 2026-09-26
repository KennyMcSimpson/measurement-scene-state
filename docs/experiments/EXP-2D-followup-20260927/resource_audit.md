# 独立数据候选的有界审计（2026-09-27）

本次只检查三个来源的官方说明、下载入口与代码；没有下载 RGB、source mask 或 target GT，没有打开 DAVIS official val/reserve，也没有修改冻结 V2。当前新确认序列数仍为 **0**。

| 候选 | 已核实 | 尚缺条件 | 判断 |
|---|---|---|---|
| YouTube-VOS 2019 train | 官方任务提供首帧对象 mask；3471 视频；官方 Drive 入口可访问 | clip 到原始 YouTube 视频及时间段映射；原生 PNG 的 void/ID 契约 | 最贴合现任务，尚未合格 |
| SegTrack-v2 | 14 个具名片段、24 对象；209 MB 官方 ZIP 支持 Range；评价函数使用 binary foreground | 原视频来源、明确的数据使用条款、原始 PNG 到 binary 的映射 | 不作为独立确认集 |
| SA-V | 官方视频 ID、CC BY 4.0、逐对象 RLE/binary mask；明确允许对象重叠 | 原始拍摄/片段分组、可用下载清单、新的重叠对象适配协议 | 仅适合另立未来协议 |

YouTube-VOS 最值得补齐的是**官方 clip→原视频 ID/时间段表和原生标注格式说明**。[官方任务](https://youtube-vos.org/dataset/vos/)与[使用条款](https://youtube-vos.org/dataset/term/)明确支持带 source 标注的研究设置；标注为 CC BY 4.0，但数据用途限定非商业研究。[作者转换器](https://github.com/youtubevos/vis2vos/blob/master/convert.py)初始化 0 背景并写入正整数对象 ID，但这是 VIS→VOS 转换代码，不能据此认定原生 VOS2019 的 255 是 void。十字符 clip 名也不能直接当原始 YouTube 视频 ID。

[SegTrack-v2 官方页面](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html)给出具名片段，但没有在本次查阅材料中建立原始拍摄来源映射。[原始 SegTrack 页面](https://sites.cc.gatech.edu/gvu/perception/projects/SegTrack/)有非商业传播的版权说明，不能把其解释为 SegTrack-v2 的明确独立数据许可。官方 ZIP 的 `compute_overlaps.m` 使用逻辑前景；原始 PNG 是否用白色255表示前景仍须核实，不能套用 DAVIS 的 void 约定。

[SA-V 官方格式说明](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md)比前两者更明确，但允许整个人与手部等对象重叠。把独立 binary masks 压成单张互斥 ID 图会改变任务。若未来选择它，应另行预注册保持重叠的逐对象适配器、manual-only 标注选择及帧处理；不能把本次调研解释为冻结 V2 已获得兼容数据。官方 `video_id` 与跨文件原始拍摄分组也要区分。

下载限于目录、README 与代码。两次完成的 SegTrack 范围读取共 1,536,300 bytes；一次过宽 MATLAB 依赖代码检查已停止，单次硬上限10 MB。加小型官方 GitHub 元数据，受控下载保守上界13 MB，低于20 MB预算；浏览工具不暴露网页传输字节数。任何图像/标注主体均未读取。完整证据边界、官方 URL 与源码哈希见 [resource_audit.json](resource_audit.json)。

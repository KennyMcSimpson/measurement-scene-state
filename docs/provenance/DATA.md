# 数据和权重 provenance

## DAVIS-2017 trainval 480p

- 官方页面：[DAVIS 2017 code](https://davischallenge.org/davis2017/code.html)
- 下载地址：`https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip`
- archive size：832,766,765 bytes；本地 SHA-256：`e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`
- 本地记录说明：HTTP Content-Length 与文件大小一致；SHA-256 是本地完整性记录，不是发布方官方 checksum。
- 解压统计：6208 JPEG + 6208 PNG；官方 train 60、val 30，二者无交集。
- probe split：train 序列名按 UTF-8 SHA-256 排序后 fit 24、discovery 12、validation 16、reserve 8；官方 val 30 全部 sealed。
- 标签：mask 只在下游评价读取，不参与写入或无标签 action selector。

## DINOv2-S/14

- 官方代码：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- 权重地址：`https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`
- 文件大小：88,283,115 bytes；本地 SHA-256：`b938bf1bc15cd2ec0feac3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- 运行方式：显式本地 checkpoint，`pretrained=False`，eval/frozen；224×224 输入，16×16×384 patch feature。

## 已清理的三维数据

旧 Hypersim 和 DL3DV 数据按用户授权清理。删除路径、文件数、总字节数和独立核验保存在本地输出目录；公开仓库不包含这些数据。删除证据为 433,375 files / 293,934,856,380 bytes，六个精确目标均确认 absent。

## 许可证和可再分发边界

数据、预训练权重和第三方源代码分别遵循各自来源条款。仓库不把下载内容复制进 Git；脚本和 manifest 只记录来源、版本、大小、hash 和 split。

# 模型仓库（MODEL_ZOO）

## 权重清单

| 文件 | 说明 | 大小量级 | 是否入库 |
|---|---|---|---|
| `checkpoints/resnet50-0676ba61.pth` | ResNet-50 ImageNet 预训练（torchvision） | ~98 MB | 否（公开可下载） |
| `checkpoints/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth` | Sparse R-CNN COCO 预训练（官方） | ~424 MB | 否（公开可下载） |
| `work_dirs/sparse_rcnn_r50_fpn_300_proposals/epoch_7.pth` | baseline（VOC-FOG trainval 训练） | ~424 MB | 否（未授权上传） |
| `work_dirs/phase1/unet_minimal_b128.pt` | 教师 U-Net（base=128） | ~18 MB | 否 |
| `work_dirs/phase2/unet_student_c128_b32.pt` | 学生 U-Net（dim=128/base=32） | ~3.2 MB | 否 |
| `work_dirs/phase4/unet_finetuned_c128_b32.pt` | 联合微调后的学生 U-Net | ~3.2 MB | 否 |
| `release/models/unet_finetuned_c128_b32.pt` | 同 phase4（归档） | ~3.2 MB | 否 |
| `release/models/sparse_rcnn_voc_rainfog_final_int8.pth` | 最终合并模型（INT8 U-Net） | ~424 MB | 否 |

## 获取方式

本仓库**不直接分发**体积大或未经授权的权重。复现者可按
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 的流水线自行训练获得，或联系作者获取
（仅在获得授权后）。

> 说明：U-Net（教师/学生）体积小（<20 MB），如作者后续获得授权可在 Release 中分发；
> 检测器 checkpoint（含 ResNet-50 权重）体积大，且依赖预训练权重许可，默认不随库发布。

## 自测 checkpoint 加载

最终合并模型（`sparse_rcnn_voc_rainfog_final_int8.pth`）为 Phase 6.1 产物：
整体 checkpoint 内 `state_dict` 同时含检测器与 `unet.*` 权重，评估时
`DiffDetDetector` 不传 `unet_path`，由 `unet.*` keys 填充 U-Net。

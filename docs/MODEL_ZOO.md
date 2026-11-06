# 模型仓库（MODEL_ZOO）

维护者联系邮箱：**yihang200407@163.com**

## 1. 已发布权重（Release `v0.1.0`，作者授权分发）

下载页：<https://github.com/yimu-15/Diffdet/releases/tag/v0.1.0>

| 文件 | 说明 | 体积 | SHA256 |
|---|---|---|---|
| `sparse_rcnn_voc_rainfog_final_int8.pth` | **最终部署模型**：检测器 + 内嵌 INT8（W8A32）U-Net，整体加载 | 409.52 MB | `0dc7eb83810e6fae887f0a0c209b5561a53bd274eebef5160425e4edfcb0ed3c` |
| `unet_teacher_b128.pt` | 教师 U-Net（base=128，4.611 M 参数） | 17.62 MB | `83b63a7b740400aaad01614063758a826079648c9b5f463ec7e2e3041bba8d30` |
| `unet_student_c128_b32.pt` | 蒸馏学生 U-Net（dim=128/base=32，0.791 M 参数） | 3.05 MB | `29607e9f238f7275724102862a35dca9fdbf481120eaf77dfbe07b39863327e2` |
| `unet_finetuned_c128_b32.pt` | 联合微调后的学生 U-Net（Phase 4 产物） | 3.04 MB | `b2270cfbc46d1acf1bd90c8da5737e917f5e21a379595e13a53fa437eec868a0` |

直链：

```
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/sparse_rcnn_voc_rainfog_final_int8.pth
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/unet_teacher_b128.pt
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/unet_student_c128_b32.pt
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/unet_finetuned_c128_b32.pt
```

校验：

```bash
sha256sum sparse_rcnn_voc_rainfog_final_int8.pth unet_*.pt
```

> `sparse_rcnn_voc_rainfog_final_int8.pth` 的 `state_dict` 同时含检测器与
> `unet.*` 权重，加载时 `DiffDetDetector` **无需** `unet_path`。
> 单独下载的 U-Net 权重用于复现蒸馏、SVD 与轻量化消融实验。

## 2. 公开可下载权重（不在本仓库分发）

| 文件 | 来源 | 体积 |
|---|---|---|
| `resnet50-0676ba61.pth` | <https://download.pytorch.org/models/resnet50-0676ba61.pth> | ~98 MB |
| `sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth` | MMDetection model zoo（见 [DATASETS.md §6](DATASETS.md)） | ~424 MB |

## 3. 未发布的训练检查点

以下检查点体积大（含优化器状态），且可由公开数据与代码完整重训，
**不随 Release 分发**：

| 文件 | 说明 | 体积 | 获取方式 |
|---|---|---|---|
| `work_dirs/sparse_rcnn_r50_fpn_300_proposals/epoch_7.pth` | baseline（VOC2007+VOC2012 trainval，12 epoch 取第 7） | 1.28 GB | 按 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) 训练，或邮件联系作者 |
| `work_dirs/phase4/finetune_diffdet/epoch_3.pth` | 联合微调最终 epoch | 1.11 GB | 同上 |
| `work_dirs/sparse_rcnn_r50_fpn_300_proposals/epoch_{1..6}.pth` | 中间 epoch | 各 1.2~1.3 GB | 同上 |

> 如需从检查点续训或复算 baseline 日志，请邮件联系 **yihang200407@163.com**。

## 4. 加载方式

最终合并模型（Phase 6.1 产物）：

```python
from mmengine.runner import load_checkpoint
from diffdet import DiffDetDetector  # 注册到 MODELS

model = DiffDetDetector(...)           # 不传 unet_path
load_checkpoint(model, 'sparse_rcnn_voc_rainfog_final_int8.pth', map_location='cpu')
```

完整推理示例见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §6 与 `tools/evaluate.py`。

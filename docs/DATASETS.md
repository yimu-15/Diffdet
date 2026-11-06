# 数据集来源与准备（DATASETS）

维护者联系邮箱：**yihang200407@163.com**

本文件完整列出 DiffDet 全部实验用到的数据集：来源地址、体积、目录结构、
本项目是否分发、以及如何自行重建。仓库**只随 Release 分发自研的 VOC-FOG**，
公开数据集一律给出官方地址，不在库内重复托管。

## 0. 总览

| 数据集 | 用途 | 图像数 | 体积 | 本项目是否分发 | 获取方式 |
|---|---|---|---|---|---|
| PASCAL VOC2007 | 全部实验的底图与标注、评估集 | 9,963（trainval 5,011 + test 4,952） | ~1.9 GB | 否 | 官方 / 镜像，见 §1 |
| PASCAL VOC2012 | baseline 训练集补充（仅用 trainval 图像） | 17,125 | ~1.9 GB | 否 | 官方，见 §2 |
| **VOC-FOG** | 雾天训练与测试（自研合成） | 10,031（light 2,510 / moderate 5,011 / dense 2,510） | 674 MB | **是（Release assets）** | 见 §3 |
| COCO2017 | 扩展实验 / 预训练权重来源 | train 118,287 + val 5,000 | ~19.6 GB（本地） | 否 | 官方，见 §4 |
| RTTS（RESIDE） | 真实雾天泛化测试（可选） | 4,322 | ~1.0 GB | 否 | 官方，见 §5 |

> 论文主表只依赖 **VOC2007 + VOC-FOG**。COCO 与 RTTS 仅用于扩展实验和
> 真实场景验证，不影响主结果复现。

## 1. PASCAL VOC2007

- 官方页面：<https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/>
- 需要下载的文件：

| 文件 | 体积 | 内容 |
|---|---|---|
| `VOCdevkit_08-Jun-2007.tar` | ~0.4 MB | 开发包与评估代码 |
| `VOCtrainval_06-Nov-2007.tar` | ~451 MB | trainval 5,011 张（含 val 2,510 张） |
| `VOCtest_06-Nov-2007.tar` | ~438 MB | test 4,952 张（本项目未使用） |

- 镜像（国内可直连，已验证可访问）：<https://pjreddie.com/projects/pascal-voc-dataset-mirror/>

解压后放置为：

```
data/VOCdevkit/VOC2007/
  JPEGImages/            # 9,963 张 jpg
  Annotations/           # 同名 xml
  ImageSets/Main/
    train.txt  val.txt  trainval.txt  test.txt
```

> 注意：本项目评估使用 **VOC2007 val 划分（2,510 张）**。若自建环境缺少
> `ImageSets/Main/val.txt`，可用官方划分文件补齐；本项目已验证的划分文件
> 为 VOC2007 官方版本（train 2,501 / val 2,510 / trainval 5,011 / test 4,952）。

## 2. PASCAL VOC2012

- 官方页面：<https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/>
- 下载文件：`VOCtrainval_11-May-2012.tar`（~1.9 GB，17,125 张）
- 仅 **baseline 训练** 会合并 VOC2007 trainval + VOC2012 trainval；
  VOC-FOG 与 DiffDet 微调只使用 VOC2007。

放置为 `data/VOCdevkit/VOC2012/`（结构与 VOC2007 相同）。

## 3. VOC-FOG（自研合成数据集，本项目分发）

### 3.1 定义

由 `tools/prepare_voc_fog.py` 对 VOC2007 图像施加大气散射合成雾：

| 等级 | β | 图像数 | 划分 | 图像尺寸 |
|---|---|---|---|---|
| light | 1.0 | 2,510 | VOC2007 val | 500×375 |
| moderate | 1.8 | 5,011 | VOC2007 trainval | 500×375 |
| dense | 3.0 | 2,510 | VOC2007 val | 500×375 |

标注与清晰图完全一致（合成雾不改变几何位置），因此直接复制 VOC2007 的
XML 标注与 `ImageSets/Main` 划分。

### 3.2 下载（Release assets）

| 文件 | 体积 | SHA256 |
|---|---|---|
| `voc_fog_light.tar.gz` | 186.73 MB | `bde307c93cbf18e0cea40897a6cc0b085953c7dff54423f262d0364e1a4eb706` |
| `voc_fog_moderate.tar.gz` | 333.56 MB | `994f23a4852997d49231eac527dd9ac870bcb7dc794cedca1496992578712f25` |
| `voc_fog_dense.tar.gz` | 142.78 MB | `684a111d935ce75b34755e329c7d65b066b9b23255602e8ba5a76c1a5c7cd972` |

下载地址（GitHub Release `v0.1.0`）：

```
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/voc_fog_light.tar.gz
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/voc_fog_moderate.tar.gz
https://github.com/yimu-15/Diffdet/releases/download/v0.1.0/voc_fog_dense.tar.gz
```

解压（每个包内顶层目录为 `VOC2007/`）：

```bash
mkdir -p data/VOC-FOG
for lvl in light moderate dense; do
  mkdir -p data/VOC-FOG/$lvl
  tar -xzf voc_fog_$lvl.tar.gz -C data/VOC-FOG/$lvl
done
```

校验：

```bash
sha256sum voc_fog_*.tar.gz
```

解压后目录：

```
data/VOC-FOG/{light,moderate,dense}/VOC2007/
  JPEGImages/{id}.jpg
  Annotations/{id}.xml
  ImageSets/Main/{val,trainval}.txt
```

该布局与 `configs/baseline/*_fog_{light,moderate,dense}.py`、
`configs/diffdet/*` 中的 `data_root` 完全一致，解压后无需改动配置。

### 3.3 自行重新合成

```bash
python tools/prepare_voc_fog.py \
    --voc-root data/VOCdevkit/VOC2007 \
    --out-root data/VOC-FOG \
    --levels light moderate dense \
    --seed 42
```

合成实现见 `diffdet/datasets/fog.py`（`LEVELS`、`synthesize_fog`）。
默认对 VOC2007 全部 5,011 张生成；本项目发布的 light/dense 包只保留
`val` 划分的 2,510 张（与论文评估口径一致），moderate 包保留全部 5,011 张。

> 重新合成使用固定种子 `--seed 42`，可得到同分布的雾图；但 JPEG 编码器版本
> 差异可能导致与发布包存在像素级细微差别。**做精确数值复现时请优先使用
> §3.2 的分发包**，自行合成仅用于验证流程。

## 4. COCO2017（可选）

- 官方页面：<https://cocodataset.org/#download>
- 直链：

```
http://images.cocodataset.org/zips/train2017.zip     # 118,287 张，~18 GB
http://images.cocodataset.org/zips/val2017.zip       #   5,000 张，~1 GB
http://images.cocodataset.org/annotations/annotations_trainval2017.zip  # ~241 MB
```

放置为 `data/coco/{train2017,val2017,annotations}/`。
本项目仅用于扩展实验与 COCO 预训练权重来源，主结果不依赖。

## 5. RTTS / RESIDE（可选，真实雾天）

- 官方页面：<https://sites.google.com/view/reside-dehaze-datasets/reside-v0>
- RTTS（Real-world Task-driven Testing Set）为真实雾天图像，4,322 张，用于
  真实场景泛化测试（论文扩展实验，不在主表内）。
- 解压后本项目使用的布局：`data/RTTS/RTTS/RTTS/*.jpg`（保留官方嵌套结构）。

> 该页面偶发访问受限，可改用 RESIDE 论文作者提供的其他镜像；无论来源如何，
> 请以官方页面标注的版本号为准。

## 6. 预训练权重来源

| 权重 | 来源 | 体积 | 说明 |
|---|---|---|---|
| ResNet-50 ImageNet | <https://download.pytorch.org/models/resnet50-0676ba61.pth> | ~98 MB | torchvision 官方，`torchvision.models.resnet50(weights='IMAGENET1K_V1')` 同源 |
| Sparse R-CNN COCO | `https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth` | ~424 MB | MMDetection 官方 model zoo |
| 本项目自训权重 | 见 [MODEL_ZOO.md](MODEL_ZOO.md) 与 Release `v0.1.0` | 3 MB ~ 410 MB | 作者授权发布，含 SHA256 |

## 7. 完整目录结构

```
data/
├── VOCdevkit/
│   ├── VOC2007/{JPEGImages,Annotations,ImageSets/Main,SegmentationClass,SegmentationObject}
│   └── VOC2012/{JPEGImages,Annotations,ImageSets/Main,...}
├── VOC-FOG/
│   ├── light/VOC2007/{JPEGImages,Annotations,ImageSets/Main}
│   ├── moderate/VOC2007/{JPEGImages,Annotations,ImageSets/Main}
│   └── dense/VOC2007/{JPEGImages,Annotations,ImageSets/Main}
├── coco/{train2017,val2017,annotations}/        # 可选
├── RTTS/RTTS/RTTS/                              # 可选
└── features/                                    # 由脚本重新提取，不分发
    ├── f_clean.pt
    └── f_haze.pt
```

`data/features/` 为 C5 特征对缓存（约 3.9 GB），可由
`tools/analyze_residuals.py --save-features` 从上述图像重新提取，
体积大且完全可重算，故不随 Release 分发。

## 8. 获取失败或数据缺失

如需旧版本划分文件、原始合成脚本参数或论文写作期的中间产物，请邮件联系
**yihang200407@163.com**，说明用途与所需等级。

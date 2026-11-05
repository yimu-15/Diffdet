# 复现指南（REPRODUCIBILITY）

## 1. 环境

- Python 3.10，PyTorch 2.x（CUDA 可选）。
- mmdetection 3.x / mmcv 2.x / mmengine 0.10.x。

```bash
conda env create -f environment.yml     # 或
pip install -r requirements.txt
```

本仓库用到的版本（已验证）：torch 2.7.1、mmdet 3.3.0、mmcv 2.1.0、mmengine 0.10.7。

## 2. 数据准备

VOC2007/VOC2012 需自行按 PASCAL VOC 官方方式获取（本仓库**不提供**数据）。

```bash
# 生成 VOC-FOG 合成雾（β=1.0/1.8/3.0）
python tools/prepare_voc_fog.py \
    --voc-root data/VOCdevkit/VOC2007 \
    --out-root data/VOC-FOG \
    --levels light moderate dense
```

预期目录：

```
data/
  VOCdevkit/VOC2007/JPEGImages & Annotations & ImageSets/Main/val.txt
  VOC-FOG/{light,moderate,dense}/JPEGImages/{level}/...
  VOC-FOG/ImageSets/Main/{level}_val.txt
```

## 3. 复现流水线

```bash
# 0. baseline 训练（Sparse R-CNN on VOC）
python tools/train_detector.py configs/baseline/sparse_rcnn_r50_fpn_300_proposals.py \
    --work-dir work_dirs/sparse_rcnn_r50_fpn_300_proposals

# 1. 特征残差分析 + 特征提取
python tools/analyze_residuals.py \
    --clean-dir data/VOCdevkit/VOC2007/JPEGImages \
    --fog-dir data/VOC-FOG/JPEGImages/moderate \
    --names data/VOCdevkit/VOC2007/ImageSets/Main/val.txt \
    --out-dir work_dirs/phase1/residuals \
    --save-features work_dirs/phase1/clean_haze_features.pt

# 1b. 教师 U-Net 训练（base=128）
python tools/train_teacher.py \
    --features work_dirs/phase1/clean_haze_features.pt \
    --out work_dirs/phase1/unet_minimal_b128.pt --base 128

# 2. 学生蒸馏（dim=128/base=32, λ=1.0, 50 epoch）
python tools/train_student.py \
    --features work_dirs/phase1/clean_haze_features.pt \
    --teacher work_dirs/phase1/unet_minimal_b128.pt \
    --out work_dirs/phase2/unet_student_c128_b32.pt

# 4. 联合微调（freeze_unet=False）
python tools/train_detector.py \
    configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_finetune.py \
    --work-dir work_dirs/finetune_diffdet

# 6. 评估（最终形态，INT8 U-Net）
python tools/evaluate.py \
    configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py \
    --checkpoint checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth
```

## 4. 随机种子

- 数据合成：`prepare_voc_fog.py --seed 42`。
- U-Net 训练：`train_teacher.py` / `train_student.py --seed 42`。
- 检测器训练：由 mmengine 配置 `sampler_seed` hook + 全局 `seed` 控制。

## 5. 已知限制

- 完整精度复现需要 VOC2007/VOC2012 数据与预训练权重，缺一不可。
- 权重文件体积大，本仓库默认不入库（`.gitignore`），需按 MODEL_ZOO 自行获取/复现。

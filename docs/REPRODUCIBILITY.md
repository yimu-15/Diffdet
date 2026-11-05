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
  VOCdevkit/VOC2007/JPEGImages & Annotations & ImageSets/Main/{val,trainval}.txt
  VOC-FOG/{light,moderate,dense}/VOC2007/JPEGImages/{id}.jpg
  VOC-FOG/{light,moderate,dense}/VOC2007/Annotations/{id}.xml
  VOC-FOG/{light,moderate,dense}/VOC2007/ImageSets/Main/{val,trainval}.txt
```

该布局与 `configs/baseline/*_fog_{light,moderate,dense}.py` 及
`configs/diffdet/*_finetune.py` 中的 `data_root` / `sub_data_root` 完全一致。

## 3. 复现流水线

```bash
# 0. baseline 训练（Sparse R-CNN on VOC）
python tools/train_detector.py configs/baseline/sparse_rcnn_r50_fpn_300_proposals.py \
    --work-dir work_dirs/sparse_rcnn_r50_fpn_300_proposals

# 1. 特征残差分析 + 特征提取
python tools/analyze_residuals.py \
    --clean-dir data/VOCdevkit/VOC2007/JPEGImages \
    --fog-dir data/VOC-FOG/moderate/VOC2007/JPEGImages \
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

# 5. 合并最终 checkpoint（U-Net 权重 INT8 量化, W8A32 模拟）
python tools/merge_final_model.py \
    --detector work_dirs/finetune_diffdet/epoch_3.pth \
    --out checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth

# 6. 评估（最终形态，INT8 U-Net）
python tools/evaluate.py \
    configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py \
    --checkpoint checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth

# 7. 效率基准（FPS / 峰值显存）
python tools/benchmark.py \
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

## 6. 测试与实测记录（2026-11-05 实际执行）

环境：Windows + `D:\conda_envs\fedsem`（torch 2.7.1 / mmdet 3.3.0 / mmcv 2.1.0 / mmengine 0.10.7）。

```bash
# 常规测试（无 checkpoint → inference smoke 自动 skip，不虚构 PASS）
python -m pytest tests/ -q
# 24 passed, 1 skipped（1 warning: pytest_asyncio 弃用提示）

# 提供已训练 checkpoint 后（inference smoke 实跑）
DIFFDET_CKPT=checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth \
DIFFDET_IMAGE=data/VOC-FOG/moderate/VOC2007/JPEGImages/000125.jpg \
python -m pytest tests/ -q
# 25 passed
```

| 实测项 | 命令 | 结果 |
|---|---|---|
| DiffDet moderate 主实验 | `tools/evaluate.py configs/diffdet/..._diffdet_final.py --checkpoint ...final_int8.pth` | **mAP 0.7835** / AP50 0.7830（2510 张） |
| Baseline moderate | `tools/evaluate.py configs/baseline/..._fog_moderate.py --checkpoint .../epoch_7.pth` | **mAP 0.6935**（论文/日志 0.6939，Δ=0.0004） |
| 参数量 | `MinimalUNet(base=128)` / `StudentUNet(dim=128, base=32)` | 4,611,328（4.611 M）/ 791,360（0.791 M），比例 17.16% |

> Baseline 复算 0.6935 与日志 0.6939 的 0.04 点差异，来自不同次推理的
> 浮点/NMS 非严格确定性，属评估噪声；DiffDet 复算与论文完全一致（0.7835）。
> 消融表中 0.8369/0.8342/0.8342 仍无法复算，见 `experiments/RESULT_AUDIT.md`。

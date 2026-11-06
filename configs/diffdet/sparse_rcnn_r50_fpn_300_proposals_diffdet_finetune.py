_base_ = '../baseline/sparse_rcnn_r50_fpn_300_proposals_fog_moderate.py'

# 加载 DiffDet 正式模块（MODELS 注册）
custom_imports = dict(imports=['diffdet'], allow_failed_imports=False)

# Phase 4 端到端联合微调: DiffDetDetector = Sparse R-CNN + C5 特征 U-Net 去噪
# - 起点: 检测器加载 epoch_7（Sparse R-CNN baseline 权重）+ 学生 U-Net c128_b32
# - freeze_unet=False: U-Net 与检测器联合微调（AMP 混合精度）
model = dict(
    type='DiffDetDetector',
    unet_path='work_dirs/phase2/unet_student_c128_b32.pt',
    freeze_unet=False,
    init_cfg=None)

# 检测器权重从 baseline 恢复（U-Net 权重由 unet_path 独立加载, missing keys 允许）
load_from = 'work_dirs/sparse_rcnn_r50_fpn_300_proposals/epoch_7.pth'

# 训练数据: VOC-FOG moderate trainval（5011 张雾天图, batch=4）
# _delete_=True: 整体替换 voc0712 的 RepeatDataset/ConcatDataset 层级
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOC-FOG/moderate/',
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline={{_base_.train_pipeline}},
        backend_args=None))

# val/test 继承 fog_moderate -> VOC-FOG moderate val (2510 张)

# AMP 混合精度训练（FP16）
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000025, weight_decay=0.0001),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=1, norm_type=2))

# 微调 3 epochs（每个 epoch 约 1253 iter, batch=4）
max_epochs = 3
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=200),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[2],
        gamma=0.1)
]

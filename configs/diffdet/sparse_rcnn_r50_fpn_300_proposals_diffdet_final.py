_base_ = '../baseline/sparse_rcnn_r50_fpn_300_proposals_fog_moderate.py'

# 加载 DiffDet 正式模块（MODELS 注册）
custom_imports = dict(imports=['diffdet'], allow_failed_imports=False)

# Phase 6.1 最终形态: 整体 checkpoint（Backbone+Head+INT8 U-Net 合并保存）
# 不传 unet_path —— U-Net 权重由主 checkpoint 的 unet.* keys 填充, 模拟真实部署
model = dict(
    type='DiffDetDetector',
    init_cfg=None)

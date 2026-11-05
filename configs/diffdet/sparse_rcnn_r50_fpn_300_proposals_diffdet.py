_base_ = '../baseline/sparse_rcnn_r50_fpn_300_proposals_fog_moderate.py'

# 加载 DiffDet 正式模块（MODELS 注册）
custom_imports = dict(imports=['diffdet'], allow_failed_imports=False)

# DiffDet Detector: Sparse R-CNN + C5 特征 U-Net 去噪（默认学生 c128_b32 0.791M，
# 冻结 U-Net）。unet_path 为独立 U-Net checkpoint（相对仓库根目录）。
model = dict(
    type='DiffDetDetector',
    unet_path='work_dirs/phase2/unet_student_c128_b32.pt',
    init_cfg=None)

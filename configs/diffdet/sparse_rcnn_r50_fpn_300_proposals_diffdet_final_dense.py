_base_ = './sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py'

# 在 VOC-FOG dense 浓度雾图上评估 DiffDet 最终模型
val_dataloader = dict(
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOC-FOG/dense/',
        ann_file='VOC2007/ImageSets/Main/val.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=None))
test_dataloader = val_dataloader

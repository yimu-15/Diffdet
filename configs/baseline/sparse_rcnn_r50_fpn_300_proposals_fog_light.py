_base_ = './sparse_rcnn_r50_fpn_300_proposals.py'

# 在 VOC-FOG light 浓度雾图上测试 baseline
val_dataloader = dict(
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOC-FOG/light/',
        ann_file='VOC2007/ImageSets/Main/val.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args=None))
test_dataloader = val_dataloader

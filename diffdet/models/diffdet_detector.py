"""DiffDet Detector: 在 Sparse R-CNN C5 特征上注入 U-Net 去噪的检测器。

- 继承 SparseRCNN, extract_feat 中对 C5 施加 denoise_features 修复, 其余结构不变
- unet 来源二选一:
  * unet_path / unet: 单独加载 U-Net checkpoint（学生/教师/SVD 版均可, 由 load 逻辑自动识别）
  * 都不传: 整体 checkpoint 模式（Phase 6.1 最终形态）——U-Net 权重由主 checkpoint
    state_dict 中的 unet.* keys 填充（DiffDetDetector 先以默认结构建占位 U-Net）
- 冻结: U-Net 默认 eval + requires_grad_(False)（Phase 4 联合微调时改为可训）
"""

import torch

from mmdet.models.detectors.sparse_rcnn import SparseRCNN
from mmdet.registry import MODELS

from ..denoising.diffusion_unet import (StudentUNet, build_diffusion_unet,
                                        denoise_features)


@MODELS.register_module()
class DiffDetDetector(SparseRCNN):
    """Sparse R-CNN + C5 特征扩散去噪（DiffDet）。"""

    def __init__(self,
                 unet_path=None,
                 unet=None,
                 freeze_unet=True,
                 unet_dim=128,
                 unet_base=32,
                 **kwargs):
        kwargs.setdefault('init_cfg', None)
        super().__init__(**kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if unet is not None:
            self.unet = unet.to(device)
            src = 'direct module'
        elif unet_path is not None:
            ckpt = torch.load(unet_path, map_location=device)
            u = build_diffusion_unet(ckpt, device=device)
            u.load_state_dict(ckpt['model'])
            self.unet = u.to(device)
            src = unet_path
        else:
            # 整体 checkpoint 模式: U-Net 权重由主 checkpoint 的 unet.* keys 填充
            self.unet = StudentUNet(in_ch=2048, dim=unet_dim,
                                    base=unet_base).to(device)
            src = 'main checkpoint (unet.* keys)'
        self.unet.eval()
        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad_(False)
        print(f'[DiffDetDetector] unet loaded from {src}')

    def extract_feat(self, batch_inputs):
        x = list(self.backbone(batch_inputs))  # [C2, C3, C4, C5]
        x[-1] = denoise_features(self.unet, x[-1])  # 只修复 C5
        if self.with_neck:
            x = self.neck(x)
        return x


__all__ = ['DiffDetDetector']

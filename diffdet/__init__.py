"""DiffDet: Foggy Object Detection through Lightweight Feature-Space Denoising.

DiffDet wraps Sparse R-CNN with a lightweight residual feature-denoising U-Net
inserted at the ResNet-50 C5 position. Importing this package registers
``DiffDetDetector`` in the mmdetection ``MODELS`` registry, so it can be used
via ``custom_imports = dict(imports=['diffdet'])`` in a config.
"""

from .utils import compat  # noqa: F401  (torch.load weights_only patch)
from .denoising.diffusion_unet import (  # noqa: F401
    MinimalUNet,
    StudentUNet,
    build_diffusion_unet,
    decompose_conv,
    denoise_features,
    svd_apply,
)
from .models.diffdet_detector import DiffDetDetector  # noqa: F401
from .quantization.int8 import (  # noqa: F401
    quantize_unet_int8,
    quantize_weight_int8,
)

__version__ = '0.1.0'

__all__ = [
    'MinimalUNet',
    'StudentUNet',
    'build_diffusion_unet',
    'decompose_conv',
    'denoise_features',
    'svd_apply',
    'DiffDetDetector',
    'quantize_unet_int8',
    'quantize_weight_int8',
]

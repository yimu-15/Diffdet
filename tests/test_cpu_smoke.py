"""CPU 端到端 smoke test（合成随机张量，不依赖数据/权重）。"""

import torch

from diffdet.denoising.diffusion_unet import StudentUNet, denoise_features
from diffdet.quantization.int8 import quantize_unet_int8


def test_end_to_end_denoise_cpu():
    torch.manual_seed(0)
    m = StudentUNet(dim=128, base=32)
    m.eval()
    x = torch.randn(1, 2048, 32, 32)
    with torch.no_grad():
        y = denoise_features(m, x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_quantized_denoise_cpu():
    torch.manual_seed(0)
    m = StudentUNet(dim=128, base=32)
    m.eval()
    x = torch.randn(1, 2048, 16, 16)
    quantize_unet_int8(m)
    with torch.no_grad():
        y = denoise_features(m, x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_fog_synthesis_shape():
    import numpy as np
    from diffdet.datasets.fog import synthesize_fog
    img = np.zeros((375, 500, 3), dtype=np.uint8)
    out = synthesize_fog(img, beta=1.8, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8

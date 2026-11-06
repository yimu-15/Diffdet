"""U-Net 前向 / C5 形状 / 残差连接测试。"""

import torch

from diffdet.denoising.diffusion_unet import (MinimalUNet, StudentUNet,
                                              denoise_features)


def _c5(device, b=1, h=16, w=16):
    return torch.randn(b, 2048, h, w, device=device)


def test_student_forward_shape(device):
    m = StudentUNet(dim=128, base=32).to(device).eval()
    x = _c5(device)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_teacher_forward_shape(device):
    m = MinimalUNet(base=128).to(device).eval()
    x = _c5(device)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_c5_feature_shape_preserved(device):
    """去噪后 C5 特征形状必须与原始完全一致，保证与 FPN 兼容。"""
    m = StudentUNet(dim=128, base=32).to(device).eval()
    for h, w in [(15, 17), (16, 16), (31, 33)]:  # 含非 4 倍数尺寸
        x = _c5(device, h=h, w=w)
        with torch.no_grad():
            y = denoise_features(m, x)
        assert y.shape == x.shape, (h, w)


def test_residual_connection_identity(device):
    """残差连接语义: denoise_features(x) == x + unet(x)。"""
    m = StudentUNet(dim=128, base=32).to(device).eval()
    x = _c5(device)
    with torch.no_grad():
        y = denoise_features(m, x)
        expected = x + m(x)
    assert torch.allclose(y, expected, atol=1e-5)


def test_denoiser_not_identity(device):
    """随机初始化下 U-Net 非恒等映射（有可学习修正量）。"""
    torch.manual_seed(0)
    m = StudentUNet(dim=128, base=32).to(device).eval()
    x = _c5(device)
    with torch.no_grad():
        y = denoise_features(m, x)
    assert not torch.allclose(y, x, atol=1e-6)

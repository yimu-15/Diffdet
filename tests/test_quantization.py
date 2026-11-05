"""weight-only INT8 量化 smoke test。"""

import torch

from diffdet.denoising.diffusion_unet import StudentUNet
from diffdet.quantization.int8 import (int8_storage_size,
                                       quantize_unet_int8, quantize_weight_int8)


def test_quantize_dequantize_bounds(device):
    torch.manual_seed(0)
    w = torch.randn(64, 16, 3, 3, device=device)
    q, scale = quantize_weight_int8(w)
    assert q.dtype == torch.int8
    assert q.shape == w.shape
    assert scale.shape[0] == w.shape[0]
    deq = q.float() * scale
    assert deq.shape == w.shape
    # 量化误差不超过 1 个 scale 步长
    assert (deq - w.float()).abs().max().item() <= scale.max().item() * 1.01


def test_quantize_unet_consistency(device):
    torch.manual_seed(0)
    m = StudentUNet(dim=128, base=32).to(device).eval()
    x = torch.randn(1, 2048, 16, 16, device=device)
    with torch.no_grad():
        ref = m(x)
    w_bytes, s_bytes = quantize_unet_int8(m)
    assert w_bytes > 0 and s_bytes > 0
    with torch.no_grad():
        out = m(x)
    # W8A32 模拟下输出与 fp32 高度一致
    rel = (out - ref).abs().mean() / (ref.abs().mean() + 1e-6)
    assert rel.item() < 0.2


def test_storage_size_reduction(device):
    m = StudentUNet(dim=128, base=32).to(device)
    fp32_b, int8_b, scale_b = int8_storage_size(m)
    assert int8_b < fp32_b  # int8 存储更小
    assert fp32_b == int8_b * 4

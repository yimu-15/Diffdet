"""模型 save/load 测试。"""

import os

import torch

from diffdet.denoising.diffusion_unet import (StudentUNet,
                                              build_diffusion_unet)


def test_student_save_load_roundtrip(device, tmp_path):
    torch.manual_seed(0)
    m = StudentUNet(dim=128, base=32).to(device).eval()
    path = os.path.join(tmp_path, 'student.pt')
    torch.save({'model': m.state_dict(), 'dim': 128, 'base': 32}, path)

    ckpt = torch.load(path, map_location=device)
    m2 = build_diffusion_unet(ckpt, device=device)
    m2.load_state_dict(ckpt['model'])
    m2.eval()

    x = torch.randn(1, 2048, 16, 16, device=device)
    with torch.no_grad():
        a, b = m(x), m2(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_teacher_save_load_roundtrip(device, tmp_path):
    from diffdet.denoising.diffusion_unet import MinimalUNet
    torch.manual_seed(0)
    m = MinimalUNet(base=128).to(device).eval()
    path = os.path.join(tmp_path, 'teacher.pt')
    torch.save({'model': m.state_dict(), 'base': 128}, path)

    ckpt = torch.load(path, map_location=device)
    m2 = build_diffusion_unet(ckpt, device=device)  # 无 'dim' -> MinimalUNet
    m2.load_state_dict(ckpt['model'])
    m2.eval()

    x = torch.randn(1, 2048, 16, 16, device=device)
    with torch.no_grad():
        a, b = m(x), m2(x)
    assert torch.allclose(a, b, atol=1e-5)

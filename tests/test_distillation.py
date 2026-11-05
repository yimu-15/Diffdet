"""蒸馏：参数统计 / 教师冻结 / 蒸馏损失测试。"""

import torch

from diffdet.distillation.distill import DistillTrainer
from diffdet.denoising.diffusion_unet import MinimalUNet, StudentUNet
from diffdet.denoising.training import masked_mse


def test_parameter_counts():
    teacher = MinimalUNet(base=128)
    student = StudentUNet(dim=128, base=32)
    t = sum(p.numel() for p in teacher.parameters())
    s = sum(p.numel() for p in student.parameters())
    assert 4.5e6 < t < 4.7e6, t          # 论文 ~4.58M
    assert 0.78e6 < s < 0.80e6, s        # 论文 ~0.791M
    ratio = s / t
    assert 0.15 < ratio < 0.20, ratio    # 论文 17.2%


def test_teacher_frozen(device):
    teacher = MinimalUNet(base=128).to(device)
    student = StudentUNet(dim=128, base=32).to(device)
    trainer = DistillTrainer(teacher, student, lam=1.0)
    trainer.set_frozen_teacher()
    assert all(not p.requires_grad for p in teacher.parameters())


def test_distill_loss_decomposition(device):
    torch.manual_seed(0)
    teacher = MinimalUNet(base=64).to(device).eval()
    student = StudentUNet(dim=128, base=32).to(device).train()
    trainer = DistillTrainer(teacher, student, lam=1.0)

    fh = torch.randn(2, 2048, 16, 16, device=device)
    noise = torch.randn(2, 2048, 16, 16, device=device)
    mask = torch.ones(2, 16, 16, device=device)

    total, parts = trainer.train_step(fh, noise, mask)
    assert total.item() > 0
    assert parts['task'].item() > 0
    assert parts['distill'].item() > 0
    # L = L_task + lam * L_distill (lam=1.0)
    assert abs(total.item() - (parts['task'].item() + parts['distill'].item())) < 1e-5


def test_masked_mse_ignores_padding(device):
    pred = torch.zeros(1, 4, 4, 4, device=device)
    target = torch.ones(1, 4, 4, 4, device=device)
    mask = torch.zeros(4, 4, device=device)
    assert masked_mse(pred, target, mask).item() == 0.0

"""教师->学生特征级蒸馏训练（StudentUNet, dim=128/base=32, ~0.79M 参数）。

用法::

    python tools/train_student.py \
        --features work_dirs/phase1/clean_haze_features.pt \
        --teacher work_dirs/phase1/unet_minimal_b128.pt \
        --out work_dirs/phase2/unet_student_c128_b32.pt \
        --epochs 50 --lam 1.0

损失: L = L_task + λ * L_distill（论文 λ=1.0）。
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffdet.distillation.distill import DistillTrainer  # noqa: E402
from diffdet.denoising.diffusion_unet import (  # noqa: E402
    StudentUNet, build_diffusion_unet)
from diffdet.denoising.training import PairDataset, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(args.features, map_location='cpu')
    clean = [v['clean'] for v in data.values()]
    haze = [v['haze'] for v in data.values()]
    n = len(clean)
    idx = np.random.RandomState(args.seed).permutation(n)
    split = int(n * 0.8)
    tr, va = idx[:split], idx[split:]

    t_ckpt = torch.load(args.teacher, map_location=args.device)
    teacher = build_diffusion_unet(t_ckpt, device=args.device)
    teacher.load_state_dict(t_ckpt['model'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = StudentUNet(dim=args.dim, base=args.base).to(args.device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    trainer = DistillTrainer(teacher, student, lam=args.lam)

    tr_loader = torch.utils.data.DataLoader(PairDataset(clean, haze, tr),
                                            batch_size=args.batch, shuffle=True)
    va_loader = torch.utils.data.DataLoader(PairDataset(clean, haze, va),
                                            batch_size=args.batch)

    best_mse = float('inf')
    for ep in range(1, args.epochs + 1):
        student.train()
        losses = []
        for fh, noise, mask in tr_loader:
            fh, noise, mask = fh.to(args.device), noise.to(args.device), mask.to(args.device)
            opt.zero_grad()
            loss, _ = trainer.train_step(fh, noise, mask)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_mse, val_cos = evaluate(student, va_loader, args.device)
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save({'model': student.state_dict(), 'dim': args.dim, 'base': args.base},
                       args.out)
        if ep % 10 == 0 or ep == 1:
            print(f'epoch {ep}/{args.epochs} loss={np.mean(losses):.5f} '
                  f'val_mse={val_mse:.5f} cos={val_cos:.4f}')

    n_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f'done. best val_mse={best_mse:.5f}, student={n_params/1e6:.3f}M, '
          f'ratio={n_params/t_params*100:.1f}% -> {args.out}')


if __name__ == '__main__':
    main()

"""训练教师 U-Net（MinimalUNet, base=128, ~4.6M 参数）。

用法::

    python tools/train_teacher.py \
        --features work_dirs/phase1/clean_haze_features.pt \
        --out work_dirs/phase1/unet_minimal_b128.pt \
        --epochs 100 --lr 1e-3 --base 128

features 文件为 {name: {'clean': C5 [2048,H,W], 'haze': C5 [2048,H,W]}} 的
torch.save 字典（可用 tools/analyze_residuals.py --save-features 生成）。
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffdet.denoising.diffusion_unet import MinimalUNet  # noqa: E402
from diffdet.denoising.training import PairDataset, evaluate, masked_mse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--base', type=int, default=128)
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

    model = MinimalUNet(base=args.base).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_loader = torch.utils.data.DataLoader(PairDataset(clean, haze, tr),
                                            batch_size=args.batch, shuffle=True)
    va_loader = torch.utils.data.DataLoader(PairDataset(clean, haze, va),
                                            batch_size=args.batch)

    best_mse = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for fh, noise, mask in tr_loader:
            fh, noise, mask = fh.to(args.device), noise.to(args.device), mask.to(args.device)
            opt.zero_grad()
            loss = masked_mse(fh + model(fh), fh + noise, mask)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_mse, val_cos = evaluate(model, va_loader, args.device)
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save({'model': model.state_dict(), 'base': args.base}, args.out)
        if ep % 10 == 0 or ep == 1:
            print(f'epoch {ep}/{args.epochs} loss={np.mean(losses):.5f} '
                  f'val_mse={val_mse:.5f} cos={val_cos:.4f}')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'done. best val_mse={best_mse:.5f}, params={n_params/1e6:.3f}M -> {args.out}')


if __name__ == '__main__':
    main()

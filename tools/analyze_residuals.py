"""特征残差分析：提取 C5 特征并统计 Δ = f_foggy - f_clean 的分布。

用法::

    python tools/analyze_residuals.py \
        --clean-dir data/VOCdevkit/VOC2007/JPEGImages \
        --fog-dir data/VOC-FOG/JPEGImages/moderate \
        --names data/VOCdevkit/VOC2007/ImageSets/Main/val.txt \
        --out-dir work_dirs/phase1/residuals \
        --save-features work_dirs/phase1/clean_haze_features.pt

输出:
    residual_stats.txt   均值/标准差/偏度/超峰度
    residual_hist.png    残差直方图
    residual_qq.png      Gaussian Q-Q plot
    clean_haze_features.pt  成对 C5 特征（供训练复用）
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffdet.utils.common import MEAN, STD, build_backbone, load_resized  # noqa: E402


def _norm(img):
    """uint8 RGB -> BGR 归一化 float 张量 [1,3,H,W]。"""
    x = img[:, :, ::-1].astype(np.float32)  # RGB -> BGR
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def _extract(forward, img, device):
    x = _norm(img).to(device)
    with torch.no_grad():
        _, _, _, c5 = forward(x)
    return c5.squeeze(0).cpu()


def _compute_stats(delta):
    d = delta.float()
    mean = d.mean().item()
    std = d.std(unbiased=False).item()
    centered = d - mean
    m2 = (centered ** 2).mean()
    m3 = (centered ** 3).mean()
    m4 = (centered ** 4).mean()
    skew = (m3 / (m2 ** 1.5)).item() if m2 > 0 else 0.0
    kurt = (m4 / (m2 ** 2)).item() - 3.0 if m2 > 0 else 0.0
    return mean, std, skew, kurt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean-dir', required=True)
    ap.add_argument('--fog-dir', required=True)
    ap.add_argument('--names', required=True, help='图像名列表（每行一个，无扩展名）')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--save-features', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-images', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    forward, _ = build_backbone(device=args.device)

    with open(args.names) as f:
        names = [l.strip() for l in f if l.strip()]
    if args.max_images:
        names = names[: args.max_images]

    deltas = []
    feats = {}
    for name in names:
        clean = load_resized(os.path.join(args.clean_dir, name + '.jpg'))
        fog = load_resized(os.path.join(args.fog_dir, name + '.jpg'))
        fc = _extract(forward, clean, args.device)
        ff = _extract(forward, fog, args.device)
        deltas.append(ff - fc)
        feats[name] = {'clean': fc, 'haze': ff}

    delta = torch.stack(deltas).flatten()
    mean, std, skew, kurt = _compute_stats(delta)

    with open(os.path.join(args.out_dir, 'residual_stats.txt'), 'w') as f:
        f.write(f'num_images={len(names)}\n')
        f.write(f'num_samples={delta.numel()}\n')
        f.write(f'mean={mean:.6f}\n')
        f.write(f'std={std:.6f}\n')
        f.write(f'skewness={skew:.2f}\n')
        f.write(f'excess_kurtosis={kurt:.2f}\n')
    print(f'mean={mean:.6f} std={std:.6f} skewness={skew:.2f} '
          f'excess_kurtosis={kurt:.2f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d = delta.numpy()
        plt.figure()
        plt.hist(d, bins=100, density=True, color='steelblue', alpha=0.8)
        plt.title('Residual histogram ($\\Delta = f_{foggy} - f_{clean}$)')
        plt.xlabel('residual value')
        plt.ylabel('density')
        plt.savefig(os.path.join(args.out_dir, 'residual_hist.png'), dpi=150)
        plt.close()

        import scipy.stats as st
        plt.figure()
        st.probplot(d, dist='norm', plot=plt)
        plt.title('Gaussian Q-Q plot of residuals')
        plt.savefig(os.path.join(args.out_dir, 'residual_qq.png'), dpi=150)
        plt.close()
        print('saved residual_hist.png / residual_qq.png')
    except Exception as e:  # noqa: BLE001
        print(f'[warn] plotting skipped: {e}')

    if args.save_features:
        torch.save(feats, args.save_features)
        print(f'saved features -> {args.save_features}')


if __name__ == '__main__':
    main()

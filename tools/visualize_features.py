"""可视化去噪前后 C5 特征与残差修复效果。

用法::

    python tools/visualize_features.py \
        --clean data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
        --fog data/VOC-FOG/moderate/VOC2007/JPEGImages/000001.jpg \
        --unet work_dirs/phase2/unet_student_c128_b32.pt \
        --out-dir work_dirs/phase1/vis
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffdet.utils.common import MEAN, STD, build_backbone, load_resized, load_unet  # noqa: E402
from diffdet.denoising.diffusion_unet import denoise_features  # noqa: E402


def _norm(img):
    x = img[:, :, ::-1].astype(np.float32)
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True)
    ap.add_argument('--fog', required=True)
    ap.add_argument('--unet', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    forward, _ = build_backbone(device=args.device)
    unet = load_unet(args.unet, device=args.device).eval()

    with torch.no_grad():
        fc = _norm(load_resized(args.clean)).to(args.device)
        ff = _norm(load_resized(args.fog)).to(args.device)
        _, _, _, c5_clean = forward(fc)
        _, _, _, c5_fog = forward(ff)
        c5_fix = denoise_features(unet, c5_fog)

    def save(t, name):
        # 取通道均值作为 2D 响应图
        m = t[0].mean(dim=0).cpu().numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        np.save(os.path.join(args.out_dir, name + '.npy'), m)

    save(c5_clean, 'c5_clean_mean')
    save(c5_fog, 'c5_fog_mean')
    save(c5_fix, 'c5_fixed_mean')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for name in ['c5_clean_mean', 'c5_fog_mean', 'c5_fixed_mean']:
            plt.figure()
            plt.imshow(np.load(os.path.join(args.out_dir, name + '.npy')), cmap='viridis')
            plt.colorbar()
            plt.title(name)
            plt.savefig(os.path.join(args.out_dir, name + '.png'), dpi=150)
            plt.close()
    except Exception as e:  # noqa: BLE001
        print(f'[warn] png export skipped: {e}')

    print(f'visualizations saved -> {args.out_dir}')


if __name__ == '__main__':
    main()

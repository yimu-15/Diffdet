"""生成 VOC-FOG 合成雾数据集。

用法::

    python tools/prepare_voc_fog.py \
        --voc-root data/VOCdevkit/VOC2007 \
        --out-root data/VOC-FOG \
        --levels light moderate dense

对 VOC2007/JPEGImages 的每张图按三种雾浓度（β=1.0/1.8/3.0）合成雾图，
输出 JPEGImages/{level}/ 与 ImageSets/Main/{level}_val.txt。
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffdet.datasets.fog import LEVELS, synthesize_fog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voc-root', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--levels', nargs='+', default=['light', 'moderate', 'dense'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-images', type=int, default=None)
    args = ap.parse_args()

    jpeg_dir = os.path.join(args.voc_root, 'JPEGImages')
    names = sorted(os.listdir(jpeg_dir))
    if args.max_images:
        names = names[: args.max_images]

    for level in args.levels:
        beta = LEVELS[level]
        out_jpeg = os.path.join(args.out_root, 'JPEGImages', level)
        os.makedirs(out_jpeg, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        for name in names:
            img = np.asarray(Image.open(os.path.join(jpeg_dir, name)).convert('RGB'))
            fog = synthesize_fog(img, beta, rng)
            Image.fromarray(fog).save(os.path.join(out_jpeg, name))
        print(f'[{level}] beta={beta}: generated {len(names)} images -> {out_jpeg}')

    # 生成 val 列表（VOC2007 官方 val set）
    val_txt = os.path.join(args.voc_root, 'ImageSets', 'Main', 'val.txt')
    os.makedirs(os.path.join(args.out_root, 'ImageSets', 'Main'), exist_ok=True)
    if os.path.exists(val_txt):
        with open(val_txt) as f:
            val_names = f.read().strip().splitlines()
        for level in args.levels:
            out_txt = os.path.join(args.out_root, 'ImageSets', 'Main', f'{level}_val.txt')
            with open(out_txt, 'w') as f:
                f.write('\n'.join(val_names) + '\n')
            print(f'[{level}] val list -> {out_txt} ({len(val_names)} images)')


if __name__ == '__main__':
    main()

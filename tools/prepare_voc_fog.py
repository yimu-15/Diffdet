"""生成 VOC-FOG 合成雾数据集。

用法::

    python tools/prepare_voc_fog.py \
        --voc-root data/VOCdevkit/VOC2007 \
        --out-root data/VOC-FOG \
        --levels light moderate dense

对 VOC2007/JPEGImages 的每张图按三种雾浓度（β=1.0/1.8/3.0）合成雾图，
输出布局与 configs/ 中的 ``data_root`` 完全一致::

    data/VOC-FOG/{level}/VOC2007/JPEGImages/{id}.jpg
    data/VOC-FOG/{level}/VOC2007/Annotations/{id}.xml
    data/VOC-FOG/{level}/VOC2007/ImageSets/Main/{val,trainval}.txt
"""

import argparse
import os
import shutil
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
    ann_dir = os.path.join(args.voc_root, 'Annotations')
    sets_dir = os.path.join(args.voc_root, 'ImageSets', 'Main')
    names = sorted(n for n in os.listdir(jpeg_dir) if n.lower().endswith('.jpg'))
    if args.max_images:
        names = names[: args.max_images]

    for level in args.levels:
        beta = LEVELS[level]
        out_voc = os.path.join(args.out_root, level, 'VOC2007')
        out_jpeg = os.path.join(out_voc, 'JPEGImages')
        out_ann = os.path.join(out_voc, 'Annotations')
        out_sets = os.path.join(out_voc, 'ImageSets', 'Main')
        for d in (out_jpeg, out_ann, out_sets):
            os.makedirs(d, exist_ok=True)

        rng = np.random.default_rng(args.seed)
        for name in names:
            img = np.asarray(Image.open(os.path.join(jpeg_dir, name)).convert('RGB'))
            fog = synthesize_fog(img, beta, rng)
            Image.fromarray(fog).save(os.path.join(out_jpeg, name))
            # 雾图沿用清晰图的标注（同一几何变换不改变框）
            xml_name = os.path.splitext(name)[0] + '.xml'
            xml_path = os.path.join(ann_dir, xml_name)
            if os.path.exists(xml_path):
                shutil.copy2(xml_path, os.path.join(out_ann, xml_name))
        print(f'[{level}] beta={beta}: generated {len(names)} images -> {out_voc}')

    # 复制划分列表（VOC2007 提供 trainval / val）
    keep = {os.path.splitext(n)[0] for n in names}
    for split in ('val', 'trainval'):
        src = os.path.join(sets_dir, f'{split}.txt')
        if not os.path.exists(src):
            print(f'[warn] {src} not found, skip {split}.txt')
            continue
        with open(src) as f:
            ids = [line.split()[0] for line in f if line.strip()]
        if args.max_images:
            ids = [i for i in ids if i in keep]
        for level in args.levels:
            out_txt = os.path.join(
                args.out_root, level, 'VOC2007', 'ImageSets', 'Main', f'{split}.txt')
            with open(out_txt, 'w') as f:
                f.write('\n'.join(ids) + '\n')
            print(f'[{level}] {split} list -> {out_txt} ({len(ids)} images)')


if __name__ == '__main__':
    main()

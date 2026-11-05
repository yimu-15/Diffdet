"""评估 DiffDet / Sparse R-CNN 在 VOC-FOG 上的 mAP。

用法::

    python tools/evaluate.py configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py \
        --checkpoint checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth \
        --work-dir work_dirs/eval

使用 mmdet 的 VOCMetric（VOC07 11-point mAP）评估。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffdet  # noqa: F401

from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--work-dir', default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()

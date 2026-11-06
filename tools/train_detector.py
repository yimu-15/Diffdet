"""训练/微调 DiffDet 检测器（mmengine Runner 封装）。

用法::

    # 训练（Phase 4 联合微调）
    python tools/train_detector.py configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_finetune.py \
        --work-dir work_dirs/finetune_diffdet

    # 断点续训
    python tools/train_detector.py configs/diffdet/... --work-dir ... --resume
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffdet  # noqa: F401  (注册 DiffDetDetector + torch.load 补丁)

from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='diffdet 配置文件路径')
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.launcher = 'none'
    if args.resume:
        cfg.resume = True
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()

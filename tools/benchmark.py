"""推理速度与显存基准（FPS / GPU 显存）。

用法::

    python tools/benchmark.py configs/diffdet/sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py \
        --checkpoint checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth \
        --input-size 500 375 --warmup 5 --iters 50

注意：本脚本测量的是 *单卡前向+后处理* 吞吐（含 U-Net 去噪），不包含
数据加载与 NMS 之外的自定义逻辑；显存为 torch 峰值显存。
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffdet  # noqa: F401

from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input-size', nargs=2, type=int, default=[500, 375])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.launcher = 'none'
    model = Runner.from_cfg(cfg).model
    model.to(args.device).eval()

    h, w = args.input_size[1], args.input_size[0]
    x = torch.randn(1, 3, h, w).to(args.device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model.extract_feat(x)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model.extract_feat(x)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters

    fps = 1.0 / dt
    if args.device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        mem = float('nan')
    print(f'input={w}x{h} fps={fps:.1f} latency={dt*1000:.1f}ms '
          f'peak_memory={mem:.0f}MB device={args.device}')


if __name__ == '__main__':
    main()

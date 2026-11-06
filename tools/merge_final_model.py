"""合并"最终形态"检测器 checkpoint: Sparse R-CNN (FP32) + INT8 权重 U-Net。

联合微调后的检测器 checkpoint 已含 ``unet.*`` keys；本脚本用独立保存的 U-Net
checkpoint 覆盖这些权重，先做 per-output-channel 对称 INT8 权重量化（W8A32
模拟，量化后反量化回 fp32），再写回整体 checkpoint，使
``configs/diffdet/..._diffdet_final*.py`` 可以只加载单个文件完成推理。

用法::

    # 直接用联合微调 checkpoint（其中已含 unet.* 权重）
    python tools/merge_final_model.py \
        --detector work_dirs/finetune_diffdet/epoch_3.pth \
        --out checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth

    # 或显式指定独立保存的 U-Net checkpoint
    python tools/merge_final_model.py \
        --detector work_dirs/finetune_diffdet/epoch_3.pth \
        --unet work_dirs/finetune_diffdet/unet_finetuned_c128_b32.pt \
        --out checkpoints/sparse_rcnn_voc_rainfog_final_int8.pth

输出 checkpoint 结构: ``{'state_dict': {...}, 'meta': {...}}``，其中 meta 记录
U-Net 结构、参数规模、真实 int8 权重字节数与每通道 scale 字节数（供部署核算）。

注意：本脚本只做 weight-only INT8 量化（激活仍为 fp32），交付的 state_dict 中
保存的是反量化后的 fp32 值；这不等价于 INT8 算子加速。
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffdet.denoising.diffusion_unet import (  # noqa: E402
    StudentUNet, build_diffusion_unet)
from diffdet.quantization.int8 import quantize_weight_int8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True,
                    help='联合微调后的检测器 checkpoint（含 unet.* keys）')
    ap.add_argument('--unet', default=None,
                    help='可选: 独立保存的 U-Net checkpoint（含 model / dim / base）')
    ap.add_argument('--unet-dim', type=int, default=128)
    ap.add_argument('--unet-base', type=int, default=32)
    ap.add_argument('--out', default=os.path.join(
        'checkpoints', 'sparse_rcnn_voc_rainfog_final_int8.pth'))
    args = ap.parse_args()

    det = torch.load(args.detector, map_location='cpu', weights_only=False)
    det_sd = det.get('state_dict', det)

    if args.unet:
        unet_ckpt = torch.load(args.unet, map_location='cpu', weights_only=False)
        model = build_diffusion_unet(unet_ckpt, device='cpu')
        model.load_state_dict(unet_ckpt['model'])
        dim = int(unet_ckpt.get('dim', args.unet_dim))
        base = int(unet_ckpt.get('base', args.unet_base))
    else:
        model = StudentUNet(in_ch=2048, dim=args.unet_dim, base=args.unet_base)
        model.load_state_dict({
            k[len('unet.'):]: v for k, v in det_sd.items()
            if k.startswith('unet.')})
        dim, base = args.unet_dim, args.unet_base
    model.eval()

    # 1) U-Net 权重量化（W8A32 模拟）
    int8_bytes = 0
    scale_bytes = 0
    quant_tensors = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            w = m.weight.data
            q, scale = quantize_weight_int8(w)
            int8_bytes += w.numel()
            scale_bytes += w.shape[0] * 4
            m.weight.data = (q.float() * scale).to(w.dtype)  # 反量化回 fp32
            quant_tensors[name + '.weight'] = q
            quant_tensors[name + '.scale'] = scale
    unet_sd = model.state_dict()  # BN running stats 原样保留
    unet_n = sum(p.numel() for p in model.parameters())

    # 2) 校验检测器 checkpoint 中的 unet.* keys 与 U-Net 结构一一对应
    det_unet_keys = sorted(k for k in det_sd if k.startswith('unet.'))
    if det_unet_keys != sorted('unet.' + k for k in unet_sd):
        raise KeyError(
            f'detector 与 U-Net keys 不匹配: detector {len(det_unet_keys)} 个 '
            f'vs U-Net {len(unet_sd)} 个')

    # 3) 组装最终 state_dict（unet.* 用刚量化后的权重替换）
    final_sd = {}
    for k, v in det_sd.items():
        final_sd[k] = unet_sd[k[len('unet.'):]] if k.startswith('unet.') else v

    det_fp32_bytes = sum(
        v.numel() * 4 for k, v in final_sd.items() if not k.startswith('unet.'))
    meta = {
        'model_type': 'DiffDetDetector',
        'quant': 'int8w',
        'unet_dim': dim,
        'unet_base': base,
        'unet_n_params': unet_n,
        'unet_int8_bytes': int8_bytes,
        'unet_scale_bytes': scale_bytes,
        'unet_fp32_bytes': unet_n * 4,
        'detector_fp32_bytes': det_fp32_bytes,
        'quant_tensors': quant_tensors,
        'note': ('整体 checkpoint: Backbone+Head(FP32)+U-Net(INT8 权重, 反量化值存储)。'
                 '部署时 U-Net 权重可存 int8, 仅多一个每通道 fp32 scale 表。'),
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'state_dict': final_sd, 'meta': meta}, args.out)

    print('=== 最终模型合并完成 ===')
    print(f'U-Net: dim={meta["unet_dim"]} base={meta["unet_base"]} '
          f'参数 {unet_n / 1e6:.3f} M')
    print(f'U-Net 存储: FP32 {unet_n * 4 / 1e6:.3f} MB -> '
          f'INT8 {int8_bytes / 1e6:.3f} MB + scales {scale_bytes / 1e6:.3f} MB '
          f'({(int8_bytes + scale_bytes) / (unet_n * 4) * 100:.1f}%)')
    print(f'检测器(FP32) 存储: {det_fp32_bytes / 1e6:.3f} MB')
    print(f'checkpoint: {args.out} '
          f'({os.path.getsize(args.out) / 1e6:.1f} MB, fp32 权重体积)')
    print(f'state_dict keys: {len(final_sd)} (含 unet.* {len(det_unet_keys)})')


if __name__ == '__main__':
    main()

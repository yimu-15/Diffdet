"""Weight-only INT8 量化（W8A32 模拟）。

对 U-Net 的 Conv2d / ConvTranspose2d / Linear 权重做 per-output-channel
对称 int8 量化，量化后反量化回 fp32 参与 forward（模拟 TensorRT 式 W8A32
推理），用于评估 INT8 权重对精度的损失并核算等效存储大小。

注意：这是 *weight-only* INT8 量化（仅权重存储为 int8，激活仍为 fp32），
     *不* 等价于 INT8 算子加速；除非在真实推理引擎上实测，否则不应宣称
     速度提升。
"""

import torch
import torch.nn as nn


@torch.no_grad()
def quantize_weight_int8(w: torch.Tensor):
    """per-output-channel 对称 int8 量化。

    返回 (q, scale): q 为 int8 权重 [out,...], scale 为每输出通道 fp32 [out,1,1,..]。
    """
    wf = w.float()
    out_c = wf.shape[0]
    shape = (out_c,) + (1,) * (wf.dim() - 1)
    amax = wf.abs().amax(dim=tuple(range(1, wf.dim())))
    scale = (amax / 127.0).clamp_min(1e-12).view(shape)
    q = torch.clamp(torch.round(wf / scale), -128, 127).to(torch.int8)
    return q, scale.float()


@torch.no_grad()
def quantize_unet_int8(unet: nn.Module):
    """把 U-Net 所有 Conv2d/ConvTranspose2d/Linear 权重量化到 INT8（W8A32 模拟）。

    返回 (int8 权重字节数, fp32 scale 字节数)。权重在原模型上就地反量化回 fp32。
    """
    w_bytes = 0
    s_bytes = 0
    for m in unet.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            w = m.weight.data
            w_bytes += w.numel()          # int8 每权重 1 字节
            s_bytes += w.shape[0] * 4     # 每输出通道一个 fp32 scale
            q, scale = quantize_weight_int8(w)
            m.weight.data = (q.float() * scale).to(w.dtype)
    return w_bytes, s_bytes


def int8_storage_size(model: nn.Module):
    """返回 (fp32_bytes, int8_bytes, scale_bytes) 供大小核算。

    仅统计 Conv2d / ConvTranspose2d / Linear 的权重；BN 的 running stats
    与偏置不纳入（FP32 原样保留）。
    """
    fp32_bytes = 0
    int8_bytes = 0
    scale_bytes = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            w = m.weight
            fp32_bytes += w.numel() * 4
            int8_bytes += w.numel()
            scale_bytes += w.shape[0] * 4
    return fp32_bytes, int8_bytes, scale_bytes


__all__ = ['quantize_weight_int8', 'quantize_unet_int8', 'int8_storage_size']

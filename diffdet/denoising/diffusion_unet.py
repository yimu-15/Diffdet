"""DiffDet 去噪 U-Net 模块（正式版，自包含，不依赖 tools 脚本）。

- MinimalUNet: 最简 2-level U-Net（Phase 1 教师，base=128）
- StudentUNet: 1x1 降维 head(2048->dim) + 小 U-Net(base) + 1x1 升维 tail（Phase 2 学生，dim=128/base=32）
- svd_apply: 按保存的逐层 rank 重建低秩分解结构（Step 2.3 产物加载）
- build_diffusion_unet: 自动识别 checkpoint 结构（教师/学生/SVD 学生）并重建
- denoise_features: C5 特征修复 x' = x + UNet(x)（推理时 pad 到 4 倍数）

注意：本模块实现的是 diffusion-inspired 的 *单次前向* 特征去噪，
不包含 DDPM 的时间步、噪声调度与迭代反向采样。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalUNet(nn.Module):
    """最简版 2-level U-Net, 全卷积, 输入输出同尺寸。

    head: in_ch->base (1x1 conv); 编码器 base->2*base (x2 下采样 x2);
    瓶颈 2*base; 解码器带 skip; tail: base->in_ch (1x1 conv).
    """

    def __init__(self, in_ch=2048, base=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base, 1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True))
        # encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 2, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True))
        # decoder
        self.u1 = nn.ConvTranspose2d(base * 2, base * 2, 2, 2)
        self.c1 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c2 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, 1, 1), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        x0 = self.head(x)                      # [B,base,H,W]
        s1 = self.e1(x0)                       # skip1
        x = self.d1(s1)                        # /2
        s2 = self.e2(x)                        # skip2
        x = self.d2(s2)                        # /4
        x = self.bottleneck(x)
        x = self.u1(x)                         # /2
        x = self.c1(torch.cat([x, s2], dim=1))
        x = self.u2(x)                         # full
        x = self.c2(torch.cat([x, s1], dim=1))
        return self.tail(x)


class StudentUNet(nn.Module):
    """轻量学生 U-Net: 1x1 降维 head(2048->dim) + 小 U-Net(base) + 1x1 升维 tail(dim->2048)。

    head: 2048 -> dim (1x1 conv + BN + ReLU)
    body: 2-level U-Net (dim -> dim, base 通道, 复用 MinimalUNet)
    tail: dim -> 2048 (1x1 conv)
    """

    def __init__(self, in_ch=2048, dim=128, base=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, dim, 1), nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))
        self.body = MinimalUNet(in_ch=dim, base=base)
        self.tail = nn.Conv2d(dim, in_ch, 1)

    def forward(self, x):
        return self.tail(self.body(self.head(x)))


def decompose_conv(conv: nn.Conv2d, rank: int):
    """把单个 Conv2d 拆成两段卷积; 无压缩空间时返回 None。"""
    w = conv.weight.detach()  # [out, in, k, k]
    out, cin, k, _ = w.shape
    w2d = w.reshape(out, -1)
    min_dim = min(out, w2d.shape[1])
    r = min(rank, min_dim)
    if r <= 0 or r >= min_dim:
        return None  # rank 达到满秩, 保留原层
    U, S, Vt = torch.linalg.svd(w2d, full_matrices=False)
    conv1 = nn.Conv2d(cin, r, k, conv.stride, conv.padding,
                      dilation=conv.dilation, bias=False)
    conv2 = nn.Conv2d(r, out, 1, bias=conv.bias is not None)
    with torch.no_grad():
        w1 = (S[:r].unsqueeze(1) * Vt[:r]).reshape(
            r, cin, k, k).contiguous()
        conv1.weight.copy_(w1)
        conv2.weight.copy_(U[:, :r].reshape(out, r, 1, 1).contiguous())
        if conv.bias is not None:
            conv2.bias.copy_(conv.bias)
    return nn.Sequential(conv1, conv2)


def svd_apply(model, ranks):
    """按 {module_path: rank} 就地替换 model 中的 Conv2d 层为低秩两段卷积。"""
    modules = dict(model.named_modules())
    for path, rank in ranks.items():
        if path not in modules:
            continue
        conv = modules[path]
        if not isinstance(conv, nn.Conv2d):
            continue
        new = decompose_conv(conv, rank)
        if new is None:
            continue
        parent_name, _, attr = path.rpartition('.')
        parent = modules[parent_name] if parent_name else model
        parent._modules[attr] = new


def build_diffusion_unet(ckpt: dict, device='cpu'):
    """按 checkpoint 结构重建 U-Net（教师/学生/SVD 学生），不加载权重。

    ckpt: torch.load 的结果。含 'dim' -> 学生; 否则 -> MinimalUNet(base)。
    svd['ranks'] 存在 -> 再应用低秩分解结构重建。
    """
    if 'dim' in ckpt:
        model = StudentUNet(dim=ckpt['dim'],
                            base=ckpt.get('base', 32)).to(device)
        svd = ckpt.get('svd')
        if svd:
            svd_apply(model, svd['ranks'])
            model.to(device)  # svd_apply 新建的层默认在 CPU
    else:
        base = ckpt.get('base', 64)  # 兼容旧版 unet_minimal.pt (base=64)
        model = MinimalUNet(base=base).to(device)
    return model


@torch.no_grad()
def denoise_features(unet: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """对 C5 特征 x [N,2048,H,W] 施加 U-Net 修复: x' = x + UNet(x)。

    推理时 pad 到 4 的倍数保证下采样对齐，再 crop 回原尺寸。
    """
    N, C, H, W = x.shape
    hp = (4 - H % 4) % 4
    wp = (4 - W % 4) % 4
    if hp or wp:
        xp = F.pad(x, (0, wp, 0, hp))
        eps = unet(xp)[:, :, :H, :W]
    else:
        eps = unet(x)
    return x + eps


__all__ = [
    'MinimalUNet',
    'StudentUNet',
    'decompose_conv',
    'svd_apply',
    'build_diffusion_unet',
    'denoise_features',
]

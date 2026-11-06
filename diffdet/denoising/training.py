"""U-Net 训练公共组件（PairDataset / masked_mse / masked_cos / evaluate）。

复用于教师 U-Net 训练（Phase 1）与学生蒸馏训练（Phase 2）。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PAD_SIZE = 32  # 特征最大空间维度（与 Phase 1 保持一致）


class PairDataset(Dataset):
    """返回 (f_haze_pad, noise_pad, mask), 全部 pad 到 PAD_SIZE x PAD_SIZE。"""

    def __init__(self, clean, haze, idx):
        self.clean = [clean[i] for i in idx]
        self.haze = [haze[i] for i in idx]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        fc, fh = self.clean[i], self.haze[i]
        H, W = fc.shape[-2:]
        assert H <= PAD_SIZE and W <= PAD_SIZE, (H, W)
        fc_p = F.pad(fc, (0, PAD_SIZE - W, 0, PAD_SIZE - H))
        fh_p = F.pad(fh, (0, PAD_SIZE - W, 0, PAD_SIZE - H))
        mask = torch.zeros(PAD_SIZE, PAD_SIZE)
        mask[:H, :W] = 1.0
        noise = fc_p - fh_p
        return fh_p, noise, mask


def masked_mse(pred, target, mask):
    diff = ((pred - target) ** 2) * mask.unsqueeze(1)
    n_valid = mask.sum() * pred.shape[1]  # 有效空间位置 x 通道数
    return diff.sum() / n_valid.clamp_min(1)


def masked_cos(fa, fb, mask):
    """按样本计算 mask 区域的余弦相似度。"""
    m = mask.unsqueeze(1)
    a = fa * m
    b = fb * m
    dot = (a * b).sum(dim=(1, 2, 3))
    na = torch.linalg.vector_norm(a, dim=(1, 2, 3))
    nb = torch.linalg.vector_norm(b, dim=(1, 2, 3))
    return dot / (na * nb + 1e-8)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_list, cos_list = [], []
    for fh, noise, mask in loader:
        fh, noise, mask = fh.to(device), noise.to(device), mask.to(device)
        eps_hat = model(fh)
        fix = fh + eps_hat
        mse_list.append(masked_mse(fix, fh + noise, mask).item())
        cos_list.append(masked_cos(fix, fh + noise, mask).mean().item())
    return float(np.mean(mse_list)), float(np.mean(cos_list))


__all__ = ['PairDataset', 'masked_mse', 'masked_cos', 'evaluate', 'PAD_SIZE']

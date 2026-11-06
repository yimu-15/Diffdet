"""公共工具：类别、归一化、图像加载、backbone、U-Net 加载。"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from ..denoising.diffusion_unet import (MinimalUNet, StudentUNet,
                                        denoise_features, svd_apply)

CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # ImageNet BGR
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_labels(path):
    """加载 PASCAL VOC 标签文件，返回每行 split 后的字符串列表。"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.split() for line in f.read().strip().splitlines()]


def load_resized(path, size=(500, 375)):
    """加载图像并 resize 到 (W, H) = size，返回 RGB uint8。"""
    img = Image.open(path).convert('RGB').resize(size)
    return np.asarray(img)


def build_backbone(device='cuda', pretrained=True, freeze_bn=True):
    """构建 ResNet-50 并截取到 C2-C5 输出（去掉 avgpool/fc）。

    返回 (backbone, feat_channels)，backbone 输入 BGR 归一化后的张量，
    输出 [c2, c3, c4, c5]。
    """
    import torchvision.models as tv
    weights = tv.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = tv.resnet50(weights=weights)
    layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
              resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    backbone = nn.Sequential(*layers).to(device).eval()
    if freeze_bn:
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    def forward(x):
        """x: [B,3,H,W] BGR 已归一化张量。返回 [c2,c3,c4,c5]。"""
        c1 = backbone[0:4](x)
        c2 = backbone[4](c1)
        c3 = backbone[5](c2)
        c4 = backbone[6](c3)
        c5 = backbone[7](c4)
        return c2, c3, c4, c5

    return forward, [256, 512, 1024, 2048]


def load_unet(ckpt_path, device='cuda'):
    """按 checkpoint 结构加载 U-Net（教师/学生/SVD 学生）。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'dim' in ckpt:
        model = StudentUNet(dim=ckpt['dim'], base=ckpt.get('base', 32))
        svd = ckpt.get('svd')
        if svd:
            svd_apply(model, svd['ranks'])
            model = model.to(device)
    else:
        model = MinimalUNet(base=ckpt.get('base', 64))
    model.load_state_dict(ckpt['model'])
    return model.to(device)


__all__ = [
    'CLASSES',
    'MEAN',
    'STD',
    'load_labels',
    'load_resized',
    'build_backbone',
    'load_unet',
    'denoise_features',
]

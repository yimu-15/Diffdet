"""VOC-FOG 合成雾生成（标准大气散射模型）。

I(x) = t(x) * I0(x) + (1 - t(x)) * A
t(x) = exp(-beta * depth(x))

depth 由随机地平线 + 指数衰减场合成；beta 对应雾浓度：
    light=1.0, moderate=1.8, dense=3.0（见论文）。A 为大气光（雾天亮度）。
"""

import numpy as np

LEVELS = {'light': 1.0, 'moderate': 1.8, 'dense': 3.0}
ATMOS_LIGHT = 0.85  # 大气光 A（论文雾天亮度设置）


def make_depth_map(h, w, rng=None):
    """生成随机地平线 + 指数衰减的深度场，取值 (0, 1]。"""
    rng = rng or np.random.default_rng()
    horizon = rng.uniform(0.2, 0.8)
    # 用每行到地平线的距离做指数衰减，产生远深近浅的透视效果
    rows = np.arange(h, dtype=np.float32)
    dist = np.abs(rows / h - horizon)
    base = np.exp(-4.0 * dist)  # (h,)
    depth = base[:, None] + rng.uniform(-0.03, 0.03, size=(h, w)).astype(np.float32)
    # 加一点水平向渐变，避免纯纵向条纹
    cols = (np.arange(w, dtype=np.float32) / w)[None, :]
    depth = depth + 0.1 * cols * (1.0 - base[:, None])
    depth = np.clip(depth, 0.05, 1.0)
    return depth


def synthesize_fog(img, beta, rng=None):
    """对 uint8 HxWx3 图像施加 beta 强度的合成雾，返回 uint8 图像。"""
    rng = rng or np.random.default_rng()
    img_f = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    depth = make_depth_map(h, w, rng)[..., None]  # (h, w, 1)
    t = np.exp(-beta * depth)
    out = t * img_f + (1.0 - t) * ATMOS_LIGHT
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def add_fog(img, beta, seed=None):
    """带随机种子的便捷接口，返回合成雾图。"""
    rng = np.random.default_rng(seed)
    return synthesize_fog(img, beta, rng)


__all__ = ['LEVELS', 'ATMOS_LIGHT', 'make_depth_map', 'synthesize_fog', 'add_fog']

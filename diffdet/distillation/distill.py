"""教师-学生特征级知识蒸馏。

训练目标（论文 λ=1.0）：

    L = L_task + λ * L_distill

* L_task    ：学生修复特征与 clean 特征的 masked MSE（去噪任务损失）。
* L_distill ：学生修复特征与教师（frozen）修复特征的 masked MSE（特征级蒸馏）。

教师在前向时冻结（no_grad + eval），仅学生更新。
"""

import torch
import torch.nn as nn

from ..denoising.training import masked_mse


def feature_distill_loss(student_feat, teacher_feat, mask):
    """特征级蒸馏损失（masked MSE）。"""
    return masked_mse(student_feat, teacher_feat, mask)


class DistillTrainer:
    """封装一次蒸馏训练步（teacher frozen, student 可训）。

    Args:
        teacher: 教师 U-Net（frozen）。
        student: 学生 U-Net（可训）。
        lam: 蒸馏权重 λ，默认 1.0。
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, lam: float = 1.0):
        self.teacher = teacher
        self.student = student
        self.lam = lam

    def set_frozen_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def train_step(self, fh, noise, mask):
        """foggy 特征 fh, 残差 noise = clean - foggy, 有效区 mask。

        返回 (total_loss, {'task': ..., 'distill': ...})。
        """
        self.set_frozen_teacher()
        with torch.no_grad():
            fix_t = fh + self.teacher(fh)  # 教师修复特征
        eps_s = self.student(fh)
        fix_s = fh + eps_s
        clean = fh + noise
        l_task = masked_mse(fix_s, clean, mask)
        l_distill = feature_distill_loss(fix_s, fix_t, mask)
        return l_task + self.lam * l_distill, {'task': l_task, 'distill': l_distill}


__all__ = ['feature_distill_loss', 'DistillTrainer']

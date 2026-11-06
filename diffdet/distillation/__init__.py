"""Teacher-student knowledge distillation."""

from .distill import DistillTrainer, feature_distill_loss  # noqa: F401

__all__ = ['DistillTrainer', 'feature_distill_loss']

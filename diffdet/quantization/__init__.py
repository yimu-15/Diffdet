"""Weight-only INT8 quantization (W8A32 simulation)."""

from .int8 import (  # noqa: F401
    int8_storage_size,
    quantize_unet_int8,
    quantize_weight_int8,
)

__all__ = ['int8_storage_size', 'quantize_unet_int8', 'quantize_weight_int8']

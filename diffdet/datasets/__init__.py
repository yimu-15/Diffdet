"""Dataset preparation (synthetic fog synthesis)."""

from .fog import (  # noqa: F401
    ATMOS_LIGHT,
    LEVELS,
    add_fog,
    make_depth_map,
    synthesize_fog,
)

__all__ = ['ATMOS_LIGHT', 'LEVELS', 'add_fog', 'make_depth_map', 'synthesize_fog']

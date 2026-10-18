"""
Shared utilities for magnet optimization.

Modules:
- field_calculator: GPU magnetic field calculation (GRFC)
- field_validation: Magpylib-based field validation and comparison
- visualize_configuration: 3D visualization and plotting
"""

from . import field_calculator
from . import field_validation
from . import visualize_configuration

# Convenience imports
from .field_validation import overlay_comparison
from .visualize_configuration import (
    rad_halbach_Design_rad,
    visualize_configuration as show_configuration
)

__all__ = [
    'field_calculator',
    'field_validation',
    'visualize_configuration',
    'overlay_comparison',
    'rad_halbach_Design_rad',
    'show_configuration',
]

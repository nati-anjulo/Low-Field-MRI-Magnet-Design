"""
GPU magnetic field calculation module.
Re-exported from utils.field_calculator for backward compatibility.
"""

from utils.field_calculator import *
from utils import field_calculator as _fc

# Re-export module-level state and functions
load_field = _fc.load_field
calc_field = _fc.calc_field
input_converter_vectorized = _fc.input_converter_vectorized

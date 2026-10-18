"""
Cylindrical Optimization Package.

All functions are exact copies from the original notebook, organized into modules.

Modules:
- field_calculator: GPU magnetic field calculation (copy of GRFC_cached)
- sampling: DSV sample point generation
- generator: Configuration generation and octant reflection
- perturbation: 3-phase hybrid SA perturbation
- constraints: Hard/soft constraint checking with bolt avoidance
- cost: Cost evaluation
- optimizer: Simulated annealing engine
- calibration: SA auto-calibration
- visualization: Plotting functions
- scad_export: Export to OpenSCAD format for manufacturing
"""

# Import all modules
from . import field_calculator
from . import sampling
from . import generator
from . import perturbation
from . import constraints
from . import cost
from . import optimizer
from . import calibration
from . import visualization
from . import scad_export

# Export key visualization function at package level
from .visualization import show_magnets_plotly
from .scad_export import export_to_scad, quick_export

__all__ = [
    'field_calculator',
    'sampling',
    'generator',
    'perturbation',
    'constraints',
    'cost',
    'optimizer',
    'calibration',
    'visualization',
    'scad_export',
    'show_magnets_plotly',
    'export_to_scad',
    'quick_export',
]

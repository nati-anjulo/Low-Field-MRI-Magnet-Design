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

# Export key visualization function at package level
from .visualization import show_magnets_plotly

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
    'show_magnets_plotly',
]

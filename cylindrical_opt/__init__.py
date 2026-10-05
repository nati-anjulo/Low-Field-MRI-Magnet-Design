"""
Cylindrical Optimization Package - STRICT COPY from GPU_SA_Octant_with_bolt_OPTIMIZED_Cylinderical.ipynb

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
]

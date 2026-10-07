"""
Ellipse Optimization Package.

All functions are exact copies from the original notebook, organized into modules.

Modules:
- field_calculator: GPU magnetic field calculation (copy of GRFC_cached)
- geometry: Ellipse radius and bolt position calculations
- sampling: DSV sample point generation
- generator: Configuration generation and 4-fold reflection
- perturbation: 3-phase hybrid SA perturbation
- constraints: Hard/soft constraint checking
- cost: Cost evaluation
- optimizer: Simulated annealing engine
- calibration: SA auto-calibration
- visualization: Plotting functions
"""

# Import all modules
from . import field_calculator
from . import geometry
from . import sampling
from . import generator
from . import perturbation
from . import constraints
from . import cost
from . import optimizer
from . import calibration
from . import visualization

# Export key functions at package level for convenience
from .geometry import ellipse_r_gpu, get_bolt_positions_ellipse_gpu
from .sampling import _generate_sample_points_small_dsv
from .generator import reflect_to_half_ellipse, generate_half_ellipse, generate_full_ellipse_reflected
from .perturbation import Hybrid_perturbation_half_ellipse
from .constraints import calculate_soft_barriers_half_ellipse, check_hard_constraints_half_ellipse
from .cost import calculate_cost_half_ellipse
from .optimizer import simulated_annealing_half_ellipse, run_n_times_half_ellipse
from .calibration import calibrate_sa_rigorous, diagnose_sa_calibration
from .visualization import Plot_positions_plotly, plot_sa_trajectory, plot_cost_progression

__all__ = [
    # Modules
    'field_calculator',
    'geometry',
    'sampling',
    'generator',
    'perturbation',
    'constraints',
    'cost',
    'optimizer',
    'calibration',
    'visualization',
    # Functions
    'ellipse_r_gpu',
    'get_bolt_positions_ellipse_gpu',
    '_generate_sample_points_small_dsv',
    'reflect_to_half_ellipse',
    'generate_half_ellipse',
    'generate_full_ellipse_reflected',
    'Hybrid_perturbation_half_ellipse',
    'calculate_soft_barriers_half_ellipse',
    'check_hard_constraints_half_ellipse',
    'calculate_cost_half_ellipse',
    'simulated_annealing_half_ellipse',
    'run_n_times_half_ellipse',
    'calibrate_sa_rigorous',
    'diagnose_sa_calibration',
    'Plot_positions_plotly',
    'plot_sa_trajectory',
    'plot_cost_progression',
]

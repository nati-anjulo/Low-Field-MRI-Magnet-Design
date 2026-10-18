# Magnet Array Optimization

Optimize permanent magnet arrays using simulated annealing on GPU.

## Project Structure

```
Magnet_design/
├── utils/                       # Shared utilities
│   ├── field_calculator.py      # GPU magnetic field calculation
│   ├── field_validation.py      # Magpylib-based validation
│   └── visualize_configuration.py  # 3D visualization
├── ellipse_opt/                 # Elliptical bore optimization
│   ├── generator.py             # Magnet placement & reflection
│   ├── optimizer.py             # Simulated annealing engine
│   ├── calibration.py           # SA auto-calibration
│   ├── scad_export.py           # Export to OpenSCAD
│   └── ...
├── cylindrical_opt/             # Cylindrical bore optimization
│   └── ...                      # Same modules as ellipse_opt
├── precompute/                  # Field precomputation notebooks
├── ellipse_optimization_usage.ipynb
├── cylindrical_optimization_usage.ipynb
└── visualize_checkpoint.ipynb
```

## Why Simulated Annealing

The choice of optimizer matters as much as the speed of each evaluation. Navigating a rugged solution space requires covering the space broadly as well as quickly.

Simulated annealing allows fine control over perturbation size - large steps early for exploration, small steps late for refinement. This is critical for magnet homogeneity tuning.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and NVIDIA GPU with CUDA. See `INSTALL.txt` for details.

## How to Use

1. **Generate a precomputed field file** (one-time)
   ```
   Run: precompute/Generate_Precomputed_Field_Batched_analytical.ipynb
   Output: Precalc_field_*.npz
   ```

2. **Run optimization**
   - Elliptical bore: `ellipse_optimization_usage.ipynb`
   - Cylindrical bore: `cylindrical_optimization_usage.ipynb`

3. **Export for manufacturing**
   ```python
   from ellipse_opt import scad_export
   scad_export.export_to_scad('design.npz', geometry_params={...})
   ```
   Opens in OpenSCAD: `view_mode="all"` for assembly, `"single_layer"` for STL export.

## 3D Viewer

Built-in:
```python
visualization.show_magnets_plotly(config, magnet_size)
```

Magpylib (optional):
```python
from utils import visualize_configuration as VS
VS.rad_halbach_Design_rad(magnet_size, config).show()
```

## Continue from Previous Run

```python
# Load previous result
prev = np.load('ellipse_design_20261007.npz')
initial_q1 = cp.asarray(prev['q1_config'])

# For refinement runs, use low temperature (T0=5-10)
USE_CALIBRATION = False
T0_MANUAL = 5
```

## Key Parameters

| Parameter | Fresh Run | Refinement |
|-----------|-----------|------------|
| `USE_CALIBRATION` | `True` | `False` |
| `T0_MANUAL` | N/A | `5-10` |
| `exploration_scale` | `4-5` | `1-2` |

---
Natnael Anjulo  
CaseMRI, Case Western Reserve University

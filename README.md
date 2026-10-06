# Magnet Array Optimization

GPU-accelerated simulated annealing for permanent magnet array design.

## Workflow

```
1. Generate Precomputed Field
   ├── Generate_Precomputed_Field_Batched_analytical.ipynb  (GPU analytical)
   └── Output: Precalc_field_*.npz

2. Run Optimization (Example Usage Notebooks)
   ├── ellipse_optimization_usage.ipynb      ← Example: elliptical bore
   └── cylindrical_optimization_usage.ipynb  ← Example: cylindrical octant
   
   These are example notebooks - modify parameters for your application.

3. Results
   ├── Optimized magnet positions/orientations
   ├── Field homogeneity metrics
   └── 3D visualization
```

## Packages

| Package | Geometry | Description |
|---------|----------|-------------|
| `ellipse_opt` | Elliptical cylinder | User-defined elliptical bore (rx, ry, gap, z_max) |
| `cylindrical_opt` | Cylindrical octant | User-defined cylindrical bounds (r_min, r_max, z_min, z_max) |
| `analytical_magnet_gpu.py` | Field calculation | GPU-accelerated analytical cuboid field (Engel-Herbert) |
| `analytical_field.py` | Field calculation | CPU analytical cuboid field |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `ellipse_optimization_usage.ipynb` | Elliptical magnet array optimization |
| `cylindrical_optimization_usage.ipynb` | Cylindrical octant optimization |
| `Generate_Precomputed_Field_Batched_analytical.ipynb` | Generate precomputed field (GPU analytical) |

## Quick Start

1. **Generate precomputed field** (if needed)
   - Run `Generate_Precomputed_Field_Batched_analytical.ipynb`
   - Adjust grid size and magnet parameters as needed

2. **Run optimization**
   - Elliptical: `ellipse_optimization_usage.ipynb`
   - Cylindrical: `cylindrical_optimization_usage.ipynb`

## File Structure

```
Magnet_design/
│
├── Optimization Packages
│   ├── ellipse_opt/                    # Elliptical geometry
│   │   ├── cost.py                     # Cost function
│   │   ├── perturbation.py             # SA perturbation (3-phase hybrid)
│   │   ├── calibration.py              # Ben-Ameur calibration
│   │   └── generator.py                # Magnet placement
│   │
│   └── cylindrical_opt/                # Cylindrical geometry
│       ├── cost.py
│       ├── perturbation.py
│       ├── calibration.py
│       └── generator.py
│
├── Field Calculation
│   ├── analytical_magnet_gpu.py        # GPU vectorized (CuPy)
│   └── analytical_field.py             # CPU version (NumPy)
│
├── Usage Notebooks
│   ├── ellipse_optimization_usage.ipynb
│   ├── cylindrical_optimization_usage.ipynb
│   └── Generate_Precomputed_Field_Batched_analytical.ipynb
│
└── Data
    └── Precalc_field_*.npz             # Precomputed B-field lookup
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA
- CuPy, NumPy, Matplotlib

## Key Features

- 3-phase hybrid perturbation (exploration → transition → exploitation)
- S-curve temperature scaling
- Auto-calibration (Ben-Ameur method)
- Bolt hole avoidance constraints
- GPU-accelerated field calculation (matches magpylib exactly)

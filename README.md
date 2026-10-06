# Magnet Array Optimization

GPU-accelerated simulated annealing for permanent magnet array design.

## Workflow

```
1. Generate Precomputed Field
   ├── precompute/Generate_Precomputed_Field_Batched_analytical.ipynb
   └── Output: Precalc_field_*.npz

2. Run Optimization
   ├── ellipse_optimization_usage.ipynb      (elliptical bore)
   └── cylindrical_optimization_usage.ipynb  (cylindrical octant)

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
| `field_calculator.py` | **Core** | GPU precompute lookup - loads Precalc_field_*.npz for fast field interpolation |
| `field_validation.py` | **Validation** | Magpylib-based validation (overlay comparison plots) |
| `analytical_magnet_gpu.py` | Field calculation | GPU analytical cuboid field (Engel-Herbert) |
| `analytical_field.py` | Field calculation | CPU analytical cuboid field |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `ellipse_optimization_usage.ipynb` | Elliptical magnet array optimization |
| `cylindrical_optimization_usage.ipynb` | Cylindrical octant optimization |
| `visualize_checkpoint.ipynb` | Load and visualize saved checkpoints |
| `precompute/Generate_Precomputed_Field_Batched_analytical.ipynb` | Generate precomputed field (GPU analytical) |

## Quick Start

1. **Generate precomputed field** (if needed)
   - Run `precompute/Generate_Precomputed_Field_Batched_analytical.ipynb`
   - Adjust grid size and magnet parameters as needed

2. **Run optimization**
   - Elliptical: `ellipse_optimization_usage.ipynb`
   - Cylindrical: `cylindrical_optimization_usage.ipynb`

3. **Continue optimization** (optional)
   - Load previous result: `np.load('*_design_*.npz')`
   - Use `q1_config` (ellipse) or `octant_config` (cylindrical) as initial config

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
├── Core Modules
│   ├── field_calculator.py             # GPU precompute lookup (loads Precalc_field_*.npz)
│   └── field_validation.py             # Magpylib-based validation (overlay comparison plots)
│
├── precompute/                         # Field generation
│   ├── Generate_Precomputed_Field_Batched_analytical.ipynb
│   ├── analytical_magnet_gpu.py        # GPU analytical (CuPy)
│   └── analytical_field.py             # CPU analytical (NumPy)
│
├── Example Usage Notebooks
│   ├── ellipse_optimization_usage.ipynb
│   ├── cylindrical_optimization_usage.ipynb
│   └── visualize_checkpoint.ipynb
│
└── Data
    └── Precalc_field_*.npz             # Precomputed B-field lookup
```

## Requirements

**Hardware:**
- NVIDIA GPU with CUDA support (for GPU-accelerated optimization)

**Software:**
- Python 3.8 or newer

**Python Packages:**
Install these packages before running:
```bash
pip install numpy cupy matplotlib magpylib plotly
```



## Key Features

- 3-phase hybrid perturbation (exploration → transition → exploitation)
- S-curve perturbation scaling
- Auto-calibration (Ben-Ameur method) — determines initial temperature and cooling rate from the cost function
- Bolt hole avoidance constraints
- GPU-accelerated field calculation

## Author

**Natnael Anjulo**  
**CaseMRI** | Department of Biomedical Engineering  
Case Western Reserve University  
2026

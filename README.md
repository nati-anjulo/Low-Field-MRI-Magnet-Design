# Magnet Array Optimization

GPU-accelerated simulated annealing for permanent magnet array design.

## Packages

| Package | Geometry | Description |
|---------|----------|-------------|
| `ellipse_opt` | Elliptical cylinder | User-defined elliptical bore (rx, ry, gap, z_max) |
| `cylindrical_opt` | Cylindrical octant | User-defined cylindrical bounds (r_min, r_max, z_min, z_max) |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `ellipse_optimization_usage.ipynb` | Run elliptical magnet array optimization with calibration and visualization |
| `cylindrical_optimization_usage.ipynb` | Run cylindrical octant magnet array optimization with calibration and visualization |
| `BenchMark/Generate_Precomputed_Field_Batched.ipynb` | Generate precomputed magnetic field lookup tables from magnet geometry |
| `BenchMark/magpylib_vs_precompute.ipynb` | Compare magpylib direct calculation vs precomputed field lookup performance |

## Quick Start

1. **Check dependencies**
   ```bash
   python check_dependencies.py
   ```

2. **Install if needed**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run optimization**
   - Elliptical: `ellipse_optimization_usage.ipynb`
   - Cylindrical: `cylindrical_optimization_usage.ipynb`

## Requirements

- Python
- NVIDIA GPU with CUDA
- Sufficient GPU memory for your configuration

## Key Features

- 3-phase hybrid perturbation (exploration → transition → exploitation)
- S-curve temperature scaling
- Auto-calibration (Ben-Ameur method)
- Bolt hole avoidance constraints
- Layer-aware overlap checking

## Precomputed Field Data

The `.npz` file contains precomputed magnetic field lookup tables required for optimization.

## File Structure

```
Magnet_design/
├── ellipse_opt/                           # Elliptical geometry package
├── cylindrical_opt/                       # Cylindrical geometry package
├── BenchMark/                             # Performance benchmarks
├── ellipse_optimization_usage.ipynb       # Elliptical optimization notebook
├── cylindrical_optimization_usage.ipynb   # Cylindrical optimization notebook
├── Precalc_field_*.npz                    # Precomputed B-field
├── requirements.txt                       # Python dependencies
├── check_dependencies.py                  # Dependency checker
└── INSTALL.txt                            # Detailed installation guide
```

## Output

- Optimized magnet configurations (positions + orientations)
- Field strength and homogeneity metrics
- Cost progression history
- 3D visualization of magnet array

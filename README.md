# Magnet Array Optimization

Optimize permanent magnet arrays using simulated annealing on GPU.

## How to Use

1. **Generate a precomputed field file** (one-time setup)
   ```
   Run: precompute/Generate_Precomputed_Field_Batched_analytical.ipynb
   Output: Precalc_field_*.npz
   ```

2. **Run optimization**
   - For elliptical bore: `ellipse_optimization_usage.ipynb`
   - For cylindrical bore: `cylindrical_optimization_usage.ipynb`

3. **View results**
   - Optimized positions saved to `*_design_*.npz`
   - 3D viewer shows magnet arrangement

## What's Inside

```
Magnet_design/
├── ellipse_opt/              # Elliptical geometry package
├── cylindrical_opt/          # Cylindrical geometry package
├── visualize_configuration.py # Magpylib 3D viewer (optional)
├── field_validation.py       # Cross-check results with magpylib
├── precompute/               # Generate field lookup tables
└── *.ipynb                   # Example notebooks
```

## 3D Viewer

Built into both packages:
```python
visualization.show_magnets_plotly(config, magnet_size)
```
Red face = North, Green face = South. Opens in browser if WebGL fails.

Magpylib viewer (if installed):
```python
import visualize_configuration as VS
VS.rad_halbach_Design_rad(magnet_size, config).show()
```

## Install

```bash
pip install -r requirements.txt
```

Needs an NVIDIA GPU with CUDA.

## Continue from Previous Run

Load a saved result and keep optimizing:
```python
prev = np.load('ellipse_design_20261007.npz')
initial_q1 = cp.asarray(prev['q1_config'])
# Then run optimization as usual
```

---
Natnael Anjulo  
CaseMRI, Case Western Reserve University

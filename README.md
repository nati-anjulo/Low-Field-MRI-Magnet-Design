# Magnet Array Optimization

Optimize permanent magnet arrays using simulated annealing on GPU.

## Why Simulated Annealing

The choice of optimizer matters as much as the speed of each evaluation. Navigating a rugged solution space requires covering the space broadly as well as quickly.

A perturbation that moves many magnets simultaneously changes the field everywhere, while a perturbation that moves only one parameter changes the field only slightly. How far each step moves a parameter through the design space shapes what the search can find.

Genetic algorithms have been common for magnet design. They explore the design space by recombining parameters of two parent configurations, and how far each new configuration moves from the parents depends on the operators. Those operators may not deliver the small single-parameter changes that homogeneity tuning needs late in a run.

Simulated annealing allows fine control over perturbation size - large steps early for exploration, small steps late for refinement.

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

3. **View results**
   - Saved to `*_design_*.npz`
   - 3D viewer shows magnet arrangement

## 3D Viewer

Built-in:
```python
visualization.show_magnets_plotly(config, magnet_size)
```

Magpylib (optional):
```python
import visualize_configuration as VS
VS.rad_halbach_Design_rad(magnet_size, config).show()
```

## Continue from Previous Run

```python
prev = np.load('ellipse_design_20261007.npz')
initial_q1 = cp.asarray(prev['q1_config'])
```

---
Natnael Anjulo  
CaseMRI, Case Western Reserve University

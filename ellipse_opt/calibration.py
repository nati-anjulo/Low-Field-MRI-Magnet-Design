"""
SA Calibration functions.
"""
import numpy as np
import cupy as cp
from scipy.optimize import brentq

from .cost import calculate_cost_half_ellipse
from .perturbation import Hybrid_perturbation_half_ellipse

# ======================================================================
# SA CALIBRATION - RIGOROUS APPROACH
# ======================================================================
# White (1984) "Concepts of Scale in Simulated Annealing"
# Ben-Ameur (2004) "Computing the Initial Temperature of Simulated Annealing"

import numpy as np
import cupy as cp
from scipy.optimize import brentq


def _random_walk_sample(
    start_config, start_cost, perturb_temp,
    z_min, z_max, rx_inner, ry_inner, gap,
    magnet_size, sample_points, weights, step_divisors,
    Precalc_Bxyz_value, Ref_grid, step_size,
    cooling_rate, n_iterations,
    n_samples,
):
    """
    True random walk — accept every feasible move so the walk explores.
    Returns (uphill_deltas, visited_configs).
    perturb_temp is (current_temp, initial_temp) passed to the perturbation.
    """
    cur_T, init_T = perturb_temp
    current, current_cost = start_config, start_cost
    uphill, visited = [], [start_config]

    for _ in range(n_samples):
        perturbed = Hybrid_perturbation_half_ellipse(
            current, z_min, z_max, rx_inner, ry_inner, gap,
            magnet_size, cur_T, init_T, cooling_rate, n_iterations,
            step_divisors
        )
        if perturbed is None:
            continue

        new_cost = calculate_cost_half_ellipse(
            perturbed, z_min, z_max, rx_inner, ry_inner, gap,
            magnet_size, sample_points, weights,
            Precalc_Bxyz_value, Ref_grid, step_size
        )
        if not cp.isfinite(new_cost):
            continue

        new_cost = float(new_cost)
        delta = new_cost - current_cost
        if delta > 0:
            uphill.append(delta)

        current, current_cost = perturbed, new_cost   # accept everything
        visited.append(perturbed)

    return np.asarray(uphill), visited


def _find_T_for_chi(target_chi, deltas, lo=None, hi=None):
    """Solve mean(exp(-dE/T)) == target_chi."""
    if deltas.size < 30:
        raise ValueError(f"only {deltas.size} uphill samples — need >=30")

    def chi(T):
        with np.errstate(under="ignore", over="ignore"):
            return float(np.mean(np.exp(-deltas / T)))

    lo = lo if lo is not None else deltas.min() / 1e3
    hi = hi if hi is not None else deltas.max() * 1e3

    n = 0
    while chi(lo) > target_chi:
        lo /= 10.0
        n += 1
        if n > 50:
            raise RuntimeError("cannot bracket below; check deltas")
    n = 0
    while chi(hi) < target_chi:
        hi *= 10.0
        n += 1
        if n > 50:
            raise RuntimeError("cannot bracket above; check deltas")

    return brentq(lambda T: chi(T) - target_chi, lo, hi, xtol=1e-12, rtol=1e-10)


def calibrate_sa_rigorous(
    initial_config,
    z_min, z_max, rx_inner, ry_inner, gap,
    magnet_size, sample_points, weights, step_divisors,
    Precalc_Bxyz_value, Ref_grid, step_size,
    cooling_rate, n_iterations,
    n_samples=200,
    chi_0=0.9,
    chi_f=0.01,
    moves_per_level=1,
    hot_temp=1e4,
    cold_ratio=0.01,
):
    print("=" * 60)
    print("SA RIGOROUS CALIBRATION")
    print("=" * 60)
    
    base_cost = float(calculate_cost_half_ellipse(
        initial_config, z_min, z_max, rx_inner, ry_inner, gap,
        magnet_size, sample_points, weights,
        Precalc_Bxyz_value, Ref_grid, step_size
    ))
    print(f"\nBase cost: {base_cost:,.2f}")

    # STEP 2: Random walk sampling (hot)
    print(f"\nStep 2: Random-walk sampling {n_samples} moves (hot)...")
    hot_deltas, visited = _random_walk_sample(
        initial_config, base_cost, (hot_temp, hot_temp),
        z_min, z_max, rx_inner, ry_inner, gap,
        magnet_size, sample_points, weights, step_divisors,
        Precalc_Bxyz_value, Ref_grid, step_size,
        cooling_rate, n_iterations, n_samples,
    )
    print(f"  {len(hot_deltas)} uphill moves from {len(visited)} configs")
    
    print(f"  ΔE  min {hot_deltas.min():,.3g}   median {np.median(hot_deltas):,.3g}"
          f"   mean {np.mean(hot_deltas):,.3g}   max {hot_deltas.max():,.3g}")
    if np.mean(hot_deltas) > 5 * np.median(hot_deltas):
        print("  ⚠️  heavy-tailed ΔE (mean >> median)")

    # STEP 2b: Sample at cold step size for T_f
    print(f"\nStep 2b: Sampling at cold step size (ratio {cold_ratio})...")
    cold_deltas, _ = _random_walk_sample(
        initial_config, base_cost, (hot_temp * cold_ratio, hot_temp),
        z_min, z_max, rx_inner, ry_inner, gap,
        magnet_size, sample_points, weights, step_divisors,
        Precalc_Bxyz_value, Ref_grid, step_size,
        cooling_rate, n_iterations, n_samples // 2,
    )
    print(f"  {len(cold_deltas)} uphill moves,  median ΔE {np.median(cold_deltas):,.3g}")

    # STEP 1: Per-term variation (std-based)
    print("\nStep 1: Per-term ΔE variation across sampled configs...")
    probe = visited[:: max(1, len(visited) // 60)]
    term_contributions = {}
    for term in weights:
        w_test = {k: (v if k == term else 0) for k, v in weights.items()}
        vals = []
        for cfg in probe:
            c = calculate_cost_half_ellipse(
                cfg, z_min, z_max, rx_inner, ry_inner, gap,
                magnet_size, sample_points, w_test,
                Precalc_Bxyz_value, Ref_grid, step_size
            )
            if cp.isfinite(c):
                vals.append(float(c))
        term_contributions[term] = float(np.std(vals)) if len(vals) > 1 else 0.0
        print(f"  {term:20s} std {term_contributions[term]:,.3g}")

    nz = [v for v in term_contributions.values() if v > 0]
    if len(nz) > 1:
        ratio = max(nz) / min(nz)
        if ratio > 1000:
            print(f"  ⚠️  imbalance {ratio:,.0f}× — the small term is invisible to SA")
        else:
            print(f"  ✓ term balance OK ({ratio:,.1f}×)")

    # STEP 3: Solve for T₀ and T_f
    print(f"\nStep 3: Solving χ(T₀)={chi_0}, χ(T_f)={chi_f} ...")
    T0 = _find_T_for_chi(chi_0, hot_deltas)
    Tf = _find_T_for_chi(chi_f, cold_deltas)

    def chi_at(T, d):
        with np.errstate(under="ignore"):
            return float(np.mean(np.exp(-d / T)))

    print(f"  T₀  = {T0:,.4g}   (verify χ = {chi_at(T0, hot_deltas)*100:.1f}%)")
    print(f"  T_f = {Tf:,.4g}   (verify χ = {chi_at(Tf, cold_deltas)*100:.2f}%)")

    if Tf >= T0:
        print("  ⚠️  T_f >= T₀ — cold ΔE not smaller than hot; check step scaling")

    med = np.median(hot_deltas)
    print(f"  sanity: T₀ / median(ΔE) = {T0/med:.2f}   (want ~0.1–10)")

    # STEP 4: Calculate cooling rate
    K = max(1, n_iterations // moves_per_level)
    alpha = (Tf / T0) ** (1.0 / K)
    print(f"\nStep 4: cooling")
    print(f"  n_iterations {n_iterations:,}  /  moves_per_level {moves_per_level}"
          f"  →  K = {K:,} cooling events")
    print(f"  α = {alpha:.8f}")
    print(f"  check: T₀·α^K = {T0 * alpha**K:,.4g}  (should equal T_f)")

    print("\n" + "=" * 60)
    print(f"initial_temp = {T0:,.6g}")
    print(f"cooling_rate = {alpha:.8f}")
    print("=" * 60)

    diagnostics = {
        "hot_deltas": hot_deltas, "cold_deltas": cold_deltas,
        "term_contributions": term_contributions,
        "T0": T0, "Tf": Tf, "alpha": alpha, "K": K,
        "chi_0": chi_0, "chi_f": chi_f,
    }
    return T0, Tf, alpha, diagnostics


def diagnose_sa_calibration(initial_config, z_min, z_max, rx_inner, ry_inner, gap,
                            magnet_size, sample_points, weights,
                            Precalc_Bxyz_value, Ref_grid, step_size,
                            initial_temp, step_divisors, n_samples=50):
    """
    Measure actual cost deltas and suggest SA parameter fixes.
    """
    print("="*60)
    print("SA CALIBRATION DIAGNOSTIC")
    print("="*60)
    
    # Calculate base cost
    base_cost = calculate_cost_half_ellipse(
        initial_config, z_min, z_max, rx_inner, ry_inner, gap,
        magnet_size, sample_points, weights,
        Precalc_Bxyz_value, Ref_grid, step_size
    )
    print(f"Initial cost: {float(base_cost):,.2f}")
    
    # Measure cost deltas from perturbations
    deltas = []
    for _ in range(n_samples):
        perturbed = Hybrid_perturbation_half_ellipse(
            initial_config, z_min, z_max, rx_inner, ry_inner, gap,
            magnet_size, initial_temp, initial_temp, 0.99995, 1000,
            step_divisors
        )
        if perturbed is None:
            continue
        new_cost = calculate_cost_half_ellipse(
            perturbed, z_min, z_max, rx_inner, ry_inner, gap,
            magnet_size, sample_points, weights,
            Precalc_Bxyz_value, Ref_grid, step_size
        )
        if not cp.isinf(new_cost):
            deltas.append(abs(float(new_cost - base_cost)))
    
    if not deltas:
        print("ERROR: No valid perturbations! Check geometry constraints.")
        return
    
    avg_delta = np.mean(deltas)
    print(f"\nMeasured from {len(deltas)} perturbations:")
    print(f"  Average cost delta: {avg_delta:,.2f}")
    print(f"  Min delta: {min(deltas):,.2f}")
    print(f"  Max delta: {max(deltas):,.2f}")
    
    # Calculate acceptance probability with current settings
    current_accept = np.exp(-avg_delta / initial_temp)
    print(f"\nCurrent settings:")
    print(f"  initial_temp = {initial_temp}")
    print(f"  Accept probability = {current_accept*100:.4f}%")
    
    if current_accept < 0.01:
        print(f"  STATUS: ❌ TOO LOW - SA is doing greedy descent, not exploring!")
    elif current_accept < 0.3:
        print(f"  STATUS: ⚠️  LOW - Limited exploration")
    elif current_accept < 0.7:
        print(f"  STATUS: ✓ GOOD - Proper SA exploration")
    else:
        print(f"  STATUS: ⚠️  HIGH - Too much random walk, slow convergence")
    
    # Calculate recommended temperature for 50% acceptance
    recommended_temp = avg_delta / 0.693  # ln(2) = 0.693
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"\nOption 1: Increase temperature (keep weights)")
    print(f"  initial_temp = {recommended_temp:,.0f}  # for 50% acceptance")
    
    weight_scale = initial_temp * 0.693 / avg_delta
    print(f"\nOption 2: Scale weights (keep temp={initial_temp})")
    print(f"  Multiply all weights by {weight_scale:.4f}")
    print(f"  field_strength = {weights['field_strength'] * weight_scale:,.0f}")
    print(f"  uniformity = {weights['uniformity'] * weight_scale:,.0f}")
    print(f"  (Keep barrier weights unchanged at {weights['z_barrier']})")
    
    print(f"\n" + "="*60)
    return {'avg_delta': avg_delta, 'recommended_temp': recommended_temp, 'weight_scale': weight_scale}

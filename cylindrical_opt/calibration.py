"""
SA Calibration functions - STRICT COPY from original notebook.
"""
import numpy as np
import cupy as cp
from scipy.optimize import brentq

from .cost import calculate_cost_test
from .perturbation import Hybrid_perturbation_vectorized_with_bolt
from .generator import reflect_to_octants

# ======================================================================
# SA CALIBRATION - RIGOROUS APPROACH (Cylindrical Octant)
# ======================================================================


def _random_walk_sample_cyl(
    start_config, start_cost, perturb_temp,
    z_min, z_max, r_min, r_max,
    magnet_size, sample_points, weights, step_divisors,
    Precalc_Bxyz_value, Ref_grid, step_size,
    cooling_rate, n_iterations,
    n_samples,
):
    """Random walk for cylindrical octant."""
    cur_T, init_T = perturb_temp
    current, current_cost = start_config, start_cost
    uphill, visited = [], [start_config]

    for _ in range(n_samples):
        perturbed = Hybrid_perturbation_vectorized_with_bolt(
            current, z_min, z_max, r_min, r_max,
            magnet_size, cur_T, init_T, cooling_rate, n_iterations,
            step_divisors
        )
        if perturbed is None:
            continue

        new_cost = calculate_cost_test(
            reflect_to_octants(perturbed), z_min, z_max, r_min, r_max,
            magnet_size, sample_points, weights,
            Precalc_Bxyz_value, Ref_grid, step_size
        )
        if not cp.isfinite(new_cost):
            continue

        new_cost = float(new_cost)
        delta = new_cost - current_cost
        if delta > 0:
            uphill.append(delta)

        current, current_cost = perturbed, new_cost
        visited.append(perturbed)

    return np.asarray(uphill), visited


def _find_T_for_chi_cyl(target_chi, deltas, lo=None, hi=None):
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


def calibrate_sa_cylindrical(
    initial_config,
    z_min, z_max, r_min, r_max,
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
    """Rigorous SA calibration for cylindrical octant."""
    print("=" * 60)
    print("SA RIGOROUS CALIBRATION (Cylindrical Octant)")
    print("=" * 60)
    
    base_cost = float(calculate_cost_test(
        reflect_to_octants(initial_config), z_min, z_max, r_min, r_max,
        magnet_size, sample_points, weights,
        Precalc_Bxyz_value, Ref_grid, step_size
    ))
    print(f"\nBase cost: {base_cost:,.2f}")

    print(f"\nStep 2: Random-walk sampling {n_samples} moves (hot)...")
    hot_deltas, visited = _random_walk_sample_cyl(
        initial_config, base_cost, (hot_temp, hot_temp),
        z_min, z_max, r_min, r_max,
        magnet_size, sample_points, weights, step_divisors,
        Precalc_Bxyz_value, Ref_grid, step_size,
        cooling_rate, n_iterations, n_samples,
    )
    print(f"  {len(hot_deltas)} uphill moves from {len(visited)} configs")
    
    print(f"  ΔE  min {hot_deltas.min():,.3g}   median {np.median(hot_deltas):,.3g}"
          f"   mean {np.mean(hot_deltas):,.3g}   max {hot_deltas.max():,.3g}")

    print(f"\nStep 2b: Sampling at cold step size (ratio {cold_ratio})...")
    cold_deltas, _ = _random_walk_sample_cyl(
        initial_config, base_cost, (hot_temp * cold_ratio, hot_temp),
        z_min, z_max, r_min, r_max,
        magnet_size, sample_points, weights, step_divisors,
        Precalc_Bxyz_value, Ref_grid, step_size,
        cooling_rate, n_iterations, n_samples // 2,
    )
    print(f"  {len(cold_deltas)} uphill moves,  median ΔE {np.median(cold_deltas):,.3g}")

    print(f"\nStep 3: Solving χ(T₀)={chi_0}, χ(T_f)={chi_f} ...")
    T0 = _find_T_for_chi_cyl(chi_0, hot_deltas)
    Tf = _find_T_for_chi_cyl(chi_f, cold_deltas)

    def chi_at(T, d):
        with np.errstate(under="ignore"):
            return float(np.mean(np.exp(-d / T)))

    print(f"  T₀  = {T0:,.4g}   (verify χ = {chi_at(T0, hot_deltas)*100:.1f}%)")
    print(f"  T_f = {Tf:,.4g}   (verify χ = {chi_at(Tf, cold_deltas)*100:.2f}%)")

    K = max(1, n_iterations // moves_per_level)
    alpha = (Tf / T0) ** (1.0 / K)
    print(f"\nStep 4: cooling")
    print(f"  n_iterations {n_iterations:,}  /  moves_per_level {moves_per_level}"
          f"  →  K = {K:,} cooling events")
    print(f"  α = {alpha:.8f}")

    print("\n" + "=" * 60)
    print(f"initial_temp = {T0:,.6g}")
    print(f"cooling_rate = {alpha:.8f}")
    print("=" * 60)

    diagnostics = {
        "hot_deltas": hot_deltas, "cold_deltas": cold_deltas,
        "T0": T0, "Tf": Tf, "alpha": alpha, "K": K,
    }
    return T0, Tf, alpha, diagnostics
# ======================================================================
# SA CALIBRATION DIAGNOSTIC + AUTO-APPLY
# ======================================================================

def diagnose_sa_calibration_cylindrical(initial_config, z_min, z_max, r_min, r_max,
                                        magnet_size, sample_points, weights,
                                        Precalc_Bxyz_value, Ref_grid, step_size,
                                        initial_temp, step_divisors, n_samples=50):
    """
    Measure actual cost deltas and auto-apply recommended temperature.
    """
    print("="*60)
    print("SA CALIBRATION DIAGNOSTIC")
    print("="*60)
    
    # Calculate base cost
    base_cost = calculate_cost_test(
        reflect_to_octants(initial_config), z_min, z_max, r_min, r_max,
        magnet_size, sample_points, weights,
        Precalc_Bxyz_value, Ref_grid, step_size
    )
    print(f"Initial cost: {float(base_cost):,.2f}")
    
    # Measure cost deltas from perturbations
    deltas = []
    for _ in range(n_samples):
        perturbed = perturb_configuration_rad_vectorized_with_bolt(
            initial_config, z_min, z_max, r_min, r_max, magnet_size,
            1e6, 1e6, step_divisors, clearance=0.0
        )
        if perturbed is None:
            continue
        new_cost = calculate_cost_test(
            reflect_to_octants(perturbed), z_min, z_max, r_min, r_max,
            magnet_size, sample_points, weights,
            Precalc_Bxyz_value, Ref_grid, step_size
        )
        if not cp.isinf(new_cost):
            deltas.append(abs(float(new_cost - base_cost)))
    
    if not deltas:
        print("ERROR: No valid perturbations!")
        return None
    
    avg_delta = np.mean(deltas)
    print(f"\nMeasured from {len(deltas)} perturbations:")
    print(f"  Average cost delta: {avg_delta:,.2f}")
    
    # Calculate acceptance probability
    current_accept = np.exp(-avg_delta / initial_temp)
    print(f"\nCurrent settings:")
    print(f"  initial_temp = {initial_temp:,.0f}")
    print(f"  Accept probability = {current_accept*100:.2f}%")
    
    if current_accept < 0.3:
        print(f"  STATUS: LOW - Limited exploration")
    elif current_accept < 0.7:
        print(f"  STATUS: GOOD - Proper SA exploration")
    else:
        print(f"  STATUS: HIGH - Too much random walk")
    
    # Calculate recommended temperature for 50% acceptance
    recommended_temp = avg_delta / 0.693  # ln(2) = 0.693
    print(f"\nRecommended temp for 50% acceptance: {recommended_temp:,.0f}")
    
    return {'avg_delta': avg_delta, 'recommended_temp': recommended_temp}

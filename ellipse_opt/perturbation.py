"""
Perturbation functions - STRICT COPY from original notebook.
"""
import math
import cupy as cp
from .geometry import ellipse_r_gpu, get_bolt_positions_ellipse_gpu
from . import field_calculator as GRFC

import math

def perturb_transition_half_ellipse(config, z_min, z_max, rx_inner, ry_inner, gap,
                                              magnet_size, temperature, n_perturb,
                                              step_divisors=None, clearance=0.0, max_attempts=15):
    """
    TRANSITION PHASE: perturbs n_perturb randomly chosen magnets, all 4 params each.
    Half-ellipse only. Uses raw temperature for perturbation magnitude.
    
    Boundary handling:
    - theta, phi: WRAP (natural 2π periodicity)
    - r, z: REFLECT (physical boundaries, no teleportation)
    """
    a_in, b_in = rx_inner, ry_inner
    a_out, b_out = rx_inner + gap, ry_inner + gap
    half_diag = magnet_size * 1.8 / 2

    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 36, 'phi': 15}

    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        gap / step_divisors['r'],
        (cp.pi) / step_divisors['theta'],
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)
    
    # Precompute upper triangular mask (used in overlap checks)
    upper_tri_precomputed = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)
    n_perturb = int(max(1, min(n_perturb, n_magnets)))

    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    # Layer discretization (z >= 0 only)
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_xy = min_dist / 2  # For 2-way z reflection
    axis_margin_z = sheet_thickness / 2  # First layer center offset
    
    # Theta bounds: ensure x,y >= axis_margin when reflected
    # At outer radius, theta must keep both x,y above margin
    r_outer_max = a_out + gap  # Maximum possible radius
    min_theta = float(cp.arcsin(axis_margin_xy / r_outer_max))  # ~1.8° at r=185mm
    max_theta = cp.pi - 0.01  # Full circle for halfz

    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)

    # Bolt positions
    bolt_number = 10
    bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(a_in + gap/2, b_in + gap/2, bolt_number)
    bolt_diameter = 0.0075  # 7.5 mm bolt
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2

    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)

    for _ in range(max_attempts):
        new_config = config.copy()

        idx = cp.random.choice(n_magnets, n_perturb, replace=False)

        perturbations = cp.random.uniform(-1, 1, (n_perturb, 4))
        new_config[idx] += perturbations * max_steps * temperature

        # z -> REFLECT within [axis_margin_z, z_max], then snap to discrete layers
        z_range = z_max - axis_margin_z
        normalized_z = new_config[idx, 0] - axis_margin_z
        in_double_z = normalized_z % (2 * z_range)
        reflected_z = cp.where(in_double_z > z_range, 2 * z_range - in_double_z, in_double_z)
        new_config[idx, 0] = axis_margin_z + reflected_z
        # Snap to discrete layers
        layer_indices = cp.round((new_config[idx, 0] - axis_margin_z) / sheet_thickness)
        layer_indices = cp.clip(layer_indices, min_layer_idx, max_layer_idx)
        new_config[idx, 0] = axis_margin_z + layer_indices * sheet_thickness

        # theta -> WRAP to [0, 2*pi] (natural periodicity)
        # theta -> REFLECT within [0, π] (half-ellipse)
        theta_val = new_config[idx, 2] % (2 * cp.pi)  # First wrap to [0, 2π]
        # Fold [0, 2π] into [0, π] using reflection
        new_config[idx, 2] = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
        theta = new_config[idx, 2]
        r_inner = ellipse_r_gpu(theta, a_in, b_in) + half_diag
        r_outer = ellipse_r_gpu(theta, a_out, b_out) - half_diag
        
        # r -> REFLECT within [r_inner, r_outer] (physical boundary)
        r_range = r_outer - r_inner
        normalized_r = new_config[idx, 1] - r_inner
        in_double_r = normalized_r % (2 * r_range)
        reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
        new_config[idx, 1] = r_inner + reflected_r

        # phi -> WRAP (natural periodicity)
        new_config[idx, 3] %= (2 * cp.pi)

        positions, _ = GRFC.input_converter_vectorized(new_config)
        x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]

        # Half-ellipse check
        # Check y margin (for x-axis reflection) and z margin
        y = positions[:, 1]
        if not (cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):  # Z margin only
            continue

        # Bolt constraints
        bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                  (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
        if cp.any(bolt_distances_squared < min_bolt_dist_squared):
            continue

        # Layer-aware overlap check - FULL XY (no quadrant reflection)
        # Magnets already in full XY space, just check same-layer distances
        # Reflect about x-axis for collision check (2 halves)
        pos_top_xy = positions[:, :2]  # y > 0
        pos_bot_xy = pos_top_xy * cp.array([1, -1])  # y < 0 (x-axis mirror)
        pos_xy_all = cp.vstack([pos_top_xy, pos_bot_xy])
        z_all = cp.tile(z_pos, 2)
        
        n_all = len(pos_xy_all)
        z_diff = cp.abs(z_all.reshape(-1, 1) - z_all.reshape(1, -1))
        same_layer_mask = z_diff < 1e-6
        upper_tri_all = upper_tri_precomputed if n_all == n_magnets else cp.triu(cp.ones((n_all, n_all), dtype=bool), k=1)
        check_mask = same_layer_mask & upper_tri_all

        # OPTIMIZED: Use sparse indices instead of full n² matrix
        i_idx, j_idx = cp.where(check_mask)
        if len(i_idx) == 0:
            return new_config
        diff = pos_xy_all[i_idx] - pos_xy_all[j_idx]
        sq_dist_sparse = cp.sum(diff**2, axis=1)
        if cp.all(sq_dist_sparse >= min_dist_squared):
            return new_config

    return None


# ======================================================================
# HYBRID DISPATCHER: 3 phases
# ======================================================================


def perturb_half_ellipse(config, z_min, z_max, rx_inner, ry_inner, gap, magnet_size,
                                   temperature, step_divisors=None,
                                   clearance=0.0, max_attempts=15):
    """
    EXPLOITATION PHASE: perturbs ONE magnet, ONE parameter per attempt.
    Half-ellipse only. Uses raw temperature for perturbation magnitude.
    
    Boundary handling:
    - theta, phi: WRAP (natural 2π periodicity)
    - r, z: REFLECT (physical boundaries, no teleportation)
    """
    a_in, b_in = rx_inner, ry_inner
    a_out, b_out = rx_inner + gap, ry_inner + gap
    half_diag = magnet_size * 1.8 / 2

    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 36, 'phi': 15}

    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        gap / step_divisors['r'],
        (cp.pi) / step_divisors['theta'],  # Only half-ellipse
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)

    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    # Layer discretization (z >= 0 only)
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_xy = min_dist / 2  # For 2-way z reflection
    axis_margin_z = sheet_thickness / 2  # First layer center offset
    
    # Theta bounds: ensure x,y >= axis_margin when reflected
    # At outer radius, theta must keep both x,y above margin
    r_outer_max = a_out + gap  # Maximum possible radius
    min_theta = float(cp.arcsin(axis_margin_xy / r_outer_max))  # ~1.8° at r=185mm
    max_theta = cp.pi - 0.01  # Full circle for halfz

    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)

    # Bolt positions on ellipse
    bolt_number = 10
    bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(a_in + gap/2, b_in + gap/2, bolt_number)
    bolt_diameter = 0.0075  # 7.5 mm bolt
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2

    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)

    for _ in range(max_attempts):
        new_config = config.copy()

        magnet_idx = int(cp.random.randint(0, n_magnets))
        param_idx = int(cp.random.randint(0, 4))

        perturbation = cp.random.uniform(-1, 1) * max_steps[param_idx] * temperature
        new_config[magnet_idx, param_idx] += perturbation

        if param_idx == 0:  # z - discrete single-layer hop (positive only)
            current_layer = cp.round((new_config[magnet_idx, 0] - axis_margin_z) / sheet_thickness)
            layer_jump = cp.random.choice(cp.array([-1, 0, 1]), size=())
            new_layer = cp.clip(current_layer + layer_jump, min_layer_idx, max_layer_idx)
            new_config[magnet_idx, 0] = axis_margin_z + new_layer * sheet_thickness

        elif param_idx == 1:  # radius - REFLECT within elliptical bounds
            theta = new_config[magnet_idx, 2]
            r_inner = ellipse_r_gpu(theta, a_in, b_in) + half_diag
            r_outer = ellipse_r_gpu(theta, a_out, b_out) - half_diag
            r_range = r_outer - r_inner
            # REFLECT instead of WRAP
            normalized_r = new_config[magnet_idx, 1] - r_inner
            in_double_r = normalized_r % (2 * r_range)
            reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
            new_config[magnet_idx, 1] = r_inner + reflected_r

        elif param_idx == 2:  # theta - REFLECT within [0, π], then adjust r
            theta_val = new_config[magnet_idx, 2] % (2 * cp.pi)
            new_config[magnet_idx, 2] = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
            new_theta = new_config[magnet_idx, 2]
            r_inner = ellipse_r_gpu(new_theta, a_in, b_in) + half_diag
            r_outer = ellipse_r_gpu(new_theta, a_out, b_out) - half_diag
            r_range = r_outer - r_inner
            # REFLECT r after theta change
            normalized_r = new_config[magnet_idx, 1] - r_inner
            in_double_r = normalized_r % (2 * r_range)
            reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
            new_config[magnet_idx, 1] = r_inner + reflected_r

        else:  # phi - WRAP (natural periodicity)
            new_config[magnet_idx, 3] %= (2 * cp.pi)

        positions, _ = GRFC.input_converter_vectorized(new_config)
        x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]

        # Half-ellipse check
        # Check y margin (for x-axis reflection) and z margin
        y = positions[:, 1]
        if not (cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):  # Z margin only
            continue

        # Bolt constraints (check against half-ellipse bolts)
        bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                  (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
        if cp.any(bolt_distances_squared < min_bolt_dist_squared):
            continue

        # Layer-aware overlap check - FULL XY (no quadrant reflection)
        # Magnets already in full XY space, just check same-layer distances
        # Reflect about x-axis for collision check (2 halves)
        pos_top_xy = positions[:, :2]  # y > 0
        pos_bot_xy = pos_top_xy * cp.array([1, -1])  # y < 0 (x-axis mirror)
        pos_xy_all = cp.vstack([pos_top_xy, pos_bot_xy])
        z_all = cp.tile(z_pos, 2)
        
        n_all = len(pos_xy_all)
        z_diff = cp.abs(z_all.reshape(-1, 1) - z_all.reshape(1, -1))
        same_layer_mask = z_diff < 1e-6
        upper_tri_all = upper_tri_precomputed if n_all == n_magnets else cp.triu(cp.ones((n_all, n_all), dtype=bool), k=1)
        check_mask = same_layer_mask & upper_tri_all

        # OPTIMIZED: Use sparse indices instead of full n² matrix
        i_idx, j_idx = cp.where(check_mask)
        if len(i_idx) == 0:
            return new_config
        diff = pos_xy_all[i_idx] - pos_xy_all[j_idx]
        sq_dist_sparse = cp.sum(diff**2, axis=1)
        if cp.all(sq_dist_sparse >= min_dist_squared):
            return new_config

    return None



def transition_n_magnets(raw_temp, t_start, t_end, n_magnets):
    """
    Magnets to perturb during the transition phase: linear ramp n_magnets -> 1.
    Linear in ITERATION index, not temperature.
    """
    if raw_temp >= t_start:
        return n_magnets
    if raw_temp <= t_end:
        return 1

    p = (math.log(raw_temp) - math.log(t_start)) / (math.log(t_end) - math.log(t_start))
    n = n_magnets + p * (1 - n_magnets)
    return int(max(1, min(n_magnets, round(n))))


# ======================================================================
# LOW TEMP: 1 magnet, 1 param
# ======================================================================


def Hybrid_perturbation_half_ellipse(config, z_min, z_max, rx_inner, ry_inner, gap, magnet_size,
                                               temperature, initial_temp, cooling_rate, n_iterations,
                                               step_divisors=None, explore_exploit_pct=0.85,
                                               fine_tune_pct=0.92, clearance=0.0, max_attempts=15):
    """
    3-phase hybrid perturbation for half-ellipse ellipse (reflected to 2 halves (z-only)).

    HIGH TEMP:   perturbs all magnets simultaneously
    TRANSITION:  perturbs n magnets, n ramping from n_magnets down to 1
    LOW TEMP:    single-magnet, single-parameter perturbation

    Half-z (z >= 0): theta in [0, 2*pi] (full circle), z >= 0 only (full XY freedom).
    
    Boundary handling:
    - theta, phi: WRAP (natural 2π periodicity)
    - r, z: REFLECT (physical boundaries, no teleportation)
    """
    a_in, b_in = rx_inner, ry_inner
    a_out, b_out = rx_inner + gap, ry_inner + gap
    half_diag = magnet_size * 1.8 / 2

    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 36, 'phi': 15}

    # Save raw temperature for phase switching
    raw_temp = temperature
    
    # LINEAR SCALE derived from temperature (avoids exponential decay for perturbation)
    # temp = T0 * alpha^iter → iter = log(temp/T0) / log(alpha)
    if temperature > 0 and initial_temp > 0:
        estimated_iter = math.log(temperature / initial_temp) / math.log(cooling_rate)
        # S-curve: stays high during explore, smooth transition, low during fine-tune
        x = estimated_iter / n_iterations
        midpoint = (explore_exploit_pct + fine_tune_pct) / 2
        steepness = 20
        temperature = 1 / (1 + cp.exp(steepness * (x - midpoint)))
        temperature = float(max(0.01, min(1.0, temperature)))
    else:
        temperature = 0.01

    # Phase thresholds
    explore_exploit = initial_temp * cooling_rate ** int(explore_exploit_pct * n_iterations)
    fine_tune_start = initial_temp * cooling_rate ** int(fine_tune_pct * n_iterations)

    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        gap / step_divisors['r'],
        (cp.pi) / step_divisors['theta'],
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)

    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    # Layer discretization (z >= 0 only)
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_xy = min_dist / 2  # For 2-way z reflection
    axis_margin_z = sheet_thickness / 2  # First layer center offset
    
    # Theta bounds: ensure x,y >= axis_margin when reflected
    # At outer radius, theta must keep both x,y above margin
    r_outer_max = a_out + gap  # Maximum possible radius
    min_theta = float(cp.arcsin(axis_margin_xy / r_outer_max))  # ~1.8° at r=185mm
    max_theta = cp.pi - 0.01  # Full circle for halfz

    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)

    # Bolt positions
    bolt_number = 10
    bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(a_in + gap/2, b_in + gap/2, bolt_number)
    bolt_diameter = 0.0075  # 7.5 mm bolt
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2

    rel_temp = float(temperature)

    # Phase switch detection
    if not hasattr(Hybrid_perturbation_half_ellipse, '_phase'):
        Hybrid_perturbation_half_ellipse._phase = 'HIGH_TEMP'

    if Hybrid_perturbation_half_ellipse._phase == 'HIGH_TEMP' and raw_temp <= explore_exploit:
        print(f">>> SWITCH to TRANSITION at T={temperature:.2f} (rel_temp={rel_temp:.6f})")
        Hybrid_perturbation_half_ellipse._phase = 'TRANSITION'

    if Hybrid_perturbation_half_ellipse._phase == 'TRANSITION' and raw_temp <= fine_tune_start:
        print(f">>> SWITCH to LOW TEMP at T={temperature:.2f} (rel_temp={rel_temp:.6f})")
        Hybrid_perturbation_half_ellipse._phase = 'LOW_TEMP'

    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)

    # ---------------- HIGH TEMPERATURE ----------------
    if raw_temp > explore_exploit:
        for _ in range(max_attempts):
            new_config = config.copy()

            perturbations = cp.random.uniform(-1, 1, config.shape)
            new_config += perturbations * max_steps * temperature

            # z -> REFLECT within range, then snap to discrete layers
            z_range = z_max - axis_margin_z
            normalized_z = new_config[:, 0] - axis_margin_z
            in_double_z = normalized_z % (2 * z_range)
            reflected_z = cp.where(in_double_z > z_range, 2 * z_range - in_double_z, in_double_z)
            new_config[:, 0] = axis_margin_z + reflected_z
            # Snap to discrete layers
            layer_indices = cp.round((new_config[:, 0] - axis_margin_z) / sheet_thickness)
            layer_indices = cp.clip(layer_indices, min_layer_idx, max_layer_idx)
            new_config[:, 0] = axis_margin_z + layer_indices * sheet_thickness

            # theta -> WRAP to [0, 2*pi] (natural periodicity)
            # theta -> REFLECT within [0, π] (half-ellipse)
            theta_val = new_config[:, 2] % (2 * cp.pi)  # First wrap to [0, 2π]
            # Fold [0, 2π] into [0, π] using reflection
            new_config[:, 2] = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
            theta = new_config[:, 2]
            r_inner = ellipse_r_gpu(theta, a_in, b_in) + half_diag
            r_outer = ellipse_r_gpu(theta, a_out, b_out) - half_diag
            
            # r -> REFLECT within [r_inner, r_outer] (physical boundary)
            r_range = r_outer - r_inner
            normalized_r = new_config[:, 1] - r_inner
            in_double_r = normalized_r % (2 * r_range)
            reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
            new_config[:, 1] = r_inner + reflected_r

            # phi -> WRAP (natural periodicity)
            new_config[:, 3] %= (2 * cp.pi)

            positions, _ = GRFC.input_converter_vectorized(new_config)
            x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]

            # Half-ellipse check: y margin and z margin
            if not (cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):
                continue

            # Bolt constraints
            bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                      (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
            if cp.any(bolt_distances_squared < min_bolt_dist_squared):
                continue

            # Layer-aware overlap check
            axis_margin_z = sheet_thickness / 2
            z_diff = cp.abs(z_pos.reshape(-1, 1) - z_pos.reshape(1, -1))
            same_layer_mask = z_diff < 1e-6

            # Reflect about x-axis for overlap check (2 halves)
            # Half-ellipse uses x-axis mirror
            pos_xy_all = positions[:, :2]
            z_all = z_pos
            
            n_all = len(pos_xy_all)
            z_diff_all = cp.abs(z_all.reshape(-1, 1) - z_all.reshape(1, -1))
            same_layer_all = z_diff_all < 1e-6
            upper_tri_all = upper_tri_precomputed if n_all == n_magnets else cp.triu(cp.ones((n_all, n_all), dtype=bool), k=1)
            check_mask = same_layer_all & upper_tri_all

            # OPTIMIZED: Use sparse indices
            i_idx, j_idx = cp.where(check_mask)
            if len(i_idx) == 0:
                return new_config
            diff = pos_xy_all[i_idx] - pos_xy_all[j_idx]
            sq_dist_sparse = cp.sum(diff**2, axis=1)
            if cp.all(sq_dist_sparse >= min_dist_squared):
                return new_config

    # ---------------- TRANSITION ----------------
    if raw_temp > fine_tune_start:
        n_perturb = transition_n_magnets(raw_temp, explore_exploit, fine_tune_start, n_magnets)

        result = perturb_transition_half_ellipse(
            config, z_min, z_max, rx_inner, ry_inner, gap, magnet_size,
            temperature, n_perturb, step_divisors, clearance, max_attempts)
        if result is not None:
            return result

    # ---------------- LOW TEMPERATURE ----------------
    return perturb_half_ellipse(
        config, z_min, z_max, rx_inner, ry_inner, gap,
        magnet_size, temperature, step_divisors, clearance, max_attempts)

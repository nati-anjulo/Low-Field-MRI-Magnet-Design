"""
Perturbation functions - STRICT COPY from original notebook.
"""
import math
import cupy as cp
from . import field_calculator as GRFC

def perturb_configuration_rad_vectorized_with_bolt(config, z_min, z_max, r_min, r_max, magnet_size, 
                                        temperature, initial_temp, step_divisors=None, clearance=0.000, max_attempts=10):
    """
    LOW TEMP PHASE: Perturbs ONE magnet, ONE parameter per attempt.
    First octant only: theta in [0, pi/2], x >= margin, y >= margin, z >= margin.
    
    Uses REFLECTION instead of clipping for boundaries.
    """
    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 15, 'phi': 15}
    
    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        (r_max - r_min) / step_divisors['r'],
        (cp.pi / 2) / step_divisors['theta'],
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)
    
    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    half_diag = magnet_size * 1.8 / 2
    axis_margin_xy = min_dist / 2
    
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_z = sheet_thickness / 2
    
    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)
    
    # Bolt positions
    bolt_number = 8
    bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
    bolt_diameter = 0.0075  # 7.5mm bolt
    bolt_rad = r_min + magnet_size * 0.5
    bolt_x = bolt_rad * cp.cos(bolt_angle)
    bolt_y = bolt_rad * cp.sin(bolt_angle)
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2
    
    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)
    
    for _ in range(max_attempts):
        new_config = config.copy()
        
        magnet_idx = int(cp.random.randint(0, n_magnets))
        param_idx = int(cp.random.randint(0, 4))
        
        perturbation = cp.random.uniform(-1, 1) * max_steps[param_idx] * temperature
        new_config[magnet_idx, param_idx] += perturbation
        
        if param_idx == 0:  # z - REFLECT then snap to layers
            z_range = z_max - axis_margin_z
            normalized_z = new_config[magnet_idx, 0] - axis_margin_z
            in_double_z = normalized_z % (2 * z_range)
            reflected_z = cp.where(in_double_z > z_range, 2 * z_range - in_double_z, in_double_z)
            new_config[magnet_idx, 0] = axis_margin_z + reflected_z
            layer_idx = cp.round((new_config[magnet_idx, 0] - axis_margin_z) / sheet_thickness)
            layer_idx = cp.clip(layer_idx, min_layer_idx, max_layer_idx)
            new_config[magnet_idx, 0] = axis_margin_z + layer_idx * sheet_thickness

        elif param_idx == 1:  # r - REFLECT within [r_min, r_max]
            r_range = r_max - r_min
            normalized_r = new_config[magnet_idx, 1] - r_min
            in_double_r = normalized_r % (2 * r_range)
            reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
            new_config[magnet_idx, 1] = r_min + reflected_r

        elif param_idx == 2:  # theta - REFLECT within [0, pi/2]
            theta_val = new_config[magnet_idx, 2] % (2 * cp.pi)
            theta_val = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
            theta_val = cp.where(theta_val > cp.pi/2, cp.pi - theta_val, theta_val)
            new_config[magnet_idx, 2] = theta_val

        else:  # phi - WRAP
            new_config[magnet_idx, 3] %= (2 * cp.pi)
        
        positions, _ = GRFC.input_converter_vectorized(new_config)
        x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]
        
        if not (cp.all(x >= axis_margin_xy) and cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):
            continue
        
        # Bolt constraints
        bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                  (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
        if cp.any(bolt_distances_squared < min_bolt_dist_squared):
            continue
        
        # Layer-aware overlap check
        z_diff = cp.abs(z_pos.reshape(-1, 1) - z_pos.reshape(1, -1))
        same_layer_mask = z_diff < 1e-6
        
        pos_xy = positions[:, :2]
        sq_distances_xy = cp.sum((pos_xy.reshape(n_magnets, 1, 2) -
                                  pos_xy.reshape(1, n_magnets, 2))**2, axis=2)
        
        check_mask = same_layer_mask & upper_tri_mask
        
        if not cp.any(check_mask):
            return new_config
        if cp.all(sq_distances_xy[check_mask] >= min_dist_squared):
            return new_config
    
    return None
import math

def Hybrid_perturbation_vectorized_with_bolt(config, z_min, z_max, r_min, r_max, magnet_size, 
                                  temperature, initial_temp, cooling_rate, n_iterations, step_divisors=None, 
                                  explore_exploit_pct=0.65, fine_tune_pct=0.90, clearance=0.0, max_attempts=10):
    """
    Fully vectorized hybrid perturbation with S-CURVE temperature scaling.
    
    HIGH TEMP: Perturbs all magnets simultaneously
    TRANSITION: Perturbs n magnets, ramping down
    LOW TEMP: Single-magnet, single-parameter perturbation
    
    Boundary handling:
    - theta: REFLECT within [0, π/2] (first octant)
    - phi: WRAP (2π periodic)
    - r, z: REFLECT (physical boundaries)
    """
    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 15, 'phi': 15}
    
    # S-CURVE temperature scaling (matches half_ellipse notebook)
    # Stays high during exploration, smooth transition, low during fine-tuning
    if temperature > 0 and initial_temp > 0:
        estimated_iter = math.log(temperature / initial_temp) / math.log(cooling_rate)
        x = estimated_iter / n_iterations
        midpoint = (explore_exploit_pct + fine_tune_pct) / 2
        steepness = 20
        scaled_temp = 1 / (1 + cp.exp(steepness * (x - midpoint)))
        scaled_temp = float(max(0.01, min(1.0, scaled_temp)))
    else:
        scaled_temp = 0.01

    # Phase thresholds
    explore_exploit = cooling_rate ** int(explore_exploit_pct * n_iterations)
    fine_tune_start = cooling_rate ** int(fine_tune_pct * n_iterations)

    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        (r_max - r_min) / step_divisors['r'],
        (cp.pi / 2) / step_divisors['theta'],
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)
    
    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    half_diag = magnet_size * 1.8 / 2
    
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_xy = min_dist / 2
    axis_margin_z = sheet_thickness / 2
    
    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)
    
    # Bolt positions
    bolt_number = 8
    bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
    bolt_diameter = 0.0075  # 7.5mm bolt
    bolt_rad = r_min + magnet_size * 0.5
    bolt_x = bolt_rad * cp.cos(bolt_angle)
    bolt_y = bolt_rad * cp.sin(bolt_angle)
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2
    
    # Phase switch detection
    if not hasattr(Hybrid_perturbation_vectorized_with_bolt, '_phase'):
        Hybrid_perturbation_vectorized_with_bolt._phase = 'HIGH_TEMP'

    if Hybrid_perturbation_vectorized_with_bolt._phase == 'HIGH_TEMP' and temperature <= explore_exploit:
        print(f">>> SWITCH to TRANSITION at T={temperature:.2f}")
        Hybrid_perturbation_vectorized_with_bolt._phase = 'TRANSITION'

    if Hybrid_perturbation_vectorized_with_bolt._phase == 'TRANSITION' and temperature <= fine_tune_start:
        print(f">>> SWITCH to LOW TEMP at T={temperature:.2f}")
        Hybrid_perturbation_vectorized_with_bolt._phase = 'LOW_TEMP'

    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)

    # ---------------- HIGH TEMPERATURE ----------------
    if temperature > explore_exploit:
        for _ in range(max_attempts):
            new_config = config.copy()

            perturbations = cp.random.uniform(-1, 1, config.shape)
            new_config += perturbations * max_steps * scaled_temp

            # z -> REFLECT then snap to layers
            z_range = z_max - axis_margin_z
            normalized_z = new_config[:, 0] - axis_margin_z
            in_double_z = normalized_z % (2 * z_range)
            reflected_z = cp.where(in_double_z > z_range, 2 * z_range - in_double_z, in_double_z)
            new_config[:, 0] = axis_margin_z + reflected_z
            layer_indices = cp.round((new_config[:, 0] - axis_margin_z) / sheet_thickness)
            layer_indices = cp.clip(layer_indices, min_layer_idx, max_layer_idx)
            new_config[:, 0] = axis_margin_z + layer_indices * sheet_thickness

            # r -> REFLECT within [r_min, r_max]
            r_range = r_max - r_min
            normalized_r = new_config[:, 1] - r_min
            in_double_r = normalized_r % (2 * r_range)
            reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
            new_config[:, 1] = r_min + reflected_r

            # theta -> REFLECT within [0, pi/2] (first octant)
            theta_val = new_config[:, 2] % (2 * cp.pi)
            theta_val = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
            theta_val = cp.where(theta_val > cp.pi/2, cp.pi - theta_val, theta_val)
            new_config[:, 2] = theta_val

            # phi -> WRAP
            new_config[:, 3] %= (2 * cp.pi)

            positions, _ = GRFC.input_converter_vectorized(new_config)
            x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]

            if not (cp.all(x >= axis_margin_xy) and cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):
                continue

            # Bolt constraints
            bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                      (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
            if cp.any(bolt_distances_squared < min_bolt_dist_squared):
                continue

            # Layer-aware overlap check
            z_diff = cp.abs(z_pos.reshape(-1, 1) - z_pos.reshape(1, -1))
            same_layer_mask = z_diff < 1e-6

            pos_xy = positions[:, :2]
            sq_distances_xy = cp.sum((pos_xy.reshape(n_magnets, 1, 2) -
                                      pos_xy.reshape(1, n_magnets, 2))**2, axis=2)

            check_mask = same_layer_mask & upper_tri_mask

            if not cp.any(check_mask):
                return new_config
            if cp.all(sq_distances_xy[check_mask] >= min_dist_squared):
                return new_config

    # ---------------- TRANSITION ----------------
    if temperature > fine_tune_start:
        n_perturb = transition_n_magnets_cyl(temperature, explore_exploit, fine_tune_start, n_magnets)

        result = perturb_transition_cylindrical(
            config, z_min, z_max, r_min, r_max, magnet_size,
            scaled_temp, n_perturb, step_divisors, clearance, max_attempts)
        if result is not None:
            return result

    # ---------------- LOW TEMPERATURE ----------------
    return perturb_configuration_rad_vectorized_with_bolt(
        config, z_min, z_max, r_min, r_max, magnet_size,
        scaled_temp, initial_temp, step_divisors, clearance, max_attempts)


def transition_n_magnets_cyl(rel_temp, t_start, t_end, n_magnets):
    """Magnets to perturb during transition phase."""
    if rel_temp >= t_start:
        return n_magnets
    if rel_temp <= t_end:
        return 1
    p = (math.log(rel_temp) - math.log(t_start)) / (math.log(t_end) - math.log(t_start))
    n = n_magnets + p * (1 - n_magnets)
    return int(max(1, min(n_magnets, round(n))))


def perturb_transition_cylindrical(config, z_min, z_max, r_min, r_max, magnet_size,
                                   temperature, n_perturb, step_divisors=None,
                                   clearance=0.0, max_attempts=10):
    """TRANSITION PHASE: perturbs n_perturb magnets."""
    if step_divisors is None:
        step_divisors = {'z': 15, 'r': 15, 'theta': 15, 'phi': 15}

    max_steps = cp.array([
        (z_max - z_min) / step_divisors['z'],
        (r_max - r_min) / step_divisors['r'],
        (cp.pi / 2) / step_divisors['theta'],
        cp.pi / step_divisors['phi']
    ])
    n_magnets = len(config)
    n_perturb = int(max(1, min(n_perturb, n_magnets)))

    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_squared = min_dist**2
    half_diag = magnet_size * 1.8 / 2

    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_xy = min_dist / 2
    axis_margin_z = sheet_thickness / 2

    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)

    bolt_number = 8
    bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
    bolt_diameter = 0.0075  # 7.5mm bolt
    bolt_rad = r_min + magnet_size * 0.5
    bolt_x = bolt_rad * cp.cos(bolt_angle)
    bolt_y = bolt_rad * cp.sin(bolt_angle)
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2

    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)

    for _ in range(max_attempts):
        new_config = config.copy()
        idx = cp.random.permutation(n_magnets)[:n_perturb]

        perturbations = cp.random.uniform(-1, 1, (n_perturb, 4))
        new_config[idx] += perturbations * max_steps * temperature

        # z -> REFLECT then snap
        z_range = z_max - axis_margin_z
        normalized_z = new_config[idx, 0] - axis_margin_z
        in_double_z = normalized_z % (2 * z_range)
        reflected_z = cp.where(in_double_z > z_range, 2 * z_range - in_double_z, in_double_z)
        new_config[idx, 0] = axis_margin_z + reflected_z
        layer_indices = cp.round((new_config[idx, 0] - axis_margin_z) / sheet_thickness)
        layer_indices = cp.clip(layer_indices, min_layer_idx, max_layer_idx)
        new_config[idx, 0] = axis_margin_z + layer_indices * sheet_thickness

        # r -> REFLECT
        r_range = r_max - r_min
        normalized_r = new_config[idx, 1] - r_min
        in_double_r = normalized_r % (2 * r_range)
        reflected_r = cp.where(in_double_r > r_range, 2 * r_range - in_double_r, in_double_r)
        new_config[idx, 1] = r_min + reflected_r

        # theta -> REFLECT within [0, pi/2]
        theta_val = new_config[idx, 2] % (2 * cp.pi)
        theta_val = cp.where(theta_val > cp.pi, 2*cp.pi - theta_val, theta_val)
        theta_val = cp.where(theta_val > cp.pi/2, cp.pi - theta_val, theta_val)
        new_config[idx, 2] = theta_val

        # phi -> WRAP
        new_config[idx, 3] %= (2 * cp.pi)

        positions, _ = GRFC.input_converter_vectorized(new_config)
        x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]

        if not (cp.all(x >= axis_margin_xy) and cp.all(y >= axis_margin_xy) and cp.all(z_pos >= axis_margin_z)):
            continue

        bolt_distances_squared = ((x.reshape(-1, 1) - bolt_x.reshape(1, -1))**2 +
                                  (y.reshape(-1, 1) - bolt_y.reshape(1, -1))**2)
        if cp.any(bolt_distances_squared < min_bolt_dist_squared):
            continue

        z_diff = cp.abs(z_pos.reshape(-1, 1) - z_pos.reshape(1, -1))
        same_layer_mask = z_diff < 1e-6

        pos_xy = positions[:, :2]
        sq_distances_xy = cp.sum((pos_xy.reshape(n_magnets, 1, 2) -
                                  pos_xy.reshape(1, n_magnets, 2))**2, axis=2)

        check_mask = same_layer_mask & upper_tri_mask

        if not cp.any(check_mask):
            return new_config
        if cp.all(sq_distances_xy[check_mask] >= min_dist_squared):
            return new_config

    return None
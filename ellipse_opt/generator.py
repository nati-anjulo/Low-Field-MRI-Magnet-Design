"""
Magnet generator functions - STRICT COPY from original notebook.
"""
import cupy as cp
from .geometry import ellipse_r_gpu, get_bolt_positions_ellipse_gpu
from . import field_calculator as GRFC

# ======================================================================
# HALF-ELLIPSE REFLECTION (4-fold: x-axis mirror + z-mirror)
# ======================================================================

def reflect_to_half_ellipse(half_ellipse_config):
    """Reflect half-ellipse (θ ∈ [0, π], z > 0) to full ellipse.
    
    Half-ellipse = top half of xy plane (y ≥ 0) with z ≥ 0
    
    4-fold symmetry:
        Sector 0: (θ, z)           - original (y > 0, z > 0)
        Sector 1: (2π - θ, z)      - x-axis mirror (y < 0, z > 0)
        Sector 2: (θ, -z)          - z-mirror (y > 0, z < 0)
        Sector 3: (2π - θ, -z)     - both mirrors (y < 0, z < 0)
    
    Magnetization (φ) transforms:
        x-axis mirror: φ → -φ (flip y-component of M)
        z-mirror: φ unchanged (z-reflection doesn't affect xy magnetization)
    """
    z, r, theta, phi = half_ellipse_config.T
    
    # Discretize z for symmetric layer mapping
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_z = sheet_thickness / 2
    
    layer_indices = cp.round((z - axis_margin_z) / sheet_thickness)
    z_discretized = axis_margin_z + layer_indices * sheet_thickness
    
    all_configs = []
    
    for sector in range(4):
        # z sign: sectors 0,1 = +z; sectors 2,3 = -z
        z_sign = 1 if sector < 2 else -1
        # x-axis mirror: sectors 1,3
        x_mirror = (sector % 2) == 1
        
        z_new = z_sign * z_discretized
        r_new = r.copy()
        
        if x_mirror:
            # x-axis reflection: θ → 2π - θ
            theta_new = (2 * cp.pi - theta) % (2 * cp.pi)
            # Magnetization: φ → -φ (flip y-component)
            phi_new = (-phi) % (2 * cp.pi)
        else:
            theta_new = theta
            phi_new = phi
        
        sector_config = cp.stack([z_new, r_new, theta_new, phi_new], axis=1)
        all_configs.append(sector_config)
    
    result = cp.vstack(all_configs)
    return result


def generate_half_ellipse(n_magnets, z_max, rx_inner, ry_inner, gap, magnet_size, max_attempts=150, clearance=0.0):
    """Generate random magnets in half-ellipse (θ ∈ [0, π], z ≥ 0).
    
    Half-ellipse = top half of xy plane where y ≥ 0.
    Uses x-axis mirror symmetry + z-mirror for 4-fold total.
    """
    config = cp.zeros((n_magnets, 4))
    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_sq = min_dist ** 2
    
    # Layer discretization
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    
    # Axis margins
    axis_margin_z = sheet_thickness / 2
    axis_margin_y = min_dist / 2  # y >= margin for x-axis reflection
    
    # Layer bounds
    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2 - axis_margin_z) / sheet_thickness)
    
    # Ellipse parameters
    a_in, b_in = rx_inner, ry_inner
    a_out, b_out = rx_inner + gap, ry_inner + gap
    half_diag = magnet_size * 1.8 / 2
    
    # Bolt positions
    bolt_number = 10
    bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(rx_inner + gap/2, ry_inner + gap/2, bolt_number)
    bolt_diameter = 0.0075  # 7.5 mm bolt
    min_bolt_dist = bolt_diameter/2 + half_diag + clearance
    min_bolt_dist_sq = min_bolt_dist ** 2
    
    print(f"Generating {n_magnets} magnets in half-ellipse (y ≥ 0)...")
    print(f"θ range: [0, π] = [0°, 180°]")
    print(f"Layer range: 0 to {max_layer_idx}")
    
    for i in range(n_magnets):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # Random discrete layer
            layer_idx = cp.random.randint(min_layer_idx, max_layer_idx + 1)
            z = axis_margin_z + layer_idx * sheet_thickness
            
            # Random theta in [0, π] (half-ellipse, y ≥ 0)
            # Small margin to keep y > axis_margin_y
            theta = cp.random.uniform(0.02, cp.pi - 0.02)
            
            # Elliptical r bounds at this theta
            r_inner_wall = ellipse_r_gpu(cp.array([theta]), a_in, b_in)[0] + half_diag
            r_outer_wall = ellipse_r_gpu(cp.array([theta]), a_out, b_out)[0] - half_diag
            
            if r_outer_wall <= r_inner_wall:
                continue
            
            rad = cp.random.uniform(r_inner_wall, r_outer_wall)
            phi = cp.random.uniform(0, 2 * cp.pi)
            
            config[i, 0] = z
            config[i, 1] = rad
            config[i, 2] = theta
            config[i, 3] = phi
            
            # Convert to Cartesian
            positions, _ = GRFC.input_converter_vectorized(config[i:i+1])
            x, y, z_cart = positions[0]
            
            # Check y margin (for x-axis reflection)
            if y < axis_margin_y:
                continue
            if z_cart < axis_margin_z:
                continue
            
            # Bolt check
            bolt_dist_sq = (x - bolt_x)**2 + (y - bolt_y)**2
            if cp.any(bolt_dist_sq < min_bolt_dist_sq):
                continue
            
            # Overlap check with existing magnets (layer-aware)
            if i > 0:
                all_positions, _ = GRFC.input_converter_vectorized(config[:i+1])
                same_layer = cp.abs(all_positions[:i, 2] - z_cart) < 1e-6
                if cp.any(same_layer):
                    dx = all_positions[:i, 0][same_layer] - x
                    dy = all_positions[:i, 1][same_layer] - y
                    xy_dist_sq = dx*dx + dy*dy
                    if cp.any(xy_dist_sq < min_dist_sq):
                        continue
            
            placed = True
        
        if not placed:
            print(f"Could not place magnet {i+1} after {max_attempts} attempts")
    
    print(f"Generated {n_magnets} magnets in half-ellipse")
    return config


def generate_full_ellipse_reflected(n_magnets, z_max, rx_inner, ry_inner, gap, magnet_size):
    """Generate full elliptical cylinder using half-ellipse reflection. Total = n × 4"""
    half_ellipse = generate_half_ellipse(n_magnets, z_max, rx_inner, ry_inner, gap, magnet_size)
    full_config = reflect_to_half_ellipse(half_ellipse)
    print(f"Full config: {len(full_config)} magnets ({n_magnets} × 4)")
    return half_ellipse, full_config
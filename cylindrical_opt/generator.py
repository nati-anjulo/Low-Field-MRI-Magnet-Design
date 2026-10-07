"""
Magnet generator functions.
"""
import cupy as cp
from . import field_calculator as GRFC

def generate_first_octant(n_magnets, z_max, r_min, r_max, magnet_size, max_attempts=100, clearance=0.0):
    """Generate random magnets in first octant with proper overlap checking, axis clearance, and bolt avoidance.
    
    Uses SAME parameters as perturbation functions for consistency:
    - min_dist = magnet_size * cp.sqrt(2) (manufacturing clearance)
    - Discrete z-layers
    - theta in [0, π/2] for first octant
    """
    config = cp.zeros((n_magnets, 4))
    
    # SAME as perturbation: 1.8 factor for manufacturing clearance
    min_dist = magnet_size * cp.sqrt(2) + clearance
    min_dist_sq = min_dist ** 2
    half_diag = magnet_size * 1.8 / 2
    
    # SAME axis margins as perturbation
    axis_margin_xy = min_dist / 2
    
    # SAME layer discretization as perturbation
    layer_clearance = 1.524e-3
    layer_thickness = 6.35e-3
    sheet_thickness = layer_thickness + layer_clearance
    axis_margin_z = sheet_thickness / 2
    
    # SAME layer bounds as perturbation
    min_layer_idx = 0
    max_layer_idx = int((z_max - magnet_size/2) / sheet_thickness)
    
    # SAME bolt setup as perturbation
    bolt_number = 8
    bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
    bolt_diameter = 0.0075  # 7.5mm bolt
    bolt_rad = r_min + magnet_size * 0.5
    bolt_x = bolt_rad * cp.cos(bolt_angle)
    bolt_y = bolt_rad * cp.sin(bolt_angle)
    min_bolt_dist_sq = (bolt_diameter/2 + half_diag) ** 2
    
    print(f"Generating {n_magnets} magnets in first octant...")
    print(f"  min_dist = {min_dist*1000:.2f} mm (magnet_size * cp.sqrt(2))")
    print(f"  axis_margin_xy = {axis_margin_xy*1000:.2f} mm")
    print(f"  axis_margin_z = {axis_margin_z*1000:.2f} mm")
    print(f"  layers: {min_layer_idx} to {max_layer_idx}")
    
    for i in range(n_magnets):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # Random discrete layer (SAME as perturbation)
            layer_idx = int(cp.random.randint(min_layer_idx, max_layer_idx + 1))
            z = axis_margin_z + layer_idx * sheet_thickness
            
            # Random r within bounds
            rad = float(cp.random.uniform(r_min + half_diag*0.1, r_max - half_diag*0.1))
            
            # Random theta in [0, π/2] for first octant (with small margin)
            theta = float(cp.random.uniform(0.05, cp.pi/2 - 0.05))
            
            # Random phi (magnetization direction)
            phi = float(cp.random.uniform(0, 2 * cp.pi))
            
            config[i, 0] = z
            config[i, 1] = rad
            config[i, 2] = theta
            config[i, 3] = phi
            
            # Convert to Cartesian for validation
            positions, _ = GRFC.input_converter_vectorized(config[i:i+1])
            x, y, z_cart = positions[0]
            
            # Check axis clearance (SAME as perturbation)
            if not (x >= axis_margin_xy and y >= axis_margin_xy and z_cart >= axis_margin_z):
                continue
            
            # Bolt check (SAME as perturbation)
            bolt_dist_sq = (x - bolt_x)**2 + (y - bolt_y)**2
            if cp.any(bolt_dist_sq < min_bolt_dist_sq):
                continue
            
            # Layer-aware overlap check (SAME as perturbation)
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
            print(f"WARNING: Could not place magnet {i+1} after {max_attempts} attempts")
    
    print(f"Generated {n_magnets} magnets in first octant")
    return config
def generate_first_octant_old(n_magnets,  z_max, r_min, r_max, magnet_size, max_attempts=100, clearance=0.0):
    """Generate random magnets in first octant with proper overlap checking and axis clearance"""
    config = cp.zeros((n_magnets, 4))
    min_dist = magnet_size * cp.sqrt(2) + clearance
    
    # For reflection across axes: reflected magnets must maintain min_dist
    # If magnet is distance 'd' from axis, reflected magnet is also distance 'd'
    # Total separation = 2d ≥ min_dist, so d ≥ min_dist/2
    axis_margin = min_dist / 2
    
   
    
    # while reflection the magenst need to be at least min_dist/2 away from the axis so that the reflected magnet does not overlap with the original magnet
    
    print(f"Generating {n_magnets} magnets in first octant...")
    print(f"Min distance between centers: {min_dist:.6f}")
    #print(f"Required distance from axes: {axis_margin:.6f}")
    
    for i in range(n_magnets):
        placed = False                                                    
        attempts = 0
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # Generate position ensuring axis clearance
            z = cp.random.uniform(axis_margin, z_max - magnet_size/2)
            rad = cp.random.uniform(r_min, r_max)
            theta = cp.random.uniform(0, cp.pi/2)  # First quadrant
            phi = cp.random.uniform(0, 2 * cp.pi)  # Random magnetization
            
            config[i, 0] = z
            config[i, 1] = rad  
            config[i, 2] = theta
            config[i, 3] = phi
            
            # Convert to Cartesian and verify axis clearance
            positions, _ = GRFC.input_converter_vectorized(config[i:i+1])
            x, y, z_cart = positions[0]
            
            # Check clearance from all coordinate axes
            if x < axis_margin or y < axis_margin or z_cart < axis_margin:
                continue
            
            # Check overlaps with existing magnets
            if i > 0:
                all_positions, _ = GRFC.input_converter_vectorized(config[:i+1])
                overlap = False
                
                for j in range(i):
                    if cp.linalg.norm(all_positions[j] - all_positions[i]) < min_dist:
                        overlap = True
                        break
                
                if not overlap:
                    placed = True
            else:
                placed = True
        
        if not placed:
            print(f"Could not place magnet {i+1} after {max_attempts} attempts")
    
    print(f"Generated {n_magnets} magnets with proper axis clearance")
    return config

def reflect_to_octants(first_octant):
    """Fully vectorized reflection in cylindrical coordinates
    we need to preserve rotaional symmetry and the magnetization direction
    we are not  refelcting anythifn we are replicating howw the magnets would look like in all octants
    by how they look like in the first octant becasue in halbach we have rotational symmetry
    and the magnetization direction is preserved in all octants
    """
    z, r, theta, phi = first_octant.T
    
    
    # Vectorized octant transforms: [theta_offset, z_sign] - Y-axis is θ=0, left-hand clockwise +y, +x,-y,-x 
    transforms = cp.array([ # clockwise from +y to -y
        # Each row corresponds to an octant:
        # [theta_offset, z_sign]
        [0, 1],           # Octant 1: (+x,+y,+z) -> theta + 0 (Y-axis is 0), z * 1
        [cp.pi/2, 1],   # Octant 2: (+x,-y,+z) -> theta + π/2 (clockwise from +y to -y), z * 1
        [cp.pi, 1],       # Octant 3: (-x,-y,+z) -> theta + π, z * 1
        [3*cp.pi/2, 1],     # Octant 4: (-x,+y,+z) -> theta + 3π/2, z * 1
        [0, -1],          # Octant 5: (+x,+y,-z) -> theta + 0, z * -1
        [cp.pi/2, -1],  # Octant 6: (+x,-y,-z) -> theta + π/2, z * -1
        [cp.pi, -1],      # Octant 7: (-x,-y,-z) -> theta + π, z * -1
        [3*cp.pi/2, -1]     # Octant 8: (-x,+y,-z) -> theta + 3π/2, z * -1
    ])
  
    
    
    theta_offsets = transforms[:, 0][:, None]  # (8, 1)
    z_signs = transforms[:, 1][:, None]        # (8, 1)
    
    # Vectorized transformations: (8, n_magnets)
    z_all = z_signs * z[None, :]                          # (8, n_magnets)
    r_all = cp.tile(r[None, :], (8, 1))                   # (8, n_magnets) 
    theta_all = (theta[None, :] + theta_offsets) % (2*cp.pi)  # (8, n_magnets)
    
    # Vectorized Halbach magnetization
    octant_indices = cp.arange(8)[:, None]                # (8, 1)
    phi_all = (phi[None, :] + octant_indices * cp.pi) % (2*cp.pi)  # (8, n_magnets)
    
    # Stack and flatten: (8*n_magnets, 4)
    result = cp.stack([z_all.flatten(), r_all.flatten(), 
                      theta_all.flatten(), phi_all.flatten()], axis=1)
    
    return result


def generate_full_configuration(n_magnets,  z_max, r_min, r_max, magnet_size):
    """Generate full configuration across all octants"""
    first_octant = generate_first_octant(n_magnets, z_max, r_min, r_max, magnet_size)
    
    # Reflect to all octants with Layer 0 handling
    full_config = reflect_to_octants(first_octant)
    
    print(f"Generated full configuration with {len(full_config)} magnets")
    return first_octant,full_config
"""
Constraint functions.
"""
import cupy as cp
from cupy.cuda import stream as cp_stream
from . import field_calculator as GRFC

def check_hard_constraints_vectorized_with_bolt(config, z_min, z_max, r_min, r_max, magnet_size, positions, clearance=0.00):
    """
    Vectorized implementation of hard constraint checks including bolt location constraints.
    Returns True if any constraint is violated, False otherwise.
    
    Layer-aware: Only checks xy-distance for magnets on the SAME z-layer.
    Different z-layers are separated by layer spacing (1.524mm gap).
    """
    # Extract z and r for all magnets at once
    z = positions[:, 2]  # Use discretized z from input_converter
    r = config[:, 1]
    n_magnets = len(config)
    
    # Check if any magnet extends beyond z bounds (vectorized)
    z_min_violation = cp.any(z - magnet_size/2 < z_min)
    z_max_violation = cp.any(z + magnet_size/2 > z_max)
    
    # Check if any magnet is outside radius range (vectorized)
    r_min_violation = cp.any(r < r_min)
    r_max_violation = cp.any(r > r_max)
    
    # Early return if any of the simple bounds are violated
    if z_min_violation or z_max_violation or r_min_violation or r_max_violation:
        return True
    
    # Check bolt location constraints (fully vectorized)
    bolt_number = 8
    bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
    bolt_diameter = 0.0075  # 7.5mm bolt  # Explicit bolt diameter (6.35mm)
    bolt_rad = r_min + magnet_size *0.5
    bolt_x = bolt_rad * cp.cos(bolt_angle)
    bolt_y = bolt_rad * cp.sin(bolt_angle)
    
    # Minimum distance = bolt_radius + magnet_rotated_radius
    min_bolt_dist_squared = (bolt_diameter/2 + magnet_size * cp.sqrt(2) / 2) ** 2 
    
    magnet_x = positions[:, 0].reshape(-1, 1)  # Shape: (n_magnets, 1)
    magnet_y = positions[:, 1].reshape(-1, 1)  # Shape: (n_magnets, 1)
    bolt_x = bolt_x.reshape(1, -1)             # Shape: (1, 8)
    bolt_y = bolt_y.reshape(1, -1)             # Shape: (1, 8)
    distances_squared = (magnet_x - bolt_x)**2 + (magnet_y - bolt_y)**2
    
    # Check if any magnet is too close to any bolt
    if cp.any(distances_squared < min_bolt_dist_squared):
        return True
    
    # LAYER-AWARE overlap checking: only check xy-distance for same z-layer
    min_xy_dist = magnet_size * cp.sqrt(2) + clearance
    min_xy_dist_squared = min_xy_dist * min_xy_dist
    
    # Extract z positions (discretized layer centers)
    z_positions = positions[:, 2]
    
    # Tolerance for comparing z-values (magnets on same layer have identical z after discretization)
    z_tolerance = 1e-6
    
    # Build same-layer pairs mask
    z_diff = cp.abs(z_positions.reshape(-1, 1) - z_positions.reshape(1, -1))
    same_layer_mask = z_diff < z_tolerance
    
    # Extract only x,y coordinates
    pos_xy = positions[:, :2]  # Shape: (n_magnets, 2)
    
    # Reshape for broadcasting
    pos1_xy = pos_xy.reshape(n_magnets, 1, 2)
    pos2_xy = pos_xy.reshape(1, n_magnets, 2)
    
    # Calculate squared xy-distances between all magnet pairs
    squared_distances_xy = cp.sum((pos1_xy - pos2_xy)**2, axis=2)
    
    # Create upper triangle mask to avoid self-comparison and duplicates
    upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)
    
    # Combine masks: check only same-layer pairs in upper triangle
    check_mask = same_layer_mask & upper_tri_mask
    
    # Check if any same-layer pair is too close in xy-plane
    if cp.any(check_mask):
        overlap_violation = cp.any(squared_distances_xy[check_mask] < min_xy_dist_squared)
        return overlap_violation
    
    return False
    
def calculate_soft_barriers_vectorized_with_bolt(config, z_min, z_max, r_min, r_max, magnet_size, positions, clearance=0.0):
    """
    Vectorized implementation of soft barrier calculations.
    Layer-aware: Only penalizes xy-distance for magnets on the SAME z-layer.
    """
    with cp_stream.Stream() as barrier_stream:
        # Define thresholds
        n_magnets = len(config)
        barrier_threshold_z = 0.12 * magnet_size
        barrier_threshold_rad = 0.20 * magnet_size
        barrier_threshold_dis = 0.12
        min_xy_dist = magnet_size * cp.sqrt(2) + clearance
        
        # Extract z and r for all magnets
        z = positions[:, 2]  # Use discretized z from input_converter
        r = config[:, 1]
        
        # Calculate z barrier costs (vectorized)
        # Lower z bound
        lower_margins = (z - magnet_size/2) - z_min
        lower_z_mask = lower_margins < barrier_threshold_z
        lower_z_cost = cp.zeros_like(lower_margins)
        lower_z_cost[lower_z_mask] = -cp.log(cp.clip(lower_margins[lower_z_mask], 1e-10, None))
        
        # Upper z bound
        upper_margins = z_max - (z + magnet_size/2)
        upper_z_mask = upper_margins < barrier_threshold_z
        upper_z_cost = cp.zeros_like(upper_margins)
        upper_z_cost[upper_z_mask] = -cp.log(cp.clip(upper_margins[upper_z_mask], 1e-10, None))
        
        z_barrier_cost = cp.sum(lower_z_cost) + cp.sum(upper_z_cost)
        
        # Calculate radius barrier costs (vectorized)
        # Lower radius bound
        lower_rad_margins = r - r_min
        lower_rad_mask = lower_rad_margins < barrier_threshold_rad
        lower_rad_cost = cp.zeros_like(lower_rad_margins)
        lower_rad_cost[lower_rad_mask] = -cp.log(cp.clip(lower_rad_margins[lower_rad_mask], 1e-10, None))
        
        # Upper radius bound
        upper_rad_margins = r_max - r
        upper_rad_mask = upper_rad_margins < barrier_threshold_rad
        upper_rad_cost = cp.zeros_like(upper_rad_margins)
        upper_rad_cost[upper_rad_mask] = -cp.log(cp.clip(upper_rad_margins[upper_rad_mask], 1e-10, None))
        
        Rad_barrier_cost = cp.sum(lower_rad_cost) + cp.sum(upper_rad_cost)
        
        # LAYER-AWARE overlap barrier: only penalize same-layer pairs
        z_positions = positions[:, 2]
        z_tolerance = 1e-6
        
        # Build same-layer pairs
        z_diff = cp.abs(z_positions.reshape(-1, 1) - z_positions.reshape(1, -1))
        same_layer_mask = z_diff < z_tolerance
        
        # Extract only x,y coordinates
        pos_xy = positions[:, :2]
        pos1_xy = pos_xy.reshape(n_magnets, 1, 2)
        pos2_xy = pos_xy.reshape(1, n_magnets, 2)
        
        # Calculate xy-distances
        distances_xy = cp.sqrt(cp.sum((pos1_xy - pos2_xy)**2, axis=2))
        
        # Upper triangle mask
        upper_tri_mask = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)
        check_mask = same_layer_mask & upper_tri_mask
        
        # Calculate margins for same-layer pairs only
        overlap_cost = 0
        if cp.any(check_mask):
            valid_distances = distances_xy[check_mask]
            margins = valid_distances - min_xy_dist
            
            # Apply threshold
            overlap_mask = margins < barrier_threshold_dis * min_xy_dist
            
            if cp.any(overlap_mask):
                overlap_cost = -cp.sum(cp.log(cp.clip(margins[overlap_mask], 1e-10, None)))
        
        # Calculate bolt barrier costs (vectorized)
        bolt_number = 8
        bolt_angle = cp.linspace(0, 2*cp.pi, bolt_number, endpoint=False)
        bolt_diameter = 0.0075  # 7.5mm bolt  # Explicit bolt diameter (6.35mm)
        bolt_rad = r_min + magnet_size * 0.5
        bolt_x = bolt_rad * cp.cos(bolt_angle)
        bolt_y = bolt_rad * cp.sin(bolt_angle)
        
        # Minimum safe distance = bolt_radius + magnet_rotated_radius
        min_bolt_dist = bolt_diameter/2 + magnet_size * cp.sqrt(2) / 2
        bolt_barrier_threshold = 0.15 * magnet_size
        
        # Vectorized distance calculation: all magnets vs all bolts
        magnet_x = positions[:, 0].reshape(-1, 1)  # Shape: (n_magnets, 1)
        magnet_y = positions[:, 1].reshape(-1, 1)  # Shape: (n_magnets, 1)
        bolt_x_reshaped = bolt_x.reshape(1, -1)    # Shape: (1, 8)
        bolt_y_reshaped = bolt_y.reshape(1, -1)    # Shape: (1, 8)
        
        # Broadcasting: (n_magnets, 8) matrix of distances
        bolt_distances = cp.sqrt((magnet_x - bolt_x_reshaped)**2 + (magnet_y - bolt_y_reshaped)**2)
        
        # Calculate margins from minimum safe distance
        bolt_margins = bolt_distances - min_bolt_dist
        
        # Apply barrier cost where margins are less than threshold
        bolt_barrier_mask = bolt_margins < bolt_barrier_threshold
        bolt_barrier_cost = 0
        
        if cp.any(bolt_barrier_mask):
            bolt_barrier_cost = -cp.sum(cp.log(cp.clip(bolt_margins[bolt_barrier_mask], 1e-10, None)))
        
        # Add bolt barrier cost to overlap cost (or could be separate)
        overlap_cost += bolt_barrier_cost
        
    return z_barrier_cost, Rad_barrier_cost, overlap_cost
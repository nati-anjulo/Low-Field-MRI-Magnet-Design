"""
Constraint functions.
"""
import cupy as cp
from cupy.cuda import stream as cp_stream
from .geometry import ellipse_r_gpu, get_bolt_positions_ellipse_gpu

def calculate_soft_barriers_half_ellipse(config, z_min, z_max, rx_inner,
                                                   ry_inner, gap, magnet_size, positions, clearance=0.0,
                                                   skip_full_overlap=False, cached_overlap=None):
    """
    Soft barrier calculations for HALF-ELLIPSE elliptical cylinder.
    
    OPTIMIZED: Uses layer-aware overlap check - only computes XY distances for
    same-layer pairs, avoiding full n² distance matrix (40x speedup).
    """
    with cp_stream.Stream() as barrier_stream:
        a_in, b_in = rx_inner, ry_inner
        a_out, b_out = rx_inner + gap, ry_inner + gap
        half_diag = magnet_size * 1.8 / 2

        n_magnets = len(config)
        barrier_threshold_z = 0.12 * magnet_size
        barrier_threshold_rad = 0.20 * magnet_size
        barrier_threshold_dis = 0.12
        min_xy_dist = magnet_size * cp.sqrt(2) + clearance

        z = positions[:, 2]
        r = config[:, 1]
        theta = config[:, 2]

        # Z barrier costs
        lower_margins = (z - magnet_size/2) - z_min
        lower_z_mask = lower_margins < barrier_threshold_z
        lower_z_cost = cp.zeros_like(lower_margins)
        lower_z_cost[lower_z_mask] = -cp.log(cp.clip(lower_margins[lower_z_mask], 1e-10, None))

        upper_margins = z_max - (z + magnet_size/2)
        upper_z_mask = upper_margins < barrier_threshold_z
        upper_z_cost = cp.zeros_like(upper_margins)
        upper_z_cost[upper_z_mask] = -cp.log(cp.clip(upper_margins[upper_z_mask], 1e-10, None))

        z_barrier_cost = cp.sum(lower_z_cost) + cp.sum(upper_z_cost)

        # Elliptical wall barrier costs
        r_inner = ellipse_r_gpu(theta, a_in, b_in) + half_diag
        r_outer = ellipse_r_gpu(theta, a_out, b_out) - half_diag

        lower_rad_margins = r - r_inner
        lower_rad_mask = lower_rad_margins < barrier_threshold_rad
        lower_rad_cost = cp.zeros_like(lower_rad_margins)
        lower_rad_cost[lower_rad_mask] = -cp.log(cp.clip(lower_rad_margins[lower_rad_mask], 1e-10, None))

        upper_rad_margins = r_outer - r
        upper_rad_mask = upper_rad_margins < barrier_threshold_rad
        upper_rad_cost = cp.zeros_like(upper_rad_margins)
        upper_rad_cost[upper_rad_mask] = -cp.log(cp.clip(upper_rad_margins[upper_rad_mask], 1e-10, None))

        rad_barrier_cost = cp.sum(lower_rad_cost) + cp.sum(upper_rad_cost)

        # Overlap barrier - skip if using cached value
        if skip_full_overlap and cached_overlap is not None:
            overlap_cost = cached_overlap
        else:
            # Reflect about x-axis (2 halves for half-ellipse)
            pos_top = positions[:, :3]  # x, y, z (y > 0)
            pos_bot = pos_top * cp.array([1, -1, 1])  # x, -y, z (y < 0)
            pos_all = cp.vstack([pos_top, pos_bot])  # 2N magnets
            
            # Layer-indexed overlap check
            z_positions = cp.tile(positions[:, 2], 2)
            pos_xy = pos_all[:, :2]
            n_all = len(pos_all)
            sheet_thickness = 6.35e-3 + 1.524e-3
            axis_margin_z = sheet_thickness / 2
            layer_indices = cp.round((z_positions - axis_margin_z) / sheet_thickness).astype(cp.int32)
            same_layer_mask = layer_indices.reshape(-1, 1) == layer_indices.reshape(1, -1)
            upper_tri = cp.triu(cp.ones((n_all, n_all), dtype=bool), k=1)
            check_mask = same_layer_mask & upper_tri
            
            i_idx, j_idx = cp.where(check_mask)
            overlap_cost = cp.float32(0.0)
            if len(i_idx) > 0:
                diff = pos_xy[i_idx] - pos_xy[j_idx]
                distances = cp.sqrt(cp.sum(diff**2, axis=1))
                margins = distances - min_xy_dist
                overlap_mask = margins < barrier_threshold_dis * min_xy_dist
                if cp.any(overlap_mask):
                    safe_margins = cp.clip(margins[overlap_mask], 1e-10, None)
                    overlap_cost = -cp.sum(cp.log(safe_margins))

            # Bolt barrier (always computed - it's cheap)
            bolt_number = 10
            bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(a_in + gap/2, b_in + gap/2, bolt_number)
            bolt_diameter = 0.0075  # 7.5 mm bolt
            min_bolt_dist = bolt_diameter/2 + half_diag
            bolt_barrier_threshold = 0.15 * magnet_size

            magnet_x = positions[:, 0].reshape(-1, 1)
            magnet_y = positions[:, 1].reshape(-1, 1)
            bolt_distances = cp.sqrt((magnet_x - bolt_x.reshape(1, -1))**2 + (magnet_y - bolt_y.reshape(1, -1))**2)
            bolt_margins = bolt_distances - min_bolt_dist

            bolt_barrier_mask = bolt_margins < bolt_barrier_threshold
            bolt_barrier_cost = cp.float32(0.0)
            if cp.any(bolt_barrier_mask):
                bolt_barrier_cost = -cp.sum(cp.log(cp.clip(bolt_margins[bolt_barrier_mask], 1e-10, None)))

            # bolt_barrier_cost kept separate (not added to overlap_cost)

    return z_barrier_cost, rad_barrier_cost, overlap_cost, bolt_barrier_cost


def check_hard_constraints_half_ellipse(config, z_min, z_max, rx_inner,
                                                  ry_inner, gap, magnet_size, positions, clearance=0.0):
    """
    Hard constraint checks for HALF-ELLIPSE elliptical cylinder.
    Half-z (z >= 0): theta in [0, pi] (half-ellipse), z >= 0 only (full XY freedom).
    """
    a_in, b_in = rx_inner, ry_inner
    a_out, b_out = rx_inner + gap, ry_inner + gap
    half_diag = magnet_size * 1.8 / 2

    z = config[:, 0]
    r = config[:, 1]
    theta = config[:, 2]
    n_magnets = len(config)

    # Z bounds (positive only)
    if cp.any(z < 0) or cp.any(z + magnet_size/2 > z_max):
        return True

    # Theta bounds (half-ellipse)
    if cp.any(theta < 0) or cp.any(theta > cp.pi):
        return True

    # Elliptical wall bounds
    r_inner = ellipse_r_gpu(theta, a_in, b_in) + half_diag
    r_outer = ellipse_r_gpu(theta, a_out, b_out) - half_diag
    if cp.any(r < r_inner) or cp.any(r > r_outer):
        return True

    # Half-ellipse position check (x, y >= margin)
    min_dist = magnet_size * cp.sqrt(2) + clearance
    sheet_thickness = 6.35e-3 + 1.524e-3
    axis_margin_xy = min_dist / 2
    axis_margin_z = sheet_thickness / 2
    x, y, z_pos = positions[:, 0], positions[:, 1], positions[:, 2]
    # XY margin removed for halfz (full XY freedom)
    # if cp.any(x < axis_margin_xy) or cp.any(y < axis_margin_xy):
    #     return True
    if cp.any(z_pos < axis_margin_z):
        return True

    # Bolt positions on ellipse
    bolt_number = 10
    bolt_x, bolt_y = get_bolt_positions_ellipse_gpu(a_in + gap/2, b_in + gap/2, bolt_number)
    bolt_diameter = 0.0075  # 7.5 mm bolt
    min_bolt_dist_squared = (bolt_diameter/2 + half_diag) ** 2

    magnet_x = x.reshape(-1, 1)
    magnet_y = y.reshape(-1, 1)
    distances_squared = (magnet_x - bolt_x.reshape(1, -1))**2 + (magnet_y - bolt_y.reshape(1, -1))**2

    if cp.any(distances_squared < min_bolt_dist_squared):
        return True

    # Layer-aware overlap checking
    min_xy_dist_squared = min_dist ** 2
    z_positions = positions[:, 2]
    
    same_layer = cp.abs(z_positions.reshape(-1, 1) - z_positions.reshape(1, -1)) < 1e-6
    upper_tri = cp.triu(cp.ones((n_magnets, n_magnets), dtype=bool), k=1)
    check_mask = same_layer & upper_tri
    
    i_idx, j_idx = cp.where(check_mask)
    
    # Reflect to 4 quadrants for continuous XY overlap check
    # Full XY - no quadrant reflection
    # Reflect about x-axis (2 halves for half-ellipse)
    pos_top_xy = positions[:, :2]  # y > 0
    pos_bot_xy = pos_top_xy * cp.array([1, -1])  # y < 0
    pos_xy_all = cp.vstack([pos_top_xy, pos_bot_xy])
    z_all = cp.tile(z_positions, 2)
    
    n_all = len(pos_xy_all)
    same_layer_all = cp.abs(z_all.reshape(-1, 1) - z_all.reshape(1, -1)) < 1e-6
    upper_tri_all = cp.triu(cp.ones((n_all, n_all), dtype=bool), k=1)
    check_mask_all = same_layer_all & upper_tri_all
    
    i_idx, j_idx = cp.where(check_mask_all)
    if len(i_idx) > 0:
        diff = pos_xy_all[i_idx] - pos_xy_all[j_idx]
        squared_distances = cp.sum(diff**2, axis=1)
        if cp.any(squared_distances < min_xy_dist_squared):
            return True

    return False

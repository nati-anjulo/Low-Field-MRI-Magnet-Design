"""
Cost function.
"""
import cupy as cp
from .generator import reflect_to_half_ellipse
from .constraints import check_hard_constraints_half_ellipse, calculate_soft_barriers_half_ellipse
from . import field_calculator as GRFC

def calculate_cost_half_ellipse(half_ellipse_config, z_min, z_max, rx_inner, ry_inner, gap,
                                          magnet_size, sample_points, weights, 
                                          Precalc_Bxyz_value, Ref_grid, step_size,
                                          skip_full_overlap=False, cached_overlap=None):
    """
    Cost function for half-ellipse reflected ellipse.
    
    Optimization: skip_full_overlap=True skips expensive n² overlap check.
    Returns (cost, overlap_cost) when skip_full_overlap is False for caching.
    """
    try:
        if not isinstance(half_ellipse_config, cp.ndarray):
            half_ellipse_config = cp.asarray(half_ellipse_config)

        # Get Q1 positions for constraint checking
        positions_q1, _ = GRFC.input_converter_vectorized(half_ellipse_config)

        # Check hard constraints on Q1 only
        if check_hard_constraints_half_ellipse(
            half_ellipse_config, z_min, z_max, rx_inner, ry_inner, gap, 
            magnet_size, positions_q1):
            return float('inf')

        # Reflect Q1 to full 2 halves (z reflection) for field calculation
        full_config = reflect_to_half_ellipse(half_ellipse_config)

        # Calculate fields on FULL configuration
        B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
            full_config, sample_points, Precalc_Bxyz_value, Ref_grid, step_size,
            batch_size=1000000, num_streams=10
        )
        
        # Guard: Check for out-of-bounds field lookup
        if not cp.all(cp.isfinite(B_magnitude_mT_GPU)):
            return float('inf')

        # Soft barriers on Q1 only - with overlap optimization
        # Now returns bolt_barrier_cost separately
        z_barrier_cost, rad_barrier_cost, overlap_cost, bolt_barrier_cost = calculate_soft_barriers_half_ellipse(
            half_ellipse_config, z_min, z_max, rx_inner, ry_inner, gap, 
            magnet_size, positions_q1,
            skip_full_overlap=skip_full_overlap, cached_overlap=cached_overlap)

        # Field statistics
        avg_field_strength = -cp.mean(B_magnitude_mT_GPU)
        mean_magnitude = cp.mean(B_magnitude_mT_GPU)
        rms_deviation = cp.mean((B_magnitude_mT_GPU - mean_magnitude)**2)

        # Cost components
        field_cost = weights['field_strength'] * avg_field_strength
        uniformity_cost = weights['uniformity'] * rms_deviation
        z_barrier_total = weights['z_barrier'] * z_barrier_cost
        overlap_total = weights['overlap_barrier'] * overlap_cost
        rad_barrier_total = weights['radius_barrier'] * rad_barrier_cost
        bolt_barrier_total = weights.get('bolt_barrier', 1.0) * bolt_barrier_cost  # Lower weight for bolts

        total_cost = field_cost + uniformity_cost + z_barrier_total + overlap_total + rad_barrier_total + bolt_barrier_total

        if not cp.isfinite(total_cost):
            return float('inf')

        return total_cost

    except Exception as e:
        print(f"Error calculating cost: {e}")
        return float('inf')
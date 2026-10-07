"""
Cost function.
"""
import cupy as cp
from cupy.cuda import stream as cp_stream
from .constraints import check_hard_constraints_vectorized_with_bolt, calculate_soft_barriers_vectorized_with_bolt
from . import field_calculator as GRFC


def calculate_cost_test(config,  z_min, z_max, r_min, r_max, magnet_size, sample_points, 
                       weights, Precalc_Bxyz_value, Ref_grid, step_size):
    """
    Calculate the cost function with a hybrid approach.
    OPTIMIZED: Uses 10 streams (3% speedup)
    """
    try:
        # Ensure config is on GPU
        if not isinstance(config, cp.ndarray):
            config = cp.asarray(config)
            
        # Convert configuration to positions for constraint checking
        positions, _ = GRFC.input_converter_vectorized(config)
        
        #t.tic()
        # Check hard constraints first (fast rejection) - now vectorized
        if check_hard_constraints_vectorized_with_bolt(config, z_min, z_max, r_min, r_max, magnet_size, positions):
            return float('inf')  # Hard constraint violation
        #hard_constraint_time = t.toc("hard constraint check")
        #print("here")
        # Calculate fields using the existing optimized function
        # OPTIMIZED: Changed from 8 to 10 streams (3% speedup)
        #t.tic()
        B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
            config, sample_points, Precalc_Bxyz_value, Ref_grid, step_size,
            batch_size=1000000, num_streams=10  # OPTIMIZED: was 8
        )
        #field_time = t.toc('field calculation')
        #print("B_field_mT_GPU shape:", B_field_mT_GPU)
        # Calculate soft barrier costs - now vectorized
       # t.tic()
        z_barrier_cost, Rad_barrier_cost, overlap_cost = calculate_soft_barriers_vectorized_with_bolt(
            config,  z_min, z_max, r_min, r_max, magnet_size, positions
        )
        #barrier_time = t.toc('soft barrier calculation')
        #print("1 here")
        # Calculate field statistics
        with cp_stream.Stream() as stat_stream:
            avg_field_strength = -cp.mean(B_magnitude_mT_GPU)
            mean_magnitude = cp.mean(B_magnitude_mT_GPU)
            rms_deviation = cp.mean((B_magnitude_mT_GPU - mean_magnitude)**2)
        
        # Free GPU memory aggressively
        #del B_field_mT_GPU, B_magnitude_mT_GPU
        #cp.get_default_memory_pool().free_all_blocks()
       
        
        # Calculate individual cost components with weights
        field_cost = weights['field_strength'] * avg_field_strength
        uniformity_cost = weights['uniformity'] * rms_deviation
        z_barrier_total = weights['z_barrier'] * z_barrier_cost
        overlap_total = weights['overlap_barrier'] * overlap_cost
        Rad_barrier_total = weights['radius_barrier'] * Rad_barrier_cost
        
        # Combine all cost components
        total_cost = field_cost + uniformity_cost + z_barrier_total + overlap_total + Rad_barrier_total
        
        # Safety: reject non-finite costs only
        if not cp.isfinite(total_cost):
            return float('inf')
        
        return total_cost
        
    except Exception as e:
        print(f"Error calculating cost: {e}")
        # Print stack trace for better debugging
        import traceback
        traceback.print_exc()
        return float('inf')  # Return infinite cost if there's an error
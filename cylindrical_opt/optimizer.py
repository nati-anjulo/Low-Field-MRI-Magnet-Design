"""
Simulated annealing optimizer - STRICT COPY from original notebook.
"""
import numpy as np
import cupy as cp
import time
from pytictoc import TicToc
t = TicToc()

from .cost import calculate_cost_test
from .perturbation import Hybrid_perturbation_vectorized_with_bolt
from .generator import reflect_to_octants
from . import field_calculator as GRFC

def simulated_annealing_optimized(initial_config,  z_min, z_max, r_min, r_max, magnet_size, sample_points, 
                       weights, Precalc_Bxyz_value, Ref_grid, step_size, step_divisors,
                       initial_temp=1.0, cooling_rate=0.999, n_iterations=1000, 
                       display_interval=100, verbose=True, debug_cost=False):
    """
    Optimized simulated annealing using precalculated fields.
    """
    print("Starting simulated annealing optimization...")
    
    # Ensure we're working with GPU arrays
    if not isinstance(initial_config, cp.ndarray):
        initial_config = cp.asarray(initial_config)
    
    # Start with the provided initial configuration
    current_config = initial_config.copy()
   
    # Calculate initial cost
    current_cost = calculate_cost_test(
        reflect_to_octants(current_config), z_min, z_max, r_min, r_max, magnet_size, 
        sample_points, weights, Precalc_Bxyz_value, Ref_grid, step_size
    )
    #print(len(current_config))
    #print(len(reflect_to_quadrants(current_config)))
    # Check if initial configuration is valid
    if cp.isinf(current_cost):
        print("ERROR: Initial configuration is invalid! Cannot proceed with optimization.")
        cp.get_default_memory_pool().free_all_blocks()  # Clean up GPU memory
        return None, float('inf'), []
    
    # Initialize best solution (only valid solutions)
    best_solution = current_config.copy()
    best_cost = current_cost
    
    # Initialize history - store as Python floats to save GPU memory
    cost_history = [float(current_cost)]
    
    # Store configurations on CPU to save GPU memory
    track_mag_cost_values = []
    track_mag_cost_values.append([cp.asnumpy(current_config), float(current_cost)])
    
    # Initialize temperature
    temp = initial_temp
    
    # Tracking variables
    invalid_count = 0
    iterations_completed = 0
    improvement_count = 0
    
    # Create buffer for next configuration to reduce allocations
    next_config = cp.zeros_like(current_config)
    
    try:
        # Main optimization loop
        while iterations_completed < n_iterations:
            # Generate a candidate solution by perturbing the current one
            
            candidate_config = Hybrid_perturbation_vectorized_with_bolt(
                current_config,  z_min, z_max, r_min, r_max, 
                magnet_size, temp, initial_temp,cooling_rate,n_iterations, step_divisors
            )
            
            # Skip this iteration if no valid perturbation found
            if candidate_config is None:
                invalid_count += 1
                if verbose and invalid_count % 100 == 0:
                    print(f"Warning: No valid perturbation found {invalid_count} times")
                
                # Still cool down the temperature and count as iteration
                temp *= cooling_rate
                iterations_completed += 1
                continue
            
            # Calculate cost for the candidate configuration
            #t.tic()
            candidate_cost = calculate_cost_test(
                reflect_to_octants(candidate_config),  z_min, z_max, r_min, r_max, magnet_size, 
                sample_points, weights, Precalc_Bxyz_value, Ref_grid, step_size
            )
            #t.toc("cost")
            #print(f"Candidate cost: {candidate_cost}")
            # Skip if candidate is invalid (should be rare with optimized perturbation)
            # Safety: Check for invalid costs (inf, nan, or extreme values)
            if cp.isinf(candidate_cost) or cp.isnan(candidate_cost) or abs(float(candidate_cost)) > 1e9:
                print(f"Warning: Unexpected invalid configuration at iteration {iterations_completed+1}")
                temp *= cooling_rate
                iterations_completed += 1
                continue
            
            # Decide whether to accept the new solution
            delta_cost = candidate_cost - current_cost
            
            # Use simpler logic for efficiency
            accept = False
            if delta_cost <= 0:  # Better solution
                accept = True
                if candidate_cost < best_cost:  # Track improvements
                    improvement_count += 1
            else:  # Worse solution - accept with probability
                accept_prob = cp.exp(-delta_cost / temp)
                accept = cp.random.random() < accept_prob
            
            # Update current solution if accepted
            if accept:
                # Avoid unnecessary copy operations - use direct assignment if possible
                current_config = candidate_config
                current_cost = candidate_cost
                
                # Update best solution if better
                if candidate_cost < best_cost:
                    best_solution = current_config.copy()  # Make a copy for best solution
                    best_cost = candidate_cost
            
            # Record history (convert to CPU and float to save GPU memory)
            # Only append to history at regular intervals to save memory
            if iterations_completed % 10 == 0 or iterations_completed == n_iterations - 1:
                cost_history.append(float(current_cost))
                track_mag_cost_values.append([cp.asnumpy(current_config), float(current_cost)])
            
            # Cool down the temperature
            temp *= cooling_rate
            iterations_completed += 1
            
            # Display progress if needed
            if verbose and ((iterations_completed % display_interval == 0) or (iterations_completed == n_iterations)):
                print(f"Iteration {iterations_completed}/{n_iterations}, Temperature: {temp:.6f}, Cost: {current_cost:.6f}")
                print(f"Improvements found: {improvement_count}, Invalid configurations: {invalid_count}")
                
                # Print best solution metrics
                if debug_cost and not cp.isinf(best_cost):
                    # Calculate individual components for the best solution
                    B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
                            reflect_to_octants(best_solution), sample_points,
                            Precalc_Bxyz_value, Ref_grid, step_size,
                            batch_size=1000000, num_streams=10  # Ensure consistent stream count
                        )
                    avg_field = -cp.mean(B_magnitude_mT_GPU)                  
                    mean_magnitude = cp.mean(B_magnitude_mT_GPU)
                    rms_dev = cp.sqrt(cp.mean((B_magnitude_mT_GPU - mean_magnitude)**2))
                    
                    # Free GPU memory quickly
                    del B_field_mT_GPU, B_magnitude_mT_GPU
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    print(f"  Best solution metrics:")
                    print(f"  → Avg field strength: {-avg_field:.2f} mT")
                    print(f"  → Field uniformity (RMS dev): {rms_dev:.2f} mT")
                    print(f"  → Weighted components - Field: {weights['field_strength']*avg_field:.2f}, " 
                        f"Uniformity: {weights['uniformity']*rms_dev:.2f}")
        
        # Final check to ensure best solution is valid
        if cp.isinf(best_cost):
            print("ERROR: No valid solution found during optimization!")
        else:
            print(f"Optimization completed. Best cost: {best_cost:.6f}")
            print(f"Total improvements found: {improvement_count}")
    
    except Exception as e:
        # Proper error handling with cleanup
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final memory cleanup - always execute this
        cp.get_default_memory_pool().free_all_blocks()
    
    return best_solution, best_cost, track_mag_cost_values # refelect the output 

def run_n_times_best_cost(n_runs, first_octant, z_min, z_max, r_min, r_max, halbach_magnet_size,
                          sample_points_small, weights, Precalc_Bxyz_value_import, Ref_grid, step_size, step_divisors,
                          initial_temp, cooling_rate, n_iterations, display_interval):
    """Run SA optimization n_runs times and return the best result."""
    best_overall_config = None
    best_overall_cost = float('inf')
    best_run = 0
    run_times = []  # Track timing for each run
    total_evaluations = 0

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")

        # Start timing for this run
        run_start_time = time.time()

        # Run optimization and get results
        best_config, best_cost, opt_track = simulated_annealing_optimized(
            first_octant, z_min, z_max, r_min, r_max, halbach_magnet_size,
            sample_points_small, weights, Precalc_Bxyz_value_import, Ref_grid, step_size, step_divisors,
            initial_temp=initial_temp, cooling_rate=cooling_rate, n_iterations=n_iterations,
            display_interval=display_interval, verbose=True, debug_cost=True
        )
        
        run_time = time.time() - run_start_time
        run_times.append(run_time)
        total_evaluations += n_iterations

        # Check if this run is the best
        if best_config is not None and best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_config = best_config.copy()
            best_run = i + 1

        print(f"Run {i+1} completed in {run_time:.1f}s, cost: {best_cost:.2f}")
        print("-" * 60)

    # ========== COMPARISON METRICS (vs GA) ==========
    print("\n" + "="*60)
    print("COMPARISON METRICS FOR SA vs GA")
    print("="*60)
    
    total_time = sum(run_times)
    
    # Calculate final field metrics
    if best_overall_config is not None:
        full_config = reflect_to_octants(best_overall_config)
        B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
            full_config, sample_points_small,
            Precalc_Bxyz_value_import, Ref_grid, step_size,
            batch_size=1000000, num_streams=10
        )
        field_strength_mT = float(cp.mean(B_magnitude_mT_GPU))
        field_std_mT = float(cp.std(B_magnitude_mT_GPU))
        homogeneity_pct = (field_std_mT / field_strength_mT) * 100
        
        print(f"\n1. FUNCTION EVALUATIONS:")
        print(f"   Total evaluations: {total_evaluations:,}")
        print(f"   Evaluations/second: {total_evaluations/total_time:.1f}")
        
        print(f"\n2. CONVERGENCE:")
        print(f"   Initial cost: {opt_track[0][1]:,.1f}")
        print(f"   Final cost: {best_overall_cost:,.1f}")
        print(f"   Improvement: {abs(best_overall_cost - opt_track[0][1])/abs(opt_track[0][1])*100:.2f}%")
        
        print(f"\n3. FIELD QUALITY:")
        print(f"   Field strength: {field_strength_mT:.2f} mT")
        print(f"   Field std dev: {field_std_mT:.4f} mT")
        print(f"   Homogeneity: {homogeneity_pct:.2f}%")
        
        print(f"\n4. COMPUTATIONAL COST:")
        print(f"   Total time: {total_time:.1f} s ({total_time/60:.1f} min)")
        print(f"   Time per iteration: {total_time/total_evaluations*1000:.2f} ms")
        print(f"   Best run: #{best_run}")
        
        # Save comparison data
        comparison_data = {
            'algorithm': 'SA',
            'total_evaluations': total_evaluations,
            'total_time_s': total_time,
            'initial_cost': float(opt_track[0][1]),
            'final_cost': float(best_overall_cost),
            'field_strength_mT': field_strength_mT,
            'field_std_mT': field_std_mT,
            'homogeneity_pct': homogeneity_pct,
            'n_iterations': n_iterations,
            'n_runs': n_runs,
        }
        import json as json_module
        with open('sa_comparison_metrics.json', 'w') as f:
            json_module.dump(comparison_data, f, indent=2)
        print(f"\nSaved comparison metrics to sa_comparison_metrics.json")
        
        # Cleanup
        del B_field_mT_GPU, B_magnitude_mT_GPU
        cp.get_default_memory_pool().free_all_blocks()

    return best_overall_config, best_overall_cost, opt_track, best_run, run_times
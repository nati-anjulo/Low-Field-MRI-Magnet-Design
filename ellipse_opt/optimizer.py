"""
Simulated annealing optimizer.
"""
import numpy as np
import cupy as cp
import time
from pytictoc import TicToc
t = TicToc()

from .generator import reflect_to_half_ellipse
from .cost import calculate_cost_half_ellipse
from .perturbation import Hybrid_perturbation_half_ellipse
from . import field_calculator as GRFC

# ======================================================================
# SIMULATED ANNEALING: FIRST QUADRANT REFLECTED ELLIPSE
# ======================================================================

def simulated_annealing_half_ellipse(initial_config, z_min, z_max, rx_inner, ry_inner, gap, 
                                               magnet_size, sample_points, weights, 
                                               Precalc_Bxyz_value, Ref_grid, step_size, step_divisors,
                                               initial_temp=1.0, cooling_rate=0.999, n_iterations = None,
                                               display_interval=100, verbose=True, debug_cost=False,
                                               explore_exploit_pct=0.85, fine_tune_pct=0.92):
    """
    Simulated annealing for half-ellipse ellipse with 2-way z reflection.
    
    Optimizes N magnets in Q1 (+x, +y, +z), reflects to 4N total for field calculation.
    Returns both Q1 config and full reflected config.
    """
    print("Starting simulated annealing (half-ellipse reflected ellipse)...")
    print(f"Optimizing {len(initial_config)} magnets in Q1 -> {len(initial_config)*2} total")
    
    # Reset phase tracking for this run
    if hasattr(Hybrid_perturbation_half_ellipse, '_phase'):
        Hybrid_perturbation_half_ellipse._phase = 'HIGH_TEMP'

    # Ensure GPU arrays
    if not isinstance(initial_config, cp.ndarray):
        initial_config = cp.asarray(initial_config)

    current_config = initial_config.copy()

    # Calculate initial cost (reflects internally)
    current_cost = calculate_cost_half_ellipse(
        current_config, z_min, z_max, rx_inner, ry_inner, gap, magnet_size,
        sample_points, weights, Precalc_Bxyz_value, Ref_grid, step_size
    )

    # Time estimate will be printed after first 1000 iterations
    import time as _time
    _run_start = _time.time()
    _estimate_printed = False

    if cp.isinf(current_cost):
        print("ERROR: Initial configuration is invalid!")
        cp.get_default_memory_pool().free_all_blocks()
        return None, None, float('inf'), []

    # Initialize best solution
    best_solution = current_config.copy()
    best_cost = current_cost

    # History
    cost_history = [float(current_cost)]
    track_mag_cost_values = []
    track_mag_cost_values.append([cp.asnumpy(current_config), float(current_cost)])

    temp = initial_temp
    invalid_count = 0
    
    # Overlap optimization: only compute full n² overlap every N iterations
    OVERLAP_CHECK_INTERVAL = 1  # Check every iteration (like backup)  # Full check every 20 iters, skip in between
    cached_overlap = None
    iterations_completed = 0
    improvement_count = 0
    accept_count = 0  # Track acceptances for logging
    
    # Timing accumulators
    total_perturb_time = 0.0
    total_cost_time = 0.0
    timing_count = 0

    try:
        while iterations_completed < n_iterations:
            # ===== TIMING: Perturbation =====
            t.tic()
            candidate_config = Hybrid_perturbation_half_ellipse(
                current_config, z_min, z_max, rx_inner, ry_inner, gap,
                magnet_size, temp, initial_temp, cooling_rate, n_iterations, step_divisors,
                explore_exploit_pct=explore_exploit_pct, fine_tune_pct=fine_tune_pct
            )
            cp.cuda.Stream.null.synchronize()
            perturb_time = t.tocvalue()
            total_perturb_time += perturb_time
            # ===== END TIMING =====

            if candidate_config is None:
                invalid_count += 1
                if verbose and invalid_count % 100 == 0:
                    print(f"Warning: No valid perturbation found {invalid_count} times")
                temp *= cooling_rate
                iterations_completed += 1
                timing_count += 1
                continue

            # ===== TIMING: Cost calculation =====
            t.tic()
            skip_overlap = (iterations_completed % OVERLAP_CHECK_INTERVAL != 0) and (cached_overlap is not None)
            candidate_cost = calculate_cost_half_ellipse(
                candidate_config, z_min, z_max, rx_inner, ry_inner, gap, magnet_size,
                sample_points, weights, Precalc_Bxyz_value, Ref_grid, step_size,
                skip_full_overlap=skip_overlap, cached_overlap=cached_overlap
            )
            cp.cuda.Stream.null.synchronize()
            cost_time = t.tocvalue()
            total_cost_time += cost_time
            timing_count += 1
            # ===== END TIMING =====

            # Check for invalid costs (inf, nan, or extreme values)
            if cp.isinf(candidate_cost) or cp.isnan(candidate_cost) or abs(float(candidate_cost)) > 1e9:
                invalid_count += 1  # Silent count, no print
                temp *= cooling_rate
                iterations_completed += 1
                continue

            # Metropolis criterion
            delta_cost = candidate_cost - current_cost

            accept = False
            if delta_cost <= 0:
                accept = True
                # Only update best if cost is reasonable
                if candidate_cost < best_cost and abs(float(candidate_cost)) < 1e9:
                    improvement_count += 1
                    best_solution = candidate_config.copy()
                    best_cost = candidate_cost
            else:
                accept_prob = cp.exp(-delta_cost / temp)
                accept = cp.random.random() < accept_prob

            if accept:
                current_config = candidate_config
                # Only update current_cost if it's valid
                if abs(float(candidate_cost)) < 1e9:
                    current_cost = candidate_cost
                accept_count += 1

            # Record history
            if iterations_completed % 10 == 0 or iterations_completed == n_iterations - 1:
                cost_history.append(float(current_cost))
                track_mag_cost_values.append([cp.asnumpy(current_config), float(current_cost)])

            temp *= cooling_rate
            iterations_completed += 1

            # Periodic GPU memory cleanup
            if iterations_completed % 5000 == 0:
                cp.get_default_memory_pool().free_all_blocks()
                # Save checkpoint every 50k iterations for visualization
                if iterations_completed % 5000 == 0 and iterations_completed > 0:
                    eith_oct_check = reflect_to_half_ellipse(best_solution)
                    full_config_check = cp.asnumpy(eith_oct_check)
                    
                    # Get current field stats
                    B_field_check, B_mag_check = GRFC.calculate_field(
                        eith_oct_check, sample_points,
                        Precalc_Bxyz_value, Ref_grid, step_size,
                        batch_size=1000000, num_streams=10
                    )
                    avg_f = float(cp.mean(B_mag_check))
                    rms_d = float(cp.sqrt(cp.mean((B_mag_check - avg_f)**2)))
                    homog_pct = rms_d / avg_f * 100
                    
                    np.savez(f'working_checkpoint_{iterations_completed}.npz',
                        full_config=full_config_check,
                        halbach_magnet_size=magnet_size,
                        iteration=iterations_completed,
                        field_mT=avg_f,
                        homogeneity=homog_pct
                    )
                    print(f"  >>> Checkpoint saved: iter {iterations_completed}, {avg_f:.2f} mT, {homog_pct:.2f}%")

            
            # Display progress
            if verbose and ((iterations_completed % display_interval == 0) or (iterations_completed == n_iterations)):
                # Get full config for field stats
                full_config = reflect_to_half_ellipse(best_solution)
                
                B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
                    full_config, sample_points,
                    Precalc_Bxyz_value, Ref_grid, step_size,
                    batch_size=1000000, num_streams=10
                )
                avg_field = cp.mean(B_magnitude_mT_GPU)
                rms_dev = cp.sqrt(cp.mean((B_magnitude_mT_GPU - avg_field)**2))
                homogeneity_pct = float(rms_dev / avg_field * 100)

                accept_rate = accept_count / display_interval * 100
                
                # Print timing breakdown
                avg_perturb = total_perturb_time / timing_count * 1000 if timing_count > 0 else 0
                avg_cost = total_cost_time / timing_count * 1000 if timing_count > 0 else 0
                
                print(f"Iter {iterations_completed}/{n_iterations}, T={temp:.2f}, Cost={current_cost:.2f}")
                print(f"  Accept: {accept_rate:.1f}%, Improvements: {improvement_count}, Invalid: {invalid_count}")
                print(f"  TIMING: Perturb={avg_perturb:.1f}ms, Cost={avg_cost:.1f}ms, Total={avg_perturb+avg_cost:.1f}ms")
                print(f"  Best: Field={float(avg_field):.2f} mT, RMS={float(rms_dev):.4f} mT, Homog={homogeneity_pct:.2f}%")
                
                # Print time estimate after first interval (based on real timing)
                if not _estimate_printed and iterations_completed >= 1000:
                    _elapsed = _time.time() - _run_start
                    _ms_per_iter = (_elapsed / iterations_completed) * 1000
                    _remaining_iter = n_iterations - iterations_completed
                    _est_hours = (_ms_per_iter * _remaining_iter) / 1000 / 3600
                    print(f"  >>> Estimated remaining time: {_est_hours:.1f} hours ({_ms_per_iter:.1f} ms/iter avg)")
                    _estimate_printed = True
                
                accept_count = 0  # Reset for next interval
                
                # Reset timing accumulators for next interval
                total_perturb_time = 0.0
                total_cost_time = 0.0
                timing_count = 0

                del B_field_mT_GPU, B_magnitude_mT_GPU, full_config

        if cp.isinf(best_cost):
            print("ERROR: No valid solution found!")
        else:
            print(f"\nOptimization completed. Best cost: {best_cost:.6f}")
            print(f"Total improvements: {improvement_count}")

            # Final stats
            full_config = reflect_to_half_ellipse(best_solution)
            B_field_mT_GPU, B_magnitude_mT_GPU = GRFC.calculate_field(
                full_config, sample_points,
                Precalc_Bxyz_value, Ref_grid, step_size,
                batch_size=1000000, num_streams=10
            )
            avg_field = cp.mean(B_magnitude_mT_GPU)
            rms_dev = cp.sqrt(cp.mean((B_magnitude_mT_GPU - avg_field)**2))
            homogeneity_pct = float(rms_dev / avg_field * 100)
            print(f"Final: Field={float(avg_field):.2f} mT, RMS={float(rms_dev):.4f} mT, Homog={homogeneity_pct:.2f}%")
            print(f"Q1 magnets: {len(best_solution)}, Full magnets: {len(full_config)}")
            del B_field_mT_GPU, B_magnitude_mT_GPU

    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cp.get_default_memory_pool().free_all_blocks()

    # Return Q1 config, full reflected config, cost, history
    best_full = reflect_to_half_ellipse(best_solution)
    return best_solution, best_full, best_cost, track_mag_cost_values



# ======================================================================
# RUN N TIMES: FIRST QUADRANT REFLECTED ELLIPSE
# ======================================================================

def run_n_times_half_ellipse(n_runs, initial_q1, z_min, z_max, rx_inner, ry_inner, gap,
                                       magnet_size, sample_points, weights, Precalc_Bxyz_value,
                                       Ref_grid, step_size, step_divisors,
                                       initial_temp=1.0, cooling_rate=0.999, n_iterations = None,
                                       display_interval=100, verbose=True, debug_cost=False,
                                       explore_exploit_pct=0.85, fine_tune_pct=0.92):
    """
    Run SA optimization multiple times for half-ellipse reflected ellipse.
    Returns best Q1 config and best full reflected config.
    """
    import time
    
    best_overall_q1 = None
    best_overall_full = None
    best_overall_cost = float('inf')
    best_run = 0
    run_times = []

    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{n_runs}")
        print(f"{'='*60}")

        # Reset phase tracking for each run
        if hasattr(Hybrid_perturbation_half_ellipse, '_phase'):
            Hybrid_perturbation_half_ellipse._phase = 'HIGH_TEMP'

        run_start_time = time.time()

        # Run optimization
        q1_config, full_config, best_cost, opt_track = simulated_annealing_half_ellipse(
            initial_config=initial_q1.copy(),
            z_min=z_min,
            z_max=z_max,
            rx_inner=rx_inner,
            ry_inner=ry_inner,
            gap=gap,
            magnet_size=magnet_size,
            sample_points=sample_points,
            weights=weights,
            Precalc_Bxyz_value=Precalc_Bxyz_value,
            Ref_grid=Ref_grid,
            step_size=step_size,
            step_divisors=step_divisors,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            n_iterations=n_iterations,
            display_interval=display_interval,
            verbose=verbose,
            debug_cost=debug_cost,
            explore_exploit_pct=explore_exploit_pct,
            fine_tune_pct=fine_tune_pct
        )

        run_end_time = time.time()
        run_elapsed = run_end_time - run_start_time
        run_times.append(run_elapsed)

        if q1_config is None:
            print(f"Run {i+1} failed: Initial configuration was invalid")
            continue

        print(f"Run {i+1} completed in {run_elapsed:.1f}s ({run_elapsed/60:.2f} min)")

        # Save this run
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"optimized_q1_ellipse_run_{i+1}_{timestamp}.npz"
        np.savez(filename,
                 q1_config=cp.asnumpy(q1_config),
                 full_config=cp.asnumpy(full_config),
                 cost=float(best_cost),
                 run_number=i+1,
                 rx_inner=rx_inner,
                 ry_inner=ry_inner,
                 gap=gap)
        print(f"Saved to {filename}")

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_q1 = q1_config.copy()
            best_overall_full = full_config.copy()
            best_run = i + 1

        print(f"Run {i+1} cost: {best_cost:.6f}")

    if best_overall_q1 is None:
        print("ERROR: All optimization runs failed.")
        return None, None, float('inf'), None, 0, []

    # Summary
    total_time = sum(run_times)
    avg_time = total_time / len(run_times) if run_times else 0

    print(f"\n{'='*20}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*20}")
    print(f"Best cost: {best_overall_cost:.6f} (from run {best_run})")
    print(f"Q1 magnets: {len(best_overall_q1)}, Full magnets: {len(best_overall_full)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Average time per run: {avg_time:.1f}s ({avg_time/60:.2f} min)")
    print(f"{'='*20}")

    # Save best overall
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    best_filename = f"optimized_q1_ellipse_BEST_run{best_run}_{timestamp}.npz"
    
    best_q1_cpu = cp.asnumpy(best_overall_q1)
    best_full_cpu = cp.asnumpy(best_overall_full)
    best_cost_cpu = float(best_overall_cost)
    
    del best_overall_q1, best_overall_full
    cp.get_default_memory_pool().free_all_blocks()
    
    np.savez(best_filename,
             q1_config=best_q1_cpu,
             full_config=best_full_cpu,
             cost=best_cost_cpu,
             run_number=best_run,
             rx_inner=rx_inner,
             ry_inner=ry_inner,
             gap=gap)

    print(f"Best configuration saved to {best_filename}")

    return best_q1_cpu, best_full_cpu, best_cost_cpu, opt_track, best_run, run_times

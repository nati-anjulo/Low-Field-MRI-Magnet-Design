"""
Visualization functions for magnet array optimization.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from . import field_calculator as GRFC

# Plotly renderer: set to "browser" if WebGL fails in VS Code
# Fix: VS Code → Command Palette → "Preferences: Configure Runtime Arguments"
#      In argv.json, set: "disable-hardware-acceleration": false
PLOTLY_RENDERER = None  # None = auto, "browser" = force browser

def Plot_positions_plotly(magnets_pos, dsv_points_seg):
    # Extract positions of each cuboid
    positions, _ = GRFC.input_converter_vectorized(magnets_pos)
    #dsv_points_seg = cp.asnumpy(dsv_points_seg)
                                            
    
    # Create figure with two scatter3d traces
    fig = go.Figure()
    
    # Convert CuPy arrays to NumPy only at the plotting step
    # Add magnet positions scatter plot
    fig.add_trace(go.Scatter3d(
        x=positions[:,0].get(),
        y=positions[:,1].get(),
        z=positions[:,2].get(),
        mode='markers',
        marker=dict(
            size=5,
            color='red',
        ),
        name='Magnet Positions'
    ))
    
    # Add DSV points scatter plot
    fig.add_trace(go.Scatter3d(
        x=dsv_points_seg[:,0].get(),
        y=dsv_points_seg[:,1].get(),
        z=dsv_points_seg[:,2].get(),
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
        ),
        name='DSV Points'
    ))

    # Update layout with labels and title
    fig.update_layout(
        title='Magnet Position Vs Dsv Points',
        scene=dict(
            xaxis_title='X in mm',
            yaxis_title='Y in mm',
            zaxis_title='Z in mm'
        ),
        width=700,
        height=500
    )

    # Show the plot (with fallback for WebGL issues)
    if PLOTLY_RENDERER:
        fig.show(renderer=PLOTLY_RENDERER)
    else:
        fig.show()

def plot_sa_trajectory(track_mag_cost_values):
    """
    Plot the trajectory of configurations from simulated annealing.
    
    Parameters:
    -----------
    track_mag_cost_values : list
        List of [configuration, cost] pairs from simulated annealing
    """
    # Extract configurations and create iteration numbers
    #track_mag_cost_values = cp.asnumpy(track_mag_cost_values_GPU)
    configs = [item[0] for item in track_mag_cost_values]
    costs = [item[1] for item in track_mag_cost_values]
    iterations = np.arange(len(configs))
    
    # Extract the values to plot
    # For example, plotting the first magnet's z and theta values
    # You might need to adjust these based on what specific values you want to plot
    x_values = [config[0, 0] for config in configs]  # First magnet's z-coordinate
    y_values = [config[0, 1] for config in configs]  # First magnet's theta value
    #x_values = [config[1, 1] for config in configs]  # Second magnet's theta value
    #y_values = [config[1, 2] for config in configs]  # Second magnet's phi value
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot scatter points with color gradient based on iteration
    scatter = plt.scatter(x_values, y_values, c=iterations, cmap='viridis', 
                         s=30, alpha=0.8, edgecolors='none')
    
    # Connect points with lines
    plt.plot(x_values, y_values, 'b-', alpha=0.3, linewidth=0.8)
    
    # Set labels and title
    plt.xlabel('(0,0) Value')
    plt.ylabel('(1,0) Value')
    plt.title('Trajectory ')
    
    # Add colorbar for iterations
    cbar = plt.colorbar(scatter)
    cbar.set_label('Iteration')
    
    plt.tight_layout()
    plt.show()
    # Run simulated annealing
# Plot the trajectory
def plot_cost_progression(track_mag_cost_values):
    """
    Plot the cost progression during simulated annealing optimization.
    
    Parameters:
    -----------
    track_mag_cost_values : list
        List of [configuration, cost] pairs from simulated annealing
    """
    costs = [item[1] for item in track_mag_cost_values]
    iterations = np.arange(len(costs)) * 10  # Multiply by 10 since we save every 10th iteration
    
    plt.figure(figsize=(14, 6))
    
    # Plot cost progression
    plt.plot(iterations, costs, 'b-', linewidth=2, alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Progression During Simulated Annealing', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add markers for key points
    min_cost_idx = np.argmin(costs)
    plt.plot(iterations[min_cost_idx], costs[min_cost_idx], 'ro', markersize=10, 
             label=f'Best: {costs[min_cost_idx]:.1f} at iter {iterations[min_cost_idx]}')
    
    # Add exploration/exploitation transition line (approximate at 65% of iterations)
    if len(iterations) > 0:
        transition_iter = int(0.65 * iterations[-1])
        plt.axvline(x=transition_iter, color='r', linestyle='--', alpha=0.5, 
                   label=f'Exploration→Exploitation (~{transition_iter})')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Cost progression summary:")
    print(f"  Initial cost: {costs[0]:.1f}")
    print(f"  Final cost: {costs[-1]:.1f}")
    print(f"  Best cost: {costs[min_cost_idx]:.1f} (at iteration {iterations[min_cost_idx]})")
    print(f"  Total improvement: {costs[-1] - costs[0]:.1f}")
    print(f"  Improvement rate: {(costs[-1] - costs[0]) / len(costs):.1f} per checkpoint")
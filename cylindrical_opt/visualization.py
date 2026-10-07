"""
Visualization functions for magnet array optimization.

Provides:
- Plot_positions_plotly: Scatter plot of magnet positions and DSV points
- plot_cost_progression: Cost vs iteration plot for SA optimization
- show_magnets_plotly: 3D magnet viewer with N/S pole coloring (magpylib-style)
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from . import field_calculator as GRFC


def safe_show(fig):
    """Display Plotly figure with automatic WebGL fallback."""
    try:
        fig.show()
    except Exception:
        try:
            fig.show(renderer="browser")
        except Exception:
            print("Could not display figure. Try: fig.show(renderer='browser')")

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
    safe_show(fig)

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
    Filters out corrupted/extreme cost values for proper visualization.
    """
    costs = [item[1] for item in track_mag_cost_values]
    iterations = np.arange(len(costs)) * 10

    # Filter out extreme values (corrupted costs)
    valid_mask = np.array([abs(c) < 1e9 for c in costs])
    valid_costs = np.array(costs)[valid_mask]
    valid_iterations = iterations[valid_mask]

    if len(valid_costs) == 0:
        print("ERROR: No valid cost values to plot!")
        return

    # Check if there were corrupted values
    n_corrupted = len(costs) - len(valid_costs)
    if n_corrupted > 0:
        print(f"WARNING: Filtered out {n_corrupted} corrupted cost values (|cost| > 1e9)")

    plt.figure(figsize=(14, 6))

    # Plot cost progression
    plt.plot(valid_iterations, valid_costs, 'b-', linewidth=2, alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Progression During Simulated Annealing', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Disable scientific notation on y-axis
    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add markers for key points
    min_cost_idx = np.argmin(valid_costs)
    plt.plot(valid_iterations[min_cost_idx], valid_costs[min_cost_idx], 'ro', markersize=10,
             label=f'Best: {valid_costs[min_cost_idx]:,.0f} at iter {valid_iterations[min_cost_idx]}')

    # Mark where corruption started (if any)
    if n_corrupted > 0:
        corruption_start = valid_iterations[-1]
        plt.axvline(x=corruption_start, color='orange', linestyle='--', alpha=0.7,
                   label=f'Corruption started (~{int(corruption_start)})')

    # Add exploration/exploitation transition line
    if len(valid_iterations) > 0:
        transition_iter = int(0.65 * iterations[-1])
        plt.axvline(x=transition_iter, color='r', linestyle='--', alpha=0.5,
                   label=f'Exploration→Exploitation (~{transition_iter})')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print(f"Cost progression summary:")
    print(f"  Initial cost: {valid_costs[0]:,.0f}")
    print(f"  Final valid cost: {valid_costs[-1]:,.0f}")
    print(f"  Best cost: {valid_costs[min_cost_idx]:,.0f} (at iteration {valid_iterations[min_cost_idx]})")


def show_magnets_plotly(config, magnet_size, show=False):
    """
    3D magnet visualization with N/S pole colors using Plotly.
    Red = North (+Y face), Green = South (-Y face), White = sides.

    Matches magpylib's visual style with proper 3D lighting.
    Automatically handles WebGL fallback to browser if needed.

    Parameters:
    -----------
    config : ndarray
        Magnet configuration array [z, r, theta, phi] per magnet (cylindrical coords)
    magnet_size : float
        Size of cubic magnet in meters
    show : bool
        If True, display figure automatically with WebGL fallback (default: True)

    Returns:
    --------
    fig : plotly Figure
        Interactive 3D figure
    """
    config_np = np.asarray(config)
    if config_np.ndim == 1:
        config_np = config_np.reshape(-1, 4)

    size_mm = magnet_size * 1000
    half = size_mm / 2
    n = len(config_np)

    # Extract cylindrical coords and convert to Cartesian
    z_pos = config_np[:, 0] * 1000  # mm
    r_pos = config_np[:, 1] * 1000  # mm
    theta_pos = config_np[:, 2]
    phi_rot = config_np[:, 3]

    x_pos = r_pos * np.cos(theta_pos)
    y_pos = r_pos * np.sin(theta_pos)

    # Base cube vertices (centered at origin)
    cube_verts = np.array([
        [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
    ]) * half

    # VECTORIZED rotation around Z-axis (much faster than loop)
    cos_phi = np.cos(phi_rot)[:, np.newaxis]  # (n, 1)
    sin_phi = np.sin(phi_rot)[:, np.newaxis]  # (n, 1)

    # Broadcast cube vertices to all magnets: (n, 8, 3)
    verts_x = cube_verts[:, 0]  # (8,)
    verts_y = cube_verts[:, 1]  # (8,)
    verts_z = cube_verts[:, 2]  # (8,)

    # Apply Z-rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
    rot_x = cos_phi * verts_x - sin_phi * verts_y  # (n, 8)
    rot_y = sin_phi * verts_x + cos_phi * verts_y  # (n, 8)
    rot_z = np.tile(verts_z, (n, 1))               # (n, 8)

    # Translate to magnet positions
    all_verts = np.stack([
        rot_x + x_pos[:, np.newaxis],
        rot_y + y_pos[:, np.newaxis],
        rot_z + z_pos[:, np.newaxis]
    ], axis=-1)  # (n, 8, 3)

    # All 6 faces per cube as one mesh with vertex-based coloring
    # Vertex intensity based on local Y coordinate creates red-white-green gradient
    # y=+1 (North) = intensity 1 = red
    # y=-1 (South) = intensity 0 = green
    # Faces in between get interpolated colors (wrapping effect)

    # All 6 faces: order matters for consistent indexing
    face_idx = [
        [1, 2, 6, 5],  # North (+Y) - all y=+1
        [0, 3, 7, 4],  # South (-Y) - all y=-1
        [0, 1, 5, 4],  # Left (-X) - mixed y
        [3, 2, 6, 7],  # Right (+X) - mixed y
        [0, 1, 2, 3],  # Bottom (-Z) - mixed y
        [4, 5, 6, 7],  # Top (+Z) - mixed y
    ]

    # Build all faces
    all_faces = np.concatenate([all_verts[:, f, :] for f in face_idx]).reshape(-1, 3)

    # Build intensity based on original (unrotated) vertex Y coordinate
    # Vertices 1,2,5,6 have y=+1 (North=red), vertices 0,3,4,7 have y=-1 (South=green)
    vert_intensity = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # 1=North(red), 0=South(green)
    face_intensities = np.concatenate([
        np.tile(vert_intensity[f], n) for f in face_idx
    ])

    def make_indices(n_quads):
        base = np.arange(n_quads) * 4
        i = np.stack([base, base], axis=1).flatten()
        j = np.stack([base + 1, base + 2], axis=1).flatten()
        k = np.stack([base + 2, base + 3], axis=1).flatten()
        return i, j, k

    fi, fj, fk = make_indices(n * 6)  # 6 faces per magnet

    # Lighting settings - smoother appearance
    lighting = dict(
        ambient=0.5,
        diffuse=0.9,
        specular=0.2,
        roughness=0.6,
        fresnel=0.1
    )
    lightposition = dict(x=200, y=200, z=400)

    # Red-White-Green colorscale (matched to magpylib)
    colorscale = [
        [0.0, 'rgb(40, 160, 40)'],     # Green (South)
        [0.5, 'rgb(240, 240, 240)'],   # Light gray (middle)
        [1.0, 'rgb(200, 50, 50)'],     # Red (North)
    ]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=all_faces[:,0], y=all_faces[:,1], z=all_faces[:,2],
            i=fi, j=fj, k=fk,
            intensity=face_intensities,
            colorscale=colorscale,
            showscale=False,
            flatshading=False,
            lighting=lighting,
            lightposition=lightposition,
        ),
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1, y=-1.5, z=1.4)),
            bgcolor='white'
        ),
        width=None, height=500,  # None = auto-fill cell width
        margin=dict(l=0, r=0, t=30, b=0),
        annotations=[
            dict(
                text=f"Collection ({n} magnets)",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=12),
                xanchor='right', yanchor='top'
            )
        ]
    )

    if show:
        safe_show(fig)

    return fig
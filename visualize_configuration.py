"""
Magnet Configuration Visualization Module.

Provides matplotlib and magpylib-based visualization for magnet arrays.
Uses the local field_calculator module for field computation.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
import plotly.graph_objects as go

# Use local field calculator (self-contained, no parent folder dependencies)
# Try ellipse_opt first, fall back to cylindrical_opt
try:
    from ellipse_opt import field_calculator as GRFC
except ImportError:
    from cylindrical_opt import field_calculator as GRFC


def calculate_field_in_plane(config, r_max, plane, z_value, n_points, z_min, z_max,
                             Precalc_Bxyz_value, Ref_grid, step_size):
    """
    Calculate magnetic field in a specified plane using precalculated fields.

    Parameters:
    -----------
    config : numpy.ndarray
        Magnet configuration [z, r, theta, phi] per magnet
    r_max : float
        Maximum radius for field of view
    plane : str
        Plane to sample ('xy', 'xz', or 'yz')
    z_value : float
        Fixed value for the third coordinate
    n_points : int
        Number of points along each axis
    z_min, z_max : float
        Z-axis limits
    Precalc_Bxyz_value, Ref_grid, step_size : field data

    Returns:
    --------
    X, Y : numpy.ndarray
        Meshgrid of coordinates
    B_mag : numpy.ndarray
        Magnitude of B field at each point in mT
    """
    fov = r_max * 1.5

    if plane == 'xy':
        x = np.linspace(-fov, fov, n_points)
        y = np.linspace(-fov, fov, n_points)
        X, Y = np.meshgrid(x, y)
        positions = np.zeros((n_points**2, 3))
        positions[:, 0] = X.flatten()
        positions[:, 1] = Y.flatten()
        positions[:, 2] = z_value
    elif plane == 'xz':
        x = np.linspace(-fov, fov, n_points)
        z = np.linspace(z_min * 1.5, z_max * 1.5, n_points)
        X, Z = np.meshgrid(x, z)
        positions = np.zeros((n_points**2, 3))
        positions[:, 0] = X.flatten()
        positions[:, 1] = z_value
        positions[:, 2] = Z.flatten()
    elif plane == 'yz':
        y = np.linspace(-fov, fov, n_points)
        z = np.linspace(z_min * 1.5, z_max * 1.5, n_points)
        Y, Z = np.meshgrid(y, z)
        positions = np.zeros((n_points**2, 3))
        positions[:, 0] = z_value
        positions[:, 1] = Y.flatten()
        positions[:, 2] = Z.flatten()
    else:
        raise ValueError(f"Invalid plane: {plane}")

    # Calculate fields using local field calculator
    config_gpu = cp.asarray(config, dtype=cp.float32)
    pos_gpu = cp.asarray(positions, dtype=cp.float32)
    _, B_mag = GRFC.calc_field(config_gpu, pos_gpu)
    B_mag_np = cp.asnumpy(B_mag).reshape(n_points, n_points)

    if plane == 'xy':
        return X, Y, B_mag_np
    elif plane == 'xz':
        return X, Z, B_mag_np
    else:
        return Y, Z, B_mag_np


def visualize_configuration(config, n_magnets, z_min, z_max, r_min, vmax_field, magnet_size,
                           sample_points, sphere_radius, Precalc_Bxyz_value, Ref_grid, step_size):
    """
    Visualize magnet configuration and resulting magnetic fields with 2D field magnitude planes.
    """
    print("Visualizing configuration...")
    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Get magnet positions from cylindrical config [z, r, theta, phi]
    config_np = np.asarray(config).reshape(-1, 4)
    z_pos = config_np[:, 0]
    r_pos = config_np[:, 1]
    theta_pos = config_np[:, 2]
    phi_rot = config_np[:, 3]

    x_pos = r_pos * np.cos(theta_pos)
    y_pos = r_pos * np.sin(theta_pos)
    positions = np.column_stack([x_pos, y_pos, z_pos])
    rotations = np.column_stack([np.zeros(len(config_np)), np.zeros(len(config_np)), phi_rot])

    # Calculate field on sample points
    config_gpu = cp.asarray(config_np, dtype=cp.float32)
    sample_gpu = cp.asarray(sample_points, dtype=cp.float32)
    _, B_magnitude = GRFC.calc_field(config_gpu, sample_gpu)
    B_magnitude = cp.asnumpy(B_magnitude)

    # 3D plot of magnet configuration
    ax1 = fig.add_subplot(gs[:2, 0], projection='3d')

    # Plot cylinder outline
    z = np.linspace(z_min, z_max, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    z_grid, theta_grid = np.meshgrid(z, theta)
    x_grid = r_min * np.cos(theta_grid)
    y_grid = r_min * np.sin(theta_grid)
    ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='gray')

    # Plot magnets as cubes
    for i in range(min(n_magnets, len(positions))):
        s = magnet_size / 2
        pos = positions[i]

        faces = [
            [[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]],
            [[-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]],
            [[-s, -s, -s], [-s, s, -s], [-s, s, s], [-s, -s, s]],
            [[s, -s, -s], [s, s, -s], [s, s, s], [s, -s, s]],
            [[-s, -s, -s], [s, -s, -s], [s, -s, s], [-s, -s, s]],
            [[-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s]]
        ]

        try:
            rot = R.from_euler('xyz', rotations[i])
            for face in faces:
                for j in range(len(face)):
                    face[j] = rot.apply(face[j]) + pos
        except:
            for face in faces:
                for j in range(len(face)):
                    face[j] = np.array(face[j]) + pos

        for face in faces:
            face = np.array(face)
            ax1.plot3D(face[:, 0], face[:, 1], face[:, 2], 'b-')
            try:
                verts = [(face[j, 0], face[j, 1], face[j, 2]) for j in range(len(face))]
                ax1.add_collection3d(art3d.Poly3DCollection([verts], alpha=0.5, color='b'))
            except:
                pass

        try:
            arrow_length = magnet_size
            magnetization_dir = rot.apply([0, 1, 0])
            ax1.quiver(pos[0], pos[1], pos[2],
                      magnetization_dir[0] * arrow_length,
                      magnetization_dir[1] * arrow_length,
                      magnetization_dir[2] * arrow_length,
                      color='r', arrow_length_ratio=0.3)
        except:
            pass

    ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                color='g', alpha=0.3, s=20)

    limit = max(r_min, r_min - z_min) * 1.5
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_zlim(-limit, limit)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('Magnet Configuration')

    # Field magnitude on sphere
    ax_comps = fig.add_subplot(gs[2:, 0], projection='3d')
    vmin = np.min(B_magnitude)
    vmax = np.max(B_magnitude)
    sc = ax_comps.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                          c=B_magnitude, cmap='viridis', s=50, vmin=vmin, vmax=min(vmax, 100))
    cbar = plt.colorbar(sc, ax=ax_comps, shrink=0.7)
    cbar.set_label('Field Magnitude [mT]')
    limit = sphere_radius * 1.2
    ax_comps.set_xlim(-limit, limit)
    ax_comps.set_ylim(-limit, limit)
    ax_comps.set_zlim(-limit, limit)
    ax_comps.set_xlabel('X [m]')
    ax_comps.set_ylabel('Y [m]')
    ax_comps.set_zlabel('Z [m]')
    ax_comps.set_title('Magnetic Field Magnitude on Sphere')

    vmin_field = 0

    # XY plane at z=0
    print("Calculating and plotting XY plane field...")
    X_xy, Y_xy, B_xy = calculate_field_in_plane(config, r_min, 'xy', 0, 50, z_min, z_max,
                                                 Precalc_Bxyz_value, Ref_grid, step_size)
    ax_xy = fig.add_subplot(gs[0, 1])
    im_xy = ax_xy.pcolormesh(X_xy, Y_xy, B_xy, cmap='viridis', shading='auto', vmin=vmin_field, vmax=vmax_field)
    plt.colorbar(im_xy, ax=ax_xy, label='Field Magnitude [mT]')
    circle = plt.Circle((0, 0), r_min, fill=False, edgecolor='r', linestyle='-')
    ax_xy.add_patch(circle)
    ax_xy.set_xlabel('X [m]')
    ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Field Magnitude in XY Plane (z=0)')
    ax_xy.set_aspect('equal')

    # XZ plane at y=0
    print("Calculating and plotting XZ plane field...")
    X_xz, Z_xz, B_xz = calculate_field_in_plane(config, r_min, 'xz', 0, 50, z_min, z_max,
                                                 Precalc_Bxyz_value, Ref_grid, step_size)
    ax_xz = fig.add_subplot(gs[1, 1])
    im_xz = ax_xz.pcolormesh(X_xz, Z_xz, B_xz, cmap='viridis', shading='auto', vmin=vmin_field, vmax=vmax_field)
    plt.colorbar(im_xz, ax=ax_xz, label='Field Magnitude [mT]')
    ax_xz.plot([r_min, r_min], [z_min, z_max], 'r-')
    ax_xz.plot([-r_min, -r_min], [z_min, z_max], 'r-')
    ax_xz.set_xlabel('X [m]')
    ax_xz.set_ylabel('Z [m]')
    ax_xz.set_title('Field Magnitude in XZ Plane (y=0)')

    # YZ plane at x=0
    print("Calculating and plotting YZ plane field...")
    Y_yz, Z_yz, B_yz = calculate_field_in_plane(config, r_min, 'yz', 0, 50, z_min, z_max,
                                                 Precalc_Bxyz_value, Ref_grid, step_size)
    ax_yz = fig.add_subplot(gs[2, 1])
    im_yz = ax_yz.pcolormesh(Y_yz, Z_yz, B_yz, cmap='viridis', shading='auto', vmin=vmin_field, vmax=vmax_field)
    plt.colorbar(im_yz, ax=ax_yz, label='Field Magnitude [mT]')
    ax_yz.plot([r_min, r_min], [z_min, z_max], 'r-')
    ax_yz.plot([-r_min, -r_min], [z_min, z_max], 'r-')
    ax_yz.set_xlabel('Y [m]')
    ax_yz.set_ylabel('Z [m]')
    ax_yz.set_title('Field Magnitude in YZ Plane (x=0)')

    plt.tight_layout()
    print("Finished creating visualization. Displaying plot...")
    plt.show()


def rad_halbach_Design_rad(halbach_magnet_size, params):
    """
    Create magpylib Collection from cylindrical config [z, r, theta, phi].

    Args:
        halbach_magnet_size: Size of cubic magnets (meters)
        params: Array of [z, r, theta, phi] per magnet

    Returns:
        magpy.Collection of Cuboid magnets
    """
    rad_magnet_coll = magpy.Collection()
    params_reshaped = params.reshape(-1, 4)

    for param in params_reshaped:
        z, rad, theta, rot = param
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        cuboid = magpy.magnet.Cuboid(
            polarization=(0, 1.48, 0),  # Br = 1.48 T (N52 NdFeB)
            dimension=(halbach_magnet_size, halbach_magnet_size, halbach_magnet_size),
            position=(x, y, z),
        )
        cuboid.orientation = R.from_rotvec((0, 0, rot))
        rad_magnet_coll.add(cuboid)

    return rad_magnet_coll


def rad_halbach_Design_cartesian(halbach_magnet_size, params):
    """
    Create magpylib Collection from Cartesian config [x, y, z, phi].

    Args:
        halbach_magnet_size: Size of cubic magnets (meters)
        params: Array of [x, y, z, phi] per magnet

    Returns:
        magpy.Collection of Cuboid magnets
    """
    magnet_coll = magpy.Collection()
    params_reshaped = params.reshape(-1, 4)

    for param in params_reshaped:
        x, y, z, rot = param
        cuboid = magpy.magnet.Cuboid(
            polarization=(0, 1.48, 0),
            dimension=(halbach_magnet_size, halbach_magnet_size, halbach_magnet_size),
            position=(x, y, z),
        )
        cuboid.orientation = R.from_rotvec((0, 0, rot))
        magnet_coll.add(cuboid)

    return magnet_coll


def cylindrical_to_cartesian(r, theta):
    """Convert cylindrical (r, theta) to Cartesian (x, y)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

"""
Field Validation Module - Magpylib-based field validation for optimized designs.

Uses magpylib's analytical calculation to cross-validate the GPU precomputed
field lookup used during optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy


def magnitude_plot(magnet, axis_range=0.07):
    """
    Plot field magnitude along X, Y, Z axes through center.

    Args:
        magnet: magpylib Collection object
        axis_range: half-length of axis to sample (default 0.07m = 70mm)

    Returns:
        mag_Bx, mag_By, mag_Bz: Field magnitudes along each axis (in Tesla)
    """
    points_z = np.linspace((0, 0, -axis_range), (0, 0, axis_range), 100)
    points_y = np.linspace((0, -axis_range, 0), (0, axis_range, 0), 100)
    points_x = np.linspace((-axis_range, 0, 0), (axis_range, 0, 0), 100)

    B_zaxis = magpy.getB(magnet, points_z)
    B_yaxis = magpy.getB(magnet, points_y)
    B_xaxis = magpy.getB(magnet, points_x)

    Bx_x, By_x, Bz_x = B_xaxis[:,0], B_xaxis[:,1], B_xaxis[:,2]
    Bx_y, By_y, Bz_y = B_yaxis[:,0], B_yaxis[:,1], B_yaxis[:,2]
    Bx_z, By_z, Bz_z = B_zaxis[:,0], B_zaxis[:,1], B_zaxis[:,2]

    mag_Bx = np.sqrt(Bx_x**2 + By_x**2 + Bz_x**2)
    mag_By = np.sqrt(Bx_y**2 + By_y**2 + Bz_y**2)
    mag_Bz = np.sqrt(Bx_z**2 + By_z**2 + Bz_z**2)

    plt.figure(figsize=(10, 8))
    plt.plot(mag_By*1000, color="g", label="Magnitude Y")
    plt.legend()

    return mag_Bx, mag_By, mag_Bz


def overlay_comparison(initial_view, optimized_view, axis_range=0.07,
                       initial_label="Initial", optimized_label="Optimized"):
    """
    Plot overlay comparison of initial vs optimized field magnitude along Y-axis.

    Args:
        initial_view: magpylib Collection for initial configuration
        optimized_view: magpylib Collection for optimized configuration
        axis_range: half-length of axis to sample (in meters)
        initial_label: label for initial plot
        optimized_label: label for optimized plot

    Returns:
        mag_init, mag_opt: Field magnitudes in mT
    """
    points_y = np.linspace((0, -axis_range, 0), (0, axis_range, 0), 100)

    B_init_y = magpy.getB(initial_view, points_y)
    B_opt_y = magpy.getB(optimized_view, points_y)

    mag_init = np.linalg.norm(B_init_y, axis=1) * 1000  # mT
    mag_opt = np.linalg.norm(B_opt_y, axis=1) * 1000    # mT

    x_mm = np.linspace(-axis_range*1000, axis_range*1000, 100)

    plt.figure(figsize=(10, 8))
    plt.plot(x_mm, mag_init, color="b", label=initial_label, linewidth=2, linestyle='--')
    plt.plot(x_mm, mag_opt, color="g", label=optimized_label, linewidth=2)
    plt.xlabel('Position along Y-axis (mm)')
    plt.ylabel('Field Magnitude (mT)')
    plt.title('Field Magnitude Comparison: Initial vs Optimized')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return mag_init, mag_opt

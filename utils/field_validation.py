"""
Field Validation Module - Magpylib-based field validation for optimized designs.

Uses magpylib's analytical calculation to cross-validate the GPU precomputed
field lookup used during optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy


def magnitude_plot(magnet, axis_range=0.07, xlim=None, ylim=None):
    """Plot field magnitude |B| along Y-axis through center."""
    points_y = np.linspace((0, -axis_range, 0), (0, axis_range, 0), 100)

    B_yaxis = magpy.getB(magnet, points_y)
    Bx_y = B_yaxis[:,0]
    By_y = B_yaxis[:,1]
    Bz_y = B_yaxis[:,2]
    mag_By = np.sqrt(Bx_y**2 + By_y**2 + Bz_y**2)

    x_mm = np.linspace(-axis_range*1000, axis_range*1000, 100)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(x_mm, mag_By*1000, 'g-', linewidth=3)
    ax.set_title(r'Field magnitude $|\mathbf{B}|$ along Y-axis', fontsize=16)
    ax.set_xlabel('Position along Y-axis (mm)', fontsize=16)
    ax.set_ylabel(r'$|\mathbf{B}|$ (mT)', fontsize=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('field_magnitude.png', dpi=600)
    plt.show()

    return mag_By


def overlay_comparison(initial_view, optimized_view, axis_range=0.07,
                       initial_label="Initial", optimized_label="Optimized"):
    """Plot overlay comparison of initial vs optimized field magnitude along Y-axis."""
    points_y = np.linspace((0, -axis_range, 0), (0, axis_range, 0), 100)

    B_init_y = magpy.getB(initial_view, points_y)
    B_opt_y = magpy.getB(optimized_view, points_y)

    # Calculate magnitude
    mag_init = np.sqrt(B_init_y[:,0]**2 + B_init_y[:,1]**2 + B_init_y[:,2]**2) * 1000  # mT
    mag_opt = np.sqrt(B_opt_y[:,0]**2 + B_opt_y[:,1]**2 + B_opt_y[:,2]**2) * 1000  # mT

    x_mm = np.linspace(-axis_range*1000, axis_range*1000, 100)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(x_mm, mag_init, 'b--', linewidth=2.5, label=initial_label)
    ax.plot(x_mm, mag_opt, 'g-', linewidth=3, label=optimized_label)
    ax.set_xlabel('Position along Y-axis (mm)', fontsize=16)
    ax.set_ylabel('|B| (mT)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('field_comparison.png', dpi=150)
    plt.show()

    return mag_init, mag_opt

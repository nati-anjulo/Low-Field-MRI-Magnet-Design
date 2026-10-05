"""
Analytical Magnetic Field Calculation - Vectorized Implementation

Implements Engel-Herbert & Hesjedal formulas for cuboid magnet stray fields.
Reference: J. Appl. Phys. 97, 074504 (2005)

Supports both NumPy (CPU) and CuPy (GPU) backends.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np


def cuboid_field(points, polarization, dimensions, position=None, use_gpu=False):
    """
    Calculate magnetic field from a uniformly magnetized cuboid.

    Fully vectorized implementation of Engel-Herbert & Hesjedal analytical formulas.

    Parameters
    ----------
    points : array, shape (N, 3)
        Observation points [x, y, z] in meters
    polarization : array, shape (3,)
        Polarization vector [Jx, Jy, Jz] in Tesla
    dimensions : array, shape (3,)
        Cuboid dimensions [dx, dy, dz] in meters (full width)
    position : array, shape (3,), optional
        Cuboid center position [x0, y0, z0] in meters (default: origin)
    use_gpu : bool, optional
        Use CuPy for GPU acceleration (default: False)

    Returns
    -------
    B : array, shape (N, 3)
        Magnetic field [Bx, By, Bz] in Tesla

    Example
    -------
    >>> points = np.array([[0.02, 0, 0], [0, 0.02, 0]])
    >>> B = cuboid_field(points, [0, 1.48, 0], [0.01, 0.01, 0.01])
    """
    xp = cp if (use_gpu and HAS_CUPY) else np

    points = xp.asarray(points)
    J = xp.asarray(polarization)
    dims = xp.asarray(dimensions)

    if position is None:
        position = xp.zeros(3)
    else:
        position = xp.asarray(position)

    # Translate to cuboid frame
    r = points - position

    # Half dimensions
    a, b, c = dims / 2

    # Corner offsets: 8 corners with signs (±a, ±b, ±c)
    signs = xp.array([
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
    ], dtype=r.dtype)

    half_dims = xp.array([a, b, c], dtype=r.dtype)

    # Initialize field
    B = xp.zeros_like(r)

    # Regularization for numerical stability
    eps = 1e-10

    # Sum over 8 corners
    for i in range(8):
        sx, sy, sz = signs[i]

        # Position relative to corner
        x = r[:, 0] - sx * a
        y = r[:, 1] - sy * b
        z = r[:, 2] - sz * c

        # Distance from corner (regularized)
        R = xp.sqrt(x**2 + y**2 + z**2 + eps**2)

        # Corner sign factor
        sign = sx * sy * sz

        # Regularized log arguments (must be positive)
        log_xR = xp.log(xp.abs(x) + R + eps)
        log_yR = xp.log(xp.abs(y) + R + eps)
        log_zR = xp.log(xp.abs(z) + R + eps)

        # Regularized coordinates for atan
        x_s = xp.where(xp.abs(x) < eps, eps * xp.sign(x + eps), x)
        y_s = xp.where(xp.abs(y) < eps, eps * xp.sign(y + eps), y)
        z_s = xp.where(xp.abs(z) < eps, eps * xp.sign(z + eps), z)

        # Field contributions (Engel-Herbert formulas)
        # Bx
        B[:, 0] += sign * (
            J[0] * xp.arctan2(y_s * z_s, x_s * R)
            - J[1] * log_zR
            - J[2] * log_yR
        )

        # By
        B[:, 1] += sign * (
            -J[0] * log_zR
            + J[1] * xp.arctan2(x_s * z_s, y_s * R)
            - J[2] * log_xR
        )

        # Bz
        B[:, 2] += sign * (
            -J[0] * log_yR
            - J[1] * log_xR
            + J[2] * xp.arctan2(x_s * y_s, z_s * R)
        )

    # Apply 1/(4π) factor
    B /= (4 * xp.pi)

    return B


def cuboid_field_multi(points, magnets, use_gpu=False):
    """
    Calculate total field from multiple cuboid magnets (superposition).

    Parameters
    ----------
    points : array, shape (N, 3)
        Observation points [x, y, z] in meters
    magnets : list of dict
        Each magnet dict has keys:
        - 'polarization': [Jx, Jy, Jz] in Tesla
        - 'dimensions': [dx, dy, dz] in meters
        - 'position': [x0, y0, z0] in meters
    use_gpu : bool, optional
        Use CuPy for GPU acceleration

    Returns
    -------
    B : array, shape (N, 3)
        Total magnetic field [Bx, By, Bz] in Tesla

    Example
    -------
    >>> magnets = [
    ...     {'polarization': [0, 1.48, 0], 'dimensions': [0.01]*3, 'position': [-0.02, 0, 0]},
    ...     {'polarization': [0, -1.48, 0], 'dimensions': [0.01]*3, 'position': [0.02, 0, 0]},
    ... ]
    >>> B = cuboid_field_multi(points, magnets)
    """
    xp = cp if (use_gpu and HAS_CUPY) else np
    points = xp.asarray(points, dtype=xp.float64)
    B_total = xp.zeros_like(points, dtype=xp.float64)

    for mag in magnets:
        B = cuboid_field(
            points,
            mag['polarization'],
            mag['dimensions'],
            mag.get('position', [0, 0, 0]),
            use_gpu=use_gpu
        )
        B_total += B

    return B_total


def generate_grid_field(grid_extent, magnet_size, step_size,
                        polarization=(0, 1.48, 0), position=(0, 0, 0), use_gpu=False):
    """
    Calculate field on a 3D grid for a single cuboid magnet.

    Parameters
    ----------
    grid_extent : float
        Grid spans [-grid_extent, +grid_extent] in all directions (meters)
    magnet_size : float
        Cubic magnet edge length (meters)
    step_size : float
        Grid spacing (meters)
    polarization : tuple, optional
        Polarization vector (Jx, Jy, Jz) in Tesla (default: +Y, 1.48T)
    position : tuple, optional
        Magnet center position (default: origin)
    use_gpu : bool, optional
        Use GPU acceleration

    Returns
    -------
    B : array, shape (N, 3)
        Field vectors at grid points
    grid : array
        1D grid coordinates
    points : array, shape (N, 3)
        Grid point coordinates

    Example
    -------
    >>> B, grid, pts = generate_grid_field(0.05, 0.01, 0.001)
    >>> print(f"Grid: {len(grid)} points, Field shape: {B.shape}")
    """
    xp = cp if (use_gpu and HAS_CUPY) else np

    grid = xp.arange(-grid_extent, grid_extent, step_size)
    X, Y, Z = xp.meshgrid(grid, grid, grid, indexing='ij')
    points = xp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    dimensions = (magnet_size, magnet_size, magnet_size)
    B = cuboid_field(points, polarization, dimensions, position, use_gpu=use_gpu)

    return B, grid, points


# Convenience aliases
single_magnet_field = generate_grid_field
multi_magnet_field = cuboid_field_multi

"""
Analytical Magnetic Field - Fully Vectorized GPU Implementation

Implements Engel-Herbert & Hesjedal formulas with NO loops.
All 8 corner contributions computed via broadcasting.

Reference: J. Appl. Phys. 97, 074504 (2005)
"""

try:
    import cupy as cp
    xp = cp
except ImportError:
    import numpy as cp
    xp = cp


def _cuboid_field_batch(r, J, half_dims, corner_signs, sigma):
    """Core vectorized computation for a batch of points."""
    # r: (N, 3), half_dims: (3,), corner_signs: (8, 3), sigma: (8,)

    # Relative positions to all 8 corners: (N, 8, 3)
    rel_pos = r[:, None, :] - corner_signs[None, :, :] * half_dims[None, None, :]

    x = rel_pos[:, :, 0]  # (N, 8)
    y = rel_pos[:, :, 1]
    z = rel_pos[:, :, 2]

    R = xp.sqrt(x**2 + y**2 + z**2)

    # Log arguments: use (x + R), not (|x| + R) - sign matters for off-axis points
    log_xR = xp.log(x + R)
    log_yR = xp.log(y + R)
    log_zR = xp.log(z + R)

    # Arctan terms
    atan_yz_xR = xp.arctan2(y * z, x * R)
    atan_xz_yR = xp.arctan2(x * z, y * R)
    atan_xy_zR = xp.arctan2(x * y, z * R)

    # Sum over 8 corners
    Bx = xp.sum(sigma * (J[0] * atan_yz_xR - J[1] * log_zR - J[2] * log_yR), axis=1)
    By = xp.sum(sigma * (-J[0] * log_zR + J[1] * atan_xz_yR - J[2] * log_xR), axis=1)
    Bz = xp.sum(sigma * (-J[0] * log_yR - J[1] * log_xR + J[2] * atan_xy_zR), axis=1)

    return xp.stack([Bx, By, Bz], axis=1) / (4 * xp.pi)


def cuboid_field_gpu(points, polarization, dimensions, position=None, batch_size=100000):
    """
    Calculate magnetic field from cuboid magnet - fully vectorized, no loops.

    Parameters
    ----------
    points : array, shape (N, 3)
        Observation points [x, y, z] in meters
    polarization : array, shape (3,)
        Polarization vector [Jx, Jy, Jz] in Tesla
    dimensions : array, shape (3,)
        Cuboid dimensions [dx, dy, dz] in meters
    position : array, shape (3,), optional
        Cuboid center (default: origin)
    batch_size : int, optional
        Process points in batches to manage memory (default: 100000)

    Returns
    -------
    B : array, shape (N, 3)
        Magnetic field [Bx, By, Bz] in Tesla
    """
    # Preserve input dtype for precision (float64 for near-surface accuracy)
    points = xp.asarray(points)
    dtype = points.dtype
    J = xp.asarray(polarization, dtype=dtype)
    dims = xp.asarray(dimensions, dtype=dtype)

    if position is None:
        pos = xp.zeros(3, dtype=dtype)
    else:
        pos = xp.asarray(position, dtype=dtype)

    # Translate to magnet frame
    r = points - pos

    # Half dimensions
    half_dims = dims / 2

    # 8 corner signs and sigma (precomputed)
    corner_signs = xp.array([
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
    ], dtype=dtype)
    sigma = corner_signs[:, 0] * corner_signs[:, 1] * corner_signs[:, 2]

    N = len(points)

    # Process in batches for memory efficiency
    if N <= batch_size:
        return _cuboid_field_batch(r, J, half_dims, corner_signs, sigma)

    # Batched processing
    B = xp.zeros((N, 3), dtype=dtype)
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        B[i:end] = _cuboid_field_batch(r[i:end], J, half_dims, corner_signs, sigma)

    return B


def multi_magnet_field_gpu(points, positions, polarizations, dimensions):
    """
    Calculate total field from multiple magnets - fully vectorized.

    Parameters
    ----------
    points : array, shape (N, 3)
        Observation points
    positions : array, shape (M, 3)
        Magnet center positions
    polarizations : array, shape (M, 3)
        Polarization vectors for each magnet
    dimensions : array, shape (3,) or (M, 3)
        Magnet dimensions (same for all, or per-magnet)

    Returns
    -------
    B : array, shape (N, 3)
        Total magnetic field
    """
    # Preserve input dtype for precision
    points = xp.asarray(points)
    dtype = points.dtype
    positions = xp.asarray(positions, dtype=dtype)
    polarizations = xp.asarray(polarizations, dtype=dtype)
    dimensions = xp.asarray(dimensions, dtype=dtype)

    N = len(points)
    M = len(positions)

    # Handle uniform dimensions
    if dimensions.ndim == 1:
        dimensions = xp.tile(dimensions, (M, 1))

    # Half dimensions: (M, 3)
    half_dims = dimensions / 2

    # Corner signs: (8, 3)
    corner_signs = xp.array([
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
    ], dtype=dtype)

    # sigma for each corner: (8,)
    sigma = corner_signs[:, 0] * corner_signs[:, 1] * corner_signs[:, 2]

    # Translate points to each magnet's frame
    # points: (N, 3), positions: (M, 3)
    # r: (N, M, 3)
    r = points[:, None, :] - positions[None, :, :]

    # Position relative to each corner of each magnet
    # r: (N, M, 3), corner_signs: (8, 3), half_dims: (M, 3)
    # corner_offsets: (8, M, 3) = corner_signs[:, None, :] * half_dims[None, :, :]
    # rel_pos: (N, M, 8, 3)
    corner_offsets = corner_signs[:, None, :] * half_dims[None, :, :]  # (8, M, 3)
    rel_pos = r[:, :, None, :] - corner_offsets[None, :, :, :]  # (N, M, 8, 3)

    x = rel_pos[:, :, :, 0]  # (N, M, 8)
    y = rel_pos[:, :, :, 1]
    z = rel_pos[:, :, :, 2]

    R = xp.sqrt(x**2 + y**2 + z**2)

    # Log and atan terms: (N, M, 8) - use (x + R), not (|x| + R)
    log_xR = xp.log(x + R)
    log_yR = xp.log(y + R)
    log_zR = xp.log(z + R)
    atan_yz_xR = xp.arctan2(y * z, x * R)
    atan_xz_yR = xp.arctan2(x * z, y * R)
    atan_xy_zR = xp.arctan2(x * y, z * R)

    # Polarizations: (M, 3) -> Jx, Jy, Jz each (M,)
    Jx = polarizations[:, 0]  # (M,)
    Jy = polarizations[:, 1]
    Jz = polarizations[:, 2]

    # Broadcast J over corners: Jx[None, :, None] is (1, M, 1)
    # sigma is (8,) -> sigma[None, None, :] is (1, 1, 8)
    # Result after sum over corners (axis=2): (N, M)

    Bx = xp.sum(sigma * (Jx[None, :, None] * atan_yz_xR - Jy[None, :, None] * log_zR - Jz[None, :, None] * log_yR), axis=2)
    By = xp.sum(sigma * (-Jx[None, :, None] * log_zR + Jy[None, :, None] * atan_xz_yR - Jz[None, :, None] * log_xR), axis=2)
    Bz = xp.sum(sigma * (-Jx[None, :, None] * log_yR - Jy[None, :, None] * log_xR + Jz[None, :, None] * atan_xy_zR), axis=2)

    # Sum over all magnets: (N, M) -> (N,)
    Bx_total = xp.sum(Bx, axis=1)
    By_total = xp.sum(By, axis=1)
    Bz_total = xp.sum(Bz, axis=1)

    B = xp.stack([Bx_total, By_total, Bz_total], axis=1) / (4 * xp.pi)

    return B


def grid_field_gpu(grid_extent, magnet_size, step_size, polarization=(0, 1.48, 0)):
    """
    Calculate field on 3D grid - fully vectorized.

    Parameters
    ----------
    grid_extent : float
        Grid spans [-extent, +extent] in meters
    magnet_size : float
        Cubic magnet edge length in meters
    step_size : float
        Grid spacing in meters
    polarization : tuple
        (Jx, Jy, Jz) in Tesla

    Returns
    -------
    B : array, shape (N, 3)
        Field at grid points
    points : array, shape (N, 3)
        Grid coordinates
    """
    grid = xp.arange(-grid_extent, grid_extent, step_size, dtype=xp.float32)
    X, Y, Z = xp.meshgrid(grid, grid, grid, indexing='ij')
    points = xp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    dims = xp.array([magnet_size, magnet_size, magnet_size], dtype=xp.float32)
    B = cuboid_field_gpu(points, polarization, dims)

    return B, points


# Convenience function matching old interface
def analytical_cuboid_field_gpu(points, polarization, dimensions, position=(0, 0, 0)):
    """Drop-in replacement for analytical_cuboid_field with GPU acceleration."""
    return cuboid_field_gpu(points, polarization, dimensions, position)

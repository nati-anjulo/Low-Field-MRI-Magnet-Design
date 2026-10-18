import cupy as cp
import numpy as np

# ============================================================================
# GRFC_cached - Unified GPU Field Calculation Module
# ============================================================================
# Combines cached field lookup with all utility functions.
# Supports both cylindrical (ellipse) and Cartesian (horseshoe) geometries.
#
# Usage:
#   1. load_field(Bxyz, Ref_grid, step_size) - call ONCE at notebook start
#
#   For CYLINDRICAL geometries (ellipse): config = [z, r, theta, phi]
#   2. calc_field(config, sample_points) - fast field lookup
#   3. input_converter_vectorized(config) - convert to Cartesian positions
#
#   For CARTESIAN geometries (horseshoe): config = [x, y, z, phi]
#   2. calc_field_cartesian(config, sample_points) - fast field lookup
#      (z must be pre-discretized to layer centers)
# ============================================================================

# Module-level cache
_field = None
_origin = None
_bounds = None
_step = None
_Ny = None
_Nz = None
_loaded = False


def load_field(Precalc_Bxyz_value, Ref_grid, step_size):
    """
    Load precomputed field to GPU. Call ONCE when notebook starts.
    """
    global _field, _origin, _bounds, _step, _Ny, _Nz, _loaded

    if isinstance(Precalc_Bxyz_value, np.ndarray):
        _field = cp.asarray(Precalc_Bxyz_value.astype(np.float32))
    else:
        _field = cp.asarray(Precalc_Bxyz_value, dtype=cp.float32)

    Ref_grid_GPU = cp.asarray(Ref_grid, dtype=cp.float32)
    Nx, Ny, Nz = Ref_grid_GPU.shape
    _origin = Ref_grid_GPU[0, 0, 0]
    _bounds = cp.array([Nx-1, Ny-1, Nz-1], dtype=cp.int32)
    _step = cp.float32(step_size)
    _Ny, _Nz = Ny, Nz
    _loaded = True

    print(f"✓ Field loaded to GPU: {_field.shape} ({_field.nbytes/1e9:.2f} GB)")
    print(f"  Grid: {Nx}x{Ny}x{Nz}, step={step_size*1000:.1f}mm")


def calc_field(config, sample_points):
    """
    Calculate magnetic field using cached lookup.

    Args:
        config: Magnet configuration [z, r, theta, phi] per magnet
        sample_points: Points to evaluate field at

    Returns:
        B_field_mT: (n_samples, 3) field vectors in mT
        B_magnitude_mT: (n_samples,) field magnitudes in mT
    """
    if not _loaded:
        raise RuntimeError("Call load_field() first!")

    if not isinstance(config, cp.ndarray):
        config = cp.asarray(config, dtype=cp.float32)
    if not isinstance(sample_points, cp.ndarray):
        sample_points = cp.asarray(sample_points, dtype=cp.float32)

    n_samples = len(sample_points)

    # Convert config to positions and rotations
    pos, orientations = input_converter_vectorized(config)
    phi = orientations[:, 2]
    nm = pos.shape[0]

    # Build rotation matrices
    sin_r, cos_r = cp.sin(phi), cp.cos(phi)
    rot = cp.zeros((nm, 3, 3), dtype=cp.float32)
    rot[:, 0, 0], rot[:, 0, 1] = cos_r, -sin_r
    rot[:, 1, 0], rot[:, 1, 1] = sin_r, cos_r
    rot[:, 2, 2] = 1.0

    # Transform sample points relative to each magnet
    rel = sample_points.reshape(1, n_samples, 3) - pos.reshape(nm, 1, 3)
    trans = cp.matmul(rel, rot)

    # Grid index lookup
    idx = (trans - _origin) / _step
    base = cp.floor(idx).astype(cp.int32)
    final = cp.clip(base + (idx - base > 0.5).astype(cp.int32), 0, _bounds)

    # Flat index for field lookup
    flat = final[:,:,0] * (_Ny * _Nz) + final[:,:,1] * _Nz + final[:,:,2]
    fv = _field[flat.reshape(-1)].reshape(nm, n_samples, 3)

    # Rotate field back to global frame
    rot_T = cp.transpose(rot, (0, 2, 1))
    rotated = cp.matmul(fv, rot_T)

    # Sum all magnet contributions
    total = cp.sum(rotated, axis=0) * 1000  # Convert to mT
    magnitude = cp.linalg.norm(total, axis=1)

    return total, magnitude


def input_converter_vectorized(params, layer_clearance=1.524e-3, layer_thickness=6.35e-3):
    """
    Convert [z, r, theta, phi] to Cartesian positions and orientations.

    Physical setup:
    - Magnets are flat cubes (6.35mm × 6.35mm × 6.35mm)
    - Only rotate around z-axis (no tilting)
    - Discrete z-layers with shim stock clearance

    Layer spacing:
    - layer_thickness = 6.35mm (magnet height)
    - layer_clearance = 1.524mm (0.06 inch shim stock)
    - sheet_thickness = 7.874mm (center-to-center spacing)

    Args:
        params: Array of [z, r, theta, phi] per magnet
        layer_clearance: Gap between layers (default 1.524mm)
        layer_thickness: Magnet thickness (default 6.35mm)

    Returns:
        positions: (N, 3) Cartesian positions [x, y, z]
        orientations: (N, 3) rotation angles [0, 0, phi]
    """
    sheet_thickness = layer_thickness + layer_clearance

    params = params.reshape(-1, 4)
    z_in, r, theta, rot = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    # Discretize to layers
    layer_index = cp.round((z_in - 0.5 * sheet_thickness) / sheet_thickness)
    z = (layer_index + 0.5) * sheet_thickness

    x = r * cp.cos(theta)
    y = r * cp.sin(theta)

    return cp.column_stack([x, y, z]), cp.column_stack([0*z, 0*z, rot])


def is_loaded():
    """Check if field is loaded."""
    return _loaded


# ============================================================================
# Cartesian field calculation - for horseshoe and other Cartesian geometries
# ============================================================================

def calc_field_cartesian(config, sample_points):
    """
    Calculate magnetic field for Cartesian config format [x, y, z, phi].

    Use this for geometries where positions are already in Cartesian coordinates
    (e.g., horseshoe magnets) instead of cylindrical [z, r, theta, phi].

    IMPORTANT: z must be pre-discretized to layer centers before calling.

    Args:
        config: Magnet configuration [x, y, z, phi] per magnet (Cartesian)
        sample_points: Points to evaluate field at

    Returns:
        B_field_mT: (n_samples, 3) field vectors in mT
        B_magnitude_mT: (n_samples,) field magnitudes in mT
    """
    if not _loaded:
        raise RuntimeError("Call load_field() first!")

    if not isinstance(config, cp.ndarray):
        config = cp.asarray(config, dtype=cp.float32)
    if not isinstance(sample_points, cp.ndarray):
        sample_points = cp.asarray(sample_points, dtype=cp.float32)

    n_samples = len(sample_points)

    # For Cartesian config: [x, y, z, phi]
    x = config[:, 0]
    y = config[:, 1]
    z = config[:, 2]
    phi = config[:, 3]

    pos = cp.column_stack([x, y, z])
    nm = pos.shape[0]

    # Build rotation matrices (same as calc_field)
    sin_r, cos_r = cp.sin(phi), cp.cos(phi)
    rot = cp.zeros((nm, 3, 3), dtype=cp.float32)
    rot[:, 0, 0], rot[:, 0, 1] = cos_r, -sin_r
    rot[:, 1, 0], rot[:, 1, 1] = sin_r, cos_r
    rot[:, 2, 2] = 1.0

    # Transform sample points relative to each magnet
    rel = sample_points.reshape(1, n_samples, 3) - pos.reshape(nm, 1, 3)
    trans = cp.matmul(rel, rot)

    # Grid index lookup
    idx = (trans - _origin) / _step
    base = cp.floor(idx).astype(cp.int32)
    final = cp.clip(base + (idx - base > 0.5).astype(cp.int32), 0, _bounds)

    # Flat index for field lookup
    flat = final[:,:,0] * (_Ny * _Nz) + final[:,:,1] * _Nz + final[:,:,2]
    fv = _field[flat.reshape(-1)].reshape(nm, n_samples, 3)

    # Rotate field back to global frame
    rot_T = cp.transpose(rot, (0, 2, 1))
    rotated = cp.matmul(fv, rot_T)

    # Sum all magnet contributions
    total = cp.sum(rotated, axis=0) * 1000  # Convert to mT
    magnitude = cp.linalg.norm(total, axis=1)

    return total, magnitude


def calculate_field_cartesian(config, sample_points, Precalc_Bxyz_value, Ref_grid, step_size,
                              batch_size=1000000, num_streams=20):
    """
    Legacy wrapper for Cartesian field calculation.
    Loads field if needed, then calculates using Cartesian coordinates.

    Args:
        config: Magnet configuration [x, y, z, phi] per magnet (Cartesian)
        sample_points: Points to evaluate field at
        Precalc_Bxyz_value: Precomputed field values
        Ref_grid: Reference grid
        step_size: Grid step size
        batch_size: (unused, for API compatibility)
        num_streams: (unused, for API compatibility)

    Returns:
        B_field_mT: (n_samples, 3) field vectors in mT
        B_magnitude_mT: (n_samples,) field magnitudes in mT
    """
    if not _loaded:
        load_field(Precalc_Bxyz_value, Ref_grid, step_size)
    return calc_field_cartesian(config, sample_points)


# ============================================================================
# Legacy compatibility - these functions match the old GRFC module interface
# ============================================================================

def calculate_field(config, sample_points, Precalc_Bxyz_value, Ref_grid, step_size,
                   batch_size=1000000, num_streams=20):
    """
    Legacy wrapper - loads field if needed, then calculates.
    For new code, use load_field() once then calc_field() per iteration.
    Uses cylindrical coordinates [z, r, theta, phi].
    """
    if not _loaded:
        load_field(Precalc_Bxyz_value, Ref_grid, step_size)
    return calc_field(config, sample_points)

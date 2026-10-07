"""
Ellipse geometry functions.
"""
import cupy as cp

def ellipse_r_gpu(theta, a, b):
    """Radius of ellipse at polar angle theta (GPU version)."""
    return (a * b) / cp.sqrt((b * cp.cos(theta))**2 + (a * cp.sin(theta))**2)


def get_bolt_positions_ellipse_gpu(rx, ry, n_bolts):
    """
    Place n_bolts at equal arc-length spacing on ellipse.
    Returns (bolt_x, bolt_y) on GPU.
    """
    # High-res theta for arc length calculation
    n_steps = 5000
    theta = cp.linspace(0, 2*cp.pi, n_steps + 1)
    
    # Arc length element: ds = sqrt((dx/dt)^2 + (dy/dt)^2) dt
    # For ellipse: x = rx*cos(t), y = ry*sin(t)
    # dx/dt = -rx*sin(t), dy/dt = ry*cos(t)
    ds = cp.sqrt((rx * cp.sin(theta))**2 + (ry * cp.cos(theta))**2)
    
    # Cumulative arc length
    dt = 2*cp.pi / n_steps
    arc_length = cp.concatenate([cp.array([0.0]), cp.cumsum(ds[:-1] * dt)])
    
    perimeter = arc_length[-1]
    
    # Find theta at equal arc-length intervals
    bolt_theta = cp.zeros(n_bolts)
    for i in range(n_bolts):
        target = i * perimeter / n_bolts
        idx = cp.searchsorted(arc_length, target)
        idx = int(cp.clip(idx, 0, n_steps))
        bolt_theta[i] = theta[idx]
    
    # Convert to xy
    bolt_r = ellipse_r_gpu(bolt_theta, rx, ry)
    bolt_x = bolt_r * cp.cos(bolt_theta)
    bolt_y = bolt_r * cp.sin(bolt_theta)
    
    return bolt_x, bolt_y

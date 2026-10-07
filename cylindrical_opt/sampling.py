"""
DSV sampling functions.
"""
import cupy as cp

def _generate_sample_points_small_dsv(n_sample_points, sphere_rad_max,z_min, z_max):
    """
    Generate uniformly distributed points on four concentric spheres with radii
    R, 0.75*R, 0.5*R, and 0.25*R, with the number of points on each sphere
    proportional to the surface area of each sphere.
    
    Returns:
    --------
    points : ndarray
        Array of shape (self.n_sample_points, 3) containing the 3D coordinates of the points
    """
    # Define the relative radii of the four spheres
    radii_factors = cp.linspace(1, 0, 8)#cp.array([1.0, 0.8,0.75, 0.5, 0.25])
    sphere_radii = sphere_rad_max * radii_factors
    
    # Calculate the relative surface areas (proportional to r²)
    relative_areas = radii_factors**2
    
    # Calculate the number of points for each sphere
    # proportional to the surface area
    points_distribution = relative_areas / cp.sum(relative_areas)
    points_per_sphere = cp.round(points_distribution * n_sample_points).astype(int)
    
    # Adjust to ensure we get exactly n_sample_points
    diff = n_sample_points - cp.sum(points_per_sphere)
    points_per_sphere[0] += diff
    
    # Generate points for each sphere
    all_points = []
    for i, (radius, n_points) in enumerate(zip(sphere_radii, points_per_sphere)):
        if n_points <= 0:
            continue
            
        # Generate points using the Fibonacci lattice method
        indices = cp.arange(0, n_points, dtype=float) + 0.5
        phi = cp.arccos(1 - 2 * indices / n_points)
        theta = cp.pi * (1 + 5**0.5) * indices
        
        x = radius * cp.cos(theta) * cp.sin(phi)
        y = radius * cp.sin(theta) * cp.sin(phi)
        z = radius * cp.cos(phi)
        z_clip = cp.clip(z,z_min,z_max)
        
        sphere_points = cp.column_stack((x, y, z_clip))
        all_points.append(sphere_points)
    
    # Combine all points
    points = cp.vstack(all_points)
    
    return points

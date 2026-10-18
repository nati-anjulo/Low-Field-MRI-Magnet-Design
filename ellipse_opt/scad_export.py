"""
OpenSCAD export module for ellipse magnet optimization results.
Converts .npz optimization output to a single self-contained OpenSCAD file.
"""

import numpy as np
from . import field_calculator as GRFC


def export_to_scad(npz_file, output_file=None, geometry_params=None, magnet_size=6.35e-3):
    """
    Convert ellipse optimization .npz results to a single OpenSCAD file.

    Parameters
    ----------
    npz_file : str
        Path to the .npz file (from optimization save)
    output_file : str, optional
        Output .scad file path. If None, uses npz filename with .scad extension
    geometry_params : dict, optional
        Geometry parameters for manufacturing:
        - rx_inner: inner ellipse semi-axis x (m)
        - ry_inner: inner ellipse semi-axis y (m)
        - gap: radial gap for magnets (m)
        - bolt_number: number of bolts
        - bolt_diameter: bolt hole diameter (m)
    magnet_size : float
        Magnet cube size in meters (default 6.35mm)

    Returns
    -------
    dict : Contains exported file path and statistics

    Example
    -------
    >>> from ellipse_opt import scad_export
    >>>
    >>> # With geometry params from notebook
    >>> params = {
    ...     'rx_inner': 170e-3,
    ...     'ry_inner': 150e-3,
    ...     'gap': 50e-3,
    ...     'bolt_number': 10,
    ...     'bolt_diameter': 7.5e-3
    ... }
    >>> scad_export.export_to_scad('ellipse_design.npz', geometry_params=params)
    """
    # Load the optimization result
    data = np.load(npz_file, allow_pickle=True)

    # Get the full config (or q1_config and reflect)
    if 'full_config' in data:
        config = data['full_config']
    elif 'q1_config' in data:
        from . import generator
        import cupy as cp
        q1 = cp.asarray(data['q1_config'])
        full = generator.reflect_to_half_ellipse(q1)
        config = cp.asnumpy(full)
    else:
        raise ValueError("npz file must contain 'full_config' or 'q1_config'")

    # Convert to Cartesian coordinates
    import cupy as cp
    config_gpu = cp.asarray(config)
    positions_gpu, rotations_gpu = GRFC.input_converter_vectorized(config_gpu)
    positions = cp.asnumpy(positions_gpu)
    rotations = cp.asnumpy(rotations_gpu)

    # Convert to mm and degrees for OpenSCAD
    positions_mm = positions * 1000
    rotations_deg = np.rad2deg(rotations[:, 2]).reshape(-1, 1)

    # Combine into magnet parameters [x, y, z, rotation]
    magnet_params = np.hstack((positions_mm, rotations_deg))

    # Get unique z positions for layers
    z_positions = np.unique(positions_mm[:, 2])
    z_positions = np.sort(z_positions)

    # Generate output filename
    if output_file is None:
        output_file = npz_file.replace('.npz', '_design.scad')

    # Use geometry params or defaults
    if geometry_params is None:
        geometry_params = {}

    # Convert to mm
    magnet_size_mm = magnet_size * 1000
    rx_inner_mm = geometry_params.get('rx_inner', 0.170) * 1000
    ry_inner_mm = geometry_params.get('ry_inner', 0.150) * 1000
    gap_mm = geometry_params.get('gap', 0.050) * 1000
    bolt_number = geometry_params.get('bolt_number', 10)
    bolt_diameter_mm = geometry_params.get('bolt_diameter', 0.0075) * 1000

    # Calculate derived values
    # Add full magnet size margin so magnets at edges don't fall off
    # (optimization places magnet centers, but physical magnets extend beyond center)
    r_inner_avg = (rx_inner_mm + ry_inner_mm) / 2 - magnet_size_mm
    r_outer_avg = (rx_inner_mm + ry_inner_mm) / 2 + gap_mm + magnet_size_mm
    bolt_rad = (rx_inner_mm + ry_inner_mm) / 2 + gap_mm / 2

    # Write complete self-contained design file
    with open(output_file, 'w') as f:
        f.write('// ================== ELLIPSE MAGNET ARRAY - MANUFACTURING DESIGN ==================\n')
        f.write('// Auto-generated from optimization results\n')
        f.write('// view_mode: "all" = full assembly, "single_layer" = one plate for STL export\n\n')

        # Magnet data
        f.write('// ================== MAGNET DATA ==================\n')
        f.write(f'// Total magnets: {len(magnet_params)}\n')
        f.write('// Format: [x_mm, y_mm, z_mm, rotation_deg]\n')
        f.write('magnet_data = [\n')
        for i, magnet in enumerate(magnet_params):
            x, y, z, rot = magnet
            comma = ',' if i < len(magnet_params) - 1 else ''
            f.write(f'    [{x:.3f}, {y:.3f}, {z:.3f}, {rot:.2f}]{comma}\n')
        f.write('];\n\n')

        # Z positions
        f.write('// ================== PLATE POSITIONS ==================\n')
        f.write(f'// Total layers: {len(z_positions)}\n')
        f.write('z_positions = [\n')
        for i, z in enumerate(z_positions):
            comma = ',' if i < len(z_positions) - 1 else ''
            f.write(f'    {z:.3f}{comma}\n')
        f.write('];\n\n')

        # Global settings
        f.write('// ================== GLOBAL SETTINGS ==================\n')
        f.write(f'magnet_size = {magnet_size_mm:.2f};\n')
        f.write(f'pocket_size = magnet_size;  // Same as magnet for tight fit\n\n')
        f.write(f'r_inner = {r_inner_avg:.2f};\n')
        f.write(f'r_outer = {r_outer_avg:.2f};\n')
        f.write(f'bolt_number = {bolt_number};\n')
        f.write(f'bolt_size = {bolt_diameter_mm/2:.2f};  // Bolt radius\n')
        f.write(f'bolt_rad = {bolt_rad:.2f};  // Bolt circle radius\n')
        f.write('angle_ratio = 360 / bolt_number;\n\n')

        # Visual options
        f.write('// ================== VISUAL OPTIONS ==================\n')
        f.write('show_rotation_dots = true;\n')
        f.write('show_layer_labels = true;\n\n')

        # Modules
        f.write('// ================== MODULES ==================\n\n')

        f.write('module rotation_indicator_dot() {\n')
        f.write('    translate([0, magnet_size*0.7, 0])\n')
        f.write('        cylinder(h = magnet_size + 0.01, r = magnet_size/20, center = true, $fn = 100);\n')
        f.write('}\n\n')

        f.write('// -------- SINGLE LAYER --------\n')
        f.write('module single_layer(layer_index) {\n')
        f.write('    if (layer_index <= len(z_positions)) {\n')
        f.write('        i = layer_index - 1;\n\n')
        f.write('        difference() {\n')
        f.write('            // Plate body\n')
        f.write('            difference() {\n')
        f.write('                cylinder(h = magnet_size, r = r_outer, center = true, $fn = 50);\n')
        f.write('                cylinder(h = magnet_size, r = r_inner, center = true, $fn = 50);\n')
        f.write('            }\n\n')
        f.write('            // Magnet pockets\n')
        f.write('            for (magnet = magnet_data) {\n')
        f.write('                if (abs(magnet[2] - z_positions[i]) < 0.001) {\n')
        f.write('                    translate([magnet[0], magnet[1], 0])\n')
        f.write('                        rotate([0, 0, magnet[3]])\n')
        f.write('                            cube([magnet_size, magnet_size, magnet_size + 1], center = true);\n')
        f.write('                }\n')
        f.write('            }\n\n')
        f.write('            // Bolt holes\n')
        f.write('            for (j = [0:angle_ratio:360-angle_ratio]) {\n')
        f.write('                translate([bolt_rad*cos(j), bolt_rad*sin(j), 0])\n')
        f.write('                    cylinder(h = magnet_size + 2, r = bolt_size, center = true, $fn = 25);\n')
        f.write('            }\n\n')
        f.write('            // "L" label\n')
        f.write('            translate([0, r_outer - 4, 0])\n')
        f.write('                linear_extrude(height = magnet_size + 1, center = true)\n')
        f.write('                    text("L", size = 3, halign = "center", valign = "center");\n\n')
        f.write('            // Layer number\n')
        f.write('            rotate([0, 0, angle_ratio/2.5])\n')
        f.write('                translate([r_outer - 6, 10, 0])\n')
        f.write('                    linear_extrude(height = magnet_size + 1, center = true)\n')
        f.write('                        text(str(len(z_positions) - layer_index + 1), size = 4, halign = "center", valign = "center");\n\n')
        f.write('            // Rotation dots\n')
        f.write('            if (show_rotation_dots) {\n')
        f.write('                for (magnet = magnet_data) {\n')
        f.write('                    if (abs(magnet[2] - z_positions[i]) < 0.001) {\n')
        f.write('                        translate([magnet[0], magnet[1], 0])\n')
        f.write('                            rotate([0, 0, magnet[3]])\n')
        f.write('                                rotation_indicator_dot();\n')
        f.write('                    }\n')
        f.write('                }\n')
        f.write('            }\n')
        f.write('        }\n')
        f.write('    }\n')
        f.write('}\n\n')

        # Render control
        f.write('// ================== RENDER CONTROL ==================\n')
        f.write('module render_inner(view_mode, layer_to_export) {\n')
        f.write('    if (view_mode == "all") {\n')
        f.write('        layers = len(z_positions);\n')
        f.write('        min_z = min(z_positions);\n')
        f.write('        max_z = max(z_positions);\n')
        f.write('        center_z = (min_z + max_z) / 2;\n\n')
        f.write('        for (i = [0:layers-1]) {\n')
        f.write('            layer_z = z_positions[i] - center_z;\n')
        f.write('            translate([0, 0, layer_z])\n')
        f.write('                single_layer(i + 1);\n')
        f.write('        }\n')
        f.write('    } else {\n')
        f.write('        single_layer(layer_to_export);\n')
        f.write('    }\n')
        f.write('}\n\n')

        # Render call
        f.write('// ================== CALL ==================\n')
        f.write('render_inner(\n')
        f.write('    view_mode = "single_layer",  // "all" or "single_layer"\n')
        f.write(f'    layer_to_export = 1          // 1 to {len(z_positions)}\n')
        f.write(');\n')

    # Get field stats if available
    field_mT = data.get('field_mT', None)
    rms_dev = data.get('rms_dev_mT', None)

    print(f"\n{'='*60}")
    print(f"SCAD EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Output:   {output_file}")
    print(f"  Magnets:  {len(magnet_params)}")
    print(f"  Layers:   {len(z_positions)}")
    if field_mT:
        print(f"  Field:    {field_mT:.2f} mT")
    if rms_dev:
        homog = rms_dev / field_mT * 100 if field_mT else 0
        print(f"  Homog:    {homog:.2f}%")
    print(f"{'='*60}")

    return {
        'output_file': output_file,
        'n_magnets': len(magnet_params),
        'n_layers': len(z_positions),
        'field_mT': field_mT,
        'rms_dev_mT': rms_dev
    }


def quick_export(npz_file, geometry_params=None):
    """Quick export with default settings."""
    return export_to_scad(npz_file, geometry_params=geometry_params)

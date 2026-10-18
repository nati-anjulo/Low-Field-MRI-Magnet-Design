"""
Clean OpenSCAD export - separates data from view logic.
Magnet data goes in JSON, SCAD file stays simple.
"""

import numpy as np
import json
from . import field_calculator as GRFC


def export_clean(npz_file, output_prefix=None, geometry_params=None, magnet_size=6.35e-3):
    """
    Export optimization results to JSON data + clean SCAD viewer.

    Parameters
    ----------
    npz_file : str
        Path to the .npz file from optimization
    output_prefix : str, optional
        Prefix for output files. Creates {prefix}_data.json and {prefix}_view.scad
    geometry_params : dict, optional
        Geometry parameters (rx_inner, ry_inner, gap, bolt_number, bolt_diameter)
    magnet_size : float
        Magnet cube size in meters (default 6.35mm)

    Returns
    -------
    dict : Paths to generated files
    """
    # Load optimization result
    data = np.load(npz_file, allow_pickle=True)

    # Get config
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

    # Convert to Cartesian
    import cupy as cp
    config_gpu = cp.asarray(config)
    positions_gpu, rotations_gpu = GRFC.input_converter_vectorized(config_gpu)
    positions = cp.asnumpy(positions_gpu)
    rotations = cp.asnumpy(rotations_gpu)

    # Convert to mm and degrees
    positions_mm = positions * 1000
    rotations_deg = np.rad2deg(rotations[:, 2])

    # Get unique z layers
    z_positions = sorted(np.unique(positions_mm[:, 2]).tolist())

    # Build magnet list
    magnets = []
    for i in range(len(positions_mm)):
        magnets.append({
            'x': round(float(positions_mm[i, 0]), 3),
            'y': round(float(positions_mm[i, 1]), 3),
            'z': round(float(positions_mm[i, 2]), 3),
            'rot': round(float(rotations_deg[i]), 2)
        })

    # Geometry defaults
    if geometry_params is None:
        geometry_params = {}

    magnet_size_mm = magnet_size * 1000
    rx_inner_mm = geometry_params.get('rx_inner', 0.170) * 1000
    ry_inner_mm = geometry_params.get('ry_inner', 0.150) * 1000
    gap_mm = geometry_params.get('gap', 0.050) * 1000
    bolt_number = geometry_params.get('bolt_number', 10)
    bolt_diameter_mm = geometry_params.get('bolt_diameter', 0.0075) * 1000

    # Full magnet margin so edges don't hang off
    r_inner_avg = (rx_inner_mm + ry_inner_mm) / 2 - magnet_size_mm
    r_outer_avg = (rx_inner_mm + ry_inner_mm) / 2 + gap_mm + magnet_size_mm
    bolt_rad = (rx_inner_mm + ry_inner_mm) / 2 + gap_mm / 2

    # JSON data
    json_data = {
        'magnets': magnets,
        'z_layers': z_positions,
        'geometry': {
            'magnet_size': magnet_size_mm,
            'r_inner': round(r_inner_avg, 2),
            'r_outer': round(r_outer_avg, 2),
            'bolt_number': bolt_number,
            'bolt_radius': round(bolt_diameter_mm / 2, 2),
            'bolt_circle_radius': round(bolt_rad, 2)
        }
    }

    # Output paths
    if output_prefix is None:
        output_prefix = npz_file.replace('.npz', '')

    json_file = f"{output_prefix}_data.json"
    scad_file = f"{output_prefix}_view.scad"

    # Write JSON
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Write clean SCAD viewer
    _write_scad_viewer(scad_file, json_file, len(z_positions))

    print(f"\nExported:")
    print(f"  Data: {json_file}")
    print(f"  View: {scad_file}")
    print(f"  Magnets: {len(magnets)}, Layers: {len(z_positions)}")

    return {
        'json_file': json_file,
        'scad_file': scad_file,
        'n_magnets': len(magnets),
        'n_layers': len(z_positions)
    }


def _write_scad_viewer(scad_file, json_file, n_layers):
    """Write the clean SCAD viewer file."""

    # Get just the filename for the include
    import os
    json_filename = os.path.basename(json_file)

    scad_content = f'''// ================== ELLIPSE MAGNET VIEWER ==================
// Clean viewer - data is in {json_filename}

// ================== VIEW SETTINGS ==================
view_mode = "all";      // "all" = full assembly, "single" = one layer
layer = 1;              // Which layer (1 to {n_layers}) when view_mode = "single"
show_dots = true;       // Show rotation indicator dots

// ================== LOAD DATA ==================
data = import("{json_filename}");

magnets = data.magnets;
z_layers = data.z_layers;
g = data.geometry;

magnet_size = g.magnet_size;
r_inner = g.r_inner;
r_outer = g.r_outer;
bolt_number = g.bolt_number;
bolt_size = g.bolt_radius;
bolt_rad = g.bolt_circle_radius;

// ================== MODULES ==================
module plate(layer_idx) {{
    z = z_layers[layer_idx];

    difference() {{
        // Ring
        difference() {{
            cylinder(h=magnet_size, r=r_outer, center=true, $fn=50);
            cylinder(h=magnet_size, r=r_inner, center=true, $fn=50);
        }}

        // Magnet pockets
        for (m = magnets) {{
            if (abs(m.z - z) < 0.001) {{
                translate([m.x, m.y, 0])
                    rotate([0, 0, m.rot])
                        cube([magnet_size, magnet_size, magnet_size+1], center=true);
            }}
        }}

        // Bolt holes
        for (a = [0 : 360/bolt_number : 359]) {{
            translate([bolt_rad*cos(a), bolt_rad*sin(a), 0])
                cylinder(h=magnet_size+2, r=bolt_size, center=true, $fn=25);
        }}

        // Layer label
        translate([0, r_outer-4, 0])
            linear_extrude(magnet_size+1, center=true)
                text(str("L", {n_layers}-layer_idx), size=4, halign="center");

        // Rotation dots
        if (show_dots) {{
            for (m = magnets) {{
                if (abs(m.z - z) < 0.001) {{
                    translate([m.x, m.y, 0])
                        rotate([0, 0, m.rot])
                            translate([0, magnet_size*0.7, 0])
                                cylinder(h=magnet_size+0.01, r=magnet_size/20, center=true, $fn=20);
                }}
            }}
        }}
    }}
}}

// ================== RENDER ==================
if (view_mode == "all") {{
    n = len(z_layers);
    center_z = (z_layers[0] + z_layers[n-1]) / 2;

    for (i = [0 : n-1]) {{
        translate([0, 0, z_layers[i] - center_z])
            plate(i);
    }}
}} else {{
    plate(layer - 1);
}}
'''

    with open(scad_file, 'w') as f:
        f.write(scad_content)

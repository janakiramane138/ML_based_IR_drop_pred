import argparse
import os
import numpy as np

"""
python3 src/data_generation.py \
    -spice_netlist benchmarks/testcase1.sp \
    -voltage_file benchmark_features/testcase1.voltage \
    -output benchmark_features/
"""

DBU = 2000.0          # per assignment note
GRID = 600            # fixed image size

def node_to_xy(node, grid=GRID):
    """
    Expect node like: <net>_<layer>_<x>_<y>
    Take the *last two* underscore-separated tokens as x,y in DBU, convert to µm,
    then map to integer pixel coordinates with clamping.
    Returns (x_idx, y_idx) or None for ground/invalid.
    """
    if node == '0' or node is None:
        return None
    toks = node.split('_')
    if len(toks) < 2:
        return None
    try:
        x_dbu = float(toks[-2])
        y_dbu = float(toks[-1])
        x_um = x_dbu / DBU
        y_um = y_dbu / DBU
        # Map µm directly to pixels (1 µm -> 1 px). If your chip > 600 µm,
        # you may want normalization; for now clamp to bounds.
        x = int(round(x_um))
        y = int(round(y_um))
        if x < 0 or y < 0:
            return None
        x = min(x, grid - 1)
        y = min(y, grid - 1)
        return (x, y)
    except Exception:
        return None

def safe_add(mat, x, y, val):
    """Write to a map only if indices are in-bounds."""
    if 0 <= y < mat.shape[0] and 0 <= x < mat.shape[1]:
        mat[y, x] += val

def parse_spice_line(line):
    """
    Return tuple (kind, n1, n2, value) or None to skip.
    kind in {'R','I','V'}
    Handles 'Vxxx n+ n- DC 1.2' and 'Ixxx n+ n- DC 3e-3'.
    """
    s = line.strip()
    if not s:
        return None
    if s[0] in ('.', '*', '+'):  # control/comment/continuation
        return None

    parts = s.split()
    if len(parts) < 4:  # too short to be an element
        return None

    el = parts[0]
    kind = el[0].upper()
    if kind not in ('R', 'I', 'V'):
        return None

    n1, n2 = parts[1], parts[2]

    # Value handling (last token OR 'DC <val>')
    val = None
    if 'DC' in (p.upper() for p in parts):
        try:
            dc_idx = [i for i, p in enumerate(parts) if p.upper() == 'DC'][0]
            val = float(parts[dc_idx + 1])
        except Exception:
            pass
    if val is None:
        # fallback: last token as value
        try:
            val = float(parts[-1])
        except Exception:
            return None

    return kind, n1, n2, val

def generate_maps(spice_netlist, voltage_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(spice_netlist))[0]

    # Load node voltages
    voltages = {}
    with open(voltage_file, 'r') as vf:
        for line in vf:
            parts = line.strip().split()
            if len(parts) == 2:
                node, v = parts
                try:
                    voltages[node] = float(v)
                except Exception:
                    continue

    # Init maps
    current_map   = np.zeros((GRID, GRID), dtype=np.float32)
    density_map   = np.zeros((GRID, GRID), dtype=np.float32)
    vsrc_map      = np.zeros((GRID, GRID), dtype=np.float32)
    ir_drop_map   = np.zeros((GRID, GRID), dtype=np.float32)

    resistors = []
    currents  = []
    vsrcs     = []

    # Parse netlist robustly
    with open(spice_netlist, 'r') as f:
        for line in f:
            rec = parse_spice_line(line)
            if rec is None:
                continue
            kind, n1, n2, val = rec
            if kind == 'R':
                resistors.append((n1, n2, val))
            elif kind == 'I':
                currents.append((n1, n2, val))
            elif kind == 'V':
                vsrcs.append((n1, n2, val))

    # Voltage source map
    for n1, n2, _vv in vsrcs:
        for node in (n1, n2):
            xy = node_to_xy(node)
            if xy:
                x, y = xy
                vsrc_map[y, x] = 1.0

    # Density & current maps
    # Density: increment both endpoints when a resistor is present
    for n1, n2, r in resistors:
        xy1 = node_to_xy(n1)
        xy2 = node_to_xy(n2)
        if xy1:
            safe_add(density_map, *xy1, 1.0)
        if xy2:
            safe_add(density_map, *xy2, 1.0)

        # Edge current magnitude via (V1 - V2)/R
        if r != 0 and xy1 and xy2:
            v1 = voltages.get(n1, None)
            v2 = voltages.get(n2, None)
            if v1 is not None and v2 is not None:
                i_mag = abs((v1 - v2) / r)
                safe_add(current_map, *xy1, i_mag)
                safe_add(current_map, *xy2, i_mag)

    # IR-drop map = Vmax - Vnode
    if voltages:
        vmax = max(voltages.values())
        for node, v in voltages.items():
            xy = node_to_xy(node)
            if xy:
                x, y = xy
                val = vmax - v
                if val < 0:  # numerical guard
                    val = 0.0
                ir_drop_map[y, x] = val

    # Save
    np.savetxt(os.path.join(output_dir, f"current_map_{base}.csv"), current_map, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"pdn_density_map_{base}.csv"), density_map, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"voltage_source_map_{base}.csv"), vsrc_map, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"ir_drop_map_{base}.csv"), ir_drop_map, delimiter=",")
    print(f"[OK] Maps generated for {base} -> {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate IR drop feature and label maps from SPICE + voltage file")
    parser.add_argument("-spice_netlist", type=str, required=True,
                        help="Path to SPICE netlist file (.sp)")
    parser.add_argument("-voltage_file", type=str, required=True,
                        help="Path to voltage output file from MNA solver")
    parser.add_argument("-output", type=str, required=True,
                        help="Directory to save generated CSV maps")
    args = parser.parse_args()

    generate_maps(args.spice_netlist, args.voltage_file, args.output)
    


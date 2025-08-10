#!/usr/bin/env python3
"""
ir_solver_sparse.py
Sparse Modified Nodal Analysis (MNA) IR-drop solver.

Provides:
- solve_ir_drop(input_file: str, output_file: str)  # callable from inference.py
- CLI: python3 ir_solver_sparse.py --input_file <netlist.sp> --output_file <outfile.voltage>
"""

import argparse
import numpy as np
from scipy.sparse import lil_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

def parse_netlist(path):
    resistors, currents, voltages, nodes = [], [], [], set()
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(('*', '.')):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            el, n1, n2, val = parts[0], parts[1], parts[2], float(parts[3])
            nodes.update([n1, n2])
            if el[0].upper() == 'R':
                resistors.append((n1, n2, val))
            elif el[0].upper() == 'I':
                currents.append((n1, n2, val))
            elif el[0].upper() == 'V':
                voltages.append((n1, n2, val))
    # Remove ground from unknowns if present
    nodes = sorted([n for n in nodes if n != '0'])
    node_index = {n: i for i, n in enumerate(nodes)}
    return resistors, currents, voltages, nodes, node_index

def build_mna_sparse(resistors, currents, voltages, nodes, node_index):
    """
    Build sparse MNA system:
        [ G  B ] [V ] = [ J ]
        [ B' 0 ] [Iv]   [ E ]
    """
    n = len(nodes)
    m = len(voltages)
# Using lil_matrix (List of Lists sparse format) for G (conductance matrix) and B (incidence matrix):
# - lil_matrix is optimized for incremental writes, making it efficient to build these matrices
    G = lil_matrix((n, n), dtype=np.float64)
    B = lil_matrix((n, m), dtype=np.float64)
    J = np.zeros(n, dtype=np.float64)
    E = np.zeros(m, dtype=np.float64)

    # Resistors -> G
    for n1, n2, r in resistors:
        if r == 0:
            continue
        g = 1.0 / r
        if n1 != '0':
            i = node_index[n1]
            G[i, i] += g
        if n2 != '0':
            j = node_index[n2]
            G[j, j] += g
        if n1 != '0' and n2 != '0':
            i, j = node_index[n1], node_index[n2]
            G[i, j] -= g
            G[j, i] -= g

    # Current sources -> J (sign convention: source from n1 -> n2)
    for n1, n2, i_val in currents:
        if n1 != '0':
            i = node_index[n1]
            J[i] -= i_val
        if n2 != '0':
            j = node_index[n2]
            J[j] += i_val

    # Voltage sources -> B and E
    for k, (n1, n2, v_val) in enumerate(voltages):
        if n1 != '0':
            i = node_index[n1]
            B[i, k] = 1.0
        if n2 != '0':
            j = node_index[n2]
            B[j, k] = -1.0
        E[k] = v_val

    # Assemble A and z
    top = hstack([G.tocsr(), B.tocsr()])
    bottom = hstack([B.transpose().tocsr(), lil_matrix((m, m), dtype=np.float64)])
    A = vstack([top, bottom]).tocsr()
    z = np.concatenate([J, E])

    return A, z

def solve_ir_drop(input_file: str, output_file: str):
    """Run sparse MNA on a SPICE netlist and write <node voltage> lines to output_file."""
    resistors, currents, voltages, nodes, node_index = parse_netlist(input_file)
    A, z = build_mna_sparse(resistors, currents, voltages, nodes, node_index)
    x = spsolve(A, z)

    V = x[:len(nodes)]  # node voltages first
    with open(output_file, 'w') as f:
        for name, v in zip(nodes, V):
            f.write(f"{name} {v:.10f}\n")
    print(f"[INFO] Voltage output written to: {output_file}")

def _main():
    ap = argparse.ArgumentParser(description="Sparse MNA IR-drop solver")
    ap.add_argument("--input_file", required=True, help="SPICE netlist")
    ap.add_argument("--output_file", required=True, help="Output voltage file")
    args = ap.parse_args()
    solve_ir_drop(args.input_file, args.output_file)

if __name__ == "__main__":
    _main()


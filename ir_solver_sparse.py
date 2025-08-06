
import numpy as np
import argparse
from scipy.sparse import lil_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

#first parse the spice netlist and get the required inputs
def parse_netlist(file_path):
    resistors = []
    current_sources = []
    voltage_sources = []
    node_set = set()

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('*', '.')):
                continue
            tokens = line.split()
            if len(tokens) != 4:
                continue
            comp, n1, n2, value = tokens
            value = float(value)

            if comp.startswith('R'):
                resistors.append((n1, n2, value))
                if n1 != '0': node_set.add(n1)
                if n2 != '0': node_set.add(n2)
            elif comp.startswith('I'):
                current_sources.append((n1, n2, value))
                if n1 != '0': node_set.add(n1)
                if n2 != '0': node_set.add(n2)
            elif comp.startswith('V'):
                voltage_sources.append((n1, n2, value))
                if n1 != '0': node_set.add(n1)
                if n2 != '0': node_set.add(n2)

    return resistors, current_sources, voltage_sources, sorted(list(node_set))

def build_mna(resistors, current_sources, voltage_sources, nodes):
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    m = len(voltage_sources)

    G = lil_matrix((n, n)) ## here first I used regular matrix, but the memory is not sufficient to run in my system, so changed to sparse matrix
    J = np.zeros((n, 1))
    B = lil_matrix((n, m)) ##similar
    E = np.zeros((m, 1))

    for n1, n2, r in resistors:
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

    for n1, n2, i_val in current_sources:
        if n1 != '0':
            i = node_index[n1]
            J[i] -= i_val
        if n2 != '0':
            j = node_index[n2]
            J[j] += i_val

    for k, (n1, n2, v_val) in enumerate(voltage_sources):
        if n1 != '0':
            i = node_index[n1]
            B[i, k] = 1
        if n2 != '0':
            j = node_index[n2]
            B[j, k] = -1
        E[k] = v_val

    A_top = hstack([G, B])
    A_bottom = hstack([B.T, lil_matrix((m, m))])
    A = vstack([A_top, A_bottom]).tocsr()
    z = np.vstack((J, E))

    return A, z, node_index, nodes

def solve_ir_drop(input_file, output_file):
    resistors, current_sources, voltage_sources, nodes = parse_netlist(input_file)
    A, z, node_index, node_names = build_mna(resistors, current_sources, voltage_sources, nodes)
    x = spsolve(A, z)
    voltages = x[:len(nodes)]

    with open(output_file, 'w') as f:
        for node, v in zip(node_names, voltages):
            f.write(f"{node} {v:.6f}\n")

    print(f"Voltage output written to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='SPICE netlist input file')
    parser.add_argument('--output_file', required=True, help='Output voltage file')
    args = parser.parse_args()

    solve_ir_drop(args.input_file, args.output_file)

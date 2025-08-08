
import argparse
import numpy as np
import os

def parse_spice_file(spice_file):
    resistors = []
    current_sources = []
    voltage_sources = []
    nodes = set()

    with open(spice_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('.', '*')):
                continue
            tokens = line.split()
            if len(tokens) != 4:
                continue
            comp, n1, n2, value = tokens
            value = float(value)
            if comp.startswith('R'):
                resistors.append((n1, n2, value))
                nodes.update([n1, n2])
            elif comp.startswith('I'):
                current_sources.append((n1, n2, value))
                nodes.update([n1, n2])
            elif comp.startswith('V'):
                voltage_sources.append((n1, n2, value))
                nodes.update([n1, n2])
    return resistors, current_sources, voltage_sources, nodes

# for finding coordinates in the node names
def extract_coordinates(node_name):
    try:
        _, _, x, y = node_name.strip().split('_')
        return int(x)//2000, int(y)//2000  # convert DBU to um
    except:
        return None


## test cases vary in size and padding with a minimum size of 600x600, so we pad the matrices to this size

def pad_to_shape(matrix, shape=(600, 600)):
    padded = np.zeros(shape)
    y, x = matrix.shape
    padded[:y, :x] = matrix
    return padded

def generate_maps(spice_file, voltage_file, output_dir):
    resistors, current_sources, voltage_sources, all_nodes = parse_spice_file(spice_file)

    coords = [extract_coordinates(n) for n in all_nodes if extract_coordinates(n) is not None]
    max_x = max(x for x, y in coords) + 1
    max_y = max(y for x, y in coords) + 1

    shape = (max(600, max_y), max(600, max_x))  # ensure maximum 600x600, similar to how we pad for mininum size

    current_map = np.zeros((max_y, max_x))
    density_map = np.zeros((max_y, max_x))
    voltage_source_map = np.zeros((max_y, max_x))
    ir_drop_map = np.zeros((max_y, max_x))

    for n1, n2, value in current_sources:
        for node in [n1, n2]:
            coord = extract_coordinates(node)
            if coord:
                y, x = coord
                current_map[y, x] += abs(value)

    for n1, n2, _ in resistors:
        for node in [n1, n2]:
            coord = extract_coordinates(node)
            if coord:
                y, x = coord
                density_map[y, x] += 1
    density_map[density_map <= 3] = 1
    density_map[(density_map > 3) & (density_map <= 6)] = 2
    density_map[density_map > 6] = 3

    for n1, n2, _ in voltage_sources:
        for node in [n1, n2]:
            coord = extract_coordinates(node)
            if coord:
                y, x = coord
                voltage_source_map[y, x] = 1

    voltage_dict = {}
    with open(voltage_file, 'r') as vf:
        for line in vf:
            parts = line.strip().split()
            if len(parts) == 2:
                voltage_dict[parts[0]] = float(parts[1])

    for node, voltage in voltage_dict.items():
        coord = extract_coordinates(node)
        if coord:
            y, x = coord
            ir_drop_map[y, x] = max(voltage_dict.values()) - voltage

    base = os.path.splitext(os.path.basename(spice_file))[0]

    # first this was erroring because of the output directory paths, later fixed
    np.savetxt(os.path.join(output_dir, f"current_map_{base}.csv"), pad_to_shape(current_map, shape), delimiter=",")
    np.savetxt(os.path.join(output_dir, f"pdn_density_map_{base}.csv"), pad_to_shape(density_map, shape), delimiter=",")
    np.savetxt(os.path.join(output_dir, f"voltage_source_map_{base}.csv"), pad_to_shape(voltage_source_map, shape), delimiter=",")
    np.savetxt(os.path.join(output_dir, f"ir_drop_map_{base}.csv"), pad_to_shape(ir_drop_map, shape), delimiter=",")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-spice_netlist', required=True)
    parser.add_argument('-voltage_file', required=True)
    parser.add_argument('-output', required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    generate_maps(args.spice_netlist, args.voltage_file, args.output)

from ttnopt.src import TreeTensorNetwork
from ttnopt.src import FactorizeTensor

import pandas as pd
import yaml
import argparse
from dotmap import DotMap
import numpy as np
from pathlib import Path


def factorize_tensor():
    parser = argparse.ArgumentParser(description="Factorize tensor")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    quantum_state = np.load(config.input)

    numerics = config.numerics
    if numerics.init_structure.tree == 1:
        psi = TreeTensorNetwork.tree(config.system.N)
    if numerics.init_structure.tree == 0:
        if config.numerics.init_random == 1:
            psi = TreeTensorNetwork.mps(config.system.N)
        else:
            psi = TreeTensorNetwork.mps(config.system.N, quantum_state, max_bond_dimension=numerics.initial_bond_dimension)

    opt_structure = numerics.opt_structure.active
    for i, (max_bond_dim, max_num_sweep) in enumerate(zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)):

        ft = FactorizeTensor(
            psi,
            quantum_state,
            init_bond_dim=numerics.initial_bond_dimension,
            max_bond_dim=max_bond_dim,
            truncation_error=numerics.truncation_error,
        )
        fidelity, entanglement = ft.run(opt_structure=opt_structure, max_num_sweep=max_num_sweep)
        opt_structure = 0


    nodes_list = []
    for edge_id in entanglement.keys():
        tmp = []
        for node_id, edges in enumerate(psi.edges):
            node_id += config.system.N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)

    for edge_id in psi.physical_edges:
        tmp = []
        tmp.append(edge_id)
        for node_id, edges in enumerate(psi.edges):
            node_id += config.system.N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)
        fidelity[edge_id] = 0.0
        entanglement[edge_id] = 0.0

    df = pd.DataFrame(nodes_list, columns=['node1', 'node2'], index=None)
    df['fidelity'] = fidelity.values()
    df['entanglement'] = entanglement.values()

    path = Path("out")
    df.to_csv(path / "mps.csv", header=True, index=None)

    return 0

factorize_tensor()

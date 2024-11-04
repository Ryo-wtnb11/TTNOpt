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

    quantum_state = np.load(config.target)
    N = len(quantum_state.shape)

    numerics = config.numerics

    if numerics.opt_structure.active == 0 and numerics.opt_fidelity.active == 0:
        print("Please check the configuration file. At least one of opt_structure and opt_fidelity should be active.")

    if numerics.opt_structure.active == 1:
        psi = TreeTensorNetwork.mps(N, quantum_state, max_bond_dimension=numerics.opt_structure.initial_bond_dimension)
        ft = FactorizeTensor(
            psi,
            quantum_state,
            max_bond_dim=max_bond_dim,
            truncation_error=numerics.truncation_error,
        )
        fidelity, entanglement = ft.run(opt_fidelity = False, opt_structure=True, max_num_sweep=numerics.opt_structure.max_num_sweeps)

    if numerics.opt_fidelity.active == 1:
        opt_structure = numerics.opt_fidelity.opt_structure.active
        if numerics.opt_fidelity.init_structure.tree == 1:
            print("Reset TN randomly as Binary TTN")
            psi = TreeTensorNetwork.tree(N)
        if numerics.opt_fidelity.init_structure.tree == 0 and numerics.opt_fidelity.init_random == 1:
            print("Reset TN randomly as MPS")
            psi = TreeTensorNetwork.mps(N)
        if numerics.opt_fidelity.init_structure.tree == 0 and numerics.opt_fidelity.init_random == 0:
            psi = TreeTensorNetwork.mps(N, quantum_state, max_bond_dimension=numerics.opt_structure.initial_bond_dimension)

        for i, (max_bond_dim, max_num_sweep) in enumerate(zip(numerics.opt_fidelity.max_bond_dimensions, numerics.opt_fidelity.max_num_sweeps)):
            ft = FactorizeTensor(
                psi,
                quantum_state,
                init_bond_dim=numerics.opt_fidelity.initial_bond_dimension,
                max_bond_dim=max_bond_dim,
                truncation_error=numerics.opt_fidelity.truncation_error,
            )
            fidelity, entanglement = ft.run(opt_fidelity=True, opt_structure=opt_structure, max_num_sweep=max_num_sweep)
            opt_structure = 0


    nodes_list = []
    for edge_id in entanglement.keys():
        tmp = []
        for node_id, edges in enumerate(psi.edges):
            node_id += N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)

    for edge_id in psi.physical_edges:
        tmp = []
        tmp.append(edge_id)
        for node_id, edges in enumerate(psi.edges):
            node_id += N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)
        fidelity[edge_id] = 0.0
        entanglement[edge_id] = 0.0

    if config.output.basic_file is not DotMap():
        df = pd.DataFrame(nodes_list, columns=['node1', 'node2'], index=None)
        if config.output.fidelity.active:
            df['energy'] = fidelity.values()
        if config.output.entanglement.active:
            df['entanglement'] = entanglement.values()

        path = Path(config.output.dir)
        df.to_csv(path / config.output.basic_file, header=True, index=None)

    return 0
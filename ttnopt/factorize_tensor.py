# ground_state_search.py
from ttnopt.src import init_structure_mps
from ttnopt.src import Observable
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch

from ttnopt.hamiltonian import hamiltonian

import pandas as pd
import yaml
import argparse
from dotmap import DotMap
from pathlib import Path
import itertools

def factorize_tensor():
    parser = argparse.ArgumentParser(description="Factorize")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    psi = TreeTensorNetwork.mps(config.system.N)

    numerics = config.numerics
    opt_structure = numerics.opt_structure.active
    edge_op_at_edge = None
    block_ham_at_edge = None
    for i, (max_bond_dim, max_num_sweep) in enumerate(zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)):
        ft = FactorizeTensor(
            psi,
            ham,
            init_bond_dim=numerics.initial_bond_dimension,
            max_bond_dim=max_bond_dim,
            truncation_error=numerics.truncation_error,
            edge_spin_operators=edge_op_at_edge,
            block_hamiltonians=block_ham_at_edge
        )

        # reset parameters for the next iteration
        opt_structure = 0
        edge_op_at_edge = gss.edge_spin_operators
        block_ham_at_edge = gss.block_hamiltonians



    return 0











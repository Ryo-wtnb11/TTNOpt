# ground_state_search.py
from ttnopt.src import init_structure_mps
from ttnopt.src import Observable
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch

from ttnopt.hamiltonian import hamiltonian

import yaml
import argparse
from dotmap import DotMap

def ground_state_search():
    parser = argparse.ArgumentParser(description="Ground state search simulation")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()


    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    psi = TreeTensorNetwork.mps(config.system.N)
    ham = hamiltonian(config.system)

    numerics = config.numerics
    opt_structure = numerics.opt_structure.active
    edge_op_at_edge = None
    block_ham_at_edge = None
    for i, (max_bond_dim, max_num_sweep) in enumerate(zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)):
        gss = GroundStateSearch(
            psi,
            ham,
            init_bond_dim=numerics.initial_bond_dimension,
            max_bond_dim=max_bond_dim,
            truncation_error=numerics.truncation_error,
            edge_spin_operators=edge_op_at_edge,
            block_hamiltonians=block_ham_at_edge
        )
        gss.run(
            opt_structure=opt_structure,
            energy_convergence_threshold=float(numerics.energy_convergence_threshold),
            entanglement_convergence_threshold=float(numerics.entanglement_convergence_threshold),
            max_num_sweep=max_num_sweep,
        )

        opt_structure = 0
        edge_op_at_edge = gss.edge_spin_operators
        block_ham_at_edge = gss.block_hamiltonians










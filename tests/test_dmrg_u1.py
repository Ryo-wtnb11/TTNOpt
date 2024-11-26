from ttnopt.src import Hamiltonian
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch

import math
import numpy as np


def open_adjacent_indexs(d: int):
    n = 2**d
    if n > 2:
        ind_list = [[i, (i + 1)] for i in range(n - 1)]
    else:
        ind_list = [[0, 1]]
    return ind_list


def heisenberg_chain_hamiltonian(d, coef_j=1.0):
    spin_size = ["S=1/2" for i in range(d**2)]
    indices = open_adjacent_indexs(d)
    interaction_coefs = [[coef_j, coef_j] for _ in indices]
    model = "XXZ"
    return Hamiltonian(
        d**2, spin_size, model, indices, interaction_coefs=interaction_coefs
    )


def test_run_dmrg():
    d = 4
    psi = TreeTensorNetwork.mps(d**2)
    hamiltonian = heisenberg_chain_hamiltonian(d)
    init_bond_dim = 4
    max_bond_dim = 10
    gss = GroundStateSearch(
        psi=psi,
        hamiltonian=hamiltonian,
        init_bond_dim=init_bond_dim,
        max_bond_dim=max_bond_dim,
    )
    gss.run(opt_structure=True, max_num_sweep=10)
    energy = gss.energy
    energy = np.mean(list(energy.values()))
    assert math.isclose(energy, -6.911621, abs_tol=1e-5)

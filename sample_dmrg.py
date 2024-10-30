from ttnopt import Hamiltonian
from ttnopt import TreeTensorNetwork
from ttnopt import GroundStateSearch


def open_adjacent_indexs(d: int):
    n = 2**d
    if n > 2:
        ind_list = [[i, (i + 1)] for i in range(n - 1)]
    else:
        ind_list = [[0, 1]]
    return ind_list


def generate_height_list(d: int):
    def assign_heights(start, end, height, heights):
        if start > end:
            return
        mid = (start + end) // 2
        heights[mid] = height
        assign_heights(start, mid - 1, height - 1, heights)
        assign_heights(mid + 1, end, height - 1, heights)

    length = 2**d - 1
    heights = [0] * length
    assign_heights(0, length - 1, d - 1, heights)
    return heights


def hierarchical_chain_hamiltonian(d, coef_j=1.0, alpha=0.5):
    spin_size = ["S=1/2" for i in range(d**2)]
    coefs = generate_height_list(d)
    coefs = [coef_j * (alpha**coef) for coef in coefs]
    interaction_coefs = [[coef, coef] for coef in coefs]
    indices = open_adjacent_indexs(d)
    model = "XXZ"
    return Hamiltonian(d**2, spin_size, model, indices, interaction_coefs=interaction_coefs)


if __name__ == "__main__":
    d = 4
    psi = TreeTensorNetwork.mps(d**2)
    hamiltonian = hierarchical_chain_hamiltonian(d)
    init_bond_dim = 4
    max_bond_dim = 100
    dmrg = GroundStateSearch(
        psi=psi,
        hamiltonian=hamiltonian,
        init_bond_dim=init_bond_dim,
        max_bond_dim=max_bond_dim,
    )
    dmrg.run(opt_structure=True)
    energy = dmrg.energy()
    print(energy)
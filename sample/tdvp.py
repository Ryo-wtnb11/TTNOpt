from ttnopt.initialTTN import init_structure_mps, init_neel_mps
from ttnopt.Observable import Observable
from ttnopt.TTN import TreeTensorNetwork
from ttnopt.TDVP import TDVP


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
    coefs = generate_height_list(d)
    adjacent_indices = open_adjacent_indexs(d)
    coefs = [coef_j * (alpha**coef) for coef in coefs]
    observables = []
    for i, coef in enumerate(coefs):
        indices = adjacent_indices[i]
        operators_list = [["S+", "S-"], ["S-", "S+"], ["Sz", "Sz"]]
        coef_list = [coef / 2.0, coef / 2.0, coef]
        ob = Observable(indices, operators_list, coef_list)
        observables.append(ob)
    return observables


def heisenberg_hamiltonian(d):
    adjacent_indices = open_adjacent_indexs(d)
    observables = []
    for i in adjacent_indices:
        indices = i
        operators_list = [["S+", "S-"], ["S-", "S+"], ["Sz", "Sz"]]
        coef_list = [1 / 2.0, 1 / 2.0, 1]
        ob = Observable(indices, operators_list, coef_list)
        observables.append(ob)
    return observables


def xxz_hamiltonian(d):
    adjacent_indices = open_adjacent_indexs(d)
    observables = []
    for i in adjacent_indices:
        indices = i
        operators_list = [["S+", "S-"], ["S-", "S+"], ["Sz", "Sz"]]
        coef_list = [1 / 2.0, 1 / 2.0, 1 / 2.0]
        ob = Observable(indices, operators_list, coef_list)
        observables.append(ob)
    return observables


if __name__ == "__main__":
    d = 4
    size = 2**d
    physical_edges, edges, top_edge_id = init_structure_mps(size)
    psi = TreeTensorNetwork(
        edges,
        top_edge_id,
    )
    # hamiltonians = heisenberg_hamiltonian(d)
    hamiltonians = xxz_hamiltonian(d)
    physical_spin_nums = {i: "S=1/2" for i in psi.physical_edges}
    max_bond_dim = 4
    tdvp = TDVP(
        psi,
        physical_spin_nums,
        hamiltonians,
        max_bond_dim=max_bond_dim,
    )
    dt = 0.01
    max_t = 4
    Mz = []

    tdvp.run(opt_structure=False, dt=dt)
    mz = tdvp.calculate_expval(size // 2, "Sz")
    Mz.append(mz)

    # csvで保存
    import pandas as pd

    df = pd.DataFrame(Mz)
    df.to_csv("Mz.csv", header=False, index=False)
    # each id of indices must be in physical_edges of TTN

from ttnopt.src import TreeTensorNetwork
from ttnopt.src import FactorizeTensor
import numpy as np
import pandas as pd
from pathlib import Path

def factorize_tensor():
    quantum_state = np.load('input/quantum_state_standard.npy')
    N = len(quantum_state.shape)
    psi = TreeTensorNetwork.mps(N, quantum_state, max_bond_dimension=4)


    opt_structure = False
    for i, (max_bond_dim, max_num_sweep) in enumerate(zip([4, 8, 20], [15, 5, 5])):
        print(i)
        ft = FactorizeTensor(
            psi,
            quantum_state,
            init_bond_dim=4,
            max_bond_dim=max_bond_dim,
            truncation_error=1e-10,
        )
        # reset parameters for the next iteration
        fidelity, entanglement = ft.run(opt_structure=opt_structure, max_num_sweep=max_num_sweep)
        opt_structure = False


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

    df = pd.DataFrame(nodes_list, columns=['node1', 'node2'], index=None)
    df['fidelity'] = fidelity.values()
    df['entanglement'] = entanglement.values()

    path = Path("out")
    df.to_csv(path / "mps.csv", header=True, index=None)

    return 0

factorize_tensor()


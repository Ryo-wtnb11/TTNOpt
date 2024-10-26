from ttnopt.src import TreeTensorNetwork
from ttnopt.src import FactorizeTensor
import numpy as np

def factorize_tensor():
    quantum_state = np.load('input/quantum_state_standard.npy')
    N = len(quantum_state.shape)
    psi = TreeTensorNetwork.mps(N, quantum_state, max_bond_dimension=4)

    ft = FactorizeTensor(
        psi,
        quantum_state,
        init_bond_dim=4,
        max_bond_dim=4,
        truncation_error=1e-10,
    )

    ft.run(opt_structure=True)

    p = psi.visualize()
    p.savefig('structure.pdf')
    return 0

factorize_tensor()










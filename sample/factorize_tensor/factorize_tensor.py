from ttnopt.src import TreeTensorNetwork
from ttnopt.src import FactorizeTensor
import numpy as np

def factorize_tensor():
    quantum_state = np.load('quantum_state_standard.npy')
    N = len(quantum_state.shape)
    psi = TreeTensorNetwork.mps(N, quantum_state, max_bond_dimension=10)

    ft = FactorizeTensor(
        psi,
        quantum_state,
        init_bond_dim=4,
        max_bond_dim=100,
        truncation_error=1e-10,
    )

    ft.run()

    # reset parameters for the next iteration
    opt_structure = 0

    return 0

factorize_tensor()










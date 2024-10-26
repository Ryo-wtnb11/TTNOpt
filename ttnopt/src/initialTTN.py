import numpy as np
import tensornetwork as tn


def init_tensors_mps(state, max_bond, truncate, min_bond=2):  # TODO
    """_summary_

    Args:
        state tensor: initial_state where the shape is (bond_dim[0], bond_dim[1], ..., bond_dim[N])
        size (_type_): _description_
    Returns:
        tensors: List[tn.Node]: List of tensors
    """
    tensors = []
    size = len(state.shape)
    for i in range(size):
        if i == 0:
            print("0")
        elif i == size - 1:
            tensor = tn.Node(state[i], backend="numpy")
        else:
            tensor = tn.Node(state[i], backend="numpy")
        tensors.append(tensor)
    return tensors


def get_upper_bond_dim(lower_bond_dims, max_bond_dim):
    bond_dim = np.prod(lower_bond_dims)
    if bond_dim > max_bond_dim:
        bond_dim = max_bond_dim
    return bond_dim
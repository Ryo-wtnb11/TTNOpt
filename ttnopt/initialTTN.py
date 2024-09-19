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


def init_neel_mps(length):
    tensors = []
    for i in range(length // 2 - 1):
        if i == 0:
            tensor = np.kron(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
            tensor = tensor.reshape(2, 2, 1)
        elif i % 2 == 0:
            tensor = np.array([1.0, 0.0])
            tensor = tensor.reshape(1, 2, 1)
        elif i % 2 == 1:
            tensor = np.array([0.0, 1.0])
            tensor = tensor.reshape(1, 2, 1)
        tensors.append(tensor)
    tmp_tensors = []
    for i in range(length // 2 - 1):
        if i == 0:
            tensor = np.kron(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
            tensor = tensor.reshape(2, 2, 1)
        elif i % 2 == 0:
            tensor = np.array([0.0, 1.0])
            tensor = tensor.reshape(2, 1, 1)
        elif i % 2 == 1:
            tensor = np.array([1.0, 0.0])
            tensor = tensor.reshape(2, 1, 1)
        tmp_tensors.append(tensor)
    tensors += reversed(tmp_tensors)

    return tensors


def init_structure_binary_tree(size):
    """_summary_

    Args:
        size int: the size of system

    Returns:
        edges List[[int, int, int]]: the edges of the structure
        center_edge_id int: the top edge id
    """
    edges = []

    layer_index = 0
    num_layer = int(np.log2(size)) - 1
    for layer in range(num_layer):
        tensor_num = int(2 ** (np.log2(size) - 1 - layer))
        nn = int(2 ** (np.log2(size) - layer))
        for i in range(tensor_num):
            if layer != num_layer - 1:
                edge_id = [
                    layer_index + i * 2,
                    layer_index + i * 2 + 1,
                    layer_index + nn + i,
                ]

            else:
                edge_id = [
                    layer_index + i * 2,
                    layer_index + i * 2 + 1,
                    layer_index + nn,
                ]

            edges.append(edge_id)

        layer_index += nn

    center_edge_id = layer_index
    physical_edges = [i for i in range(size)]

    return physical_edges, edges, center_edge_id


def init_structure_mps(size):
    """_summary_

    Args:
        size int: the size of system

    Returns:
        edges List[[int, int, int]]: the edges of the structure
        center_edge_id int: the top edge id
    """
    edges = []
    upper_edge_id = size
    edges.append([0, 1, upper_edge_id])

    for i in range(2, (size - 2) // 2 + 1):
        edges.append([upper_edge_id, i, upper_edge_id + 1])
        upper_edge_id += 1

    center_edge_id = upper_edge_id
    upper_edge_id = upper_edge_id + 1

    tmp_edges = []
    tmp_edges.append([size - 2, size - 1, upper_edge_id])

    for i in reversed(range((size - 2) // 2 + 2, size - 2)):
        tmp_edges.append([i, upper_edge_id, upper_edge_id + 1])
        upper_edge_id += 1

    tmp_edges.append([(size - 2) // 2 + 1, upper_edge_id, center_edge_id])
    edges += reversed(tmp_edges)

    physical_edges = [i for i in range(size)]

    return physical_edges, edges, center_edge_id


def get_upper_bond_dim(lower_bond_dims, max_bond_dim):
    bond_dim = np.prod(lower_bond_dims)
    if bond_dim > max_bond_dim:
        bond_dim = max_bond_dim
    return bond_dim


"""
def init_structure_binary_tree(size, physical_bond_dims, max_bond_dim):
    edges = []

    ranks = []
    bond_dims = physical_bond_dims
    new_bond_dims = []

    layer_index = 0
    num_layer = int(np.log2(size)) - 1
    for layer in range(num_layer):
        tensor_num = int(2 ** (np.log2(size) - 1 - layer))
        nn = int(2 ** (np.log2(size) - layer))
        for i in range(tensor_num):

            lower_bond_dim_l = bond_dims[i * 2]
            lower_bond_dim_r = bond_dims[i * 2 + 1]
            upper_bond_dim = get_upper_bond_dim(
                [lower_bond_dim_l, lower_bond_dim_r], max_bond_dim
            )
            new_bond_dims.append(upper_bond_dim)

            rank = [lower_bond_dim_l, lower_bond_dim_r, upper_bond_dim]
            ranks.append(rank)

            if layer != num_layer - 1:
                edge_id = [
                    layer_index + i * 2,
                    layer_index + i * 2 + 1,
                    layer_index + nn + i,
                ]

            else:
                edge_id = [
                    layer_index + i * 2,
                    layer_index + i * 2 + 1,
                    layer_index + nn,
                ]

            edges.append(edge_id)

        layer_index += nn

        new_bond_dims = []

    center_edge_id = layer_index

    return edges, ranks, center_edge_id


def init_structure_mps(size, physical_bond_dims, max_bond_dim):
    edges = []
    upper_edge_id = size
    edges.append([0, 1, upper_edge_id])

    ranks = []
    upper_bond_dim = get_upper_bond_dim(physical_bond_dims[:2], max_bond_dim)
    ranks.append([physical_bond_dims[0], physical_bond_dims[1], upper_bond_dim])

    for i in range(2, (size - 2) // 2 + 1):
        edges.append([upper_edge_id, i, upper_edge_id + 1])
        upper_edge_id += 1

        lower_bond_dim = upper_bond_dim
        upper_bond_dim = get_upper_bond_dim(
            [physical_bond_dims[i - 1], lower_bond_dim], max_bond_dim
        )
        rank = [physical_bond_dims[i], lower_bond_dim, upper_bond_dim]
        ranks.append(rank)

    top_edge_id = upper_edge_id
    upper_edge_id = upper_edge_id + 1
    top_bond_dim = upper_bond_dim

    lower_bond_dim = physical_bond_dims[size - 1]
    upper_bond_dim = get_upper_bond_dim(physical_bond_dims[size - 2 :], max_bond_dim)

    tmp_edges = []
    tmp_edges.append([size - 2, size - 1, upper_edge_id])

    tmp_ranks = []
    tmp_ranks.append(
        [physical_bond_dims[size - 2], physical_bond_dims[size - 1], upper_bond_dim]
    )
    for i in reversed(range((size - 2) // 2 + 2, size - 2)):
        tmp_edges.append([i, upper_edge_id, upper_edge_id + 1])
        upper_edge_id += 1

        bottom_bond_dim = upper_bond_dim
        upper_bond_dim = get_upper_bond_dim(
            [physical_bond_dims[i], bottom_bond_dim], max_bond_dim
        )
        rank = [physical_bond_dims[i], bottom_bond_dim, upper_bond_dim]
        tmp_ranks.append(rank)

    tmp_edges.append([(size - 2) // 2 + 1, upper_edge_id, top_edge_id])
    tmp_ranks.append(
        [
            physical_bond_dims[(size - 2) // 2 + 1],
            upper_bond_dim,
            top_bond_dim,
        ]
    )
    edges += reversed(tmp_edges)
    ranks += reversed(tmp_ranks)

    return edges, ranks, top_edge_id


def get_upper_bond_dim(lower_bond_dims, max_bond_dim):
    bond_dim = np.prod(lower_bond_dims)
    if bond_dim > max_bond_dim:
        bond_dim = max_bond_dim
    return bond_dim
"""

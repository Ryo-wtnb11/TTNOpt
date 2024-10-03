import tensornetwork as tn
import numpy as np


def get_renormalization_sequence(edges, top_edge_id):
    tensor_flag = [0 for _ in range(len(edges))]
    top_child_ids = [i for i in range(len(edges)) if edges[i][2] == top_edge_id]

    stack = [(-1, top_child_ids)]
    sequence = []
    while stack:
        parent_id, child_ids = stack[-1]
        if all(tensor_flag[child_id] == 1 for child_id in child_ids) or not child_ids:
            if parent_id != -1:
                sequence.append(parent_id)
                tensor_flag[parent_id] = 1
            stack.pop()
        else:
            for child_id in child_ids:
                if tensor_flag[child_id] == 0:
                    stack.append((child_id, get_child_tensor_id(child_id, edges)))
    return sequence


def get_child_tensor_id(tensor_id, edges):
    child_tensor_id = []
    for i, edge in enumerate(edges):
        if edge[2] in edges[tensor_id][:2]:
            child_tensor_id.append(i)
    return child_tensor_id


def get_bare_edges(edge_id, edges, physical_edges):
    bare_spins = []
    if edge_id in physical_edges:
        return [edge_id]
    else:
        child_ids = [i for i, edge in enumerate(edges) if edge_id == edge[2]]
        while len(child_ids) > 0:
            new_child = []
            for child_id in child_ids:
                for edge in edges[child_id][:2]:
                    if edge in physical_edges:
                        bare_spins.append(edge)
                new_child += get_child_tensor_id(child_id, edges)
            child_ids = new_child
    return bare_spins


def inner_product(u, v):
    u = u.copy(conjugate=True)
    u[0] ^ v[0]
    u[1] ^ v[1]
    u[2] ^ v[2]
    u[3] ^ v[3]
    return tn.contractors.auto([u, v]).get_tensor()


def inner_product_sparse(u, v):
    u = u.copy(conjugate=True)
    u[0] ^ v[0]
    u[1] ^ v[1]
    u[2] ^ v[2]
    u[3] ^ v[3]
    u[4] ^ v[4]
    prod = tn.contractors.auto([u, v]).tensor.todense()
    return np.real(prod)[0]

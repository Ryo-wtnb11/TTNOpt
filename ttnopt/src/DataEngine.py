import tensornetwork as tn
import numpy as np

from ttnopt.src.TTN import TreeTensorNetwork
from ttnopt.src.TwoSiteUpdater import TwoSiteUpdater
from ttnopt.src.functionTTN import (
    get_renormalization_sequence,
)

tn.set_default_backend("numpy")


class DataEngine(TwoSiteUpdater):
    def __init__(
        self,
        psi: TreeTensorNetwork,
        target: np.ndarray,
        init_bond_dim: int,
        max_bond_dim: int,
        truncation_error: float,
    ):
        """Initialize a PhysicsEngine object.

        Args:
            psi (TreeTensorNetwork): The quantum state.
            target (np.ndarray): Tensor of target_data.
            init_bond_dim (int): Initial bond dimension.
            max_bond_dim (int): Maximum bond dimension.
            truncation_error (float): Maximum truncation error.
        """
        super().__init__(psi)
        self.target = target / self.psi.norm
        self.init_bond_dim = init_bond_dim
        self.max_bond_dim = max_bond_dim
        self.truncation_error = truncation_error
        self.environment_tensor = None
        self.environment_edges = None

        init_tensor_flag = False
        if self.psi.tensors is None:
            print("No initial tensors in TTN object.")
            self.psi.tensors = []
            for _ in self.psi.edges:
                self.psi.tensors.append(None)
            init_tensor_flag = True
        else:
            for i, dim in enumerate(target.shape):
                if dim != self.psi.edge_dims[i]:
                    print("Initial tensors are not valid for given Target")
                    init_tensor_flag = True
                    break

        if init_tensor_flag:
            print("Initialize tensors rank")
            for i, dim in enumerate(target.shape):
                self.psi.edge_dims[i] = dim
            self.init_tensors()

    def get_fidelity(self):
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        target = tn.Node(self.target)
        dangling_edges_dict = {i: target[i] for i in self.psi.physical_edges}
        for tensor_id in sequence:
            iso = tn.Node(self.psi.tensors[tensor_id])
            for i, edge_id in enumerate(self.psi.edges[tensor_id][:2]):
                dangling_edges_dict[edge_id] ^ iso[i]
                # remove dangling edge
                dangling_edges_dict.pop(edge_id)
            out_edge_orders = list(dangling_edges_dict.values()) + [iso[2]]
            target = tn.contractors.auto(
                [target, iso], output_edge_order=out_edge_orders
            )
            dangling_edges_dict[self.psi.edges[tensor_id][2]] = target[-1]
        gauge = tn.Node(self.psi.gauge_tensor)
        target[0] ^ gauge[0]
        target[1] ^ gauge[1]
        inner_prod = tn.contractors.auto([target, gauge])
        return np.abs(inner_prod.tensor) ** 2

    def update_tensor(self, central_tensor_ids):
        environment, out_edge_orders = self.environment(central_tensor_ids)
        output_order = (
            self.psi.edges[central_tensor_ids[0]][:2]
            + self.psi.edges[central_tensor_ids[1]][:2]
        )
        environment.reorder_edges(
            [out_edge_orders[edge_id] for edge_id in output_order]
        )
        return environment

    def environment(self, tensor_ids):
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        if self.environment_tensor is None:
            environment = tn.Node(self.target)
            dangling_edges_dict = {i: environment[i] for i in self.psi.physical_edges}
            for tensor_id in sequence:
                if tensor_id not in tensor_ids:
                    iso = tn.Node(self.psi.tensors[tensor_id])
                    for i, edge_id in enumerate(self.psi.edges[tensor_id][:2]):
                        dangling_edges_dict[edge_id] ^ iso[i]
                        # remove dangling edge
                        dangling_edges_dict.pop(edge_id)
                    out_edge_orders = list(dangling_edges_dict.values()) + [iso[2]]
                    environment = tn.contractors.auto(
                        [environment, iso], output_edge_order=out_edge_orders
                    )
                    dangling_edges_dict[self.psi.edges[tensor_id][2]] = environment[-1]
            return environment, dangling_edges_dict
        else:
            environment = tn.Node(self.environment_tensor)
            dangling_edges_dict = {
                edge: environment[i] for i, edge in enumerate(self.environment_edges)
            }
            for tensor_id in sequence:
                if set(self.psi.edges[tensor_id][:2]).issubset(self.environment_edges):
                    iso = tn.Node(self.psi.tensors[tensor_id])
                    for i, edge_id in enumerate(self.psi.edges[tensor_id][:2]):
                        dangling_edges_dict[edge_id] ^ iso[i]
                        # remove dangling edge
                        dangling_edges_dict.pop(edge_id)
                    out_edge_orders = list(dangling_edges_dict.values()) + [iso[2]]
                    environment = tn.contractors.auto(
                        [environment, iso], output_edge_order=out_edge_orders
                    )
                    dangling_edges_dict[self.psi.edges[tensor_id][2]] = environment[-1]
                if self.psi.edges[tensor_id][
                    2
                ] == self.psi.canonical_center_edge_id and any(
                    elem not in dangling_edges_dict.keys()
                    for elem in self.psi.edges[tensor_id][:2]
                ):
                    iso = tn.Node(self.psi.tensors[tensor_id])
                    iso = iso.copy(conjugate=True)
                    dangling_edges_dict[self.psi.edges[tensor_id][2]] ^ iso[2]
                    dangling_edges_dict.pop(self.psi.edges[tensor_id][2])
                    out_edge_orders = list(dangling_edges_dict.values()) + [
                        iso[0],
                        iso[1],
                    ]
                    environment = tn.contractors.auto(
                        [environment, iso], output_edge_order=out_edge_orders
                    )
                    dangling_edges_dict[self.psi.edges[tensor_id][0]] = environment[-2]
                    dangling_edges_dict[self.psi.edges[tensor_id][1]] = environment[-1]
            return environment, dangling_edges_dict

    def init_tensors(self):
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        for tensor_id in sequence:
            m = (
                self.psi.edge_dims[self.psi.edges[tensor_id][0]]
                * self.psi.edge_dims[self.psi.edges[tensor_id][1]]
            )
            n = np.min([m, self.init_bond_dim])
            random_matrix = np.random.normal(0, 1, (m, n))
            Q, _ = np.linalg.qr(random_matrix)
            self.psi.tensors[tensor_id] = np.reshape(
                Q,
                (
                    self.psi.edge_dims[self.psi.edges[tensor_id][0]],
                    self.psi.edge_dims[self.psi.edges[tensor_id][1]],
                    n,
                ),
            )
            self.psi.edge_dims[self.psi.edges[tensor_id][2]] = n
        self.psi.gauge_tensor = np.eye(n)

    def move_center(self, opt_structure=False):
        (
            edge_id,
            selected_tensor_id,
            connected_tensor_id,
            not_selected_tensor_id,
        ) = self.local_two_tensor()

        # absorb gauge tensor
        iso = tn.Node(self.psi.tensors[selected_tensor_id])
        gauge = tn.Node(self.psi.gauge_tensor)
        iso[2] ^ gauge[0]
        iso = tn.contractors.auto(
            [iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]]
        )
        self.psi.tensors[selected_tensor_id] = iso.get_tensor()

        self.set_flag(not_selected_tensor_id)

        self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

        psi1 = tn.Node(self.psi.tensors[selected_tensor_id])
        psi2 = tn.Node(self.psi.tensors[connected_tensor_id])
        psi1[2] ^ psi2[2]
        psi = tn.contractors.auto(
            [psi1, psi2], output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1]]
        )
        u, s, v, edge_order = self.decompose_two_tensors(
            psi,
            self.max_bond_dim,
            opt_structure=opt_structure,
            operate_degeneracy=False,
        )
        psi_edges = (
            self.psi.edges[selected_tensor_id][:2]
            + self.psi.edges[connected_tensor_id][:2]
        )

        self.psi.tensors[selected_tensor_id] = u
        self.psi.tensors[connected_tensor_id] = v
        self.psi.gauge_tensor = s
        (
            self.psi.edges[selected_tensor_id][0],
            self.psi.edges[selected_tensor_id][1],
        ) = (
            psi_edges[edge_order[0]],
            psi_edges[edge_order[1]],
        )
        (
            self.psi.edges[connected_tensor_id][0],
            self.psi.edges[connected_tensor_id][1],
        ) = (
            psi_edges[edge_order[2]],
            psi_edges[edge_order[3]],
        )
        self.distance = self.initial_distance()

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
        max_bond_dim: int,
    ):
        """Initialize a PhysicsEngine object.

        Args:
            psi (TreeTensorNetwork): The quantum state.
            target (np.ndarray): Tensor of target_data.
            max_bond_dim (int): Maximum bond dimension.
        """
        super().__init__(psi)
        self.max_bond_dim = max_bond_dim
        self.environment_tensor = None
        self.environment_edges = None

    def get_fidelity(self, target):
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        target_node = tn.Node(target)
        target_node = target_node.copy(conjugate=True)
        dangling_edges_dict = {i: target_node[i] for i in self.psi.physical_edges}
        for tensor_id in sequence:
            iso = tn.Node(self.psi.tensors[tensor_id])
            for i, edge_id in enumerate(self.psi.edges[tensor_id][:2]):
                dangling_edges_dict[edge_id] ^ iso[i]
                # remove dangling edge
                dangling_edges_dict.pop(edge_id)
            out_edge_orders = list(dangling_edges_dict.values()) + [iso[2]]
            target_node = tn.contractors.auto(
                [target_node, iso], output_edge_order=out_edge_orders
            )
            dangling_edges_dict[self.psi.edges[tensor_id][2]] = target_node[-1]
        gauge = tn.Node(self.psi.gauge_tensor)
        target_node[0] ^ gauge[0]
        target_node[1] ^ gauge[1]
        inner_prod = tn.contractors.auto([target_node, gauge])
        return np.abs(inner_prod.tensor) ** 2

    def update_tensor(self, target, central_tensor_ids):
        environment, out_edge_orders = self.environment(target, central_tensor_ids)
        output_order = (
            self.psi.edges[central_tensor_ids[0]][:2]
            + self.psi.edges[central_tensor_ids[1]][:2]
        )
        environment.reorder_edges(
            [out_edge_orders[edge_id] for edge_id in output_order]
        )
        return environment

    def environment(self, target, tensor_ids):
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        environment = tn.Node(target)
        environment = environment.copy(conjugate=True)
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

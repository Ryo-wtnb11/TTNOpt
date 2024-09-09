import tensornetwork as tn
import numpy as np
from src.TwoSiteUpdater import TwoSiteUpdater
import copy


class DataEngine(TwoSiteUpdater):
    def __init__(
        self,
        psi,
        max_bond_dim=100,
        max_truncation_err=1e-11,
    ):
        """_summary_

        Args:
            psi TreeTensorNetwork: the initial state
            hamiltonian List[Model]: the hamiltonian
        """
        super().__init__(psi)
        self.max_bond_dim = max_bond_dim
        self.max_truncation_err = max_truncation_err

    def opt_structure(
        self,
        ee_threshold=1e-8,
        converged_count=1,
    ):
        ee_at_edge, _ee_at_edge = {}, {}
        edges, _edges = copy.deepcopy(self.psi.edges), copy.deepcopy(self.psi.edges)

        converged_num = 0

        sweep_num = 0
        while converged_num < converged_count:
            ee_at_edge = copy.deepcopy(_ee_at_edge)
            edges = copy.deepcopy(_edges)

            self.distance = self.initial_distance()
            self.flag = self.initial_flag()

            plt = self.psi.visualize()
            plt.show()
            while self.candidate_edge_ids() != []:
                self.move_center(opt_structure=True)
                ee = self.entanglement_entropy()
                _ee_at_edge[self.psi.canonical_center_edge_id] = ee

            _edges = copy.deepcopy(self.psi.edges)
            # 終了判定
            sweep_num += 1
            if sweep_num > 2:
                diff_ee = [
                    np.abs(ee_at_edge[key] - _ee_at_edge[key])
                    for key in ee_at_edge.keys()
                ]
                if all(
                    [
                        set(edge[:2]) == set(_edge[:2]) and edge[2] == _edge[2]
                        for edge, _edge in zip(edges, _edges)
                    ]
                ):
                    if all([ee < ee_threshold for ee in diff_ee]):
                        converged_num += 1

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
            psi, opt_structure=opt_structure
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

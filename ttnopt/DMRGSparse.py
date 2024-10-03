import tensornetwork as tn
import numpy as np
from ttnopt.PhysicsEngineSparse import PhysicsEngineSparse
import copy
from ttnopt.functionTTN import (
    inner_product_sparse,
)


class DMRGSparse(PhysicsEngineSparse):

    def __init__(
        self,
        psi,
        physical_spin_nums: list[int],
        hamiltonians,
        u1_num: int,
        init_bond_dim: int = 4,
        max_bond_dim: int = 100,
        max_truncation_err: float = 1e-11,
    ):
        super().__init__(
            psi,
            physical_spin_nums,
            hamiltonians,
            u1_num,
            init_bond_dim,
            max_bond_dim,
            max_truncation_err,
        )

    def run(
        self,
        energy_threshold: float = 1e-8,
        ee_threshold: float = 1e-8,
        converged_count: int = 1,
        opt_structure: bool = False,
    ):
        energy_at_edge, _energy_at_edge = {}, {}
        ee_at_edge, _ee_at_edge = {}, {}
        edges, _edges = copy.deepcopy(self.psi.edges), copy.deepcopy(self.psi.edges)

        converged_num = 0

        sweep_num = 0
        while converged_num < converged_count:
            # energy
            energy_at_edge = copy.deepcopy(_energy_at_edge)
            ee_at_edge = copy.deepcopy(_ee_at_edge)
            edges = copy.deepcopy(_edges)

            self.distance = self.initial_distance()
            self.flag = self.initial_flag()

            print("Sweep count: " + str(sweep_num))
            while self.candidate_edge_ids() != []:
                (
                    edge_id,
                    selected_tensor_id,
                    connected_tensor_id,
                    not_selected_tensor_id,
                ) = self.local_two_tensor()
                # absorb gauge tensor
                iso = tn.Node(self.psi.tensors[selected_tensor_id])
                gauge = tn.Node(self.psi.gauge_tensor)
                if iso.tensor.flat_flows[2]:
                    out = gauge[1]
                    iso[2] ^ gauge[0]
                else:
                    out = gauge[0]
                    iso[2] ^ gauge[1]
                iso = tn.contractors.auto(
                    [iso, gauge], output_edge_order=[iso[0], iso[1], out, gauge[2]]
                )
                self.psi.tensors[selected_tensor_id] = iso.get_tensor()

                self.set_flag(not_selected_tensor_id)

                self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

                self._set_block_hamiltonian(not_selected_tensor_id)

                ground_state = self.lanczos([selected_tensor_id, connected_tensor_id])
                ground_state = ground_state / np.sqrt(
                    inner_product_sparse(ground_state, ground_state)
                )
                psi_edges = (
                    self.psi.edges[selected_tensor_id][:2]
                    + self.psi.edges[connected_tensor_id][:2]
                )

                u, s, v, edge_order, ee = self.decompose_two_tensors(
                    ground_state,
                    self.max_bond_dim,
                    self.max_truncation_err,
                    opt_structure=opt_structure,
                    operate_degeneracy=True,
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

                energy = self.energy()
                print(energy)
                _energy_at_edge[self.psi.canonical_center_edge_id] = energy
                _ee_at_edge[self.psi.canonical_center_edge_id] = ee

            _edges = copy.deepcopy(self.psi.edges)

            # 終了判定
            sweep_num += 1
            if sweep_num > 2:
                diff_energy = [
                    np.abs(energy_at_edge[key] - _energy_at_edge[key])
                    for key in energy_at_edge.keys()
                ]
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
                    if all(
                        [energy < energy_threshold for energy in diff_energy]
                    ) and all([ee < ee_threshold for ee in diff_ee]):
                        converged_num += 1
        print("Converged")

    def set_ttn_properties_at_one_tensor(self, edge_id, selected_tensor_id):
        # update_ttn_properties
        self.psi.canonical_center_edge_id = edge_id
        out_selected_inds = []
        for i, e in enumerate(self.psi.edges[selected_tensor_id]):
            if e == edge_id:
                canonical_center_ind = i
            else:
                out_selected_inds.append(i)
        self.psi.tensors[selected_tensor_id] = self.psi.tensors[
            selected_tensor_id
        ].transpose(
            out_selected_inds + [canonical_center_ind] + [3],
        )
        self.psi.edges[selected_tensor_id] = [
            self.psi.edges[selected_tensor_id][i] for i in out_selected_inds
        ] + [edge_id]
        for i, e in enumerate(self.psi.edges[selected_tensor_id]):
            self.psi.edge_dims[e] = self.psi.tensors[selected_tensor_id].shape[i]
        return

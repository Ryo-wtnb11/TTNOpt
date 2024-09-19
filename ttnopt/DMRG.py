import tensornetwork as tn
import numpy as np
from ttnopt.PhysicsEngine import PhysicsEngine
import copy


class DMRG(PhysicsEngine):
    def __init__(
        self,
        psi,
        physical_spin_nums,
        hamiltonians,
        init_bond_dim=4,
        max_bond_dim=100,
        max_truncation_err=1e-11,
    ):
        super().__init__(
            psi,
            physical_spin_nums,
            hamiltonians,
            init_bond_dim,
            max_bond_dim,
            max_truncation_err,
        )

    def run(
        self,
        energy_threshold=1e-8,
        ee_threshold=1e-8,
        converged_count=1,
        opt_structure=False,
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
                iso[2] ^ gauge[0]
                iso = tn.contractors.auto(
                    [iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]]
                )
                self.psi.tensors[selected_tensor_id] = iso.get_tensor()

                self.set_flag(not_selected_tensor_id)

                self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

                self._set_edge_spin(not_selected_tensor_id)
                self._set_block_hamiltonian(not_selected_tensor_id)

                ground_state = self.lanczos([selected_tensor_id, connected_tensor_id])
                psi_edges = (
                    self.psi.edges[selected_tensor_id][:2]
                    + self.psi.edges[connected_tensor_id][:2]
                )

                u, s, v, edge_order = self.decompose_two_tensors(
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
                ee = self.entanglement_entropy(self.max_truncation_err)
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

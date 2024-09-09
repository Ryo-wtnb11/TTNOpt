import tensornetwork as tn
from src.PhysicsEngine import PhysicsEngine


class TDVP(PhysicsEngine):
    def __init__(
        self,
        psi,
        physical_spin_nums,
        hamiltonians,
        max_bond_dim=100,
        max_truncation_err=1e-11,
    ):
        super().__init__(
            psi, physical_spin_nums, hamiltonians, max_bond_dim, max_truncation_err
        )

    def run(
        self,
        dt,
        opt_structure=False,
    ):
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

            psi = self.lanczos_exp_multiply(
                [selected_tensor_id, connected_tensor_id], dt
            )
            psi_edges = (
                self.psi.edges[selected_tensor_id][:2]
                + self.psi.edges[connected_tensor_id][:2]
            )

            u, s, v, edge_order = self.decompose_two_tensors(
                psi, opt_structure=opt_structure, operate_degeneracy=True
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

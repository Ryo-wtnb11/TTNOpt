import numpy as np
import tensornetwork as tn
from tensornetwork import BlockSparseTensor, U1Charge

from ttnopt.src.TwoSiteUpdater import TwoSiteUpdaterMixin


class TwoSiteUpdaterSparse(TwoSiteUpdaterMixin):
    def __init__(self, psi):
        self.psi = psi
        self.backend = "symmetric"
        self.flag = self.initial_flag()
        self.distance = self.initial_distance()

    def decompose_two_tensors(
        self,
        psi,
        max_bond_dim,
        opt_structure=0,
        epsilon=1e-8,
        delta=1e-8,
        temperature=0.0,
    ):
        edge_order = [0, 1, 2, 3]
        psi_last = psi.copy()
        if not opt_structure:
            a = psi[0]
            b = psi[1]
            c = psi[2]
            d = psi[3]
            e = psi[4]
            (u, s, v, _) = tn.split_node_full_svd(psi, [e, a, b], [c, d])
            p = s.tensor.data
            p = np.sort(p)[::-1]
            p = p[p > 0.0]
            ind = np.min([max_bond_dim, len(p)])
            if ind < len(p):
                while ind > 1:
                    if np.abs(p[ind] - p[ind - 1]) < delta:
                        ind -= 1
                    else:
                        break
        else:
            candidates = [[0, 1, 2, 3], [2, 1, 0, 3], [0, 2, 1, 3]]
            candidate_index = 0
            ps = []
            inds = []
            for edges in candidates:
                psi_ = psi.copy()
                a = psi_[edges[0]]
                b = psi_[edges[1]]
                c = psi_[edges[2]]
                d = psi_[edges[3]]
                e = psi_[4]
                (u_, s_, v_, terr) = tn.split_node_full_svd(psi_, [e, a, b], [c, d])

                p_ = s_.tensor.data
                p_ = np.sort(p_)[::-1]
                p_ = p_[p_ > 0.0]
                ps.append(p_)
                # diagonal
                ind = np.min([max_bond_dim, len(p_)])
                if ind < len(p_):
                    while ind > 1:
                        if np.abs(p_[ind] - p_[ind - 1]) < delta:
                            ind -= 1
                        else:
                            break
                inds.append(ind)

            ees = [self.entanglement_entropy(probability=p) for p in ps]
            p_truncates = [p[:max_bond_dim] for p in ps]
            errors = [1.0 - np.real(np.sum(p_t**2)) for p_t in p_truncates]
            if opt_structure == 1:
                if temperature == 0.0:
                    candidate_index = np.argmin(ees)
                else:
                    weights = np.array(ees) / temperature
                    weights = np.exp(-weights)
                    weights = np.array(weights) / np.sum(weights)
                    candidate_index = np.random.choice(len(ees), p=weights)
            elif opt_structure == 2:
                candidate_index = np.argmin(errors)
                if np.isclose(errors[candidate_index], errors[0], atol=1e-14):
                    candidate_index = np.argmin(ees)

            if np.isclose(ees[candidate_index], ees[0], atol=epsilon):
                candidate_index = 0
            edge_order = candidates[candidate_index]
            p = ps[candidate_index]
            ind = inds[candidate_index]

        a = psi_last[edge_order[0]]
        b = psi_last[edge_order[1]]
        c = psi_last[edge_order[2]]
        d = psi_last[edge_order[3]]
        e = psi_last[4]
        (u, s, v, terr) = tn.split_node_full_svd(
            psi_last, [e, a, b], [c, d], max_singular_values=ind
        )
        u = u.reorder_edges([u[1], u[2], u[3], u[0]])
        v = v.reorder_edges([v[1], v[2], v[0]])

        u_tensor = u.tensor
        u_tensor.contiguous(inplace=True)
        s_tensor = s.tensor / np.linalg.norm(s.tensor.data)
        s_tensor.contiguous(inplace=True)
        v_tensor = v.tensor
        v_tensor.contiguous(inplace=True)
        p_truncate = s.tensor.data

        s = tn.Node(s_tensor, backend=self.backend)
        u = tn.Node(u_tensor, backend=self.backend)
        a, ss, b, terr = tn.split_node_full_svd(
            u,
            [
                u[0],
                u[1],
            ],
            [u[2], u[3]],
        )
        u_tensor = a.tensor
        u_tensor.contiguous(inplace=True)
        u_tensor = BlockSparseTensor(
            data=u_tensor.data,
            charges=[
                U1Charge(u_tensor.flat_charges[0].charges.flatten()),
                U1Charge(u_tensor.flat_charges[1].charges.flatten()),
                U1Charge(-u_tensor.flat_charges[2].charges.flatten()),
            ],
            flows=[True, True, False],
        )
        s[0] ^ b[1]
        s = tn.contractors.auto([s, b], output_edge_order=[b[0], s[1], b[2]])
        s_tensor = s.tensor
        s_tensor.contiguous(inplace=True)
        s_tensor = BlockSparseTensor(
            data=s_tensor.data,
            charges=[
                U1Charge(-s_tensor.flat_charges[0].charges.flatten()),
                U1Charge(s_tensor.flat_charges[1].charges.flatten()),
                U1Charge(s_tensor.flat_charges[2].charges.flatten()),
            ],
            flows=[True, True, False],
        )
        err = 1.0 - np.real(np.sum(p_truncate**2))
        return (
            u_tensor,
            s_tensor,
            v_tensor,
            p,
            err,
            edge_order,
        )

    def entanglement_entropy_at_physical_bond(self, psi, psi_edges):
        ee_at_physical_bond = {}
        for i, edge in enumerate(psi_edges):
            if edge in self.psi.physical_edges:
                ee_at_physical_bond[edge] = 0.0
        for edge_id in ee_at_physical_bond.keys():
            psi_ = psi.copy()
            matching_index = list({i for i, v in enumerate(psi_edges) if v == edge_id})[
                0
            ]
            non_matching_indices = list(
                {i for i in range(len(psi_edges)) if i != matching_index}
            )
            psi_dag = psi_.copy(conjugate=True)
            for i in non_matching_indices:
                psi_[i] ^ psi_dag[i]

            psi_[4] ^ psi_dag[4]

            rho = tn.contractors.auto(
                [psi_, psi_dag],
                output_edge_order=[psi_[matching_index], psi_dag[matching_index]],
            )
            (u, s, v, terr) = tn.split_node_full_svd(rho, [rho[0]], [rho[1]])
            p_ = s.tensor.data
            p_ = p_[p_ > 0.0]
            p_ = np.sqrt(p_)
            ee_at_physical_bond[edge_id] = self.entanglement_entropy(p_)
        return ee_at_physical_bond

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

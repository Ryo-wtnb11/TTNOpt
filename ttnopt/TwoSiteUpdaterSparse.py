import tensornetwork as tn
import itertools
import numpy as np
from collections import deque, defaultdict

from ttnopt.TwoSiteUpdater import TwoSiteUpdaterMixin

class TwoSiteUpdaterSparse(TwoSiteUpdaterMixin):
    def __init__(self, psi):
        self.psi = psi
        self.backend = "symmetric"
        self.flag = self.initial_flag()
        self.distance = self.initial_distance()

    def entanglement_entropy(self, probability):
        el = probability**2 / np.sum(probability**2)
        el = el[el > 0.0]
        ee = -np.sum(el * np.log2(el))
        return np.real(ee)

    def decompose_two_tensors(
        self,
        psi,
        max_bond_dim,
        max_truncation_err,
        opt_structure=False,
        operate_degeneracy=False,
        epsilon=1e-8,
    ):
        psi_last = psi.copy()
        if opt_structure is False:
            a = psi[0]
            b = psi[1]
            c = psi[2]
            d = psi[3]
            e = psi[4]
            (u, s, v, terr) = tn.split_node_full_svd(psi, [e, a, b], [c, d])

            p = np.diagonal(s.tensor.todense())
            ee = self.entanglement_entropy(p)

            edge_order = [0, 1, 2, 3]
        else:
            candidates = [[0, 1, 2, 3], [0, 2, 1, 3], [1, 2, 3, 0]]
            ee = 1e10
            for edges in candidates:
                psi_ = psi.copy()
                a = psi_[edges[0]]
                b = psi_[edges[1]]
                c = psi_[edges[2]]
                d = psi_[edges[3]]
                e = psi_[4]
                (u_, s_, v_, terr) = tn.split_node_full_svd(psi_, [e, a, b], [c, d])

                p_ = np.diagonal(s_.tensor.todense())
                ee_tmp = self.entanglement_entropy(p_)
                if ee_tmp < ee:
                    u = u_
                    s = s_
                    v = v_
                    ee = ee_tmp
                    edge_order = edges
                    p = p_
        # 縮退を解消
        ind = np.min([max_bond_dim, len(p)])
        indices = np.argsort(-p)
        if operate_degeneracy:
            if ind < len(p):
                while ind > 1:
                    if (
                        np.abs(p[indices[ind]] - p[indices[ind] - 1]) / p[indices[ind]]
                    ) * 100 < 0.1:
                        ind -= 1
                    else:
                        break
        a = psi_last[edge_order[0]]
        b = psi_last[edge_order[1]]
        c = psi_last[edge_order[2]]
        d = psi_last[edge_order[3]]
        e = psi_last[4]
        (u, s, v, terr) = tn.split_node_full_svd(
            psi_last, [e, a, b], [c, d], max_singular_values=ind
        )
        u = u.reorder_edges([u[1], u[2], u[3], u[0]])
        u_tensor = u.tensor
        s_data = s.tensor.data
        s_tensor = s.tensor / np.linalg.norm(s_data)

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
        s[0] ^ b[1]
        s = tn.contractors.auto([s, b], output_edge_order=[b[0], s[1], b[2]])
        s_tensor = s.tensor
        v = v.reorder_edges([v[1], v[2], v[0]])
        v_tensor = v.tensor
        return (
            u_tensor,
            s_tensor,
            v_tensor,
            edge_order,
            ee,
        )
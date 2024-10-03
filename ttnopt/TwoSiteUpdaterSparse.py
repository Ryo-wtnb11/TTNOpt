import tensornetwork as tn
import itertools
import numpy as np
from collections import deque, defaultdict


class TwoSiteUpdaterSparse:
    def __init__(self, psi):
        self.psi = psi
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

        s = tn.Node(s_tensor)
        u = tn.Node(u_tensor)
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

    def initial_flag(self):
        edge_ids = set(itertools.chain.from_iterable(self.psi.edges))
        flag = {ind: 0 if ind not in self.psi.physical_edges else 1 for ind in edge_ids}
        return flag

    def initial_distance(self):
        # 隣接リストを作成
        adjacency_list = defaultdict(list)
        for node in self.psi.edges:
            child1, child2, parent = node
            adjacency_list[child1].append(parent)
            adjacency_list[parent].append(child1)
            adjacency_list[child2].append(parent)
            adjacency_list[parent].append(child2)
            adjacency_list[child1].append(child2)
            adjacency_list[child2].append(child1)
        for key, val in adjacency_list.items():
            adjacency_list[key] = list(set(val))

        distances = {self.psi.top_edge_id: 0}
        queue = deque([self.psi.top_edge_id])

        while queue:
            current_edge = queue.popleft()
            current_distance = distances[current_edge]
            for neighbor in adjacency_list[current_edge]:
                if neighbor not in distances.keys():
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return distances

    def candidate_edge_ids(self):
        child_tensor_ids = [
            i
            for i, edge in enumerate(self.psi.edges)
            if edge[2] == self.psi.canonical_center_edge_id
        ]
        candidate_edge_ids = (
            self.psi.edges[child_tensor_ids[0]][:2]
            + self.psi.edges[child_tensor_ids[1]][:2]
        )
        candidate_edge_ids = [e for e in candidate_edge_ids if self.flag[e] == 0]
        return candidate_edge_ids

    def local_two_tensor(self):
        candidate_edge_ids = self.candidate_edge_ids()
        max_v = np.max([self.distance[e] for e in candidate_edge_ids])
        candidate_edge_ids = [
            e for e in candidate_edge_ids if self.distance[e] == max_v
        ]
        edge_id = candidate_edge_ids[0]  # select one

        for i, edge in enumerate(self.psi.edges):
            if edge_id == edge[2]:
                connected_tensor_id = i
            if edge_id in edge[:2]:
                selected_tensor_id = i

        child_tensor_ids = [
            i
            for i, edge in enumerate(self.psi.edges)
            if edge[2] == self.psi.canonical_center_edge_id
        ]

        for child_tensor_id in child_tensor_ids:
            if child_tensor_id != selected_tensor_id:
                not_selected_tensor_id = child_tensor_id

        return (
            edge_id,
            selected_tensor_id,
            connected_tensor_id,
            not_selected_tensor_id,
        )

    def set_flag(self, not_selected_tensor_id):
        if (
            self.flag[self.psi.edges[not_selected_tensor_id][0]] == 1
            and self.flag[self.psi.edges[not_selected_tensor_id][1]] == 1
        ):
            if self.psi.canonical_center_edge_id != self.psi.top_edge_id:
                self.flag[self.psi.edges[not_selected_tensor_id][2]] = 1
        return

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
            out_selected_inds + [canonical_center_ind],
        )
        self.psi.edges[selected_tensor_id] = [
            self.psi.edges[selected_tensor_id][i] for i in out_selected_inds
        ] + [edge_id]
        for i, e in enumerate(self.psi.edges[selected_tensor_id]):
            self.psi.edge_dims[e] = self.psi.tensors[selected_tensor_id].shape[i]
        return

    def contract_central_tensors(self):
        central_tensor_ids = self.psi.central_tensor_ids()

        psi1 = tn.Node(self.psi.tensors[central_tensor_ids[0]])
        psi2 = tn.Node(self.psi.tensors[central_tensor_ids[1]])
        gauge = tn.Node(self.psi.gauge_tensor)

        psi1[2] ^ gauge[0]
        gauge[1] ^ psi2[2]

        psi = tn.contractors.auto(
            [psi1, gauge, psi2],
            output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1]],
        )
        return psi

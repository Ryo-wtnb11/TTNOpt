import numpy as np
import tensornetwork as tn
import itertools
from collections import deque, defaultdict


class TwoSiteUpdaterMixin:
    def initial_flag(self):
        edge_ids = set(itertools.chain.from_iterable(self.psi.edges))
        flag = {ind: 0 if ind not in self.psi.physical_edges else 1 for ind in edge_ids}
        return flag

    def entanglement_entropy(self, probability):
        el = probability**2 / np.sum(probability**2)
        el = el[el > 0.0]
        ee = -np.sum(el * np.log2(el))
        return np.real(ee)

    def initial_distance(self):
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
                self.flag[self.psi.canonical_center_edge_id] = 1
        return

    def contract_central_tensors(self):
        central_tensor_ids = self.psi.central_tensor_ids()

        psi1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
        psi2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)

        psi1[2] ^ gauge[0]
        gauge[1] ^ psi2[2]

        psi = tn.contractors.auto(
            [psi1, gauge, psi2],
            output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1]],
        )
        return psi


class TwoSiteUpdater(TwoSiteUpdaterMixin):
    def __init__(self, psi):
        self.psi = psi
        self.backend = "numpy"
        self.flag = self.initial_flag()
        self.distance = self.initial_distance()

    def decompose_two_tensors(
        self,
        psi,
        max_bond_dim,
        opt_structure=0,
        operate_degeneracy=False,
        epsilon=1e-8,
        delta=0.1,
        truncated_singularvalues=0.0,
        beta=0.0,
    ):
        edge_order = [0, 1, 2, 3]
        if opt_structure == 0:
            a = psi[0]
            b = psi[1]
            c = psi[2]
            d = psi[3]
            (u, s, v, _) = tn.split_node_full_svd(psi, [a, b], [c, d])
            p = np.diagonal(s.get_tensor())
        else:
            candidates = [[0, 1, 2, 3], [0, 2, 1, 3], [1, 2, 3, 0]]
            candidate_index = 0
            ps = []
            for edges in candidates:
                psi_ = psi.copy()
                a = psi_[edges[0]]
                b = psi_[edges[1]]
                c = psi_[edges[2]]
                d = psi_[edges[3]]
                (u_, s_, v_, _) = tn.split_node_full_svd(
                    psi_,
                    [a, b],
                    [c, d],
                )
                p_ = np.diagonal(s_.get_tensor())
                ps.append(p_)
            ees = [self.entanglement_entropy(probability=p) for p in ps]
            p_truncates = [p[:max_bond_dim] for p in ps]
            errors = [1.0 - np.real(np.sum(p_t**2)) for p_t in p_truncates]
            if opt_structure == 1:
                if beta == 0.0:
                    candidate_index = np.argmin(ees)
                else:
                    weights = np.array(ees) * beta
                    weights = np.array(weights) / np.sum(weights)
                    candidate_index = np.random.choice(len(ees), p=weights)
            elif opt_structure == 2:
                candidate_index = np.argmin(errors)

            if np.isclose(ees[candidate_index], ees[0], atol=epsilon):
                candidate_index = 0
            edge_order = candidates[candidate_index]
            p = ps[candidate_index]
        # Degeneracy
        ind = np.min([max_bond_dim, len(p)])
        if truncated_singularvalues > 0.0:
            p_max = p[0]
            for i in range(1, ind):
                if p[i] / p_max < truncated_singularvalues:
                    ind = i
                    break
        if operate_degeneracy:
            if ind < len(p):
                while ind > 1:
                    if (np.abs(p[ind] - p[ind - 1]) / (p[ind - 1] + 1e-10)) < delta:
                        ind -= 1
                    else:
                        break
        a = psi[edge_order[0]]
        b = psi[edge_order[1]]
        c = psi[edge_order[2]]
        d = psi[edge_order[3]]
        (u, s, v, terr) = tn.split_node_full_svd(
            psi, [a, b], [c, d], max_singular_values=ind
        )
        v = v.reorder_edges([v[1], v[2], v[0]])
        u = u.get_tensor()
        v = v.get_tensor()
        s = s.get_tensor()
        p_truncate = np.diagonal(s)
        err = 1.0 - np.real(np.sum(p_truncate**2))
        s = s / np.linalg.norm(s)
        return u, s, v, p, err, edge_order

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
            rho = tn.contractors.auto(
                [psi_, psi_dag],
                output_edge_order=[psi_[matching_index], psi_dag[matching_index]],
            )
            (u, s, v, terr) = tn.split_node_full_svd(rho, [rho[0]], [rho[1]])
            p_ = np.diagonal(s.get_tensor())
            p_ = np.sqrt(p_)
            ee_at_physical_bond[edge_id] = self.entanglement_entropy(probability=p_)
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
            out_selected_inds + [canonical_center_ind],
        )
        self.psi.edges[selected_tensor_id] = [
            self.psi.edges[selected_tensor_id][i] for i in out_selected_inds
        ] + [edge_id]
        for i, e in enumerate(self.psi.edges[selected_tensor_id]):
            self.psi.edge_dims[e] = self.psi.tensors[selected_tensor_id].shape[i]
        return

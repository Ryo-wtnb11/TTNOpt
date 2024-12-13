from typing import List, Optional

import numpy as np
import tensornetwork as tn
import networkx as nx
import matplotlib.pyplot as plt


class TreeTensorNetwork:
    """A class for Tree Tensor Network (TTN)."""

    def __init__(
        self,
        edges: List[List[int]],
        top_edge_id: int,
        tensors: Optional[List[np.ndarray]] = None,
        gauge_tensor: Optional[np.ndarray] = None,
        norm: Optional[float] = None,
    ):
        """Initialize a TreeTensorNetwork object.

        Args:
            edges (List[List[int]]): Edge id list for each tensor in the order [left, right, top].
            top_edge_id (int): edge id that connects to the top tensor.
            tensors (Optional[List[np.ndarray]]): tensors for each node.
            gauge_tensor (Optional[np.ndarray]): gauge tensor at top_edge_bond.
                This parameter is not required for some algorithms (Ground State Search, etc.)
        """
        self.edges = edges

        self.tensor_num = len(edges)
        self.top_edge_id = top_edge_id
        self.canonical_center_edge_id = top_edge_id
        self.physical_edges = self._physical_edges()
        self.size = len(self.physical_edges)

        self.tensors = None
        self.gauge_tensor = None
        self.norm = 1.0
        self.edge_dims = {}
        if tensors is not None:
            self.tensors = tensors
            self.edge_dims = self._edge_dims()
        if gauge_tensor is not None:
            self.gauge_tensor = gauge_tensor
        if norm is not None:
            self.norm = norm

    @classmethod
    def mps(
        cls,
        size: int,
        target: Optional[np.ndarray] = None,
        max_bond_dimension: Optional[int] = None,
    ):
        """Initialize an State object with matrix product structure.
        Args:
            size : The size of system.
        """

        edges = []
        upper_edge_id = size
        edges.append([0, 1, upper_edge_id])
        for i in range(2, (size - 2) // 2 + 1):
            edges.append([upper_edge_id, i, upper_edge_id + 1])
            upper_edge_id += 1

        center_edge_id = upper_edge_id
        upper_edge_id = upper_edge_id + 1

        tmp_edges = []
        tmp_edges.append([size - 2, size - 1, upper_edge_id])

        for i in reversed(range((size - 2) // 2 + 2, size - 2)):
            tmp_edges.append([i, upper_edge_id, upper_edge_id + 1])
            upper_edge_id += 1

        tmp_edges.append([(size - 2) // 2 + 1, upper_edge_id, center_edge_id])
        edges += reversed(tmp_edges)

        if target is not None and max_bond_dimension is not None:
            norm = np.linalg.norm(target)
            normed_target = target / norm

            if len(target.shape) != size:
                raise ValueError(
                    f"The shape of the tensor is not correct. tensor.shape={target.shape}, size={size}"
                )
            target_node = tn.Node(normed_target)
            U, S, V, _ = tn.split_node_full_svd(
                target_node,
                target_node[:2],
                target_node[2:],
                max_singular_values=max_bond_dimension,
            )
            tensors = [U.tensor]
            V = tn.contractors.auto([V, S], [S[0]] + V[1:])
            for i in range(2, (size - 2) // 2 + 1):
                U, S, V, _ = tn.split_node_full_svd(
                    V, [V[0], V[1]], V[2:], max_singular_values=max_bond_dimension
                )
                tensors.append(U.tensor)
                V = tn.contractors.auto([V, S], [S[0]] + V[1:])

            U, S, V, _ = tn.split_node_full_svd(
                V, V[:-2], [V[-2], V[-1]], max_singular_values=max_bond_dimension
            )
            tmp_tensors = [V.reorder_edges([V[1], V[2], V[0]]).tensor]
            U = tn.contractors.auto([U, S], U[:-1] + [S[1]])
            for i in reversed(range((size - 2) // 2 + 2, size - 2)):
                U, S, V, _ = tn.split_node_full_svd(
                    U, U[:-2], [U[-2], U[-1]], max_singular_values=max_bond_dimension
                )
                U = tn.contractors.auto([U, S], U[:-1] + [S[1]])
                tmp_tensors.append(V.reorder_edges([V[1], V[2], V[0]]).tensor)
            U, S, V, _ = tn.split_node_full_svd(
                U, U[:-2], [U[-2], U[-1]], max_singular_values=max_bond_dimension
            )
            tmp_tensors.append(V.reorder_edges([V[1], V[2], V[0]]).tensor)
            tensors += reversed(tmp_tensors)
            return cls(edges, center_edge_id, tensors, S.tensor, norm.item())

        return cls(edges, center_edge_id)

    @classmethod
    def tree(cls, size: int):
        """Initialize an object with binary tree structure.

        Args:
            size : The size of system.
        """
        edges = []

        layer_index = 0
        num_layer = int(np.log2(size)) - 1
        for layer in range(num_layer):
            tensor_num = int(2 ** (np.log2(size) - 1 - layer))
            nn = int(2 ** (np.log2(size) - layer))
            for i in range(tensor_num):
                if layer != num_layer - 1:
                    edge_id = [
                        layer_index + i * 2,
                        layer_index + i * 2 + 1,
                        layer_index + nn + i,
                    ]

                else:
                    edge_id = [
                        layer_index + i * 2,
                        layer_index + i * 2 + 1,
                        layer_index + nn,
                    ]

                edges.append(edge_id)

            layer_index += nn

        center_edge_id = layer_index

        return cls(edges, center_edge_id)

    def visualize(self):
        """Visualize the TreeTensorNetwork."""
        g = nx.DiGraph()
        logs = self._get_parent_child_pairs()

        small_black_nodes = []
        large_red_nodes = []
        default_nodes = []
        for log in logs:  # log=[self node, parent node]
            g.add_node(log[0])
            g.add_edge(log[1], log[0])
            # ノードの種類に応じて分類
            if "bare" in log[0]:
                small_black_nodes.append(log[0])
            else:
                default_nodes.append(log[0])
            if log[1] == "top":
                large_red_nodes.append(log[1])
        pos = nx.nx_pydot.graphviz_layout(g, prog="twopi")
        # matplotlib settings
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        # isometry node
        nx.draw(
            g,
            ax=ax,
            pos=pos,
            nodelist=default_nodes,
            with_labels=False,
            arrows=False,
            node_size=20,
            node_shape="o",
            width=0.5,
            node_color="blue",
        )

        # bare node
        nx.draw_networkx_nodes(
            g,
            ax=ax,
            pos=pos,
            nodelist=small_black_nodes,
            node_size=8,
            node_shape="o",
            node_color="black",
        )

        # top node
        # nx.draw_networkx_nodes(
        #     g,
        #     ax=ax,
        #     pos=pos,
        #     nodelist=large_red_nodes,
        #     node_size=25,
        #     node_shape="o",
        #     node_color="red",
        # )
        return plt

    def central_tensor_ids(self):
        central_tensor_ids = [
            tensor_id
            for tensor_id, edges in enumerate(self.edges)
            if edges[2] == self.canonical_center_edge_id
        ]
        return central_tensor_ids

    def _physical_edges(self):
        count_dict = {}
        for edge in self.edges:
            for i in edge:
                if i not in count_dict.keys():
                    count_dict[i] = 0
                count_dict[i] += 1
        physical_edges = [k for k, v in count_dict.items() if v == 1]
        return physical_edges

    def _edge_dims(self):
        edge_dims = {}
        for i, t in enumerate(self.tensors):
            for j, e in enumerate(self.edges[i]):

                edge_dims[e] = t.shape[j]
        return edge_dims

    def _get_parent_child_pairs(self):
        parent_child_pairs = []
        structure_edges = self.edges

        for i, edges in enumerate(structure_edges):
            child1_edge, child2_edge, parent_edge = edges
            if child1_edge in self.physical_edges:
                parent_child_pairs.append([f"bare{child1_edge}", str(i)])
            if child2_edge in self.physical_edges:
                parent_child_pairs.append([f"bare{child2_edge}", str(i)])
            for j, _edges in enumerate(structure_edges):
                if j != i and parent_edge in _edges[:2]:
                    parent_child_pairs.append([str(i), str(j)])
                if parent_edge == self.canonical_center_edge_id:
                    parent_child_pairs.append([str(i), "top"])
        return parent_child_pairs

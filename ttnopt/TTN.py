import networkx as nx
import matplotlib.pyplot as plt


class TreeTensorNetwork:

    def __init__(self, edges, top_edge_id, tensors=None):
        """_summary_
        Args:
            edges List[[int, int, int]]: edge_id_list of each tensor [left, right, top]
            top_edge_id : edge_id which connects top_tensor
            tensors List[np.array]: tensors is not necessary initially because we can generate it in some algorithms (DMRG, etc.)
        """
        self.edges = edges
        self.tensor_num = len(edges)
        self.top_edge_id = top_edge_id
        self.canonical_center_edge_id = top_edge_id
        self.physical_edges = self._physical_edges()
        self.size = len(self.physical_edges)

        self.tensors = None
        self.gauge_tensor = None
        self.edge_dims = {}
        if tensors is not None:
            self.tensors = tensors
            self.edge_dims = self._edge_dims()

    def visualize(self):
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
        pos = nx.nx_agraph.graphviz_layout(g, prog="twopi")
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
        nx.draw_networkx_nodes(
            g,
            ax=ax,
            pos=pos,
            nodelist=large_red_nodes,
            node_size=25,
            node_shape="o",
            node_color="red",
        )
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

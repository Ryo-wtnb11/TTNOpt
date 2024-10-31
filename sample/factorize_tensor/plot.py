import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

# Sample data with N=24
N = 24
data_tree = pd.read_csv('out/tree.csv').values
data_mps = pd.read_csv('out/mps.csv').values

def visualize():
    """Visualize two TreeTensorNetworks with entanglement color-coded on edges and bare nodes colored by index"""

    # Create figure with two subplots, with more space between them
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1.0, 1.0], 'wspace': 0.1}, dpi=100, tight_layout=True)

    def create_graph(data, ax, title):
        """Create a graph for a given dataset and axis"""
        g = nx.DiGraph()

        # Prepare node and edge data
        bare_nodes = list(range(N))
        default_nodes = set()  # We'll collect the non-bare nodes here
        edge_colors = []

        # Add edges to the graph and distinguish between bare and non-bare nodes
        for node1, node2, fidelity, entanglement in data:
            g.add_edge(node1, node2, weight=entanglement)
            # Distinguish between bare and non-bare nodes
            if node1 < N:
                g.nodes[node1]['type'] = 'bare'
            else:
                default_nodes.add(node1)

            if node2 < N:
                g.nodes[node2]['type'] = 'bare'
            else:
                default_nodes.add(node2)

            # Track entanglement values for coloring edges
            edge_colors.append(entanglement)

        # Define positions for nodes using a layout
        pos = nx.nx_pydot.graphviz_layout(g, prog="twopi")

        # Adjust the positions of bare nodes to make edges shorter, but not too close
        for node in bare_nodes:
            if node in pos:
                connected_node = next(g.neighbors(node))  # Get the node connected to this bare node
                # Move the bare node closer to its connected node by adjusting the position, but keep some distance
                pos[node] = (
                    pos[node][0] * 0.7 + pos[connected_node][0] * 0.3,  # A little closer but not too much
                    pos[node][1] * 0.7 + pos[connected_node][1] * 0.3
                )

        # Normalize the edge color range
        edge_cmap = plt.cm.viridis
        norm = Normalize(vmin=min(edge_colors), vmax=max(edge_colors))

        # Draw edges with entanglement values as color (without arrows)
        edges = nx.draw_networkx_edges(
            g, pos, edge_color=edge_colors, edge_cmap=edge_cmap, width=2, edge_vmin=min(edge_colors), edge_vmax=max(edge_colors), arrows=False, ax=ax
        )

        # Use the 'plasma' colormap for bare nodes
        node_norm = Normalize(vmin=min(bare_nodes), vmax=max(bare_nodes))
        bare_node_colors = [plt.cm.cool(node_norm(n)) for n in bare_nodes]

        # Draw bare nodes (colored by index)
        bare_nodes_list = [n for n, d in g.nodes(data=True) if d.get('type') == 'bare']
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=bare_nodes_list,
            node_size=40,
            node_color=bare_node_colors
        )

        # Draw default nodes (black)
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=list(default_nodes),
            node_size=10,
            node_color="black"
        )

        # Set title
        ax.set_title(title)
        ax.axis('off')

        return norm, node_norm

    # Create graphs
    norm_tree, node_norm_tree = create_graph(data_tree, ax1, "TTN")
    norm_mps, node_norm_mps = create_graph(data_mps, ax2, "MPS")

    # Add colorbar for bare node index values using the 'plasma' colormap (on the left)
    sm_bare = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=node_norm_tree)
    sm_bare.set_array([])

    # Position the left colorbar (bare node index) outside the figure
    cbar_bare = fig.add_axes([0.05, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(sm_bare, cax=cbar_bare)

    # Add colorbar for entanglement values on edges (on the right)
    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_tree)
    sm_edges.set_array([])

    # Position the right colorbar (entanglement) outside the figure
    cbar_edges = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(sm_edges, cax=cbar_edges, label='Entanglement')

    # Save the plot
    plt.tight_layout()
    plt.savefig('out/visualization_side_by_side.pdf')

# Call the function to visualize
visualize()

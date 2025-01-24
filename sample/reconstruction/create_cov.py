import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def create_cov(D, rho, distance_matrix):
    cov = np.eye(D, dtype=np.float64)
    for i in range(D):
        for j in range(i+1, D):
            cov[i, j] = cov[j, i] = rho ** distance_matrix[i][j]
    return cov

def plot_tree(G, save_dir):
    # adjust node positions
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    leaf_nodes = [node for node in G.nodes if node < 0]
    default_nodes = [node for node in G.nodes if node >= 0]
    node_labels = {}
    for node in G.nodes:
        if node < 0:
            node_labels[node] = f"{abs(node)-1}"

    # draw the graph
    # # draw graph
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")
    nx.draw_networkx_nodes(G, pos, nodelist=default_nodes, node_size=10, node_color="black")
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, font_family="Times New Roman")
    plt.savefig(f"{save_dir}/cov_tree.pdf")
    plt.close()


def output_cov_as_latex(distance_matrix, save_dir):
    # output as latex
    # each element is represented as \rho^{distance}
    with open(f"{save_dir}/cov.md", "w") as f:
        f.write("# Covariance matrix\n")
        f.write("$$")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"\Sigma = \begin{pmatrix}" + "\n")
        for i in range(D):
            for j in range(D):
                f.write(r"\rho^{" + str(distance_matrix[i][j]) + "}")
                if j < D-1:
                    f.write(" & ")
                else:
                    f.write(r" \\")
            f.write("\n")
        f.write(r"\end{pmatrix}" + "\n")
        f.write(r"\end{equation}" + "\n")
        f.write("$$")


def create_tree_distance_matrix(D, seed=0, save_dir=None):
    np.random.seed(seed)
    # generate random binary tree with D nodes
    tree = [[-1, -1]]
    free_edges = [[0, 0], [0, 1]]

    while len(tree) < D-1:
        edge = random.choice(free_edges)
        free_edges.remove(edge)
        node, child = edge
        k = len(tree)
        tree[node][child] = k
        tree.append([-1, -1])
        free_edges.append([k, 0])
        free_edges.append([k, 1])

    # assign random values to the leaf nodes
    leaf_index = np.random.permutation([-(i+1) for i in range(D)])
    leaf_idx = 0
    for i in range(len(tree)):
        for j in range(2):
            if tree[i][j] == -1:
                tree[i][j] = leaf_index[leaf_idx]
                leaf_idx += 1

    # convert variable tree to networkx graph
    G = nx.Graph()
    G.add_edge(tree[0][0], tree[0][1])
    for i in range(1, len(tree)):
        for j in range(2):
            G.add_edge(i, tree[i][j])

    if save_dir is not None:
        plot_tree(G, save_dir)

    # calculate the distance matrix
    distance_matrix = np.zeros((D, D)).astype(int)
    for i in range(D):
        for j in range(D):
            distance_matrix[i, j] = int(nx.shortest_path_length(G, -(i+1), -(j+1)))
    
    return distance_matrix


def create_tree_cov(D, rho, seed=0, save_dir=None):
    distance_matrix = create_tree_distance_matrix(D, seed, save_dir)
    if save_dir is not None:
        output_cov_as_latex(distance_matrix, save_dir)

    return create_cov(D, rho, distance_matrix)

D = 16
seed = 0
rho = 0.2

cov = create_tree_cov(D, rho, save_dir="normal_distribution_data/cov/")

np.save("normal_distribution_data/cov/cov.npy", cov)
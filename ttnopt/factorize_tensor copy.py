import os
import pandas as pd
import numpy as np
from dotmap import DotMap
from pathlib import Path
from ttnopt.src import TreeTensorNetwork, FactorizeTensor


def load_file(file_path, error_message):
    if not os.path.exists(file_path):
        raise FileNotFoundError(error_message)
    return np.load(file_path)


def prepare_output_directory(output_dir):
    path = Path(output_dir)
    os.makedirs(path, exist_ok=True)
    return path


def initialize_tensor_network(quantum_state, numerics):
    N = len(quantum_state.shape)
    init_bond_dim = int(numerics.initial_bond_dimension or 4)
    psi = TreeTensorNetwork.mps(
        N, target=quantum_state, max_bond_dimension=init_bond_dim
    )
    return psi, N


def run_factorization(
    ft, quantum_state, numerics, opt_structure=False, opt_fidelity=False
):
    max_num_sweep = int(numerics.max_num_sweep or 1)
    ft.run(
        quantum_state,
        opt_fidelity=opt_fidelity,
        opt_structure=opt_structure,
        max_num_sweep=max_num_sweep,
    )


def process_nodes(ft, psi, N):
    nodes_list = {
        edge_id: [
            node_id + N for node_id, edges in enumerate(psi.edges) if edge_id in edges
        ]
        for edge_id in ft.fidelity.keys()
    }
    for edge_id in psi.physical_edges:
        nodes_list.setdefault(edge_id, []).insert(0, edge_id)
        ft.fidelity[edge_id] = 0.0
        ft.error[edge_id] = 0.0
    return nodes_list


def save_results(nodes_list, ft, psi, path, state_norm, run_id=None):
    all_keys = set(nodes_list.keys())
    df = pd.DataFrame([nodes_list[k] for k in all_keys], columns=["node1", "node2"])
    df["entanglement"] = [ft.entanglement[k] for k in all_keys]
    df["fidelity"] = [ft.fidelity[k] for k in all_keys]
    df["error"] = [ft.error[k] for k in all_keys]

    run_path = path / (f"run{run_id}" if run_id else "")
    os.makedirs(run_path, exist_ok=True)
    df.to_csv(run_path / "basic.csv", header=True, index=None)

    file_psi = run_path / "tensors"
    os.makedirs(file_psi, exist_ok=True)
    for i, iso in enumerate(ft.psi.tensors):
        np.save(file_psi / f"isometry{i}.npy", iso)
    np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
    np.save(file_psi / "norm.npy", state_norm)
    np.savetxt(file_psi / "edges.dat", ft.psi.edges, fmt="%d", delimiter=",")


def factorize_tensor(config, path):
    quantum_state = load_file(config.target, f"{config.target} does not exist.")
    state_norm = np.linalg.norm(quantum_state)
    quantum_state /= state_norm

    psi, N = initialize_tensor_network(quantum_state, config.numerics)
    ft = FactorizeTensor(
        psi,
        max_bond_dim=int(config.numerics.initial_bond_dimension or 4),
        truncation_error=float(config.numerics.truncation_error or 1e-11),
    )

    run_factorization(ft, quantum_state, config.numerics, opt_structure=True)
    nodes_list = process_nodes(ft, psi, N)
    save_results(nodes_list, ft, psi, path, state_norm)

    if config.numerics.fidelity.opt_structure:
        for i, (max_bond_dim, max_num_sweep) in enumerate(
            zip(
                config.numerics.fidelity.max_bond_dimensions,
                config.numerics.fidelity.max_num_sweeps,
            )
        ):
            ft.max_bond_dim = max_bond_dim
            run_factorization(ft, quantum_state, config.numerics, opt_fidelity=True)
            nodes_list = process_nodes(ft, psi, N)
            save_results(nodes_list, ft, psi, path, state_norm, run_id=i + 1)

    return 0

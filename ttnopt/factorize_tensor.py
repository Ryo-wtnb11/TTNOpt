from ttnopt.src import TreeTensorNetwork
from ttnopt.src import FactorizeTensor

import os
import pandas as pd
import yaml
import argparse
from dotmap import DotMap
import numpy as np
from pathlib import Path


def factorize_tensor():
    parser = argparse.ArgumentParser(description="Factorize tensor")
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    quantum_state = np.load(config.target)
    N = len(quantum_state.shape)

    numerics = config.numerics

    psi = TreeTensorNetwork.mps(
        N,
        target=quantum_state,
        max_bond_dimension=numerics.initial_bond_dimension,
    )

    ft = FactorizeTensor(
        psi,
        quantum_state,
        max_bond_dim=numerics.initial_bond_dimension,
    )

    path = Path(config.output.dir)

    if numerics.opt_structure:
        ft.run(
            opt_fidelity=False,
            opt_structure=True,
            max_num_sweep=numerics.max_num_sweep,
        )

        nodes_list = {}
        for edge_id in ft.fidelity.keys():
            tmp = []
            for node_id, edges in enumerate(psi.edges):
                node_id += N
                if edge_id in edges:
                    tmp.append(node_id)
            nodes_list[edge_id] = tmp

        for edge_id in psi.physical_edges:
            tmp = []
            tmp.append(edge_id)
            for node_id, edges in enumerate(psi.edges):
                node_id += N
                if edge_id in edges:
                    tmp.append(node_id)
            nodes_list[edge_id] = tmp
            ft.fidelity[edge_id] = 0.0
            ft.error[edge_id] = 0.0

        all_keys = set(nodes_list.keys())
        df = pd.DataFrame(
            [nodes_list[k] for k in all_keys], columns=["node1", "node2"], index=None
        )
        df["entanglement"] = [ft.entanglement[k] for k in all_keys]
        df["fidelity"] = [ft.fidelity[k] for k in all_keys]
        df["error"] = [ft.error[k] for k in all_keys]
        df.to_csv(path / "basic.csv", header=True, index=None)

    file_psi = path / "tensors"
    os.makedirs(file_psi, exist_ok=True)
    for i, iso in enumerate(ft.psi.tensors):
        np.save(file_psi / f"isometry{i}.npy", iso)

    opt_fidelity = (
        True if not isinstance(numerics.fidelity.opt_structure, DotMap) else False
    )

    if opt_fidelity:
        opt_structure = True if numerics.fidelity.opt_structure == 1 else False
        for i, (max_bond_dim, max_num_sweep) in enumerate(
            zip(
                numerics.fidelity.max_bond_dimensions,
                numerics.fidelity.max_num_sweeps,
            )
        ):
            ft.max_bond_dim = max_bond_dim
            ft.run(
                opt_fidelity=True,
                opt_structure=opt_structure,
                max_num_sweep=max_num_sweep,
            )
            opt_structure = 0

            nodes_list = {}
            for edge_id in ft.fidelity.keys():
                tmp = []
                for node_id, edges in enumerate(psi.edges):
                    node_id += N
                    if edge_id in edges:
                        tmp.append(node_id)
                nodes_list[edge_id] = tmp

            for edge_id in psi.physical_edges:
                tmp = []
                tmp.append(edge_id)
                for node_id, edges in enumerate(psi.edges):
                    node_id += N
                    if edge_id in edges:
                        tmp.append(node_id)
                nodes_list[edge_id] = tmp
                ft.fidelity[edge_id] = 0.0
                ft.error[edge_id] = 0.0

            all_keys = set(nodes_list.keys())
            df = pd.DataFrame(
                [nodes_list[k] for k in all_keys],
                columns=["node1", "node2"],
                index=None,
            )
            df["entanglement"] = [ft.entanglement[k] for k in all_keys]
            df["fidelity"] = [ft.fidelity[k] for k in all_keys]
            df["error"] = [ft.error[k] for k in all_keys]

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)
            df.to_csv(path_ / "basic.csv", header=True, index=None)

            file_psi = Path(path_) / "tensors"
            os.makedirs(file_psi, exist_ok=True)
            for i, iso in enumerate(ft.psi.tensors):
                np.save(file_psi / f"isometry{i}.npy", iso)

    return 0

from typing import List
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
    numerics = config.numerics

    path = Path(config.output.dir)
    os.makedirs(path, exist_ok=True)

    if not isinstance(config.target, DotMap):
        if not os.path.exists(str(config.target)):
            raise FileNotFoundError(f"{str(config.target)} does not exist.")
        else:
            quantum_state = np.load(config.target)
            state_norm = np.linalg.norm(quantum_state)
            quantum_state = quantum_state / state_norm
            N = len(quantum_state.shape)
            psi = TreeTensorNetwork.mps(
                N,
                target=quantum_state,
                max_bond_dimension=numerics.initial_bond_dimension,
            )

            init_bond_dim = 4
            if not isinstance(numerics.initial_bond_dimension, DotMap):
                init_bond_dim = int(numerics.initial_bond_dimension)

            truncated_singularvalues = 0.0
            if not isinstance(numerics.truncated_singularvalues, DotMap):
                truncated_singularvalues = float(numerics.truncated_singularvalues)

            ft = FactorizeTensor(
                psi,
                max_bond_dim=init_bond_dim,
            )

            if not isinstance(numerics.opt_structure.type, DotMap):
                beta = (
                    numerics.opt_structure.beta
                    if isinstance(numerics.opt_structure.beta, List)
                    else [0.0, 0.0]
                )
                seed = (
                    numerics.opt_structure.seed
                    if isinstance(numerics.opt_structure.beta, int)
                    else 0
                )
                np.random.seed(seed)
                if numerics.opt_structure.type:
                    max_num_sweep = 1
                    if not isinstance(numerics.max_num_sweep, DotMap):
                        max_num_sweep = numerics.max_num_sweep
                    ft.run(
                        target=quantum_state,
                        opt_fidelity=False,
                        opt_structure=numerics.opt_structure.type,
                        beta=beta,
                        max_num_sweep=max_num_sweep,
                        truncated_singularvalues=truncated_singularvalues,
                    )
                else:
                    ft.run(
                        target=quantum_state,
                        opt_fidelity=False,
                        opt_structure=0,
                        max_num_sweep=1,
                        truncated_singularvalues=truncated_singularvalues,
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
                    [nodes_list[k] for k in all_keys],
                    columns=["node1", "node2"],
                    index=None,
                )
                df["entanglement"] = [ft.entanglement[k] for k in all_keys]
                df["fidelity"] = [ft.fidelity[k] for k in all_keys]
                df["error"] = [ft.error[k] for k in all_keys]
                df.to_csv(path / "basic.csv", header=True, index=None)

            file_psi = path / "tensors"
            os.makedirs(file_psi, exist_ok=True)
            for i, iso in enumerate(ft.psi.tensors):
                np.save(file_psi / f"isometry{i}.npy", iso)
            np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
            np.save(file_psi / "norm.npy", state_norm)
            np.savetxt(file_psi / "edges.dat", ft.psi.edges, fmt="%d", delimiter=",")

            opt_fidelity = (
                True
                if not isinstance(numerics.fidelity.opt_structure.type, DotMap)
                else False
            )

            if opt_fidelity:
                opt_structure = numerics.fidelity.opt_structure.type
                beta = (
                    numerics.fidelity.opt_structure.beta
                    if isinstance(numerics.fidelity.opt_structure.beta, List)
                    else [0.0, 0.0]
                )
                seed = (
                    numerics.opt_structure.seed
                    if isinstance(numerics.fidelity.opt_structure.beta, int)
                    else 0
                )
                np.random.seed(seed)
                for i, (max_bond_dim, max_num_sweep) in enumerate(
                    zip(
                        numerics.fidelity.max_bond_dimensions,
                        numerics.fidelity.max_num_sweeps,
                    )
                ):
                    ft.max_bond_dim = max_bond_dim
                    ft.run(
                        target=quantum_state,
                        opt_fidelity=True,
                        opt_structure=opt_structure,
                        beta=beta,
                        max_num_sweep=max_num_sweep,
                        truncated_singularvalues=truncated_singularvalues,
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
                        ft.bond_dim[edge_id] = quantum_state.shape[edge_id]

                    all_keys = set(nodes_list.keys())
                    df = pd.DataFrame(
                        [nodes_list[k] for k in all_keys],
                        columns=["node1", "node2"],
                        index=None,
                    )
                    df["entanglement"] = [ft.entanglement[k] for k in all_keys]
                    df["fidelity"] = [ft.fidelity[k] for k in all_keys]
                    df["error"] = [ft.error[k] for k in all_keys]
                    df["bond"] = [ft.bond_dim[k] for k in all_keys]

                    path_ = path / f"run{i + 1}"
                    os.makedirs(path_, exist_ok=True)
                    df.to_csv(path_ / "basic.csv", header=True, index=None)

                    file_psi = Path(path_) / "tensors"
                    os.makedirs(file_psi, exist_ok=True)
                    for i, iso in enumerate(ft.psi.tensors):
                        np.save(file_psi / f"isometry{i}.npy", iso)
                    np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
                    np.save(file_psi / "norm.npy", state_norm)
                    np.savetxt(
                        file_psi / "edges.dat",
                        ft.psi.edges,
                        fmt="%d",
                        delimiter=",",
                    )

    elif not isinstance(config.target.dir, DotMap):
        if not os.path.exists(str(config.target.dir)):
            raise FileNotFoundError(f"{str(config.target.dir)} does not exist.")
        if not isinstance(config.target.tensors_name, str):
            raise ValueError("Please specify the name of tensor files.")
        input_path = Path(config.target.dir)
        isometries = list(input_path.glob(f"{config.target.tensors_name}*.npy"))
        if isometries == []:
            raise FileNotFoundError(
                f"No files found files named as {str(config.target.tensors_name)} in {str(config.target.dir)}"
            )
        isometries.sort(
            key=lambda x: int(x.stem.split(f"{config.target.tensors_name}")[-1])
        )
        isometries = [np.load(iso) for iso in isometries]
        singular_values = np.load(input_path / "singular_values.npy")
        state_norm = np.load(input_path / "norm.npy")
        if not isinstance(config.target.graph_file, str):
            raise ValueError("Please specify the name of connectivity file")
        edges = pd.read_csv(
            input_path / config.target.graph_file, delimiter=",", header=None
        ).values
        edges = [list(edge.tolist()) for edge in edges]
        psi = TreeTensorNetwork(
            edges, tensors=isometries, gauge_tensor=singular_values, norm=state_norm
        )
        N = len(psi.physical_edges)

        max_bond_dim = np.max([iso.shape[2] for iso in isometries])

        init_bond_dim = 4
        if not isinstance(numerics.initial_bond_dimension, DotMap):
            init_bond_dim = int(numerics.initial_bond_dimension)

        truncated_singularvalues = 0.0
        if not isinstance(numerics.truncated_singularvalues, DotMap):
            truncated_singularvalues = float(numerics.truncated_singularvalues)

        ft = FactorizeTensor(
            psi,
            max_bond_dim=max_bond_dim,
        )

        ft.run(
            opt_fidelity=False,
            opt_structure=1,
            max_num_sweep=numerics.max_num_sweep,
            truncated_singularvalues=truncated_singularvalues,
        )
        nodes_list = {}
        for edge_id in ft.error.keys():
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
            ft.error[edge_id] = 0.0
            for t, edges in enumerate(ft.psi.edges):
                if edge_id == edges[0]:
                    ft.bond_dim[edge_id] = ft.psi.tensors[t].shape[0]
                if edge_id == edges[1]:
                    ft.bond_dim[edge_id] = ft.psi.tensors[t].shape[1]

        all_keys = set(nodes_list.keys())
        df = pd.DataFrame(
            [nodes_list[k] for k in all_keys],
            columns=["node1", "node2"],
            index=None,
        )
        df["entanglement"] = [ft.entanglement[k] for k in all_keys]
        df["error"] = [ft.error[k] for k in all_keys]
        df["bond"] = [ft.bond_dim[k] for k in all_keys]
        df.to_csv(path / "basic.csv", header=True, index=None)

        file_psi = path / "tensors"
        os.makedirs(file_psi, exist_ok=True)
        for i, iso in enumerate(ft.psi.tensors):
            np.save(file_psi / f"isometry{i}.npy", iso)
        np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
        np.save(file_psi / "norm.npy", state_norm)
        np.savetxt(file_psi / "edges.dat", ft.psi.edges, fmt="%d", delimiter=",")

    return 0

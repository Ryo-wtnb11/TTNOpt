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

    path = (
        Path(config.output.dir)
        if not isinstance(config.output.dir, DotMap)
        else Path("./")
    )
    os.makedirs(path, exist_ok=True)

    if isinstance(config.target.tensor, DotMap):
        print("=" * 50)
        print("⚠️  Error:")
        print("     Please set file or folder of the target.")
        print("=" * 50)
        exit()
    if not os.path.exists(str(config.target.tensor)):
        print("=" * 50)
        print("⚠️  Error:")
        print("     Please guarantee the file or folder of the target exist.")
        print("=" * 50)
        exit()

    save_tensors = (
        config.output.tensors
        if not isinstance(config.output.tensors, DotMap)
        else False
    )

    if os.path.isfile(str(config.target.tensor)):
        quantum_state = np.load(config.target.tensor)
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

        max_truncation_error = 0.0
        if not isinstance(numerics.max_truncation_error, DotMap):
            max_truncation_error = float(numerics.max_truncation_error)

        ft = FactorizeTensor(
            psi,
            max_bond_dim=init_bond_dim,
        )

        if not isinstance(numerics.opt_structure.type, DotMap):
            temperature = (
                float(numerics.opt_structure.temperature)
                if (
                    isinstance(numerics.opt_structure.temperature, float)
                    or isinstance(numerics.opt_structure.temperature, int)
                )
                else 0.0
            )
            seed = (
                numerics.opt_structure.seed
                if isinstance(numerics.opt_structure.seed, int)
                else 0
            )
            np.random.seed(seed)

            tau = (
                numerics.opt_structure.tau
                if isinstance(numerics.opt_structure.tau, int)
                else numerics.max_bond_dimension // 2
            )

            if numerics.opt_structure.type:
                max_num_sweep = 1
                if not isinstance(numerics.max_num_sweep, DotMap):
                    max_num_sweep = numerics.max_num_sweep
                ft.run(
                    target=quantum_state,
                    opt_fidelity=False,
                    opt_structure=numerics.opt_structure.type,
                    temperature=temperature,
                    tau=tau,
                    max_num_sweep=max_num_sweep,
                    max_truncation_error=max_truncation_error,
                )
            else:
                ft.run(
                    target=quantum_state,
                    opt_fidelity=False,
                    opt_structure=0,
                    max_num_sweep=1,
                    max_truncation_error=max_truncation_error,
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

        np.savetxt(path / "graph.dat", ft.psi.edges, fmt="%d", delimiter=",")
        if save_tensors:
            file_psi = path / "tensors_info"
            os.makedirs(file_psi, exist_ok=True)
            for i, iso in enumerate(ft.psi.tensors):
                np.save(file_psi / f"tensor{i}.npy", iso)
            np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
            np.save(file_psi / "norm.npy", state_norm)

        opt_fidelity = (
            True
            if not isinstance(numerics.fidelity.opt_structure.type, DotMap)
            else False
        )

        if opt_fidelity:
            opt_structure = numerics.fidelity.opt_structure.type
            temperature = (
                float(numerics.fidelity.opt_structure.temperature)
                if (
                    isinstance(numerics.fidelity.opt_structure.temperature, float)
                    or isinstance(numerics.fidelity.opt_structure.temperature, int)
                )
                else 0.0
            )
            seed = (
                numerics.opt_structure.seed
                if isinstance(numerics.fidelity.opt_structure.seed, int)
                else 0
            )
            np.random.seed(seed)

            tau = (
                numerics.fidelity.opt_structure.tau
                if isinstance(numerics.fidelity.opt_structure.tau, int)
                else numerics.fidelity.max_bond_dimensions[0] // 2
            )

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
                    temperature=temperature,
                    tau=tau,
                    max_num_sweep=max_num_sweep,
                    max_truncation_error=max_truncation_error,
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

                np.savetxt(
                    path_ / "graph.dat",
                    ft.psi.edges,
                    fmt="%d",
                    delimiter=",",
                )
                if save_tensors:
                    file_psi = Path(path_) / "tensors_info"
                    os.makedirs(file_psi, exist_ok=True)
                    for i, iso in enumerate(ft.psi.tensors):
                        np.save(file_psi / f"isometry{i}.npy", iso)
                    np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
                    np.save(file_psi / "norm.npy", state_norm)

    elif os.path.isdir(str(config.target.tensor)):
        input_path = Path(config.target.tensor)
        isometries = list(input_path.glob("isometry*.npy"))
        if isometries == []:
            raise FileNotFoundError(
                f"No files found files named as 'tensor' in {str(config.target.tensor)}"
            )
        isometries.sort(key=lambda x: int(x.stem.split("isometry")[-1]))
        isometries = [np.load(iso) for iso in isometries]
        singular_values = np.load(input_path / "singular_values.npy")
        state_norm = np.load(input_path / "norm.npy")
        graph_file = config.target.graph
        if os.path.isfile(str(config.target.graph)):
            edges = pd.read_csv(graph_file, delimiter=",", header=None).values
        else:
            print("=" * 50)
            print("⚠️  Error:")
            print("     target.graph is must be graph file.")
            print("=" * 50)
            exit()
        edges = [list(edge.tolist()) for edge in edges]
        psi = TreeTensorNetwork(
            edges, tensors=isometries, gauge_tensor=singular_values, norm=state_norm
        )
        N = len(psi.physical_edges)

        max_bond_dim = np.max([iso.shape[2] for iso in isometries])

        init_bond_dim = 4
        if not isinstance(numerics.initial_bond_dimension, DotMap):
            init_bond_dim = int(numerics.initial_bond_dimension)

        max_truncation_error = 0.0
        if not isinstance(numerics.max_truncation_error, DotMap):
            max_truncation_error = float(numerics.max_truncation_error)

        ft = FactorizeTensor(
            psi,
            max_bond_dim=max_bond_dim,
        )

        opt_structure = numerics.fidelity.opt_structure.type

        temperature = (
            float(numerics.opt_structure.temperature)
            if (
                isinstance(numerics.opt_structure.temperature, float)
                or isinstance(numerics.opt_structure.temperature, int)
            )
            else 0.0
        )
        seed = (
            numerics.opt_structure.seed
            if isinstance(numerics.opt_structure.seed, int)
            else 0
        )
        np.random.seed(seed)

        tau = (
            numerics.opt_structure.tau
            if isinstance(numerics.opt_structure.tau, int)
            else numerics.max_num_sweep // 2
        )

        ft.run(
            opt_fidelity=False,
            opt_structure=opt_structure,
            temperature=temperature,
            tau=tau,
            max_num_sweep=numerics.max_num_sweep,
            max_truncation_error=max_truncation_error,
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

        path
        df.to_csv(path / "basic.csv", header=True, index=None)
        np.savetxt(path / "graph.dat", ft.psi.edges, fmt="%d", delimiter=",")

        if save_tensors:
            file_psi = path / "tensors_info"
            os.makedirs(file_psi, exist_ok=True)
            for i, iso in enumerate(ft.psi.tensors):
                np.save(file_psi / f"isometry{i}.npy", iso)
            np.save(file_psi / "singular_values.npy", ft.psi.gauge_tensor)
            np.save(file_psi / "norm.npy", state_norm)

    return 0

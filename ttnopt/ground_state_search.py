# ground_state_search.py
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch
from ttnopt.src import GroundStateSearchSparse

from ttnopt.hamiltonian import hamiltonian

import pandas as pd
import itertools
import yaml
import argparse
from dotmap import DotMap
from pathlib import Path
import os


def ground_state_search():
    parser = argparse.ArgumentParser(description="Ground state search simulation")
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    psi = TreeTensorNetwork.mps(config.system.N)
    ham = hamiltonian(config.system)

    numerics = config.numerics
    opt_structure = 1 if numerics.opt_structure.active else 0
    edge_op_at_edge = None
    block_ham_at_edge = None

    path = Path(config.output.dir)

    save_basic = True if config.output.basic.active else False
    save_onesite_expval = True if config.output.single_site.active else False
    save_twosite_expval = True if config.output.two_site.active else False
    if save_basic is True and config.output.basic.file is DotMap():
        raise ValueError("Please input basic path file")
    if save_onesite_expval is True and config.output.single_site.file is DotMap():
        raise ValueError("Please input single site path file")
    if save_twosite_expval is True and config.output.two_site.file is DotMap():
        raise ValueError("Please input two site path file")

    for i, (max_bond_dim, max_num_sweep) in enumerate(
        zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)
    ):
        if numerics.U1_symmetry.active:
            gss = GroundStateSearchSparse(
                psi,
                ham,
                numerics.U1_symmetry.magnetization,
                init_bond_dim=numerics.initial_bond_dimension,
                max_bond_dim=max_bond_dim,
                truncation_error=numerics.truncation_error,
                edge_spin_operators=edge_op_at_edge,
                block_hamiltonians=block_ham_at_edge,
            )

            energy, entanglement, _ = gss.run(
                opt_structure=opt_structure,
                energy_convergence_threshold=float(
                    numerics.energy_convergence_threshold
                ),
                entanglement_convergence_threshold=float(
                    numerics.entanglement_convergence_threshold
                ),
                max_num_sweep=max_num_sweep,
            )

        else:
            gss = GroundStateSearch(
                psi,
                ham,
                init_bond_dim=numerics.initial_bond_dimension,
                max_bond_dim=max_bond_dim,
                truncation_error=numerics.truncation_error,
                edge_spin_operators=edge_op_at_edge,
                block_hamiltonians=block_ham_at_edge,
            )

            if i == 0:
                gss.run(
                    opt_structure=opt_structure,
                    energy_convergence_threshold=float(
                        numerics.energy_convergence_threshold
                    ),
                    entanglement_convergence_threshold=float(
                        numerics.entanglement_convergence_threshold
                    ),
                    max_num_sweep=max_num_sweep,
                )
                # re-run the first iteration to save the expectation values
                gss.run(
                    opt_structure=False,
                    max_num_sweep=1,
                    eval_onesite_expval=save_onesite_expval,
                    eval_twosite_expval=save_twosite_expval,
                )
            else:
                gss.run(
                    opt_structure=False,
                    energy_convergence_threshold=float(
                        numerics.energy_convergence_threshold
                    ),
                    entanglement_convergence_threshold=float(
                        numerics.entanglement_convergence_threshold
                    ),
                    max_num_sweep=max_num_sweep,
                    eval_onesite_expval=save_onesite_expval,
                    eval_twosite_expval=save_twosite_expval,
                )

        # reset parameters for the next iteration
        edge_op_at_edge = gss.edge_spin_operators
        block_ham_at_edge = gss.block_hamiltonians

        nodes_list = {}
        for edge_id in gss.energy.keys():
            tmp = []
            for node_id, edges in enumerate(psi.edges):
                node_id += config.system.N
                if edge_id in edges:
                    tmp.append(node_id)
            nodes_list[edge_id] = tmp

        for edge_id in psi.physical_edges:
            tmp = []
            tmp.append(edge_id)
            for node_id, edges in enumerate(psi.edges):
                node_id += config.system.N
                if edge_id in edges:
                    tmp.append(node_id)
            nodes_list[edge_id] = tmp
            gss.energy[edge_id] = 0.0
            gss.error[edge_id] = 0.0

        if save_basic is True:
            all_keys = set(nodes_list.keys())
            df = pd.DataFrame(
                [nodes_list[k] for k in all_keys],
                columns=["node1", "node2"],
                index=None,
            )
            df["energy"] = [gss.energy[k] for k in all_keys]
            df["entanglement"] = [gss.entanglement[k] for k in all_keys]
            df["error"] = [gss.error[k] for k in all_keys]

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)

            df.to_csv(path_ / config.output.basic.file, header=True, index=None)

        if save_onesite_expval is True:
            df = pd.DataFrame(psi.physical_edges, columns=["site"], index=None)
            df["Sx"] = [
                gss.one_site_expval[edge_id]["Sx"] for edge_id in psi.physical_edges
            ]
            df["Sy"] = [
                gss.one_site_expval[edge_id]["Sy"] for edge_id in psi.physical_edges
            ]
            df["Sz"] = [
                gss.one_site_expval[edge_id]["Sz"] for edge_id in psi.physical_edges
            ]
            df.to_csv(path_ / config.output.single_site.file, header=True, index=None)

        if save_twosite_expval is True:
            pairs = [(i, j) for i, j in itertools.combinations(psi.physical_edges, 2)]
            df = pd.DataFrame(pairs, columns=["site1", "site2"], index=None)
            df["SxSx"] = [gss.two_site_expval[pair]["SxSx"] for pair in pairs]
            df["SySy"] = [gss.two_site_expval[pair]["SySy"] for pair in pairs]
            df["SzSz"] = [gss.two_site_expval[pair]["SzSz"] for pair in pairs]
            df["SxSy"] = [gss.two_site_expval[pair]["SxSy"] for pair in pairs]
            df["SySx"] = [gss.two_site_expval[pair]["SySx"] for pair in pairs]
            df["SySz"] = [gss.two_site_expval[pair]["SySz"] for pair in pairs]
            df["SzSy"] = [gss.two_site_expval[pair]["SzSy"] for pair in pairs]
            df["SzSx"] = [gss.two_site_expval[pair]["SzSx"] for pair in pairs]
            df["SxSz"] = [gss.two_site_expval[pair]["SxSz"] for pair in pairs]
            df.to_csv(path_ / config.output.two_site.file, header=True, index=None)

    return 0

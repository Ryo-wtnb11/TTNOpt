# ground_state_search.py
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch
from ttnopt.src import GroundStateSearchSparse

from ttnopt.hamiltonian import hamiltonian

import numpy as np
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
    if config.numerics.init_tree:
        if config.system.N > 0 and (config.system.N & (config.system.N - 1)) == 0:
            psi = TreeTensorNetwork.tree(config.system.N)
        else:
            print("=" * 50)
            print("⚠️  Note: N is not a power of 2.")
            print("     Using MPS structure as the default.")
            print("=" * 50)

    ham = hamiltonian(config.system)

    numerics = config.numerics
    opt_structure = 1 if numerics.opt_structure else 0
    edge_op_at_edge = None
    block_ham_at_edge = None

    path = Path(config.output.dir)

    save_basic = True if not isinstance(config.output.basic.file, DotMap) else False
    save_onesite_expval = (
        True if not isinstance(config.output.single_site.file, DotMap) else False
    )
    save_twosite_expval = (
        True if not isinstance(config.output.two_site.file, DotMap) else False
    )
    u1_symmetry = (
        True if not isinstance(numerics.U1_symmetry.magnitude, DotMap) else False
    )
    if u1_symmetry and config.model.type == "XYZ":
        raise ValueError(
            "U1 symmetry is not supported for the XYZ model. Please set the U1 symmetry to False or change the model to XXZ."
        )

    for i, (max_bond_dim, max_num_sweep) in enumerate(
        zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)
    ):
        if u1_symmetry:
            gss = GroundStateSearchSparse(
                psi,
                ham,
                numerics.U1_symmetry.magnitude,
                init_bond_dim=numerics.initial_bond_dimension,
                max_bond_dim=max_bond_dim,
                truncation_error=numerics.truncation_error,
                edge_spin_operators=edge_op_at_edge,
                block_hamiltonians=block_ham_at_edge,
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

        if opt_structure:
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
            print("Calculating the expectation values for the initial structure")
            # re-run the first iteration to save the expectation values
            gss.run(
                opt_structure=False,
                max_num_sweep=1,
                eval_onesite_expval=save_onesite_expval,
                eval_twosite_expval=save_twosite_expval,
            )
            opt_structure = 0
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

            if not isinstance(config.output.basic.file, DotMap):
                df.to_csv(path_ / config.output.basic.file, header=True, index=None)
            else:
                df.to_csv(path_ / "basic", header=True, index=None)

        if save_onesite_expval is True:
            df = pd.DataFrame(psi.physical_edges, columns=["site"], index=None)
            sp = np.zeros(len(psi.physical_edges))
            sm = np.zeros(len(psi.physical_edges))
            if not u1_symmetry:
                sp = np.array(
                    [
                        gss.one_site_expval[edge_id]["S+"]
                        for edge_id in psi.physical_edges
                    ]
                )
                sm = np.array(
                    [
                        gss.one_site_expval[edge_id]["S-"]
                        for edge_id in psi.physical_edges
                    ]
                )
            df["Sx"] = (sp + sm) / 2.0
            df["Sy"] = (sp - sm) / 2.0j
            df["Sz"] = [
                gss.one_site_expval[edge_id]["Sz"] for edge_id in psi.physical_edges
            ]

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)
            df.to_csv(path_ / config.output.single_site.file, header=True, index=None)

        if save_twosite_expval is True:
            pairs = [(i, j) for i, j in itertools.combinations(psi.physical_edges, 2)]
            df = pd.DataFrame(pairs, columns=["site1", "site2"], index=None)
            spp = np.zeros(len(pairs))
            smm = np.zeros(len(pairs))
            szp = np.zeros(len(pairs))
            spz = np.zeros(len(pairs))
            szm = np.zeros(len(pairs))
            smz = np.zeros(len(pairs))
            if not u1_symmetry:
                spp = np.array([gss.two_site_expval[pair]["S+S+"] for pair in pairs])
                smm = np.array([gss.two_site_expval[pair]["S-S-"] for pair in pairs])
                szp = np.array([gss.two_site_expval[pair]["SzS+"] for pair in pairs])
                spz = np.array([gss.two_site_expval[pair]["S+Sz"] for pair in pairs])
                szm = np.array([gss.two_site_expval[pair]["SzS-"] for pair in pairs])
                smz = np.array([gss.two_site_expval[pair]["S-Sz"] for pair in pairs])

            szz = np.array([gss.two_site_expval[pair]["SzSz"] for pair in pairs])
            spm = np.array([gss.two_site_expval[pair]["S+S-"] for pair in pairs])
            smp = np.array([gss.two_site_expval[pair]["S-S+"] for pair in pairs])

            df["SzSz"] = szz
            df["SxSx"] = (spp + spm + smp + smm) / 4.0
            df["SySy"] = -(spp - spm - smp + smm) / 4.0

            df["SxSy"] = (spp - spm + smp - smm) / 4.0j
            df["SySx"] = (spp + spm - smp - smm) / 4.0j

            df["SzSx"] = (szp + szm) / 2.0
            df["SxSz"] = (spz - smz) / 2.0

            df["SzSy"] = (szp - szm) / 2.0j
            df["SySz"] = (spz + smz) / 2.0j

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)
            df.to_csv(path_ / config.output.two_site.file, header=True, index=None)

    return 0

from typing import List

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

    path = (
        Path(config.output.dir)
        if not isinstance(config.output.dir, DotMap)
        else Path("./")
    )

    save_onesite_expval = (
        config.output.single_site
        if not isinstance(config.output.single_site, DotMap)
        else False
    )
    save_twosite_expval = (
        config.output.two_site
        if not isinstance(config.output.two_site, DotMap)
        else False
    )

    u1_symmetry = True if not isinstance(numerics.U1_symmetry, DotMap) else False

    if u1_symmetry and config.model.type == "XYZ":
        print("=" * 50)
        print("⚠️  Error: U1 symmetry is not supported for the XYZ model.")
        print("     Please set the U1 symmetry to False or change the model to XXZ.")
        print("=" * 50)
        exit()
    if u1_symmetry and not (isinstance(config.system.MF_X, DotMap)):
        print("=" * 50)
        print("⚠️  Error: U1 symmetry is not allowed for the X magnetic field.")
        print("=" * 50)
        exit()
    if u1_symmetry and not (isinstance(config.system.MF_Y, DotMap)):
        print("=" * 50)
        print("⚠️  Error: U1 symmetry is not allowed for the Y magnetic field.")
        print("=" * 50)
        exit()
    if u1_symmetry and not isinstance(config.system.DM_X, DotMap):
        print("=" * 50)
        print(
            "⚠️  Error: U1 symmetry is not allowed for the X term of Dzyaloshinskii-Moriya interaction."
        )
        print("=" * 50)
        exit()
    if u1_symmetry and not isinstance(config.system.DM_Y, DotMap):
        print("=" * 50)
        print(
            "⚠️  Error: U1 symmetry is not allowed for the Y term of Dzyaloshinskii-Moriya interaction."
        )
        print("=" * 50)
        exit()
    if u1_symmetry and not isinstance(config.system.SOD_X, DotMap):
        print("=" * 50)
        print(
            "⚠️  Error: U1 symmetry is not allowed for the X term of symmetric off-diagonal exchange interaction."
        )
        print("=" * 50)
        exit()
    if u1_symmetry and not isinstance(config.system.SOD_Y, DotMap):
        print("=" * 50)
        print(
            "⚠️  Error: U1 symmetry is not allowed for the Y term of symmetric off-diagonal exchange interaction."
        )
        print("=" * 50)
        exit()
    if u1_symmetry and not isinstance(config.system.SOD_Z, DotMap):
        print("=" * 50)
        print(
            "⚠️  Error: U1 symmetry is not allowed for the Y term of symmetric off-diagonal exchange interaction."
        )
        print("=" * 50)
        exit()

    opt_structure = numerics.opt_structure.type
    temperature = (
        numerics.opt_structure.temperature
        if isinstance(numerics.opt_structure.temperature, List)
        else [0.0, 0.0]
    )

    seed = (
        numerics.opt_structure.seed
        if isinstance(numerics.opt_structure.beta, int)
        else 0
    )
    if isinstance(numerics.energy_degeneracy_threshold, DotMap):
        energy_degeneracy_threshold = 1.0e-8
    else:
        energy_degeneracy_threshold = float(numerics.energy_degeneracy_threshold)
    if isinstance(numerics.entanglement_degeneracy_threshold, DotMap):
        entanglement_degeneracy_threshold = 1.0e-8
    else:
        entanglement_degeneracy_threshold = float(
            numerics.entanglement_degeneracy_threshold
        )
    if isinstance(numerics.energy_convergence_threshold):
        energy_convergence_threshold = 1.0e-8
    else:
        energy_convergence_threshold = float(numerics.energy_convergence_threshold)
    if isinstance(numerics.entanglement_convergence_threshold):
        entanglement_convergence_threshold = 1.0e-8
    else:
        entanglement_convergence_threshold = float(
            numerics.entanglement_convergence_threshold
        )

    np.random.seed(seed)

    if u1_symmetry:
        gss = GroundStateSearchSparse(
            psi,
            ham,
            numerics.U1_symmetry,
            init_bond_dim=numerics.initial_bond_dimension,
            max_bond_dim=numerics.max_bond_dimensions[0],
            energy_degeneracy_threshold=energy_degeneracy_threshold,
            entanglement_degeneracy_threshold=entanglement_degeneracy_threshold,
        )
        if gss.init_u1_num != gss.u1_num:
            sz_sign = np.sign(gss.u1_num - gss.init_u1_num)
            sz_diff = abs(gss.u1_num - gss.init_u1_num) // 2
            gss.run(
                opt_structure=0,
                max_num_sweep=sz_diff,
                energy_convergence_threshold=0.0,
                entanglement_convergence_threshold=0.0,
                sz_sign=sz_sign,
            )
            print(f"Complete the warmup sweeps for M={gss.u1_num}/2.")
    else:
        gss = GroundStateSearch(
            psi,
            ham,
            init_bond_dim=numerics.initial_bond_dimension,
            max_bond_dim=numerics.initial_bond_dimension,
            energy_degeneracy_threshold=energy_degeneracy_threshold,
            entanglement_degeneracy_threshold=entanglement_degeneracy_threshold,
        )

    for i, (max_bond_dim, max_num_sweep) in enumerate(
        zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)
    ):
        gss.max_bond_dim = max_bond_dim
        if opt_structure:
            gss.run(
                opt_structure=opt_structure,
                energy_convergence_threshold=energy_convergence_threshold,
                entanglement_convergence_threshold=entanglement_convergence_threshold,
                max_num_sweep=max_num_sweep,
                beta=beta,
            )
            print("Calculating the expectation values for the initial structure")
            # re-run the first iteration to save the expectation values
            gss.run(
                opt_structure=0,
                max_num_sweep=1,
                eval_onesite_expval=save_onesite_expval,
                eval_twosite_expval=save_twosite_expval,
            )
        else:
            gss.run(
                opt_structure=0,
                energy_convergence_threshold=energy_convergence_threshold,
                entanglement_convergence_threshold=entanglement_convergence_threshold,
                max_num_sweep=max_num_sweep,
                eval_onesite_expval=save_onesite_expval,
                eval_twosite_expval=save_twosite_expval,
            )

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
        df.to_csv(path_ / "basic.csv", header=True, index=None)

        if save_onesite_expval:
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
            df["Sx"] = np.real((sp + sm) / 2.0)
            df["Sy"] = np.real((sp - sm) / 2.0j)
            df["Sz"] = np.real(
                [gss.one_site_expval[edge_id]["Sz"] for edge_id in psi.physical_edges]
            )

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)
            df.to_csv(path_ / "single_site.csv", header=True, index=None)

        if save_twosite_expval:
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

            df["SxSx"] = np.real((spp + spm + smp + smm) / 4.0)
            df["SySy"] = np.real((-spp + spm + smp - smm) / 4.0)
            df["SzSz"] = np.real(szz)

            df["SySz"] = np.real((spz - smz) / 2.0j)
            df["SzSy"] = np.real((szp - szm) / 2.0j)

            df["SzSx"] = np.real((szp + szm) / 2.0)
            df["SxSz"] = np.real((spz + smz) / 2.0)

            df["SxSy"] = np.real((spp - spm + smp - smm) / 4.0j)
            df["SySx"] = np.real((spp + spm - smp - smm) / 4.0j)

            path_ = path / f"run{i + 1}"
            os.makedirs(path_, exist_ok=True)
            df.to_csv(path_ / "two_site.csv", header=True, index=None)

    return 0

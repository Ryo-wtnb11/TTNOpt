# ground_state_search.py
from ttnopt.src import TreeTensorNetwork
from ttnopt.src import GroundStateSearch
from ttnopt.src import GroundStateSearchSparse

from ttnopt.hamiltonian import hamiltonian

import pandas as pd
import yaml
import argparse
from dotmap import DotMap
from pathlib import Path
import itertools

def ground_state_search():
    parser = argparse.ArgumentParser(description="Ground state search simulation")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()


    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = DotMap(config)

    psi = TreeTensorNetwork.mps(config.system.N)
    ham = hamiltonian(config.system)

    numerics = config.numerics
    opt_structure = numerics.opt_structure.active
    edge_op_at_edge = None
    block_ham_at_edge = None
    for i, (max_bond_dim, max_num_sweep) in enumerate(zip(numerics.max_bond_dimensions, numerics.max_num_sweeps)):
        if numerics.U1_symmetry.active:
            gss = GroundStateSearchSparse(
                psi,
                ham,
                numerics.U1_symmetry.magnetization,
                init_bond_dim=numerics.initial_bond_dimension,
                max_bond_dim=max_bond_dim,
                truncation_error=numerics.truncation_error,
                edge_spin_operators=edge_op_at_edge,
                block_hamiltonians=block_ham_at_edge
            )

            energy, entanglement = gss.run(
                opt_structure=opt_structure,
                energy_convergence_threshold=float(numerics.energy_convergence_threshold),
                entanglement_convergence_threshold=float(numerics.entanglement_convergence_threshold),
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
                block_hamiltonians=block_ham_at_edge
            )
            energy, entanglement = gss.run(
                opt_structure=opt_structure,
                energy_convergence_threshold=float(numerics.energy_convergence_threshold),
                entanglement_convergence_threshold=float(numerics.entanglement_convergence_threshold),
                max_num_sweep=max_num_sweep,
            )

        # reset parameters for the next iteration
        opt_structure = 0
        edge_op_at_edge = gss.edge_spin_operators
        block_ham_at_edge = gss.block_hamiltonians


    nodes_list = []
    for edge_id in energy.keys():
        tmp = []
        for node_id, edges in enumerate(psi.edges):
            node_id += config.system.N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)

    for edge_id in psi.physical_edges:
        tmp = []
        tmp.append(edge_id)
        for node_id, edges in enumerate(psi.edges):
            node_id += config.system.N
            if edge_id in edges:
                tmp.append(node_id)
        nodes_list.append(tmp)
        energy[edge_id] = 0.0

    if config.output.basic_file is not DotMap():
        df = pd.DataFrame(nodes_list, columns=['node1', 'node2'], index=None)
        if config.output.energy.active:
            df['energy'] = energy.values()
        if config.output.entanglement.active:
            df['entanglement'] = entanglement.values()

        path = Path(config.output.dir)
        df.to_csv(path / config.output.basic_file, header=True, index=None)

    if config.output.single_site_file is not DotMap():
        df = pd.DataFrame(psi.physical_edges, columns=['site'], index=None)

        if config.output.single_site.Sx.active:
            sx = [gss.calculate_expval(edge_id, "Sx") for edge_id in psi.physical_edges]
            df['Sx'] = sx
        if config.output.single_site.Sy.active:
            sy = [gss.calculate_expval(edge_id, "Sy") for edge_id in psi.physical_edges]
            df['Sy'] = sy
        if config.output.single_site.Sz.active:
            sz = [gss.calculate_expval(edge_id, "Sz") for edge_id in psi.physical_edges]
            df['Sz'] = sz
        df.to_csv(path / config.output.single_site_file, header=True, index=None)

    if config.output.two_site_file is not DotMap():
        pairs =  [(i, j) for i, j in itertools.combinations(psi.physical_edges, 2)]
        df = pd.DataFrame(pairs, columns=['site1', 'site2'], index=None)
        if config.output.two_site.SxSx.active:
            xx = [gss.calculate_expval(pair, ["Sx", "Sx"]) for pair in pairs]
            df['SxSx'] = xx
        if config.output.two_site.SySy.active:
            yy = [gss.calculate_expval(pair, ["Sy", "Sy"]) for pair in pairs]
            df['SySy'] = yy
        if config.output.two_site.SzSz.active:
            zz = [gss.calculate_expval(pair, ["Sz", "Sz"]) for pair in pairs]
            df['SzSz'] = zz
        if config.output.two_site.SySz.active:
            yz = [gss.calculate_expval(pair, ["Sy", "Sz"]) for pair in pairs]
            df['SySz'] = yz
        if config.output.two_site.SzSy.active:
            zy = [gss.calculate_expval(pair, ["Sz", "Sy"]) for pair in pairs]
            df['SzSy'] = zy
        if config.output.two_site.SzSx.active:
            zx = [gss.calculate_expval(pair, ["Sz", "Sx"]) for pair in pairs]
            df['SzSx'] = zx
        if config.output.two_site.SzSx.active:
            xz = [gss.calculate_expval(pair, ["Sx", "Sz"]) for pair in pairs]
            df['SxSz'] = xz
        if config.output.two_site.SxSy.active:
            xy = [gss.calculate_expval(pair, ["Sx", "Sy"]) for pair in pairs]
            df['SxSy'] = xy
        if config.output.two_site.SySx.active:
            yx = [gss.calculate_expval(pair, ["Sy", "Sx"]) for pair in pairs]
            df['SySx'] = yx
        df.to_csv(path / config.output.two_site_file, header=True, index=None)

    return 0
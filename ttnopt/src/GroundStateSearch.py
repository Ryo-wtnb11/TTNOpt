from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import tensornetwork as tn

from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.PhysicsEngine import PhysicsEngine
from ttnopt.src.TTN import TreeTensorNetwork


class GroundStateSearch(PhysicsEngine):
    """A class for ground state search algorithm based on DMRG.
    Args:
        psi (TreeTensorNetwork): The quantum state.
        hamiltonians (Hamiltonian): Hamiltonian which is list of Observable.
        init_bond_dim (int, optional): Initial bond dimension. Defaults to 4.
        max_bond_dim (int, optional): Maximum bond dimension. Defaults to 16.
    """

    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        init_bond_dim: int = 4,
        max_bond_dim: int = 16,
        energy_degeneracy_threshold: float = 1e-13,
        entanglement_degeneracy_threshold: float = 0.1,
    ):
        """Initialize a DMRG object.

        Args:
            psi : The quantum state.
            hamiltonians : The Hamiltonian.
            init_bond_dim : Initial bond dimension.
            max_bond_dim : Maximum bond dimension.
            truncation_error : Maximum truncation error.
        """
        self.energy: Dict[int, float] = {}
        self.entanglement: Dict[int, float] = {}
        self.error: Dict[int, float] = {}
        self.one_site_expval: Dict[int, Dict[str, float]] = {}
        self.two_site_expval: Dict[Tuple[int, int], Dict[str, float]] = {}

        super().__init__(
            psi,
            hamiltonian,
            init_bond_dim,
            max_bond_dim,
            energy_degeneracy_threshold,
            entanglement_degeneracy_threshold,
        )

    def run(
        self,
        opt_structure: int = 0,
        energy_convergence_threshold: float = 1e-8,
        entanglement_convergence_threshold: float = 1e-8,
        max_num_sweep: int = 10,
        converged_count: int = 2,
        eval_onesite_expval: bool = False,
        eval_twosite_expval: bool = False,
        temperature: float = 0.0,
        tau: int = 0,
    ):
        """Run DMRG algorithm.

        Args:
            opt_structure (bool, optional): If optimize the tree structure or not. Defaults to False.
            energy_convergence_threshold (float, optional): Energy threshold for convergence. Defaults to 1e-8.
            entanglement_convergence_threshold (float, optional): Entanglement entropy threshold for automatic optimization. Defaults to 1e-8.
            converged_count (int, optional): Converged count. Defaults to 1.
            eval_onesite_expval (bool): If evaluate one-site expectation value or not.
            eval_twosite_expval (bool): If evaluate two-site expectation value or not.
        """
        energy_at_edge: Dict[int, float] = {}
        _energy_at_edge: Dict[int, float] = {}
        ee_at_edge: Dict[int, float] = {}
        _ee_at_edge: Dict[int, float] = {}
        _error_at_edge: Dict[int, float] = {}
        onesite_expval: Dict[int, Dict[str, float]] = {}
        twosite_expval: Dict[Tuple[int, int], Dict[str, float]] = {}

        edges, _edges = deepcopy(self.psi.edges), deepcopy(self.psi.edges)

        converged_num = 0

        if tau == 0:
            tau = max_num_sweep // 2 + 1
        for sweep_num in range(max_num_sweep):
            temp = temperature * (2 ** (-sweep_num / tau))
            if converged_num > converged_count:
                break

            energy_at_edge = deepcopy(_energy_at_edge)
            ee_at_edge = deepcopy(_ee_at_edge)
            edges = deepcopy(_edges)

            self.distance = self.initial_distance()
            self.flag = self.initial_flag()

            (
                _edge_id,
                _selected_tensor_id,
                _connected_tensor_id,
                _not_selected_tensor_id,
            ) = self.local_two_tensor()

            print("Sweep count: " + str(sweep_num + 1))
            while True:
                edge_id = _edge_id
                selected_tensor_id = _selected_tensor_id
                connected_tensor_id = _connected_tensor_id
                not_selected_tensor_id = _not_selected_tensor_id
                # absorb gauge tensor
                iso = tn.Node(self.psi.tensors[selected_tensor_id])
                gauge = tn.Node(self.psi.gauge_tensor)
                iso[2] ^ gauge[0]
                iso = tn.contractors.auto(
                    [iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]]
                )
                self.psi.tensors[selected_tensor_id] = iso.get_tensor()

                self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

                self._set_edge_spin(not_selected_tensor_id)

                self._set_block_hamiltonian(not_selected_tensor_id)

                ground_state_order = [selected_tensor_id, connected_tensor_id]
                ground_state, energy = self.lanczos(ground_state_order)
                psi_edges = (
                    self.psi.edges[selected_tensor_id][:2]
                    + self.psi.edges[connected_tensor_id][:2]
                )

                u, s, v, probability, error, edge_order = self.decompose_two_tensors(
                    ground_state,
                    self.max_bond_dim,
                    opt_structure=opt_structure,
                    temperature=temp,
                    operate_degeneracy=True,
                    epsilon=entanglement_convergence_threshold,
                    delta=self.entanglement_degeneracy_threshold,
                )

                self.psi.tensors[selected_tensor_id] = u
                self.psi.tensors[connected_tensor_id] = v
                self.psi.gauge_tensor = s
                (
                    self.psi.edges[selected_tensor_id][0],
                    self.psi.edges[selected_tensor_id][1],
                ) = (
                    psi_edges[edge_order[0]],
                    psi_edges[edge_order[1]],
                )
                (
                    self.psi.edges[connected_tensor_id][0],
                    self.psi.edges[connected_tensor_id][1],
                ) = (
                    psi_edges[edge_order[2]],
                    psi_edges[edge_order[3]],
                )

                self.distance = self.initial_distance()
                _energy_at_edge[self.psi.canonical_center_edge_id] = energy
                print(energy)
                ee = self.entanglement_entropy(probability)
                _ee_at_edge[self.psi.canonical_center_edge_id] = ee
                ee_dict = self.entanglement_entropy_at_physical_bond(
                    ground_state, psi_edges
                )
                for key in ee_dict.keys():
                    _ee_at_edge[key] = ee_dict[key]
                _error_at_edge[self.psi.canonical_center_edge_id] = error

                if self.candidate_edge_ids() == []:
                    break

                (
                    _edge_id,
                    _selected_tensor_id,
                    _connected_tensor_id,
                    _not_selected_tensor_id,
                ) = self.local_two_tensor()
                self.set_flag(_not_selected_tensor_id)
                # eval expval
                if self.flag[_not_selected_tensor_id]:
                    if eval_onesite_expval:
                        onesite_expval_dict = self.expval_onesite(
                            _not_selected_tensor_id,
                            ground_state,
                            ground_state_order,
                        )
                        for key in onesite_expval_dict.keys():
                            onesite_expval[key] = onesite_expval_dict[key]
                    if eval_twosite_expval:
                        twosite_expval_dict = self.expval_twosite(
                            _not_selected_tensor_id,
                            ground_state,
                            ground_state_order,
                        )
                        for key in twosite_expval_dict.keys():
                            twosite_expval[key] = twosite_expval_dict[key]

            if eval_onesite_expval:
                for i in self.psi.central_tensor_ids():
                    onesite_expval_dict = self.expval_onesite(
                        i, ground_state, ground_state_order
                    )
                    for key in onesite_expval_dict.keys():
                        onesite_expval[key] = onesite_expval_dict[key]
            if eval_twosite_expval:
                for i in self.psi.central_tensor_ids():
                    twosite_expval_dict = self.expval_twosite(
                        i, ground_state, ground_state_order
                    )
                    for key in twosite_expval_dict.keys():
                        twosite_expval[key] = twosite_expval_dict[key]
                twosite_expval_dict = self.expval_twosite_origin(
                    twosite_expval.keys(), ground_state, ground_state_order
                )
                for key in twosite_expval_dict.keys():
                    twosite_expval[key] = twosite_expval_dict[key]

            _edges = deepcopy(self.psi.edges)

            sweep_num += 1
            if sweep_num > 2:
                diff_energy = [
                    np.abs(energy_at_edge[key] - _energy_at_edge[key])
                    for key in energy_at_edge.keys()
                ]
                diff_ee = [
                    np.abs(ee_at_edge[key] - _ee_at_edge[key])
                    for key in ee_at_edge.keys()
                ]
                if all(
                    [
                        set(edge[:2]) == set(_edge[:2]) and edge[2] == _edge[2]
                        for edge, _edge in zip(edges, _edges)
                    ]
                ):
                    if all(
                        [
                            energy < energy_convergence_threshold
                            for energy in diff_energy
                        ]
                    ) and all(
                        [ee < entanglement_convergence_threshold for ee in diff_ee]
                    ):
                        converged_num += 1
        print("Converged")

        self.energy = _energy_at_edge
        self.entanglement = _ee_at_edge
        self.error = _error_at_edge
        self.one_site_expval = onesite_expval
        self.two_site_expval = twosite_expval
        return 0

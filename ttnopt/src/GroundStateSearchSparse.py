from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import tensornetwork as tn
from tensornetwork import BlockSparseTensor, Index, U1Charge

from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.PhysicsEngineSparse import PhysicsEngineSparse
from ttnopt.src.TTN import TreeTensorNetwork

# tn.block_sparse.enable_caching()


class GroundStateSearchSparse(PhysicsEngineSparse):
    """A class for ground state search algorithm based on DMRG using Sparse Tensor.
    Args:
        psi (TreeTensorNetwork): The quantum state.
        hamiltonians (Hamiltonian): Hamiltonian which is list of Observable.
        u1_num (str): The number of preserving total spin in U(1) symmetry.
        init_bond_dim (int, optional): Initial bond dimension. Defaults to 4.
        max_bond_dim (int, optional): Maximum bond dimension. Defaults to 16.
        truncation_error (float, optional): Maximum truncation error. Defaults to 1e-11.
    """

    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        u1_num: Union[int, str],
        init_bond_dim: int = 4,
        max_bond_dim: int = 16,
        energy_degeneracy_threshold: float = 1e-13,
        entanglement_degeneracy_threshold: float = 1e-8,
    ):
        """Initialize a DMRG object.

        Args:
            psi : The quantum state.
            hamiltonians : The Hamiltonian.
            u1_num : The number of preserving total spin in U(1) symmetry.
            init_bond_dim : Initial bond dimension.
            max_bond_dim : Maximum bond dimension.
            truncation_error : Maximum truncation error.
        """
        self.energy: Dict[int, float] = {}
        self.entanglement: Dict[int, float] = {}
        self.error: Dict[int, float] = {}
        self.one_site_expval: Dict[int, Dict[str, float]] = {}
        self.two_site_expval: Dict[Tuple[int, int], Dict[str, float]] = {}

        # set backend only in this function
        super().__init__(
            psi,
            hamiltonian,
            u1_num,
            init_bond_dim,
            max_bond_dim,
            energy_degeneracy_threshold,
            entanglement_degeneracy_threshold,
        )

    def run(
        self,
        opt_structure: int = 0,
        energy_convergence_threshold: float = 1e-10,
        entanglement_convergence_threshold: float = 1e-10,
        max_num_sweep: int = 10,
        converged_count: int = 2,
        eval_onesite_expval: bool = False,
        eval_twosite_expval: bool = False,
        sz_sign: int = 0,
        temperature: float = 0.0,
        tau: int = 1,
    ):
        """Run Ground State Search algorithm using Sparse Tensor with total spin.

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
            tau = max_num_sweep // 2
        for sweep_num in range(max_num_sweep):
            temp = temperature * (2 ** (-sweep_num / tau))
            if converged_num > converged_count:
                break
            # energy
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
                iso = tn.Node(
                    self.psi.tensors[selected_tensor_id], backend=self.backend
                )
                gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
                if selected_tensor_id == self.previous_id:
                    out = gauge[1]
                    iso[2] ^ gauge[0]
                else:
                    out = gauge[0]
                    iso[2] ^ gauge[1]
                iso = tn.contractors.auto(
                    [iso, gauge], output_edge_order=[iso[0], iso[1], out, gauge[2]]
                )

                self.psi.tensors[selected_tensor_id] = iso.tensor

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

                self.previous_id = selected_tensor_id

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

                if sz_sign != 0 and self.candidate_edge_ids() == []:
                    # absorb gauge tensor
                    iso = tn.Node(
                        self.psi.tensors[selected_tensor_id], backend=self.backend
                    )
                    c0 = (
                        self.psi.tensors[selected_tensor_id]
                        .flat_charges[2]
                        .charges.flatten()
                    )
                    c1 = (
                        self.psi.tensors[connected_tensor_id]
                        .flat_charges[2]
                        .charges.flatten()
                    )
                    gauge_tensor = BlockSparseTensor.random(
                        [
                            Index(U1Charge(c0), flow=True),
                            Index(U1Charge(c1), flow=True),
                            Index(
                                U1Charge([self.init_u1_num + int(sz_sign * 2)]),
                                flow=False,
                            ),
                        ]
                    )
                    gauge_tensor /= np.linalg.norm(gauge_tensor.data)
                    gauge = tn.Node(gauge_tensor, backend=self.backend)
                    self.init_u1_num += int(sz_sign * 2)
                    out = gauge[1]
                    iso[2] ^ gauge[0]
                    iso = tn.contractors.auto(
                        [iso, gauge], output_edge_order=[iso[0], iso[1], out, gauge[2]]
                    )
                    self.psi.tensors[selected_tensor_id] = iso.tensor
                    ground_state, energy = self.lanczos(
                        [selected_tensor_id, connected_tensor_id]
                    )
                    self.psi.tensors[selected_tensor_id] = iso.tensor
                    u, s, v, probability, error, edge_order = (
                        self.decompose_two_tensors(
                            ground_state,
                            self.max_bond_dim,
                            opt_structure=False,
                            epsilon=entanglement_convergence_threshold,
                            delta=self.entanglement_degeneracy_threshold,
                        )
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

            # eval expval
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
                    np.abs(1 - _energy_at_edge[key] / energy_at_edge[key])
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
